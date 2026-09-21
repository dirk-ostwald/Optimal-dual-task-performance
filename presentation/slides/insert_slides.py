"""Insert the rendered Beamer model slides as full-bleed PNGs into the deck.

Usage
-----
    python insert_slides.py

Reads the PNGs in ``png/`` produced by ``build.sh`` and inserts them as
full-bleed pictures on blank slides, after the "Agentic behavioral model"
overview slide. The deck is rewritten in place; copies are kept in
``backups/``. Close the deck in PowerPoint before running this.

Why this edits the zip container directly
-----------------------------------------
python-pptx cannot round-trip this deck. Slide 37 carries a dangling image
relationship (``Target="NULL"``) that PowerPoint tolerates but python-pptx
silently drops on save, leaving the slide's ``r:embed`` pointing at nothing.
PowerPoint then reports the file as damaged and discards the added slides
during repair. So instead of re-serializing the package, every original part
is copied byte for byte and only these are touched:

  * a new ``ppt/slides/slideN.xml`` plus its ``.rels`` per inserted slide
  * a new ``ppt/media/imageK.png`` per inserted slide
  * ``[Content_Types].xml``      - one Override per new slide part
  * ``ppt/_rels/presentation.xml.rels`` - one relationship per new slide
  * ``ppt/presentation.xml``     - one ``p:sldId`` per new slide

Reruns are idempotent: slides carrying ``SHAPE_TAG`` in their picture shape
name are recognized as ours and removed, together with their parts and
references, before the current PNGs go in.
"""

import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESENTATION_DIR = HERE.parent

DECK = PRESENTATION_DIR / "TP3-Kick-Off-Presentation.pptx"
BACKUP_DIR = HERE / "backups"
KEEP_BACKUPS = 3

# Slide number (1-based) the model slides are placed behind, counted after
# any previously inserted model slides have been removed. None appends them
# at the end of the deck.
INSERT_AFTER_SLIDE = 40

# PNGs to insert, in rendering order. pdftoppm zero-pads the page number to the
# width of the page count, so the names are read from disk rather than built.
PNG_DIR = HERE / "png"
PNG_GLOB = "slide-*.png"

BLANK_LAYOUT_NAME = "Leer"

# Placement of the rendered slide on the PowerPoint slide, in inches. The
# picture is inset so that the master's furniture stays visible: the blue
# vertical rule at 0.36in, the black horizontal rule at 5.17in, and the logo
# in the bottom-left corner, whose artwork reaches to about 0.70in. A side
# margin of 0.80in clears all three; the picture keeps the slide's 16:9 ratio,
# sits centred horizontally, and is centred vertically above the lower rule.
EMU_PER_INCH = 914400
MASTER_RULE_BOTTOM_IN = 5.17
SIDE_MARGIN_IN = 0.80

# Marks the picture shapes this script owns, so reruns can replace them.
SHAPE_TAG = "beamer-model-slide"

R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
SLIDE_CT = (
    "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
)

SLIDE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"\
 xmlns:r="{r}" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">\
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>\
</p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>\
<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>\
<p:pic><p:nvPicPr><p:cNvPr id="2" name="{name}"/><p:cNvPicPr>\
<a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>\
<p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>\
<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>\
<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>\
</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""

SLIDE_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\
<Relationship Id="rId1" Type="{r}/slideLayout" Target="../slideLayouts/{layout}"/>\
<Relationship Id="rId2" Type="{r}/image" Target="../media/{image}"/>\
</Relationships>"""


def part_number(name):
    return int(re.search(r"(\d+)\.xml$", name).group(1))


def blank_layout_for(parts, slide_part):
    """Blank layout belonging to the same master as ``slide_part``."""
    rels = parts[rels_name(slide_part)].decode("utf-8")
    layout = re.search(r'Target="\.\./slideLayouts/(slideLayout\d+\.xml)"', rels).group(1)
    master = re.search(
        r'Target="\.\./slideMasters/(slideMaster\d+\.xml)"',
        parts[rels_name(f"ppt/slideLayouts/{layout}")].decode("utf-8"),
    ).group(1)

    for name in sorted(parts):
        if not re.fullmatch(r"ppt/slideLayouts/slideLayout\d+\.xml", name):
            continue
        xml = parts[name].decode("utf-8")
        cSld = re.search(r"<p:cSld[^>]*name=\"([^\"]*)\"", xml)
        if not cSld or cSld.group(1) != BLANK_LAYOUT_NAME:
            continue
        own_master = re.search(
            r'Target="\.\./slideMasters/(slideMaster\d+\.xml)"',
            parts[rels_name(name)].decode("utf-8"),
        ).group(1)
        if own_master == master:
            return name.rsplit("/", 1)[1]

    raise SystemExit(f"no {BLANK_LAYOUT_NAME!r} layout found for {master}")


def picture_box(slide_cx, slide_cy):
    """Inset picture rectangle in EMU, as (x, y, cx, cy).

    Keeps the slide's aspect ratio, leaves SIDE_MARGIN_IN on either side, and
    centres the result vertically in the band above the master's lower rule.
    """
    cx = slide_cx - 2 * round(SIDE_MARGIN_IN * EMU_PER_INCH)
    cy = round(cx * slide_cy / slide_cx)

    band = round(MASTER_RULE_BOTTOM_IN * EMU_PER_INCH)
    if cy > band:  # too tall for the band: shrink about the slide centre
        cy = band
        cx = round(cy * slide_cx / slide_cy)

    return (slide_cx - cx) // 2, (band - cy) // 2, cx, cy


def rels_name(part):
    head, tail = part.rsplit("/", 1)
    return f"{head}/_rels/{tail}.rels"


def slide_order(presentation_xml):
    """Slide part numbers in presentation order, with their rIds."""
    lst = re.search(r"<p:sldIdLst>.*?</p:sldIdLst>", presentation_xml, re.S)
    return re.findall(r'<p:sldId id="(\d+)" r:id="(rId\d+)"\s*/>', lst.group())


def main():
    lock = DECK.with_name(f"~${DECK.name}")
    if lock.exists():
        raise SystemExit(f"{DECK.name} is open in PowerPoint - close it first")

    slide_pngs = sorted(PNG_DIR.glob(PNG_GLOB))
    if not slide_pngs:
        raise SystemExit(f"no rendered slides in {PNG_DIR}")

    BACKUP_DIR.mkdir(exist_ok=True)
    pristine = BACKUP_DIR / f"{DECK.stem}.orig{DECK.suffix}"
    if not pristine.exists():
        shutil.copy2(DECK, pristine)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(DECK, BACKUP_DIR / f"{DECK.stem}.{stamp}{DECK.suffix}")
    # The deck is ~9 MB, so keep only the most recent timestamped copies
    # alongside the pristine one.
    stamped = sorted(BACKUP_DIR.glob(f"{DECK.stem}.20*{DECK.suffix}"))
    for stale in stamped[:-KEEP_BACKUPS]:
        stale.unlink()

    with zipfile.ZipFile(DECK) as zin:
        infos = list(zin.infolist())
        parts = {i.filename: zin.read(i.filename) for i in infos}

    content_types = parts["[Content_Types].xml"].decode("utf-8")
    pres_rels = parts["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    pres_xml = parts["ppt/presentation.xml"].decode("utf-8")

    # --- drop slides inserted by an earlier run -----------------------------
    dropped = 0
    for name in sorted(
        (n for n in parts if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
        key=part_number,
    ):
        if SHAPE_TAG not in parts[name].decode("utf-8"):
            continue
        dropped += 1
        target = name[len("ppt/"):]
        rid = re.search(
            rf'<Relationship Id="(rId\d+)"[^>]*Target="{re.escape(target)}"\s*/>',
            pres_rels,
        ).group(1)
        pres_rels = re.sub(
            rf'<Relationship Id="{rid}"[^>]*/>', "", pres_rels
        )
        pres_xml = re.sub(
            rf'<p:sldId id="\d+" r:id="{rid}"\s*/>', "", pres_xml
        )
        content_types = re.sub(
            rf'<Override PartName="/{re.escape(name)}"[^>]*/>', "", content_types
        )
        for image in re.findall(
            r'Target="\.\./media/(image\d+\.\w+)"',
            parts[rels_name(name)].decode("utf-8"),
        ):
            parts.pop(f"ppt/media/{image}", None)
        parts.pop(name)
        parts.pop(rels_name(name))

    # --- work out the insertion point and fresh part numbers ---------------
    order = slide_order(pres_xml)
    target_by_rid = dict(
        re.findall(
            r'<Relationship Id="(rId\d+)"[^>]*Target="(slides/slide\d+\.xml)"\s*/>',
            pres_rels,
        )
    )
    position = len(order) if INSERT_AFTER_SLIDE is None else INSERT_AFTER_SLIDE
    if not 1 <= position <= len(order):
        raise SystemExit(
            f"cannot insert after slide {position}: deck has {len(order)} slides"
        )
    anchor_rid = order[position - 1][1]
    layout = blank_layout_for(parts, f"ppt/{target_by_rid[anchor_rid]}")

    next_slide = max(
        (part_number(n) for n in parts if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
        default=0,
    ) + 1
    next_image = max(
        int(re.search(r"image(\d+)", n).group(1))
        for n in parts
        if re.match(r"ppt/media/image\d+", n)
    ) + 1
    next_rid = max(int(r[3:]) for r in re.findall(r'Id="(rId\d+)"', pres_rels)) + 1
    next_sld_id = max(int(i) for i, _ in order) + 1

    slide_cx, slide_cy = (
        int(v) for v in
        re.search(r'<p:sldSz cx="(\d+)" cy="(\d+)"', pres_xml).groups()
    )
    x, y, cx, cy = picture_box(slide_cx, slide_cy)

    # --- add the rendered slides -------------------------------------------
    for offset, png in enumerate(slide_pngs):
        slide_part = f"ppt/slides/slide{next_slide + offset}.xml"
        image_part = f"image{next_image + offset}.png"
        rid = f"rId{next_rid + offset}"

        parts[f"ppt/media/{image_part}"] = png.read_bytes()
        parts[slide_part] = SLIDE_XML.format(
            r=R_NS, name=f"{SHAPE_TAG}-{offset + 1}", x=x, y=y, cx=cx, cy=cy
        ).encode("utf-8")
        parts[rels_name(slide_part)] = SLIDE_RELS_XML.format(
            r=R_NS, layout=layout, image=image_part
        ).encode("utf-8")

        content_types = content_types.replace(
            "</Types>",
            f'<Override PartName="/{slide_part}" ContentType="{SLIDE_CT}"/></Types>',
        )
        pres_rels = pres_rels.replace(
            "</Relationships>",
            f'<Relationship Id="{rid}" Type="{R_NS}/slide"'
            f' Target="{slide_part[len("ppt/"):]}"/></Relationships>',
        )
        pres_xml = re.sub(
            rf'(<p:sldId id="\d+" r:id="{anchor_rid}"\s*/>)',
            rf'\1<p:sldId id="{next_sld_id + offset}" r:id="{rid}"/>',
            pres_xml,
        )
        anchor_rid = rid  # keep the inserted slides in order

    parts["[Content_Types].xml"] = content_types.encode("utf-8")
    parts["ppt/_rels/presentation.xml.rels"] = pres_rels.encode("utf-8")
    parts["ppt/presentation.xml"] = pres_xml.encode("utf-8")

    # --- write the package, original entries byte for byte ------------------
    tmp = DECK.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        written = set()
        for info in infos:
            if info.filename not in parts:
                continue
            zout.writestr(info, parts[info.filename])
            written.add(info.filename)
        for name in parts:
            if name not in written:
                zout.writestr(name, parts[name])
    tmp.replace(DECK)

    total = len(slide_order(pres_xml))
    print(
        f"{DECK.name}: replaced {dropped}, inserted {len(slide_pngs)} "
        f"as slides {position + 1}-{position + len(slide_pngs)} "
        f"({total} slides total)"
    )


if __name__ == "__main__":
    main()
