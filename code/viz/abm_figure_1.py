"""
Visualize the experimental design with the original PTD and RMC stimuli.
"""

# initialization
# -----------------------------------------------------------------------------
import sys                                                                      # system tools
from pathlib import Path                                                        # path management

import matplotlib.pyplot as plt                                                 # plotting interface
import numpy as np                                                              # numerical operations
import pandas as pd                                                             # stimulus catalogue loading
from matplotlib.image import imread                                             # image loading

# directory management
# -----------------------------------------------------------------------------
src  = Path(__file__).resolve().parent                                          # visualization directory
udir = src.parent / 'abm'                                                       # project utilities directory
root = src.parent.parent                                                        # project root directory
sdir = root / 'data' / 'experiment' / 'derivatives' / 'stimuli'                 # original stimulus directory
fdir = root / 'figures'                                                         # figure output directory
sys.path.append(str(udir))                                                      # add project utilities to module search path

# project utility import
# -----------------------------------------------------------------------------
from abm_figure import abm_figure                                               # shared figure defaults

# stimulus catalogue
# -----------------------------------------------------------------------------
cat = pd.read_csv(sdir / 'stimuli.csv')                                         # experimental stimulus catalogue
ptd = cat.loc[cat.task_type == 'PTD', 'stimulus_file'].tolist()                 # all PTD image paths
rmc = cat.loc[cat.task_type == 'RMC', 'stimulus_file'].tolist()                 # all RMC image paths
exa = ptd[0]                                                                    # displayed PTD trial stimulus
exb = rmc[2]                                                                    # displayed RMC trial stimulus
tar = [ptd[i] for i in [2,9,12]]                                                # distinct memorized PTD targets
hnd = imread(sdir / 'images' / 'hand.bmp').astype(float)                        # supplied right-hand icon
hnd = hnd / 255 if hnd.max() > 1 else hnd                                       # normalized icon colors
alp = 1.0 - hnd.mean(axis = 2)                                                  # white-background transparency
rgb = np.zeros((*alp.shape, 3))                                                 # black icon color layer
hnd = np.dstack((rgb, alp))                                                     # transparent hand icon

# figure canvas and colors
# -----------------------------------------------------------------------------
plt = abm_figure(plt)                                                           # shared pyplot defaults
fig = plt.figure(figsize = (12, 6.75), facecolor = 'white')                     # widescreen design figure
ax  = fig.add_axes([0, 0, 1, 1])                                                # full-canvas annotation axis
ax.set_xlim(0, 1)                                                               # normalized horizontal coordinates
ax.set_ylim(0, 1)                                                               # normalized vertical coordinates
ax.axis('off')                                                                  # hide canvas axes
blu = 'blue'                                                                    # participant-figure PTD color
red = 'red'                                                                     # participant-figure RMC color
grn = 'green'                                                                   # participant-figure target color
ink = '#111111'                                                              # primary text and outline color

# main section structure
# -----------------------------------------------------------------------------
ax.text(                                                                        # first task name
0.245,                                                                          # horizontal position
0.925,                                                                          # vertical position
'Pattern target detection (PTD)',                                               # task name
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 17,                                                                  # task-name font size
color    = blu)                                                                 # PTD color
ax.text(                                                                        # first task decision
0.245,                                                                          # horizontal position
0.675,                                                                          # vertical position
'Target present? (Yes/No)',                                                     # decision prompt
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 15,                                                                  # prompt font size
color    = ink)                                                                 # prompt color
ax.text(                                                                        # second task name
0.245,                                                                          # horizontal position
0.585,                                                                          # vertical position
'Rotation/mirror classification (RMC)',                                         # task name
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 17,                                                                  # task-name font size
color    = red)                                                                 # RMC color
ax.text(                                                                        # second task decision
0.245,                                                                          # horizontal position
0.335,                                                                          # vertical position
'Rotation only? (Yes/No)',                                                      # decision prompt
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 15,                                                                  # prompt font size
color    = ink)                                                                 # prompt color

# exemplary trial stimuli
# -----------------------------------------------------------------------------
for pth, pos in [(exa, [0.135, 0.715, 0.22, 0.16]),                             # PTD exemplar specification
                 (exb, [0.135, 0.375, 0.22, 0.16])]:                            # RMC exemplar specification
    iax = fig.add_axes(pos)                                                     # exemplar image axis
    iax.imshow(imread(sdir / pth), interpolation = 'nearest')                   # original experimental image
    iax.set_xticks([])                                                          # omit horizontal ticks
    iax.set_yticks([])                                                          # omit vertical ticks
    for spn in iax.spines.values():                                             # exemplar frame spines
        spn.set_visible(True)                                                   # show the complete frame
        spn.set_color(ink)                                                      # thin black outline
        spn.set_linewidth(0.6)                                                  # restrained outline width

# response format
# -----------------------------------------------------------------------------
ax.text(                                                                        # response heading
0.245,                                                                          # horizontal position
0.242,                                                                          # vertical position
'Response format',                                                              # heading text
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 18,                                                                  # heading font size
color    = ink)                                                                 # heading color
bxs = [0.145, 0.205, 0.285, 0.345]                                              # response-button centers
bcs = [blu, blu, red, red]                                                      # task-specific button colors
bts = ['PTD\nYes', 'PTD\nNo', 'RMC\nYes', 'RMC\nNo']                            # button labels
for x, col, lab in zip(bxs, bcs, bts):                                          # button iterations
    ax.scatter(                                                                 # circular response button
    x,                                                                          # horizontal position
    0.172,                                                                      # vertical position
    s         = 1100,                                                           # button area in points
    color     = col,                                                            # task-specific button fill
    edgecolor = ink,                                                            # button outline color
    linewidth = 1.0)                                                            # button outline width
    ax.text(                                                                    # response-button label
    x,                                                                          # horizontal position
    0.172,                                                                      # vertical position
    lab,                                                                        # task and answer label
    ha       = 'center',                                                        # horizontal alignment
    va       = 'center',                                                        # vertical alignment
    fontsize = 9,                                                               # button-label font size
    color    = 'white',                                                         # button-label color
    usetex   = False)                                                           # native multiline text
for x, img in [(0.175, np.fliplr(hnd)), (0.315, hnd)]:                          # mirrored left and original right hands
    iax = fig.add_axes([x - 0.025, 0.040, 0.050, 0.115])                        # hand-icon axis
    iax.imshow(img, interpolation = 'bilinear')                                 # supplied hand visualization
    iax.axis('off')                                                             # hide hand-image axes

# stimulus-pool headings
# -----------------------------------------------------------------------------
ax.text(                                                                        # right-column heading
0.745,                                                                          # horizontal position
0.935,                                                                          # vertical position
'Stimulus pools',                                                               # heading text
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 18,                                                                  # heading font size
color    = ink)                                                                 # heading color
ax.text(                                                                        # PTD pool heading
0.745,                                                                          # horizontal position
0.885,                                                                          # vertical position
'Pattern target detection (PTD)',                                               # pool heading text
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 15,                                                                  # pool-heading font size
color    = blu)                                                                 # PTD color
ax.text(                                                                        # RMC pool heading
0.745,                                                                          # horizontal position
0.445,                                                                          # vertical position
'Rotation/mirror classification (RMC)',                                         # pool heading text
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 15,                                                                  # pool-heading font size
color    = red)                                                                 # RMC color

# complete PTD stimulus pool
# -----------------------------------------------------------------------------
for idx, pth in enumerate(ptd):                                                 # PTD image iterations
    row = idx // 6                                                              # pool-grid row
    col = idx % 6                                                               # pool-grid column
    x   = 0.535 + 0.075 * col                                                   # image-axis left position
    y   = 0.785 - 0.090 * row                                                   # image-axis lower position
    iax = fig.add_axes([x, y, 0.064, 0.050])                                    # compact PTD image axis
    img = imread(sdir / pth).astype(float)                                      # original pattern image
    img = img / 255 if img.max() > 1 else img                                   # normalized image colors
    if pth in tar:                                                              # memorized target image
        img = 0.62 * img + 0.38 * np.array([0.25, 1.00, 0.25])                  # participant-figure target tint
    iax.imshow(img, interpolation = 'nearest')                                  # pattern visualization
    iax.set_xticks([])                                                          # omit horizontal ticks
    iax.set_yticks([])                                                          # omit vertical ticks
    for spn in iax.spines.values():                                             # stimulus-pool frame spines
        spn.set_visible(True)                                                   # show the complete frame
        spn.set_color(grn if pth == tar else ink)                               # target or standard outline
        spn.set_linewidth(1.6 if pth == tar else 0.5)                           # target-emphasis width
ax.text(                                                                        # target legend
0.745,                                                                          # horizontal position
0.575,                                                                          # vertical position
'Green: memorized target',                                                      # legend text
ha       = 'center',                                                            # horizontal alignment
va       = 'center',                                                            # vertical alignment
fontsize = 12,                                                                  # legend font size
color    = grn)                                                                 # target color

# complete RMC stimulus pool
# -----------------------------------------------------------------------------
for idx, pth in enumerate(rmc):                                                 # RMC image iterations
    row = idx // 4                                                              # pool-grid row
    col = idx % 4                                                               # pool-grid column
    x   = 0.545 + 0.110 * col                                                   # image-axis left position
    y   = 0.300 - 0.145 * row                                                   # image-axis lower position
    iax = fig.add_axes([x, y, 0.090, 0.075])                                    # compact RMC image axis
    iax.imshow(imread(sdir / pth), interpolation = 'nearest')                   # original object-pair image
    iax.set_xticks([])                                                          # omit horizontal ticks
    iax.set_yticks([])                                                          # omit vertical ticks
    for spn in iax.spines.values():                                             # RMC stimulus frame spines
        spn.set_visible(True)                                                   # show the complete frame
        spn.set_color(ink)                                                      # thin black outline
        spn.set_linewidth(0.5)                                                  # restrained outline width

# figure saving
# -----------------------------------------------------------------------------
fdir.mkdir(parents = True, exist_ok = True)                                     # ensure output directory exists
for ext in ('png', 'pdf'):                                                      # raster and vector formats
    file = fdir / f'abm_figure_1.{ext}'                                         # output figure path
    fig.savefig(file, dpi = 400, format = ext, facecolor = 'white')             # publication-resolution figure
plt.close(fig)                                                                  # release figure resources
