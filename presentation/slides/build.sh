#!/usr/bin/env bash
# Render the Beamer model slides and insert them into the kick-off deck.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-/c/Users/dirko/AppData/Local/Programs/Python/Python310/python.exe}"
DPI="${DPI:-400}"

quarto render model-slides.qmd
mkdir -p png
pdftoppm -r "$DPI" -png model-slides.pdf png/slide
"$PYTHON" insert_slides.py
