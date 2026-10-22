"""Export the camera-ready manuscript figures for the README.

The figures themselves are not modified: each vector source is rendered at high
resolution, uniform white margins are trimmed, and the result is written as a PNG at
the original aspect ratio. Colors, labels, connectors and layout are untouched.

Sources are the figure PDFs included by the manuscript (see FIGURES below). Run from
the repository root with a checkout of the manuscript available:

    python scripts/export_paper_figures.py --source /path/to/manuscript/Figures
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"

# (source pdf relative to --source, output stem, target pixel width)
FIGURES = [
    ("fig1/Fig1-v10-neurips.pdf", "figure1_overview", 2000),
    ("fig2/Fig2-v14.pdf", "figure2_architecture", 2000),
]

MARGIN_PX = 6  # uniform padding kept around the trimmed content
WHITE_CUTOFF = 250  # pixel values at or above this count as background


def trim(image: Image.Image) -> Image.Image:
    """Crop uniform white margins, keeping a small symmetric padding."""
    gray = image.convert("L")
    mask = gray.point(lambda v: 0 if v >= WHITE_CUTOFF else 255)
    box = mask.getbbox()
    if box is None:
        return image
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, left - MARGIN_PX),
            max(0, top - MARGIN_PX),
            min(image.width, right + MARGIN_PX),
            min(image.height, bottom + MARGIN_PX),
        )
    )


def export(pdf_path: Path, stem: str, width: int) -> None:
    page = pymupdf.open(pdf_path)[0]
    # Render wider than the target so the trimmed figure still lands on `width`.
    scale = (width * 1.25) / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    before = image.size
    image = trim(image)
    image = image.resize((width, round(width * image.height / image.width)), Image.LANCZOS)
    image.save(OUT / f"{stem}.png", optimize=True)
    shutil.copy(pdf_path, OUT / f"{stem}.pdf")
    kb = (OUT / f"{stem}.png").stat().st_size / 1024
    print(f"{pdf_path.name:28s} {before[0]}x{before[1]} -> {image.size[0]}x{image.size[1]}  ({kb:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "overleaf-strand" / "Figures",
        help="directory holding the manuscript's Figures/ tree",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for rel, stem, width in FIGURES:
        path = args.source / rel
        if not path.exists():
            raise SystemExit(f"missing figure source: {path}")
        export(path, stem, width)


if __name__ == "__main__":
    main()
