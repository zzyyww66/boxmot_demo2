#!/usr/bin/env python3
"""Build a single-page 2x3 PDF from fixed images in this folder."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
OUTPUT_PDF = ROOT / "screen_grid.pdf"

# Row-major order (2 rows x 3 columns)
IMAGE_GRID = [
    ["ori.png", "ori2.png", "imp2.jpeg"],
    ["ori3.png", "ori4.png", "imp4.jpeg"],
]

# Layout: fixed-size tiles with equal horizontal/vertical spacing.
# This guarantees every image appears in the same display size.
TILE_WIDTH = 720
TILE_HEIGHT = 1080
MARGIN = 60
GAP = 60  # same for horizontal and vertical spacing
BG_COLOR = (255, 255, 255)


def _check_inputs() -> list[list[Path]]:
    """Validate input files and return their resolved paths."""
    paths: list[list[Path]] = []
    missing: list[Path] = []
    for row in IMAGE_GRID:
        row_paths: list[Path] = []
        for name in row:
            p = ROOT / name
            row_paths.append(p)
            if not p.exists():
                missing.append(p)
        paths.append(row_paths)

    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing image(s):\n{missing_str}")
    return paths


def _to_uniform_tile(src: Image.Image) -> Image.Image:
    """Convert image to a fixed-size tile while preserving aspect ratio."""
    contained = ImageOps.contain(src, (TILE_WIDTH, TILE_HEIGHT), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), BG_COLOR)
    offset_x = (TILE_WIDTH - contained.width) // 2
    offset_y = (TILE_HEIGHT - contained.height) // 2
    tile.paste(contained, (offset_x, offset_y))
    return tile


def main() -> None:
    paths = _check_inputs()
    rows = len(paths)
    cols = len(paths[0])

    page_width = 2 * MARGIN + cols * TILE_WIDTH + (cols - 1) * GAP
    page_height = 2 * MARGIN + rows * TILE_HEIGHT + (rows - 1) * GAP

    canvas = Image.new("RGB", (page_width, page_height), BG_COLOR)

    for r, row in enumerate(paths):
        for c, img_path in enumerate(row):
            x0 = MARGIN + c * (TILE_WIDTH + GAP)
            y0 = MARGIN + r * (TILE_HEIGHT + GAP)

            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                tile = _to_uniform_tile(im_rgb)
                canvas.paste(tile, (x0, y0))

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PDF, "PDF", resolution=300.0)
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
