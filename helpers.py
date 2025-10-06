"""
puzzle_formats.py
Utility functions to render ARC-style grid puzzles in multiple representations:
- ascii
- coords
- rle
- png (as base64 markdown image)
"""

from collections import defaultdict
from PIL import Image, ImageDraw
import io, base64, json


# CONFIGS


DEFAULT_MAP = {0: '.', 1: '1', 2: '2', 3: '3', 4: '4', 8: '#'}
PALETTE = {
    0: (255, 255, 255, 0),  
    1: (255, 0, 0, 255),
    2: (0, 128, 0, 255),
    3: (0, 0, 255, 255),
    4: (255, 165, 0, 255),
    8: (0, 0, 0, 255),
}


# ASCII REPRESENTATION


def grid_to_ascii(grid, cmap=DEFAULT_MAP):

    return "\n".join("".join(cmap.get(cell, '?') for cell in row) for row in grid)


def example_to_ascii_block(ex, idx):

    w = len(ex['input'][0])
    h = len(ex['input'])
    in_block = grid_to_ascii(ex['input'])
    out_block = grid_to_ascii(ex['output'])
    return f"EX #{idx} — size {w}x{h}\nINPUT:\n{in_block}\nOUTPUT:\n{out_block}"



# COORDINATE REPRESENTATION

def coords_by_color(grid):
    d = defaultdict(list)
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val != 0:
                d[val].append((x, y))
    return dict(d)


def example_to_coords_block(ex, idx):

    h = len(ex['input'])
    w = len(ex['input'][0])
    in_coords = coords_by_color(ex['input'])
    out_coords = coords_by_color(ex['output'])
    return f"EX #{idx} size={w}x{h}\nIN_COORDS: {in_coords}\nOUT_COORDS: {out_coords}"



# IMAGE (PNG) REPRESENTATION


def grid_to_png_bytes(grid, cell_size=12, palette=PALETTE):
    """Render grid to PNG bytes."""
    h = len(grid)
    w = len(grid[0])
    img = Image.new("RGBA", (w * cell_size, h * cell_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            color = palette.get(val, (128, 128, 128, 255))
            x0, y0 = x * cell_size, y * cell_size
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def example_to_image_block(ex, idx):

    in_png = grid_to_png_bytes(ex['input'])
    out_png = grid_to_png_bytes(ex['output'])
    in_b64 = base64.b64encode(in_png).decode('ascii')
    out_b64 = base64.b64encode(out_png).decode('ascii')
    in_md = f"![input](data:image/png;base64,{in_b64})"
    out_md = f"![output](data:image/png;base64,{out_b64})"
    return f"EX#{idx}\nINPUT_IMAGE:\n{in_md}\nOUTPUT_IMAGE:\n{out_md}"



# RLE REPRESENTATION


def rle_row(row):
    out = []
    prev = row[0]
    cnt = 1
    for cell in row[1:]:
        if cell == prev:
            cnt += 1
        else:
            out.append(f"{cnt}x{prev}")
            prev = cell
            cnt = 1
    out.append(f"{cnt}x{prev}")
    return ",".join(out)


def grid_to_rle(grid):
    return ";".join(rle_row(row) for row in grid)


def example_to_rle_block(ex, idx):

    w = len(ex['input'][0])
    h = len(ex['input'])
    return (
        f"EX#{idx}\nSIZE:{w}x{h}\n"
        f"IN_RLE:{grid_to_rle(ex['input'])}\n"
        f"OUT_RLE:{grid_to_rle(ex['output'])}"
    )



# UNIFIED INTERFACE


def examples_to_str(data, style='ascii'):

    style_map = {
        'ascii': example_to_ascii_block,
        'coords': example_to_coords_block,
        'rle': example_to_rle_block,
        'images': example_to_image_block,
        'json': lambda ex, i: f"EX#{i}\nINPUT:\n{json.dumps(ex['input'])}\nOUTPUT:\n{json.dumps(ex['output'])}"
    }

    if style not in style_map:
        raise ValueError(f"Unknown style: {style}")

    converter = style_map[style]
    blocks = [converter(ex, i) for i, ex in enumerate(data['train'])]
    return "\n\n".join(blocks)

