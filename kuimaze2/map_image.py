"""
Extension module for converting a kuimaze2.map.Map to and from a bitmap image.
"""

import os
from PIL import Image

from kuimaze2.map import Map, Cell, State, Role


COLOR_FOR_ROLE_DEFAULT = {
    Role.EMPTY: (255, 255, 255),
    Role.WALL: (0, 0, 0),
    Role.GOAL: (0, 255, 0),  # Make a desired goal green
    Role.START: (0, 0, 255),  # Make a start point blue
    Role.DANGER: (255, 0, 0),  # Make a dangerous area red
}

ROLE_FROM_COLOR_DEFAULT = {value: key for key, value in COLOR_FOR_ROLE_DEFAULT.items()}



def map_from_image(
        image_fpath: os.PathLike, 
        role_from_color: dict[tuple[int, int, int], Role] | None = None
    ) -> Map:
    """Load a map from a bitmap image."""
    role_from_color = role_from_color or ROLE_FROM_COLOR_DEFAULT
    image = Image.open(image_fpath)
    cells = []
    for r in range(image.height):
        for c in range(image.width):
            pix_color = image.getpixel((c, r)) # PIL assumes (x, y), not (r, c)
            if pix_color not in role_from_color:
                print(f"--- WARNING: No role specified for color: {pix_color} at ({r=}, {c=}). Using Role.EMPTY.")
            role = role_from_color.get(pix_color, Role.EMPTY)
            cell = Cell(position=State(r, c), role=role)
            cells.append(cell)
    return Map(cells=cells)


def image_from_map(
        map: Map, 
        image_fpath: os.PathLike, 
        color_for_role: dict[Role, tuple[int, int, int]] | None = None
    ) -> None:
    """Save a map as a bitmap image."""
    color_for_role = color_for_role or COLOR_FOR_ROLE_DEFAULT
    image = Image.new("RGB", (map.width, map.height))
    for r in range(map.height):
        for c in range(map.width):
            cell = map[State(r, c)]
            if cell.role not in color_for_role:
                print(f"--- WARNING: No color specified for role: {cell.role} at ({r=}, {c=}). Using white.")
            color = color_for_role.get(cell.role, (255, 255, 255))
            image.putpixel((c, r), color)
    image.save(image_fpath)
