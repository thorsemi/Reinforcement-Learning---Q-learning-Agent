from dataclasses import astuple, dataclass
import tkinter as tk
from typing import Mapping, Self, Callable
import random


from kuimaze2.map import Border, Map, State, Action, Role, Cell
from kuimaze2.typing import QTable

MAX_WIN_WIDTH = 800
MAX_WIN_HEIGHT = 800
SQUARE_SIZE = 100
MARGIN_SIZE = 50
CIRCLE_DIAMETER = 0.5  # Relative to SQUARE_SIZE
FONT_FAMILY = "Helvetica"


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

    def mix(self, other: Self, factor: float) -> Self:
        return Color(
            int(self.r * (1 - factor) + other.r * factor),
            int(self.g * (1 - factor) + other.g * factor),
            int(self.b * (1 - factor) + other.b * factor),
        )

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    @staticmethod
    def random() -> Self:
        return Color(
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        )


COLOR_FROM_ROLE: Mapping[Role, Color] = {
    Role.EMPTY: Color(255, 255, 255),  # "#ffffff",
    Role.WALL: Color(0, 0, 0),  # "#000000",
    Role.START: Color(0, 0, 255).mix(Color(255, 255, 255), 0.5),  # "#0000ff",
    Role.GOAL: Color(0, 255, 0).mix(Color(255, 255, 255), 0.5),  # "#00ff00",
    Role.DANGER: Color(255, 0, 0).mix(Color(255, 255, 255), 0.5),  # "#ff0000",
}
DIVIDER_COLOR = Color(204, 204, 204)  # "#cccccc"
CURRENT_CIRCLE_COLOR = Color(255, 0, 255).mix(Color(255, 255, 255), 0.5)
NEXT_CIRCLE_COLOR = Color(255, 255, 0)
FRONTIER_CIRCLE_COLOR = COLOR_FROM_ROLE[Role.GOAL].mix(Color(255, 255, 255), 0.5)


class ColorFromValue:
    def __init__(self, value_range: tuple[float, float]):
        self.scale = (
            (value_range[0], COLOR_FROM_ROLE[Role.DANGER]),
            (0, Color(255, 255, 255)),
            (value_range[1], COLOR_FROM_ROLE[Role.GOAL]),
        )

    def __call__(self, value: float) -> Color:
        if value <= self.scale[0][0]:
            return self.scale[0][1]
        if value >= self.scale[-1][0]:
            return self.scale[-1][1]
        min_value, min_color = self.scale[0]
        mid_value, mid_color = self.scale[1]
        max_value, max_color = self.scale[-1]
        if min_value < value <= mid_value:
            factor = (value - min_value) / (mid_value - min_value)
            return min_color.mix(mid_color, factor)
        else:
            factor = (value - mid_value) / (max_value - mid_value)
            return mid_color.mix(max_color, factor)


@dataclass(frozen=True)
class RectCoords:
    left: int
    top: int
    right: int
    bottom: int

    def __iter__(self):
        return iter(astuple(self))

    def of_border_line(self, border: Border):
        match border:
            case Border.TOP:
                return self.left, self.top, self.right, self.top
            case Border.RIGHT:
                return self.right, self.top, self.right, self.bottom
            case Border.BOTTOM:
                return self.left, self.bottom, self.right, self.bottom
            case Border.LEFT:
                return self.left, self.top, self.left, self.bottom


class MapCanvas(tk.Canvas):

    def __init__(self, parent: tk.Tk, map: Map, sq_size: int = 0, **kwargs):
        if sq_size == 0:
            max_sq_width = (MAX_WIN_WIDTH - 2 * MARGIN_SIZE) // map.width
            max_sq_height = (MAX_WIN_HEIGHT - 2 * MARGIN_SIZE) // map.height
            sq_size = min([max_sq_width, max_sq_height, SQUARE_SIZE])
        canvas_width = sq_size * map.width + 2 * MARGIN_SIZE
        canvas_height = sq_size * map.height + 2 * MARGIN_SIZE
        super().__init__(parent, width=canvas_width, height=canvas_height, **kwargs)
        self.map: Map = map
        self.square_size = sq_size
        self.texts = {}
        self.font_size = max(2, int(0.2 * self.square_size))

    def draw(self):
        self.draw_borders()
        self.draw_row_indices()
        self.draw_col_indices()
        self.draw_square_texts()

    def _coords_from_position(self, pos: State) -> RectCoords:
        left = MARGIN_SIZE + self.square_size * pos.c
        top = MARGIN_SIZE + self.square_size * pos.r
        return RectCoords(left, top, left + self.square_size, top + self.square_size)

    def _center_from_position(self, pos: State) -> tuple[int, int]:
        rect = self._coords_from_position(pos)
        return int((rect.left + rect.right) / 2), int((rect.top + rect.bottom) / 2)

    def draw_borders(self):
        for sq in self.map:
            self.draw_border(sq)

    def draw_border(self, square: Cell):
        coords = self._coords_from_position(square.position)
        color = COLOR_FROM_ROLE[Role.WALL].to_hex()
        for border in Border:
            if border in square.border:
                line_coords = coords.of_border_line(border)
                self.create_line(*line_coords, fill=color, width=5, capstyle="round")

    def draw_row_indices(self):
        for row in range(self.map.height):
            self.create_text(
                MARGIN_SIZE // 2,
                MARGIN_SIZE + self.square_size * row + self.square_size // 2,
                text=str(row),
                font=(FONT_FAMILY, self.font_size),
            )
            self.create_text(
                MARGIN_SIZE + self.square_size * self.map.width + MARGIN_SIZE // 2,
                MARGIN_SIZE + self.square_size * row + self.square_size // 2,
                text=str(row),
                font=(FONT_FAMILY, self.font_size),
            )

    def draw_col_indices(self):
        for col in range(self.map.width):
            self.create_text(
                MARGIN_SIZE + self.square_size * col + self.square_size // 2,
                MARGIN_SIZE // 2,
                text=str(col),
                font=(FONT_FAMILY, self.font_size),
            )
            self.create_text(
                MARGIN_SIZE + self.square_size * col + self.square_size // 2,
                MARGIN_SIZE + self.square_size * self.map.height + MARGIN_SIZE // 2,
                text=str(col),
                font=(FONT_FAMILY, self.font_size),
            )

    def draw_square_texts(self, texts=None, default_text=""):
        texts = texts or {}
        for sq in self.map:
            text = texts.get(sq.position, default_text)
            coords = self._coords_from_position(sq.position)
            left = coords.left + self.square_size // 2
            top = coords.top + self.square_size // 2
            self.texts[sq.position] = self.create_text(
                left, top, text=text, font=(FONT_FAMILY, self.font_size)
            )

    def set_square_text(self, position: State, text: str):
        self.itemconfig(self.texts[position], text=text)

    def update_square_texts(self, texts: dict[State, str]):
        for position, text in texts.items():
            self.set_square_text(position, text)


class SquareCanvas(MapCanvas):

    def __init__(self, parent: tk.Tk, map: Map, sq_size: int = 0, **kwargs):
        super().__init__(parent, map, sq_size, **kwargs)
        self.squares = {}
        self.circles = {}
        self.path_line = None
        self.state_action_arrow = None
        self.draw()

    def draw(self):
        self.draw_squares()
        self.draw_borders()
        self.draw_row_indices()
        self.draw_col_indices()
        self.draw_circles()
        self.draw_init_path()
        self.draw_init_state_action_arrow()
        self.draw_square_texts()

    def draw_squares(self):
        """Draw all squares in the map, parts that can change color based on value"""
        for sq in self.map:
            self.draw_square(sq)

    def draw_square(self, square: Cell):
        coords = self._coords_from_position(square.position)
        self.squares[square.position] = self.create_rectangle(
            *coords,
            fill=COLOR_FROM_ROLE[square.role].to_hex(),
            outline=DIVIDER_COLOR.to_hex(),
            width=3,
        )

    def draw_circles(self):
        """Draw all squares in the map, parts that can change color based on value"""
        for sq in self.map:
            self.draw_circle(sq)

    def draw_circle(self, square: Cell):
        coords = self._coords_from_position(square.position)
        self.circles[square.position] = self.create_oval(
            *coords,
            fill="",
            outline="",
            state=tk.HIDDEN,
        )

    def draw_init_state_action_arrow(self):
        state = State(0, 0)
        action = Action.UP
        coords = self.state_action_arrow_coords(state, action)
        self.state_action_arrow = self.create_line(
            coords,
            fill="green",
            width=10,
            capstyle="round",
            arrow=tk.LAST,
            arrowshape=(12, 20, 6),
            state=tk.HIDDEN,
        )

    def state_action_arrow_coords(self, state, action):
        arrow_length = 0.45
        center = self._center_from_position(state)  # this is x, y
        difvec = action.to_vec()  # this is r, c
        end = (
            center[0] + difvec[1] * self.square_size * arrow_length,
            center[1] + difvec[0] * self.square_size * arrow_length,
        )  # 0-1 and 1-0 because of x + k1*c, y + k2*r
        coords = list(center) + list(end)
        return coords

    def draw_state_action_arrow(self, state: State, action: Action):
        coords = self.state_action_arrow_coords(state, action)
        self.coords(self.state_action_arrow, coords)
        self.itemconfig(self.state_action_arrow, state=tk.NORMAL)

    def hide_state_action_arrow(self):
        self.itemconfig(self.state_action_arrow, state=tk.HIDDEN)

    def draw_init_path(self):
        points = [0, 0, 0, 0]
        self.path_line = self.create_line(
            points,
            fill="red",
            width=7,
            capstyle="round",
            arrow=tk.LAST,
            arrowshape=(12, 20, 6),
        )

    def draw_path(self, path: list[State]):
        points = []
        for state in path:
            sq_center = self._center_from_position(state)
            points.extend(sq_center[:])
        self.coords(self.path_line, points)
        self.itemconfig(self.path_line, state=tk.NORMAL)

    def hide_path(self):
        self.itemconfig(self.path_line, state=tk.HIDDEN)

    def set_square_color(self, position: State, color: Color):
        self.itemconfig(self.squares[position], fill=color.to_hex())

    def set_circle_color(self, position: State, color: Color, visible: bool = True):
        self.itemconfig(
            self.circles[position],
            fill=color.to_hex(),
            state=tk.NORMAL if visible else tk.HIDDEN,
        )

    def set_circle_visibility(self, position: State, visible: bool = True):
        self.itemconfig(
            self.circles[position], state=tk.NORMAL if visible else tk.HIDDEN
        )

    def set_square_colors(self, colors: dict[State, Color], keep_role_colors=False):
        for position, color in colors.items():
            if keep_role_colors and self.map[position].role != Role.EMPTY:
                continue
            self.set_square_color(position, color)


class TriangleCanvas(MapCanvas):

    def __init__(self, parent: tk.Tk, map: Map, sq_size: int = 0, **kwargs):
        super().__init__(parent, map, sq_size, **kwargs)
        self.triangles = {}
        self.triangle_texts = {}
        self.draw()

    def draw(self):
        self.draw_triangles()
        self.draw_borders()
        self.draw_row_indices()
        self.draw_col_indices()
        self.draw_triangle_texts()
        self.draw_square_texts()

    def draw_triangles(self):
        for sq in self.map:
            if sq.is_free():
                self.triangles[sq.position] = {}
                for action in Action:
                    self.draw_triangle(sq, action)
            elif sq.role == Role.WALL:
                self.draw_square(sq)
            else:
                assert False, "Unexpected role"

    def draw_triangle(self, square: Cell, action: Action):
        coords = self._tr_coords(square, action)
        self.triangles[square.position][action] = self.create_polygon(
            *coords,
            fill=Color(255, 255, 255).to_hex(),
            outline=DIVIDER_COLOR.to_hex(),
        )

    def _tr_coords(self, square: Cell, action: Action):
        coords = self._coords_from_position(square.position)
        border = coords.of_border_line(square.border.corresponding_to(action))
        center = self._center_from_position(square.position)
        return (*border, *center)

    def draw_square(self, square: Cell):
        coords = self._coords_from_position(square.position)
        # We will not store the squares, they are not needed
        self.create_rectangle(
            *coords,
            fill=COLOR_FROM_ROLE[square.role].to_hex(),
            outline=DIVIDER_COLOR.to_hex(),
            width=3,
        )

    def set_triangle_color(self, position: State, action: Action, color: Color):
        self.itemconfig(self.triangles[position][action], fill=color.to_hex())

    def set_triangle_colors(self, colors: dict[(State, Action), Color]):
        for (position, action), color in colors.items():
            self.set_triangle_color(position, action, color)

    def draw_triangle_texts(self, texts=None, default_text=""):
        texts = texts or {}
        for sq in self.map:
            middle = self._center_from_position(sq.position)
            for action in Action:
                text = texts.get((sq.position, action), default_text)
                left = middle[0] + action.to_vec().c * self.square_size * 0.3
                top = middle[1] + action.to_vec().r * self.square_size * 0.3
                self.triangle_texts[(sq.position, action)] = self.create_text(
                    left,
                    top,
                    text=text,
                    font=(FONT_FAMILY, round(self.font_size * 0.6)),
                )

    def set_triangle_text(self, position: State, action: Action, text: str):
        self.itemconfig(self.triangle_texts[position, action], text=text)

    def update_triangle_texts(self, texts: dict[(State, Action), str]):
        for (position, action), text in texts.items():
            self.set_triangle_text(position, action, text)


class SearchCanvas(SquareCanvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_colors = (
            (0, COLOR_FROM_ROLE[Role.START].mix(Color(255, 255, 255), 0.5)),
            (self.map.number_of_accessible_states, Color(255, 255, 255)),
        )
        self.current_circle = None
        self.next_circles = []
        self.frontier_circles = []

    def color_from_value(self, value: float = None, factor: float = None) -> Color:
        ((min_value, color1), (max_value, color2)) = self.value_colors
        if not factor:
            if value <= min_value:
                return color1
            if value >= max_value:
                return color2
            factor = (value - min_value) / (max_value - min_value)
        color = color1.mix(color2, factor)
        return color

    def set_square_colors_from_values(self, colors: dict[State, float]):
        colors = {
            position: self.color_from_value(value) for position, value in colors.items()
        }
        self.set_square_colors(colors, keep_role_colors=True)

    def set_square_colors_from_visited(self, visited):
        color = COLOR_FROM_ROLE[Role.START].mix(Color(255, 255, 255), 0.5)
        colors = {position: color for position in visited}
        self.set_square_colors(colors, keep_role_colors=True)

    def set_current_state(self, state: State = None):
        if self.current_circle:
            self.set_circle_visibility(self.current_circle, False)
        self.current_circle = state
        if state:
            self.set_circle_color(state, CURRENT_CIRCLE_COLOR)
            self.set_circle_visibility(state, True)

    def set_next_states(self, next_states: list[State] = []):
        self.reset_next_states()
        for state in next_states:
            self.add_next_state(state)

    def add_next_state(self, state: State):
        self.set_circle_color(state, NEXT_CIRCLE_COLOR)
        self.set_circle_visibility(state, True)
        self.next_circles.append(state)

    def reset_next_states(self):
        for state in self.next_circles:
            if state in self.frontier_circles:
                self.set_circle_color(state, FRONTIER_CIRCLE_COLOR)
            else:
                self.set_circle_visibility(state, False)
        self.next_circles = []

    def set_frontier_states(self, frontier):
        self.reset_frontier_states()
        for state in frontier:
            self.add_frontier_state(state)

    def add_frontier_state(self, state: State):
        self.set_circle_color(state, FRONTIER_CIRCLE_COLOR)
        self.set_circle_visibility(state, True)
        self.frontier_circles.append(state)

    def reset_frontier_states(self):
        for state in self.frontier_circles:
            self.set_circle_visibility(state, False)
        self.frontier_circles = []


class ValueCanvas(SquareCanvas):

    def __init__(
        self,
        parent: tk.Tk,
        map: Map,
        value_range: tuple[float, float],
        sq_size: int = 0,
        **kwargs,
    ):
        if sq_size == 0:
            # Allow for 2 canvases above each other
            max_sq_width = (MAX_WIN_WIDTH - 2 * MARGIN_SIZE) // map.width
            max_sq_height = (MAX_WIN_HEIGHT - 4 * MARGIN_SIZE) // (2 * map.height)
            sq_size = min([max_sq_width, max_sq_height, SQUARE_SIZE])
        # print(f"Value canvas size: {sq_size}")
        super().__init__(parent=parent, map=map, sq_size=sq_size, **kwargs)
        self.color_from_value = ColorFromValue(value_range)

    def set_square_colors_from_values(self, values: dict[State, float]):
        colors = {
            position: self.color_from_value(value) for position, value in values.items()
        }
        self.set_square_colors(colors)


class QValueCanvas(TriangleCanvas):

    def __init__(
        self,
        parent: tk.Tk,
        map: Map,
        value_range: tuple[float, float],
        sq_size: int = 0,
        **kwargs,
    ):
        if not sq_size:
            # Allow for 2 canvases above each other
            max_sq_width = (MAX_WIN_WIDTH - 2 * MARGIN_SIZE) // map.width
            max_sq_height = (MAX_WIN_HEIGHT - 2 * MARGIN_SIZE) // (2 * map.height)
            sq_size = min([max_sq_width, max_sq_height, SQUARE_SIZE])
        # print(f"QValue canvas size: {sq_size}")
        super().__init__(parent=parent, map=map, sq_size=sq_size, **kwargs)
        self.color_from_value = ColorFromValue(value_range)

    def set_triangle_colors_from_qvalues(self, qvalues: QTable):
        colors = {}
        for state, action_values in qvalues.items():
            for action, value in action_values.items():
                colors[state, action] = self.color_from_value(value)
        self.set_triangle_colors(colors)
