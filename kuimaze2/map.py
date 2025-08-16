import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag
from functools import cached_property
from pathlib import Path
from typing import Self, Sequence


@dataclass(frozen=True)
class MazeVec:
    """A 2D vector-like class used to represent maze coordinates (r,c)."""

    r: int = 0
    c: int = 0

    def __len__(self):
        return 2

    def __getitem__(self, key):
        if key == 0:
            return self.r
        if key == 1:
            return self.c
        raise IndexError(f"MazeVec index out of range: {key}")

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.r + other.r, self.c + other.c)

    def __sub__(self, other: Self) -> Self:
        return self.__class__(self.r - other.r, self.c - other.c)

    @cached_property
    def norm(self):
        return (self.r**2 + self.c**2) ** 0.5


class State(MazeVec):
    """Representation of State in all maze-related environments."""

    def __str__(self):
        return f"S({self.r},{self.c})"


class Action(IntEnum):
    """Enum representing possible actions in the maze environment."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def to_vec(self) -> MazeVec:
        """Return a delta MazeVec corresponding to the action.

        Can be added to/subtracted from a State to get a new State.
        """
        if self == Action.UP:
            return MazeVec(r=-1, c=0)
        elif self == Action.RIGHT:
            return MazeVec(r=0, c=1)
        elif self == Action.DOWN:
            return MazeVec(r=1, c=0)
        elif self == Action.LEFT:
            return MazeVec(r=0, c=-1)
        assert False, "Unreachable"

    def __str__(self):
        """Return a character represeting the action."""
        if self == Action.UP:
            return "⮝"  # "⮉"  # "↑" # "🠝" #
        elif self == Action.RIGHT:
            return "⮞"  # "⮊"  # "→" # "🠞" #
        elif self == Action.DOWN:
            return "⮟"  # "⮋"  # "↓" # "🠟" #
        elif self == Action.LEFT:
            return "⮜"  # "⮈"  # "←" # "🠜" #
        assert False, "Unreachable"


class Role(Enum):
    """Role of a cell in a map."""

    EMPTY = "."
    WALL = "#"
    START = "S"
    GOAL = "G"
    DANGER = "D"


class Border(IntFlag):
    """Flags representing existing borders of a map cell."""

    NONE = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 4
    LEFT = 8

    @classmethod
    def corresponding_to(cls, action: Action) -> Self:
        """Return the border corresponding to action."""
        return cls(2**action.value)

    def prevents_action(self, action: Action) -> bool:
        """Return True if the border setting prevents a specific action."""
        return Border.corresponding_to(action) in self


@dataclass(frozen=True)
class Cell:
    position: State = field(default_factory=State)
    role: Role = Role.EMPTY
    border: Border = Border.NONE

    def is_free(self) -> bool:
        return self.role != Role.WALL

    def is_terminal(self) -> bool:
        return self.role in (Role.GOAL, Role.DANGER)

    # def is_dead_end(self) -> bool:
    #     return self.border.bit_count() == 3
    #
    # def is_crossing(self) -> bool:
    #     return self.border.bit_count() < 2


class Map:
    """Map of a maze represented as a collection of cells.

    Map contains the constant part of a maze (the obstacles, walls, and free/empty cells),
    and also the default start and goal states if specified.
    """

    def __init__(self, cells: Sequence[Cell] | None = None):
        cells = cells or [Cell()]
        self.cell_at: dict[State, Cell] = {}
        for cell in cells:
            self.cell_at[cell.position] = cell
        self._complete()

    def __str__(self) -> str:
        """Return a string representation of the map."""
        rows = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                cell = self[State(r, c)]
                row.append(cell.role.value)
            rows.append("".join(row))
        return "\n".join(rows)

    @cached_property
    def number_of_accessible_states(self):
        return len([state for state, cell in self.cell_at.items() if cell.is_free()])

    @classmethod
    def from_string(cls, input: str) -> Self:
        """Create a map from a string.

        Example:
        >>> map_string = '''
        ...G
        .#.D
        ....
        '''
        >>> map = Map.from_string(map_string)
        """
        cells = []
        # Remove empty rows and white space at the beginning and end of each row
        rows = [r.strip() for r in input.splitlines() if r.strip()]
        for r, row in enumerate(rows):
            if not row.strip():
                continue
            for c, char in enumerate(row):
                cells.append(Cell(position=State(r, c), role=Role(char)))
        return cls(cells=cells)

    @classmethod
    def from_file(cls, fpath: os.PathLike) -> Self:
        """Create a map from a text file.

        The text file contents shall obey format for `Map.from_string`.
        """
        with Path(fpath).open("rt", encoding="utf-8") as f:
            content = f.read()
        return cls.from_string(content)

    @cached_property
    def width(self) -> int:
        return max(cell.position.c for cell in self.cell_at.values()) + 1

    @cached_property
    def height(self) -> int:
        return max(cell.position.r for cell in self.cell_at.values()) + 1

    @cached_property
    def start(self) -> State | None:
        """Return the start state as specified in the map, or None."""
        starts = [cell.position for cell in self if cell.role == Role.START]
        if len(starts) > 1:
            raise ValueError(f"Map: Multiple start squares are prohibited: {starts}")
        return starts[0] if starts else None

    @cached_property
    def goals(self) -> set[State]:
        """Return the set goal states as specified in the map."""
        return {cell.position for cell in self if cell.role == Role.GOAL}

    @cached_property
    def dangers(self) -> set[State]:
        """Return the set danger states as specified in the map."""
        return {cell.position for cell in self if cell.role == Role.DANGER}

    def accessible_neighbor_states(self, state: State) -> list[State]:
        return [
            self.get_transition_result(state, action)
            for action in Action
            if self.transition_possible(state, action)
        ]

    def transition_possible(self, state: State, action: Action) -> bool:
        return not self[state].border.prevents_action(action)

    def get_transition_result(self, state: State, action: Action) -> State:
        """Return the result of applying action to state.

        If the action is not possible, return the same state.
        """
        if not self.transition_possible(state, action):
            return state
        return state + action.to_vec()

    def __len__(self):
        return len(self.cell_at)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = State(*key)
        if cell := self.cell_at.get(key, None):
            return cell
        # If cell not defined, return a wall
        return Cell(position=key, role=Role.WALL, border=Border.NONE)

    def __iter__(self):
        """Iterate over map squres"""
        return iter(self.cell_at.values())

    def neighbor_cell(self, cell: Cell, direction: Action) -> Cell:
        return self[cell.position + direction.to_vec()]

    def all_states(self):
        """Generate all states, including walls."""
        for row in range(self.height):
            for col in range(self.width):
                yield State(c=col, r=row)

    # def __iter__(self):
    #     for pos in self.all_states():
    #         yield self[pos]

    def _complete(self) -> None:
        for state in self.all_states():
            square = self[state]
            assert state == square.position
            border: Border = Border.NONE
            for direction in Action:
                if self._shall_have_border(square, direction):
                    border |= Border.corresponding_to(direction)
            self.cell_at[state] = Cell(state, square.role, border)

    def _shall_have_border(self, square: Cell, direction: Action) -> bool:
        # If the square is a wall, it shall have border in any direction
        if square.role == Role.WALL:
            return True
        # If there is a border, leave it there
        if Border.corresponding_to(direction) in square.border:
            return True
        # Signal border if there is a boundary between EMPTY and WALL
        neighbor = self.neighbor_cell(square, direction)
        if square.is_free() != neighbor.is_free():
            return True
        return False
