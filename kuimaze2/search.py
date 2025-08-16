from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Mapping
import tkinter as tk

from kuimaze2.map import Map, State, Action, Role
from kuimaze2.rendering import SearchCanvas
from kuimaze2 import keyboard

StateRoleCosts = Mapping[Role, float]

DEFAULT_COSTS: StateRoleCosts = {
    Role.EMPTY: 1,
    Role.START: 1,
    Role.GOAL: 0,
    Role.DANGER: 10,
    Role.WALL: float("inf"),
}


class SearchProblem:

    def __init__(
        self, map: Map, costs: Optional[StateRoleCosts] = None, graphics: bool = False
    ):
        """
        Create a SearchProblem environment given a Map.

        Arguments:

            map: Map
                The Map instance. Create using `kuimaze2.Map.from_string()`
                or using `kuimaze2.map_image.map_from_image()`.

            costs: StateRoleCosts = None
                An optional dictionary of (Role, float) pairs, i.e., an assignment
                of cost when leaving a cell with the given Role.

            graphics: bool = False
                If True, a graphical interface will be created and displayed.
        """
        self.map = map
        self._start: State = self.map.start
        self._goals: set[State] = self.map.goals
        self._costs: StateRoleCosts = costs or DEFAULT_COSTS
        self._visited: set[State] = set()
        self._view: SearchView = (
            NullSearchView(self) if not graphics else TkSearchView(self)
        )

    @classmethod
    def from_string(cls, map):
        return cls(Map.from_string(map))

    def get_start(self) -> State:
        """Retrun the start state as specified in the map."""
        self._visited.add(self._start)
        return self._start

    def get_goals(self) -> set[State]:
        """Return the set of goal states as specified in the map."""
        return deepcopy(self._goals)

    def is_goal(self, state):
        """Return True if the state is one of the goals."""
        return state in self._goals

    def get_actions(self, state) -> list[Action]:
        """Return all actions that can be applied in the given state.

        For this maze problem, all 4 actions are possible in every state.
        """
        return list(Action) if self.map[state].is_free() else []

    def get_transition_result(
        self, state: State, action: Action
    ) -> tuple[State, float]:
        """Return the new state and transition cost by applying the given action in the given state.

        If the action is not possible, return the same state.
        """
        successor = self.map.get_transition_result(state, action)
        self._visited.add(successor)
        return (successor, self._get_cost(state))

    def _get_cost(self, state: State) -> float:
        return self._costs[self.map[state].role]

    def render(self, *args, **kwargs):
        """Display/update the graphical representation of the environment

        Arguments (all optional):

            current_state: State = None
                The provided State is marked with a circle of color 1.
                Intended for emphasizing the current State being expanded.

            next_states: list[State] = []
                The States in the provided list are marked with a circle of color 2.
                Intended for emphasizing the successors of the current state.

            frontier_states: list[State] = []
                The States in the provided list are marked with a circle of color 3.
                Intended for emphasizing the states in the frontier.

            texts: dict[State, str] = {}
                A dictionary of (State, str) pairs; assignment of texts to States.
                Each cell in the maze can display a text string. It can be an ID, cost,
                heuristic value, all of it.

            colors: dict[State, float] = {},
                A dictionary of (State, float) pairs; assignment of values to States.
                The values are used to determine the color intensity of each State.
                If not given, colors of cells (except walls, start, goal and danger)
                are given by the fact whether the cell was already visited.
                Start is visited once you call `get_start()`.
                Other state are visited once they appear as a result of `_get_transition_result(...)`.

            path: list[State] = []
                A list of States that shall form a continuous path in the maze.
                Intended to display the found path.

            wait: bool = False
                If True, wait for key press before continuing.
                The terminal should contain instructions on what keys can be pressed.

            use_keyboard: bool = None
                After the keys were switched off using some keyboard key (presumably 's' - skip),
                you can switch it on again by setting 'use_keyboard' to True.
                You can also programmatically skip all subsequent 'wait's by setting 'use_keyboard'
                to False.
        """
        self._view.render(*args, **kwargs)


class SearchView(ABC):

    def __init__(self, env: SearchProblem):
        pass

    @abstractmethod
    def render(self, *args, **kwargs):
        pass


class NullSearchView(SearchView):
    """'Fake' SearchView subclass that does nothing"""

    def render(self, *args, **kwargs):
        """Does not render anything, immediatelly returns"""
        pass


class TkSearchView(SearchView):
    """SearchView with Tkinter as backend"""

    def __init__(self, env: SearchProblem):
        self.env = env
        self.tk = tk.Tk()
        self.tk.title("Search Problem Visualization")
        self.tk.geometry("+0+0")
        self.canvas = SearchCanvas(self.tk, map=self.env.map)
        self.canvas.pack()

    def render(
        self,
        texts: dict[State, str] = {},
        colors: dict[State, float] = {},
        path: list[State] = [],
        current_state: State = None,
        next_states: list[State] = [],
        frontier_states: list[State] = [],
        wait: bool = False,
        use_keyboard: bool | None = None,
    ):
        if colors:
            self.canvas.set_square_colors_from_values(colors)
        else:
            self.canvas.set_square_colors_from_visited(self.env._visited)
        self.canvas.set_frontier_states(frontier_states)
        self.canvas.set_next_states(next_states)
        self.canvas.set_current_state(current_state)
        if path:
            self.canvas.draw_path(path)
        if texts:
            self.canvas.update_square_texts(texts)
        self.tk.update()
        if use_keyboard is not None:
            keyboard.STEPS_TO_SKIP = 0
            keyboard.SKIP = not use_keyboard
        if wait:
            keyboard.wait()
