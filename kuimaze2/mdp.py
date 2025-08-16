import random
import tkinter as tk
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from kuimaze2 import keyboard
from kuimaze2.map import Action, Map, State
from kuimaze2.rendering import QValueCanvas, ValueCanvas
from kuimaze2.typing import ActionValues


@dataclass(frozen=True)
class Rewards:
    """Rewards for relevant state roles"""

    goal: float = 1.0
    danger: float = -1.0
    normal: float = -0.04


class Confusion(IntEnum):
    """Possible confusions of actions when performing stochastic actions."""

    NONE = 0
    RIGHT = 1
    BACKWARD = 2
    LEFT = 3

    def apply_to(self, action: Action) -> Action:
        return Action((action + self) % 4)


class ActionsModel(ABC):
    """Base class for deterministic and stochastic action models."""

    @abstractmethod
    def get_actions_probs(self, action: Action) -> ActionValues: ...

    @abstractmethod
    def sample_action(self, action: Action) -> Action: ...


class DeterministicActions(ActionsModel):
    """Model of deterministic actions; no confusion."""

    def get_actions_probs(self, action: Action) -> ActionValues:
        """Return the original action with probability 1.0."""
        return {action: 1.0}

    def sample_action(self, action: Action) -> Action:
        """Return the original action."""
        return action


class StochasticActions(ActionsModel):
    """Model for stochastic actions with possible confusions."""

    def __init__(self, forward: float, left: float, right: float, backward: float):
        total = forward + left + right + backward
        assert abs(1 - total) < 1e-6, "Sum of confusion probabilities must be 1.0"
        # Renormalize in any case
        left, right, backward = left / total, right / total, backward / total
        forward = 1 - left - right - backward
        self.confusion_probs = {
            Confusion.NONE: forward,
            Confusion.RIGHT: right,
            Confusion.BACKWARD: backward,
            Confusion.LEFT: left,
        }

    def _sample_confusion(self) -> Confusion:
        return random.choices(
            list(self.confusion_probs.keys()),
            weights=list(self.confusion_probs.values()),
        )[0]

    def get_actions_probs(self, action: Action) -> ActionValues:
        """Return the actual possible actions after applying the confusion, with their probs."""
        return {
            confusion.apply_to(action): prob
            for confusion, prob in self.confusion_probs.items()
        }

    def sample_action(self, action: Action) -> Action:
        """Return an actual action after applying the confusion."""
        confusion = self._sample_confusion()
        return confusion.apply_to(action)


class MDP:
    """MDP problem class defined over a map with deterministic or stochastic actions."""

    def __init__(
        self,
        map: Map,
        action_probs: Optional[dict[str, float]] = None,
        rewards: Optional[Rewards] = None,
    ):
        """Initialize the MDP problem.

        Arguments:

             map: Map
                 The Map instance. Create using `kuimaze2.Map.from_string()`
                 or using `kuimaze2.map_image.map_from_image()`.

             action_probs: dict[str, float]
                 The probabilty distribution for action confusions. Defaults to deterministic actions.
                 Stochastic actions can be specified by providing a dictionary
                 with keys 'forward', 'left', 'right' and 'backward', e.g.,
                 `action_probs=dict(forward=0.8, left=0.1, right=0.1, backward=0.0)`.

             rewards: Rewards
                 The rewards for leaving states with certain relevant state roles.
                 Defaults to `Rewards(goal=1.0, danger=-1.0, normal=-0.04)`.
        """
        self._map = map
        self._actions_model = (
            DeterministicActions()
            if not action_probs
            else StochasticActions(**action_probs)
        )
        self._rewards = rewards or Rewards()
        # MDP may redefine goal and danger state in the future
        # Right now, just use those in the map
        self._goals = self._map.goals
        self._dangers = self._map.dangers

    def get_states(self) -> list[State]:
        """Return all free states in the MDP problem, including terminals."""
        return [cell.position for cell in self._map if cell.is_free()]

    def get_non_terminal_states(self) -> list[State]:
        """Return all free states in the MDP problem, excluding terminals."""
        return [state for state in self.get_states() if not self.is_terminal(state)]

    def get_actions(self, state: State) -> list[Action]:
        """Return a list of all possible actions in the given state."""
        return list(Action)

    def get_reward(self, state: State) -> float:
        """Return reward for leaving a state."""
        if self._is_goal(state):
            return self._rewards.goal
        if self._is_danger(state):
            return self._rewards.danger
        return self._rewards.normal

    def _get_transition_result(self, state: State, action: Action) -> State | None:
        """Apply a DETERMINISTIC action to a state."""
        if self.is_terminal(state):
            return None
        return self._map.get_transition_result(state, action)

    def get_next_states_and_probs(
        self, state: State, action: Action
    ) -> list[tuple[State | None, float]]:
        """Return a list of possible next states and their probabilities, after applying the action in the state."""
        actions_probs = self._actions_model.get_actions_probs(action)
        return [
            (self._get_transition_result(state, action), prob)
            for action, prob in actions_probs.items()
        ]

    def is_terminal(self, state: State) -> bool:
        """Return True for terminal states, False otherwise."""
        return self._is_goal(state) or self._is_danger(state)

    def _is_goal(self, state: State) -> bool:
        """Return True for goal states, False otherwise."""
        return state in self._goals

    def _is_danger(self, state: State) -> bool:
        """Return True for danger states, False otherwise."""
        return state in self._dangers


class MDPProblem(MDP):
    def __init__(
        self,
        map: Map,
        action_probs: Optional[dict[str, float]] = None,
        rewards: Optional[Rewards] = None,
        graphics: bool = False,
    ):
        super().__init__(map, action_probs, rewards)
        self._view = NullMDPView(self) if not graphics else TkMDPView(self)

    def render(self, *args, **kwargs):
        """Display/update the graphical representation of the environment

        Arguments (all optional):

            square_texts: dict[State, str] = {}
                A dictionary of (State, str) pairs; assignment of texts to States.
                Each cell in the maze can display a text string. It can be an ID, cost,
                heuristic value, all of it.

            square_colors: dict[State, float] = {}
                A dictionary of (State, float) pairs; assignment of values to States.
                The values are used to determine the color intensity of each State.

            triangle_texts: dict[(State, Action), str] = {}
                A dictionary of ((State, Action), str) pairs; assignment of texts
                to (State, Action) pairs displayed in the triangle corresponding to State-Action.

            triangle_colors: dict[(State, Action), float] = {}
                A dictionary of ((State, Action), float) pairs; assignment of values to State-Action.
                The values are used to determine the color intensity of each State-Action triangle.

            middle_texts: dict[State, str] = {},
                Texts displayed in the middle of the square (state) in the triangle view.

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


class MDPView(ABC):
    def __init__(self, env: MDPProblem):
        pass

    @abstractmethod
    def render(self, *args, **kwargs):
        pass


class NullMDPView(MDPView):
    """'Fake' MDPView subclass that does nothing"""

    def render(self, *args, **kwargs):
        """Does not render anything, immediatelly returns"""
        pass


class TkMDPView(MDPView):
    """MDPView with Tkinter as backend"""

    def __init__(self, env: MDPProblem):
        self.env = env
        self.tk = tk.Tk()
        self.tk.title("MDP Problem Visualization")
        self.tk.geometry("+0+0")
        self.v_canvas = ValueCanvas(
            self.tk,
            map=self.env._map,
            value_range=(self.env._rewards.danger, self.env._rewards.goal),
        )
        self.v_canvas.pack()
        self.q_canvas = QValueCanvas(
            self.tk,
            map=self.env._map,
            value_range=(self.env._rewards.danger, self.env._rewards.goal),
        )
        self.q_canvas.pack()

    def render(
        self,
        square_texts: dict[State, str] = {},
        square_colors: dict[State, float] = {},
        path: list[State] = [],
        state_action_arrow: tuple[State, Action] = None,
        triangle_texts: dict[(State, Action), str] = {},
        triangle_colors: dict[(State, Action), float] = {},
        middle_texts: dict[State, str] = {},
        wait: bool = False,
        use_keyboard: bool | None = None,
    ):
        if square_colors:
            self.v_canvas.set_square_colors_from_values(square_colors)
        if state_action_arrow:
            self.v_canvas.draw_state_action_arrow(*state_action_arrow)
        else:
            self.v_canvas.hide_state_action_arrow()
        if path:
            self.v_canvas.draw_path(path)
        if square_texts:
            self.v_canvas.update_square_texts(square_texts)
        if triangle_colors:
            self.q_canvas.set_triangle_colors_from_qvalues(triangle_colors)
        if triangle_texts:
            self.q_canvas.update_triangle_texts(triangle_texts)
        if middle_texts:
            self.q_canvas.update_square_texts(middle_texts)
        self.tk.update()
        if use_keyboard is not None:
            keyboard.STEPS_TO_SKIP = 0
            keyboard.SKIP = not use_keyboard
        if wait:
            keyboard.wait()
