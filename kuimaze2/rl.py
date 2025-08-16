import random
from typing import Optional

from kuimaze2.mdp import MDP, Action, State, ActionValues, NullMDPView, TkMDPView
from kuimaze2.exceptions import NeedsResetError, ResetImpossibleError


class RLProblem:
    """Fully observable RL problem, i.e., the observation is the actual state."""

    def __init__(self, *args, graphics: bool = False, **kwargs):
        self._mdp = MDP(*args, **kwargs)
        self._current_state = next(iter(self._mdp._goals))
        self._episode_finished = True
        self._view = NullMDPView(self._mdp) if not graphics else TkMDPView(self._mdp)

    def reset(
        self, /, state: Optional[State] = None, random_start: bool = False
    ) -> State:
        """Reset the environment, return a new random initial (non-terminal) state for an episode.

        You can specify the initial state for testing purposes.
        """
        self._episode_finished = False
        if state:
            self._current_state = state
        elif random_start:
            self._current_state = random.choice(self._mdp.get_states())
        else:
            self._current_state = self._mdp._map.start
        if not self._current_state:
            raise ResetImpossibleError(
                "RLProblem: Unable to set new start state. "
                "Specify a particular state, define a start state on a map, or set random_start to True."
            )
        return self._current_state

    def get_states(self) -> list[State]:
        """Return all states in the MDP problem"""
        return self._mdp.get_states()

    def get_action_space(self) -> list[Action]:
        """Return the union of all actions applicable in any state (except the EXIT action)"""
        return [action for action in Action]

    def sample_action(self, action_probs: Optional[ActionValues] = None) -> Action:
        """Return a random action from the action space"""
        if not action_probs:
            return random.choice(self.get_action_space())
        assert sum(action_probs.values()) == 1
        return random.choices(
            list(action_probs.keys()), weights=list(action_probs.values())
        )[
            0
        ]  # Intentional: take the first item, random.choices() always returns a list

    def step(self, action: Action) -> tuple[State, float, bool]:
        """Take a single step in the environment"""
        if self._episode_finished:
            raise NeedsResetError(
                "RLProblem: Episode terminated. You must call reset() first."
            )
        # If we are going to make a step from terminal, then we are done
        if self._mdp.is_terminal(self._current_state):
            self._episode_finished = True
        reward = self._mdp.get_reward(self._current_state)
        actual_action = self._mdp._actions_model.sample_action(action)
        self._current_state = self._mdp._get_transition_result(
            self._current_state, actual_action
        )
        return self._current_state, reward, self._episode_finished

    def render(self, *args, **kwargs):
        """Display/update the graphical representation of the environment

        Arguments (all optional):

            square_texts: dict[State, str] = {}
                A dictionary of (State, str) pairs; assignment of texts to States.
                Each cell in the maze can display a text string. It can be an ID,
                value, you name it.

            square_colors: dict[State, float] = {}
                A dictionary of (State, float) pairs; assignment of values to States.
                The values are used to determine the color intensity of each State.

            path: list[State] = []
                A list of states forming a path.

            state_action_arrow: tuple[State, Action]
                A state-action pair for which an arrow should be drawn.

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
