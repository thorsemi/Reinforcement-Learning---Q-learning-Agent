from collections import Counter

import pytest
from kuimaze2 import Action, Map, RLProblem, State
from kuimaze2.exceptions import NeedsResetError
from kuimaze2.mdp import MDP, Rewards


def test_creation():
    map = Map.from_string("G")
    env = RLProblem(map)
    assert env


def test_reset():
    map = Map.from_string("SG")
    env = RLProblem(map)
    init_state = env.reset(random_start=True)
    assert init_state in env.get_states()


def test_get_action_space():
    map = Map.from_string("G")
    env = RLProblem(map)
    actions = env.get_action_space()
    assert len(actions) == 4
    assert Action.UP in actions
    assert Action.RIGHT in actions
    assert Action.DOWN in actions
    assert Action.LEFT in actions


class TestSampleAction:
    def test_sample_action_uniformly(self):
        # This is a stochastic test. Not good.
        # How to test that sample_action returns all possible actions uniformly?
        map = Map.from_string("SG")
        env = RLProblem(map)
        init_state = env.reset()
        action = env.sample_action()
        assert action in env.get_action_space()

    def test_sample_action_uniformly_action_probs(self):
        # This is a stochastic test. Not good.
        # How to test that sample_action returns all possible actions uniformly?
        map = Map.from_string("SG")
        env = RLProblem(map)
        init_state = env.reset()
        action = env.sample_action(
            {Action.UP: 0.25, Action.RIGHT: 0.25, Action.DOWN: 0.25, Action.LEFT: 0.25}
        )
        assert action in env.get_action_space()

    def test_sample_action_UP(self):
        map = Map.from_string("SG")
        env = RLProblem(map)
        init_state = env.reset()
        action = env.sample_action(Counter({Action.UP: 1}))
        assert action == Action.UP

    def test_sample_action_UP_multi(self):
        map = Map.from_string("SG")
        env = RLProblem(map)
        init_state = env.reset()
        action = env.sample_action(
            {Action.UP: 1, Action.RIGHT: 0, Action.DOWN: 0, Action.LEFT: 0}
        )
        assert action == Action.UP

    def test_sample_action_RIGHT(self):
        map = Map.from_string("SG")
        env = RLProblem(map)
        init_state = env.reset()
        action = env.sample_action(Counter({Action.RIGHT: 1}))
        assert action == Action.RIGHT

    def test_sample_action_DOWN(self):
        map = Map.from_string("SG")
        env = RLProblem(map)
        init_state = env.reset()
        action = env.sample_action(Counter({Action.DOWN: 1}))
        assert action == Action.DOWN

    def test_sample_action_LEFT(self):
        map = Map.from_string("SG")
        env = RLProblem(map)
        init_state = env.reset()
        action = env.sample_action(Counter({Action.LEFT: 1}))
        assert action == Action.LEFT


class TestStep:
    def test_terminal_state(self):
        map = Map.from_string("G")
        env = RLProblem(map)
        with pytest.raises(NeedsResetError):
            new_state, reward, terminated = env.step(Action.UP)

    def test_nonterminal_state(self):
        map = Map.from_string("S.G")
        env = RLProblem(map, rewards=Rewards(goal=10, danger=-10, normal=-5))
        init_state = env.reset(State(0, 0))
        assert init_state == State(0, 0)
        new_state, reward, terminated = env.step(Action.RIGHT)
        assert new_state == State(0, 1)
        assert reward == -5
        assert terminated == False

    def test_reaching_goal(self):
        map = Map.from_string("SG")
        env = RLProblem(map, rewards=Rewards(goal=10, danger=-10, normal=-5))
        init_state = env.reset(state=State(0, 0))
        assert init_state == State(0, 0)
        new_state, reward, terminated = env.step(Action.RIGHT)
        assert new_state == State(0, 1)
        assert reward == -5  # Include also reward for reaching the goal state
        assert terminated == False
        new_state, reward, terminated = env.step(Action.RIGHT)
        assert new_state == None
        assert reward == 10  # Include also reward for reaching the goal state
        assert terminated == True
