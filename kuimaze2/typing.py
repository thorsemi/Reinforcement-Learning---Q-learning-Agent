from collections.abc import MutableMapping

from kuimaze2.map import Action, State

type Path = list[State]
"""Path type annotation: a sequence of states. May be useful for solvers."""

type ActionValues = MutableMapping[Action, float]
"""ActionValues type annotation: representation of a distribution over actions."""

type VTable = MutableMapping[State, float]
"""VTable type annotation: a tabular representation of V function. May be useful for solvers."""

type QTable = MutableMapping[State, ActionValues]
"""QTable type annotation: a tabular representation of Q function. May be useful for solvers."""

type Policy = MutableMapping[State, Action]
"""Policy type annotation: a representation of a policy, the return type of MDP and RL problems."""
