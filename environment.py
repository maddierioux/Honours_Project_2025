"""Building blocks for simulating an RL agent/mouse's environment.

Classes and functions for simulating trace conditioning experiments and other
mousey experiences.

"""
from typing import List, Optional, Union, Iterable, SupportsInt
from numbers import Integral
from dataclasses import dataclass

import numpy as np

class State:
    def __init__(self, id_: int, label: Optional[str] = None):
        self._id = int(id_)
        if label is not None:
            self._label = str(label)
        else:
            self._label = None

    def __repr__(self) -> str:
        if self._label is not None:
            return f'State(id_={self._id}, label={self._label})'
        return f'State(id_={self._id})'

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other) -> bool:
        if isinstance(other, State):
            if (self.id == other.id) and (self.label != other.label):
                raise RuntimeError(
                    'Found two states with same id and different labels: '
                    f'{self}, {other}'
                )
        return int(self) == int(other)

    def __lt__(self, other) -> bool:
        return int(self) < int(other)

    def __le__(self, other) -> bool:
        return (self < other) or (self == other)

    def __gt__(self, other) -> bool:
        return int(self) > int(other)

    def __ge__(self, other) -> bool:
        return (self > other) or (self == other)

    def __int__(self) -> int:
        return self._id

    @property
    def id(self) -> int:
        return self._id

    @property
    def label(self) -> str:
        return self._label


@dataclass(frozen=True)
class Experience:
    states: np.ndarray
    rewards: np.ndarray

    def __init__(
        self,
        states: Union[State, Iterable[State]],
        rewards: Union[float, Iterable[float]],
    ):
        object.__setattr__(self, 'states', self._to_immutable_array(states))
        object.__setattr__(
            self, 'rewards', self._to_immutable_array(rewards)
        )
        if len(self.states) != len(self.rewards):
            raise ValueError('Expected states and rewards to be equal len')

    def __len__(self):
        assert len(self.states) == len(self.rewards)
        return len(self.states)

    def state_domain(self) -> (State, State):
        """Return a tuple of the minimum and maximum States."""
        return (min(self.states), max(self.states))

    def iter_one_hot(self):
        """Iterate over one-hot vector states and rewards.

        Returns
        -------
        Generator that yields (state, reward) pairs where the state is
        represented as a one-hot vector instead of as a State object.

        """
        min_state, max_state = self.state_domain()
        min_state_ind, max_state_ind = int(min_state), int(max_state)
        state_vector_len = max_state_ind - min_state_ind + 1
        def one_hot_state_reward_iterator():
            for s, r in zip(self.states, self.rewards):
                state_vector = np.zeros(state_vector_len, dtype=np.float16)
                state_vector[int(s) - min_state_ind] = 1.
                yield state_vector, r
        return one_hot_state_reward_iterator()

    @staticmethod
    def _to_immutable_array(x) -> np.ndarray:
        # Coerce scalar arguments to iterable
        try:
            iter(x)
        except TypeError:
            x = [x]

        arr = np.array(x, copy=True)
        arr.setflags(write=False)
        return arr


def concat_experiences(experiences: Iterable[Experience]) -> Experience:
    states = np.concatenate([e.states for e in experiences])
    rewards = np.concatenate([e.rewards for e in experiences])
    return Experience(states, rewards)


class TraceConditioningTrial:
    def __init__(
        self,
        *,
        cs_length: int,
        trace_length: int,
        us_length: int,
        us_value,
        base_state=State(0),
    ):
        if us_length < 1:
            raise ValueError(f'Expected us_length >= 1, got {us_length}.')

        self.cs_length = int(cs_length)
        self.trace_length = int(trace_length)
        self.us_length = int(us_length)
        self.us_value = us_value
        self.base_state = base_state

    def __repr__(self) -> str:
        return (
            'TraceConditioningTrial('
            f'cs_length={self.cs_length}, '
            f'trace_length={self.trace_length}, '
            f'us_length={self.us_length}, '
            f'us_value={self.us_value}, '
            f'base_state={self.base_state}'
            ')'
        )

    def __len__(self) -> int:
        return self.cs_length + self.trace_length + self.us_length

    def _cs_states(self) -> List[State]:
        result = [
            State(i, 'cs')
            for i in range(
                int(self.base_state), int(self.base_state) + self.cs_length
            )
        ]
        return result

    def _trace_states(self) -> List[State]:
        base = int(self.base_state) + self.cs_length
        return [
            State(i, 'trace') for i in range(base, base + self.trace_length)
        ]

    def _us_states(self) -> List[State]:
        base = int(self.base_state) + self.cs_length + self.trace_length
        result = [State(i, 'us') for i in range(base, base + self.us_length)]
        return result

    def ground_truth(self) -> Experience:
        """One state for each time step.

        Also called 'complete serial compound'.

        """
        return Experience(
            [*self._cs_states(), *self._trace_states(), *self._us_states(),],
            self._rewards(),
        )

    def trial(self) -> Experience:
        """No distinction between CS, trace, and US states."""
        return Experience(
            [State(int(self.base_state), 'trial')] * len(self), self._rewards()
        )

    def no_time_perception(self) -> Experience:
        """Distinct CS, trace, and US states, but no sub-states.

        Nothing to distinguish the start of the CS from end of the CS, for
        example.

        """
        return Experience(
            [
                *[State(int(self.base_state), 'cs')] * self.cs_length,
                *[State(int(self.base_state) + 1, 'trace')]
                * self.trace_length,
                *[State(int(self.base_state) + 2, 'us')] * self.us_length,
            ],
            self._rewards(),
        )

    def sensory(self, iti_state) -> Experience:
        """Distinct CS and US states but no trace state.

        No time perception within CS and US states and trace is
        indistinguishable from ITI.

        """
        return Experience(
            [
                *[State(int(self.base_state), 'cs')] * self.cs_length,
                *[iti_state] * self.trace_length,
                *[State(int(self.base_state) + 1, 'us')] * self.us_length,
            ],
            self._rewards(),
        )

    def blind(self, iti_state) -> Experience:
        """Trial is indistinguishable from ITI."""
        return Experience([iti_state] * len(self), self._rewards())

    def _rewards(self) -> List[float]:
        return [
            *[0.0] * (self.cs_length + self.trace_length),
            *[self.us_value] * self.us_length,
        ]


class UncuedTrial:
    def __init__(
        self, *, us_length: int, us_value: float, base_state=State(0)
    ):
        self.us_length = int(us_length)
        self.us_value = us_value
        self.base_state = base_state

    def __repr__(self) -> str:
        return (
            f'UncuedTrial(us_length={self.us_length}, '
            f'us_value={self.us_value}, base_state={self.base_state})'
        )

    def __len__(self) -> int:
        return self.us_length

    def ground_truth(self) -> Experience:
        """One state for each time step.

        Also called 'complete serial compound'.

        """
        return [
            State(i, 'us')
            for i in range(
                int(self.base_state), int(self.base_state) + self.us_length
            )
        ]

    def no_time_perception(self) -> Experience:
        """Distinct CS, trace, and US states, but no sub-states.

        Nothing to distinguish the start of the CS from end of the CS, for
        example.

        """
        return Experience(
            [State(int(self.base_state), 'us') for _ in range(self.us_length)],
            self._rewards(),
        )

    def sensory(self, iti_state) -> Experience:
        """Distinct CS and US states but no trace state.

        No time perception within CS and US states and trace is
        indistinguishable from ITI.

        """
        return Experience(self.no_time_perception(), self._rewards())

    def trial(self) -> Experience:
        """No distinction between CS, trace, and US states."""
        return Experience(self.no_time_perception(), self._rewards())

    def blind(self, iti_state) -> Experience:
        """Trial is indistinguishable from ITI."""
        return Experience([iti_state] * len(self), self._rewards())

    def _rewards(self) -> List[float]:
        return [self.us_value] * self.us_length


def geometric_dwell(
    state: State,
    reward: float,
    exit_probability: Optional[float] = None,
    mean_dwell_time: Optional[float] = None,
) -> Experience:
    """Dwell in a state according to a Bernoulli process.

    Dwell in the specified state, collecting the specified reward at each time
    step, and exit according to a Bernoulli process such that the dwell time
    follows a geometric distribution.

    Parameters
    ----------
    state
        State to dwell in.
    reward
        Reward to collect at each time step.
    exit_probability, mean_dwell_time
        Set the dwell time either in terms of the exit probability at each time
        step or the mean number of time steps before exiting.

    Returns
    -------
    Experience containing states and rewards.

    """
    if exit_probability is not None and mean_dwell_time is None:
        if exit_probability < 0.0 or exit_probability > 1.0:
            raise ValueError(
                'Expected exit_probability to be between 0 and 1, got '
                + str(exit_probability)
            )
    elif exit_probability is None and mean_dwell_time is not None:
        if mean_dwell_time < 1.0:
            raise ValueError(
                f'Expected mean_dwell_time to be >= 1, got {mean_dwell_time}'
            )
        exit_probability = 1.0 / mean_dwell_time
    else:
        raise ValueError(
            'Either exit_probability or mean_dwell_time must be specified'
        )

    dwell_time = np.random.geometric(exit_probability)
    return Experience([state] * dwell_time, [reward] * dwell_time)


def fixed_dwell(state: State, reward: float, dwell_time: int) -> Experience:
    """Dwell in a state for a fixed number of time steps."""
    if dwell_time < 0.:
        raise ValueError(f'Expected dwell_time >= 0, got {dwell_time}')
    return Experience([state] * dwell_time, [reward] * dwell_time)


def one_hot(
    states: Union[Experience, Iterable[SupportsInt]], domain='auto'
) -> np.ndarray:
    """Map an M-length list of states to an MxN one-hot matrix.

    Parameters
    ----------
    states
        List of states to encode.
    domain: 'auto', int, or (int, int)
        Control how states are mapped to the column space of the one-hot
        matrix.
            - If 'auto', int(state) is used as the column index and the number
              of columns is set to max([int(s) for s in states]).
            - If a single int, int(state) is still used as the column index but
              the argument specifies the number of columns. Out of bounds
              states will result in an error.
            - If a pair of ints, the first int is subtracted from int(state) to
              get the column index  and the difference between the two ints
              gives the number of columns.

    """
    if isinstance(states, Experience):
        states = states.states
        if domain == 'auto':
            domain = states.state_domain()
    col_indices = np.array([int(s) for s in states])
    # Infer/set matrix shape based on col_indices and domain argument, plus
    # perform shape checks.
    if domain == 'auto':
        shape = (len(col_indices), col_indices.max() + 1)
    elif isinstance(domain, Integral):
        if np.any(col_indices >= domain):
            raise IndexError(
                f'Found state(s) above domain upper bound {domain}'
            )
        shape = (len(col_indices), domain)
    elif len(domain) == 2:
        if not (
            isinstance(domain[0], Integral) and isinstance(domain[1], Integral)
        ):
            raise TypeError(
                f"Expected 'auto', int, or (int, int) for domain, got {domain}"
            )
        shape = (len(col_indices), domain[1] - domain[0])
        if np.any(col_indices < domain[0]):
            raise IndexError(
                f'Found state(s) below domain lower bound {domain[0]}'
            )
        if np.any(col_indices >= domain[1]):
            raise IndexError(
                f'Found state(s) above domain upper bound {domain[1]}'
            )
        col_indices -= domain[0]
    else:
        raise ValueError(
            f"Expected 'auto', int, or (int, int) for domain, got {domain}"
        )
    # Construct one-hot matrix.
    state_matrix = np.zeros(shape)
    state_matrix[np.arange(len(states)), col_indices] = 1
    return state_matrix
