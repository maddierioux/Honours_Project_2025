from typing import Union, Set
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import numba as nb

from environment import State


def discount(x: np.ndarray, discounting_factor: float) -> np.ndarray:
    """Apply time discounting to a time-dependent signal."""
    discounted_output = np.empty_like(x, dtype=np.float64)
    discounted_output[-1] = x[-1]
    for i in range(2, len(x) + 1):
        discounted_output[-i] = (
            x[-i] + discounting_factor * discounted_output[-i + 1]
        )
    return discounted_output


@nb.jit(nopython=True)
def iti_to_max_value_ratio(
    effective_trial_length: float, iti_to_trial_ratio: float
) -> float:
    """Exact ratio of ITI value to trial end value under trace conditioning.

    Parameters
    ----------
    effective_trial_length
        Ratio of trial duration to discounting time constant.
    iti_to_trial_ratio
        Ratio of mean ITI duration to trial duration.

    Returns
    -------
    Ratio of value during the ITI to value just before reward delivery.

    See Also
    --------
    analytical_iti_value.ipynb

    """
    return np.exp(-effective_trial_length) / (
        effective_trial_length * iti_to_trial_ratio + 1.0
    )


@nb.jit(nopython=True)
def iti_to_max_value_ratio_dimensional(
    trial_length: float, discount_tau: float, iti_length: float
) -> float:
    """Alternate implementation of iti_to_max_value_ratio.

    This implementation is still valid when trial_length == 0 because it does
    not use the iti_to_trial_ratio.

    """
    return np.exp(-trial_length / discount_tau) / (
        iti_length / discount_tau + 1.0
    )


def perceptual_lag(x, lag_timesteps=150):
    """Lag a signal to simulate perceptual delay."""
    if lag_timesteps < 0:
        raise ValueError(f'Expected lag_timesteps >= 0, got {lag_timesteps}')
    if lag_timesteps == 0:
        return x
    return np.concatenate([[x[0]] * lag_timesteps, x[:-lag_timesteps]])


def normative_value(
    experience,
    iti_state: State,
    trial,
    iti_duration,
    discount_tau,
    dt,
):
    value_map = {}

    # Store ITI value.
    # Note: US duration does not count towards trial duration for purposes of
    # calculating normalized ITI value.
    trial_duration = (len(trial) - trial.us_length) * dt
    value_map[iti_state] = iti_to_max_value_ratio_dimensional(
        trial_duration, discount_tau, iti_duration
    )

    # Store exponentially-discounted pre-reward state values
    num_pre_reward_states = len(trial) - trial.us_length
    if num_pre_reward_states > 0:
        for i, s in enumerate(trial.ground_truth().states[:num_pre_reward_states]):
            time_to_reward = (num_pre_reward_states - i) * dt
            # Value v_t is calculated based on time_to_reward - dt because
            # v_t = R_{t+1} + ...
            # In other words, a reward obtained dt in the future should not be
            # discounted at all.
            value_map[s] = np.exp(-(time_to_reward - dt) / discount_tau)
        # The last state to be evaluated should be dt before reward.
        assert np.isclose(time_to_reward - dt, 0)

    # Store reward state values
    num_reward_states = trial.us_length
    scaling_factor = 1.0 - value_map[iti_state]
    # offset represents the minimum value during the reward period, ie right at
    # the end of reward. This is the ITI value discounted by one time step
    # because v_t = R_{t+1} + gamma * v_{t+1}, where R_{t+1} = 0 and
    # v_{t+1} = v_iti
    offset = value_map[iti_state] * np.exp(-dt / discount_tau)
    for i, s in enumerate(trial.ground_truth().states[num_pre_reward_states:]):
        time_to_reward_end = (num_reward_states - i) * dt
        # Value is calculated based on time_to_reward_end - dt for same reason
        # as above.
        norm_value_during_reward = (
            1.0 - np.exp(-(time_to_reward_end - dt) / discount_tau)
        ) / (1.0 - np.exp(-(num_reward_states * dt) / discount_tau))
        value_map[s] = scaling_factor * norm_value_during_reward + offset
    assert np.isclose(time_to_reward_end - dt, 0)

    # Compute value signal
    v_t = np.empty_like(experience.states, dtype=np.float64)
    for i, s in enumerate(experience.states):
        v_t[i] = value_map[s]

    return v_t


def imperfect_value(trial_duration: float, reward_duration: float, discount_tau: float, learning_fraction: float, dt: float) -> np.ndarray:
    """Variation of normative value for Luo lab experiments.
    
    Compared with the standard `normative_value` function, this function
    has the following differences:
    - ITI duration is considered to be infinite, such that ITI value is zero.
      The reason for doing this is that serotonin neurons have extremely low
      firing rates outside of the reward zone in the Li data.
    - We allow the value leading up to the reward to be only a fraction of
      the true value. The reason is that the animals may not have fully learned
      to predict the reward in this particular task (no reward zone cue, freely-
      moving animals may perceive delays differently from stationary animals).
    
    """
    eps = 0.01 * dt
    t_to_trial_end = trial_duration - np.arange(0, trial_duration - eps, dt)
    pre_reward_value = np.exp(-(t_to_trial_end - dt) / discount_tau)
    t_to_reward_end = reward_duration - np.arange(0, reward_duration - eps, dt)
    reward_epoch_value = (
        (1.0 - np.exp(-(t_to_reward_end - dt) / discount_tau)) 
        / (1.0 - np.exp(-reward_duration / discount_tau))
    )
    # NB: Since ITI value is zero, reward_epoch_value doesn't have to be scaled or
    # offset as in `normative_value`
    return np.concatenate([learning_fraction * pre_reward_value, reward_epoch_value])


def stack_trials(
    values: np.ndarray,
    states: np.ndarray,
    iti_states: Union[State, Set[State]],
    trial_length: int,
    baseline_length: int = 0,
) -> np.ndarray:
    """Extract a time slice around each trial and stack into a matrix.

    Parameters
    ----------
    values
        Values to stack (usually value in the RL sense).
    states
        Array of State objects. Must have same length as values.
    iti_states
        Set of States that indicate an inter-trial interval. All timesteps for
        which the state is not in this set are considered to be in a trial.
    trial_length
        Trial duration in timesteps. All trials must have the same length in
        order to be stackable.
    baselin_length
        Number of timesteps to include as a baseline before the start and after
        the end of each trial.

    Returns
    -------
    2D array of values during each trial where each row is one trial.

    """
    if not isinstance(iti_states, set):
        # Coerce iti_states argument to a set.
        iti_states = {iti_states}
    iti_state_mask = np.array([s in iti_states for s in states])
    trial_start_mask = np.diff((~iti_state_mask).astype(int))
    if not iti_state_mask[0]:
        # First time step is in a trial
        trial_start_mask = np.concatenate([[1], trial_start_mask]) > 0.5
    else:
        trial_start_mask = np.concatenate([[0], trial_start_mask]) > 0.5
    trials = []
    for i, trial_start_flag in enumerate(trial_start_mask):
        if trial_start_flag:
            # Find the lower and upper bounds of the trial slice to stack, but
            # do not allow the slice to extend before the start or after the
            # end of the timeseries.
            lb = max(0, i - baseline_length)
            ub = min(i + trial_length + baseline_length, len(values))
            # If trial is at the start or end of the timeseries and a full
            # baseline period is not available, pad the baseline with np.nan.
            lpad = [np.nan] * (lb - (i - baseline_length))
            upad = [np.nan] * ((i + trial_length + baseline_length) - ub)
            trial = np.concatenate([lpad, values[lb:ub], upad,])
            assert len(trial) == (2 * baseline_length + trial_length), (
                f'Expected trial length {2 * baseline_length + trial_length} '
                f'for trial starting at {i}, got {len(trial)}'
            )
            trials.append(trial)
    return np.array(trials)


def compose(*funcs):
    """Compose functions from right to left.

    Examples
    --------
    >>> f = compose(lambda x: 2*x, lambda x: x - 1, lambda x: 2 * x)
    >>> f(5)
    18

    """

    def compose_two(g, f):
        def composed(*args, **kwargs):
            return f(g(*args, **kwargs))

        return composed

    return reduce(compose_two, reversed(funcs))


class NbSubplots:
    """Matplotlib subplots context manager for Jupyter notebooks.

    In 2022, matplotlib leaks memory each time a reference to a figure is
    dropped. On machines with limited memory, it's easy to use up the available
    memory and crash the Python interpreter when generating many plots to test
    new code. Here's an example of code that causes this problem:

    >>> fig, axes = plt.subplots(1, 1)
    >>> axes.plot([1, 2, 3])
    >>> fig.show()
    >>> fig, axes = plt.subplots(1, 1)  # Memory for original figure leaked.

    This context manager shows the figure, cleans up its memory when it exits,
    and can be used exactly like plt.subplots().

    >>> with NbSubplots(1, 1) as (fig, axes):
    >>>     axes.plot([1, 2, 3])
    >>>     plt.show()
    >>>     # No memory is leaked!

    """
    def __init__(
        self,
        nrows=1,
        ncols=1,
        *,
        sharex=False,
        sharey=False,
        squeeze=True,
        subplot_kw=None,
        gridspec_kw=None,
        **fig_kw
    ):
        """Initialize Subplots.

        Parameters are the same as for plt.subplots().

        """
        self._fig, self._axes = plt.subplots(
            nrows,
            ncols,
            sharex=sharex,
            sharey=sharey,
            squeeze=True,
            subplot_kw=None,
            gridspec_kw=None,
            **fig_kw
        )

    def __enter__(self):
        return self._fig, self._axes

    def __exit__(self, type, value, traceback):
        # Clean up figure memory.
        self._fig.clf()
        plt.close(self._fig)
