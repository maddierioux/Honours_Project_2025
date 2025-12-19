"""DRN encoding filter.

See Also
--------
leaky_integration_of_derivative.ipynb

"""
from typing import Callable

import numpy as np
import numba as nb


def auto_encode(
    input_signal: np.ndarray,
    encoder: Callable[[np.ndarray], np.ndarray],
    decoder: Callable[[np.ndarray], np.ndarray],
) -> dict:
    """Encode then decode a signal."""
    encoded = encoder(input_signal)
    decoded = decoder(encoded)
    return {'input': input_signal, 'encoded': encoded, 'decoded': decoded}


def neural_filter(
    input_signal: np.ndarray,
    adaptation_timescale: float,
    adaptation_strength: float,
    normalize_steady_state: bool = True,
    return_adaptation: bool = False,
) -> np.ndarray:
    """Neural rate model with rectification and subtractive adaptation.

    Rate model is defined as

        y_t = relu(x_t - a u_t)
        du/dt = (y_t - u_t) / tau

    where y_t is the output, x_t is the input, u_t is adaptation, and a and tau
    parameterize the strength and timescale of adaptation, respectively.

    """
    if adaptation_timescale < 0.0:
        raise ValueError(
            f'Expected 0 <= adaptation_timescale, got {adaptation_timescale}'
        )
    if adaptation_strength < 0.0:
        raise ValueError(
            f'Expected 0 <= adaptation_strength, got {adaptation_strength}'
        )

    if normalize_steady_state:
        # Increase the gain of the signal so that steady state with adaptation
        # is same as without adaptation.
        input_signal = input_signal * (1.0 + adaptation_strength)

    integrated_dynamics = _integrate_neural_dynamics(
        input_signal, adaptation_timescale, adaptation_strength,
    )

    if return_adaptation:
        return integrated_dynamics.T
    return integrated_dynamics[:, 0]


@nb.jit(nopython=True)
def _integrate_neural_dynamics(
    input_signal: np.ndarray,
    adaptation_timescale: float,
    adaptation_strength: float,
) -> np.ndarray:
    """Integrate dynamics of neural rate model.

    Dynamics of adaptation are integrated using the second order Runge-Kutta
    method. I wrote this function instead of using scipy.integrate.solve_ivp in
    order to have fast and accurate integration with a fixed step size. In my
    tests, this function is ~1000X faster than solve_ivp and ~4X slower than
    numba-accelerated forward Euler. Compared with forward Euler, this function
    can be used with a step size at least 10X larger, yielding a net ~2X
    speedup and nicer looking results.

    Reference for RK2 method: https://web.mit.edu/10.001/Web/Course_Notes/Differential_Equations_Notes/node5.html

    """
    solution = np.empty((len(input_signal), 2), dtype=np.float64)
    solution[0, :] = [
        max(input_signal[0], 0.0) / (1.0 + adaptation_strength)
    ] * 2
    for t, (x_t, x_tpp) in enumerate(zip(input_signal[:-1], input_signal[1:])):
        # Integrate adaptation dynamics using second order Runge-Kutta.
        k1 = _neural_adaptation_derivative(
            x_t, solution[t, 1], adaptation_strength, adaptation_timescale
        )
        k2 = _neural_adaptation_derivative(
            x_tpp,
            solution[t, 1] + k1,
            adaptation_strength,
            adaptation_timescale,
        )
        next_adaptation = solution[t, 1] + (k1 + k2) / 2.0
        solution[t + 1, :] = [
            _neural_output(x_tpp, next_adaptation, adaptation_strength),
            next_adaptation,
        ]
    return solution


@nb.jit(nopython=True)
def _neural_output(
    x: float, adaptation: float, adaptation_strength: float
) -> float:
    """Output of DRN rate model with subtractive adaptation."""
    return max(x - adaptation_strength * adaptation, 0.0)


@nb.jit(nopython=True)
def _neural_adaptation_derivative(
    x: float,
    adaptation: float,
    adaptation_strength: float,
    adaptation_timescale: float,
):
    """Dynamics of adaptation in DRN rate model."""
    return (
        _neural_output(x, adaptation, adaptation_strength) - adaptation
    ) / adaptation_timescale


def exponential_filter(
    input_signal: np.ndarray, time_constant: float
) -> np.ndarray:
    """Convolve the input with an exponential filter.

    Parameters
    ----------
    input_signal
        Signal to be filtered.
    time_constant
        Time constant of exponential filter in units of time steps.

    Returns
    -------
    Filtered array.

    """
    filtered_output = np.empty_like(input_signal)
    decay_factor = np.exp(-1.0 / time_constant)
    filtered_output[0] = input_signal[0]
    for i in range(1, len(input_signal)):
        filtered_output[i] = filtered_output[
            i - 1
        ] * decay_factor + input_signal[i - 1] * (1.0 - decay_factor)

    return filtered_output


def rectifying_filter(input_signal: np.ndarray) -> np.ndarray:
    """Filter that clips input at zero."""
    return np.clip(input_signal, 0.0, np.inf)


def ideal_deriv_filter(
    input_signal: np.ndarray, deriv_weight: float
) -> np.ndarray:
    """Transform input into a weighted sum of its intensity and derivative."""
    deriv = np.concatenate([[0.0], np.diff(input_signal)])
    return np.asarray(input_signal) + deriv_weight * deriv


def rectified_ideal_deriv_filter(
    input_signal: np.ndarray, deriv_weight: float
) -> np.ndarray:
    rectified_input = np.clip(input_signal, 0, np.inf)
    deriv = np.concatenate([[0.0], np.diff(rectified_input)])
    return np.clip(rectified_input + deriv_weight * deriv, 0, np.inf)

    out = [np.clip(input_signal[0], 0, np.inf)]
    for x in np.clip(input_signal[:-1], 0, np.inf):
        deriv = x - out[-1]
        out.append(np.clip(x + deriv_weight * deriv, 0, np.inf))
    return np.array(out)
