"""
True-online distributional TD(lambda) with expectile-weighted TD errors.
"""
from dataclasses import dataclass

import numpy as np
import numba as nb


@dataclass(frozen=True)
class DistTDLambdaResult:
    values: np.ndarray        # (T, K)
    rpe: np.ndarray           # (T, K)
    weights: np.ndarray       # (T, K, F)
    eligibility: np.ndarray   # (T, K, F)

@dataclass(frozen=True)
class DistTDLambdaResult:
    values: np.ndarray       # shape (T, K)
    rpe: np.ndarray          # shape (T, K) or (T,)
    weights: np.ndarray      # shape (T, K, F)
    eligibility: np.ndarray  # shape (T, K, F)

    def __init__(self, values, rpe, weights, eligibility):
        object.__setattr__(self, 'values', self._to_immutable_array(values))
        object.__setattr__(self, 'rpe', self._to_immutable_array(rpe))
        object.__setattr__(self, 'weights', self._to_immutable_array(weights))
        object.__setattr__(self, 'eligibility', self._to_immutable_array(eligibility))
        self.__check_len()

    def __len__(self):
        """Number of time steps T"""
        self.__check_len()
        return self.values.shape[0]

    def final_weights(self) -> np.ndarray:
        """
        Final weights for each expectile channel.
        Returns shape (K, F)
        """
        return self.weights[-1]

    def __check_len(self):
        T = self.values.shape[0]
        if not all(
            x.shape[0] == T
            for x in [self.rpe, self.weights, self.eligibility]
        ):
            raise ValueError(
                "Expected values, rpe, weights, and eligibility "
                "to have the same time dimension T"
            )

    @staticmethod
    def _to_immutable_array(x) -> np.ndarray:
        arr = np.array(x, copy=True)
        arr.setflags(write=False)
        return arr

def dist_true_online_td(
    state_matrix: np.ndarray,   # (T, F)
    rewards: np.ndarray,        # (T,)
    taus: np.ndarray,           # (K,)
    gamma: float,
    lambda_: float,
    learning_rate: float,
):
    T, F = state_matrix.shape
    taus = np.asarray(taus)
    K = len(taus)

    # Allocate
    weights = np.zeros((T, K, F))
    eligibility = np.zeros((T+1, K, F))
    values = np.zeros((T, K))
    rpe_hist = np.zeros((T, K))


    for t in range(T - 1):
        x_t = state_matrix[t]
        x_tp1 = state_matrix[t + 1]
        r_tp1 = rewards[t + 1]

        for k in range(K):
            w = weights[t, k]
            e = eligibility[t, k]

            v_t = np.dot(w, x_t)
            v_tp1 = np.dot(w, x_tp1)

            delta = r_tp1 + gamma * v_tp1 - v_t
            rpe_hist[t, k] = delta

            # Expectile-specific step size
            if delta >= 0:
                alpha_k = learning_rate * taus[k]
            else:
                alpha_k = learning_rate * (1.0 - taus[k])

            # Dutch eligibility trace 
            e_new = (
                gamma * lambda_ * e
                + alpha_k * x_t
                - alpha_k * gamma * lambda_ * np.dot(e, x_t) * x_t
            )

            # TRUE-ONLINE TD WEIGHT UPDATE 
            w_new = (
                w
                + delta * e_new
                + alpha_k * (v_t - np.dot(w, x_t))* x_t
            )

            weights[t+1, k] = w_new
            eligibility[t+1, k] = e_new
            values[t, k] = v_tp1
            rpe_hist[t, k] = delta

       # values[t+1] = np.einsum("kf,f->k", weights[t+1], x_tp1)

    rpe_hist[-1] = np.nan
    eligibility = eligibility[1:]

    return DistTDLambdaResult(
        values=values,
        rpe=rpe_hist,
        weights=weights,
        eligibility=eligibility,
    )


def heterogeneous_discount_true_online_td(
state_matrix: np.ndarray,   # (T, F)
        rewards: np.ndarray,        # (T,)
        taus: np.ndarray,           # (K,) discount timescales
        dt: float,
        lambda_: float,
        learning_rate: float
) -> DistTDLambdaResult:
    
    """
    True-online TD(lambda) with heterogeneous discounting timescales.
    Each channel k uses gamma_k = exp(-dt / taus[k]).
    """

    T, F = state_matrix.shape
    taus = np.asarray(taus, dtype=float)
    K = len(taus)

    if np.any(taus <= 0):
        raise ValueError("All discount timescales taus must be > 0.")

    # Per-channel discount factors
    gammas = np.exp(-dt / taus)  # (K,)

    # Allocate
    weights = np.zeros((T, K, F))
    eligibility = np.zeros((T + 1, K, F))
    values = np.zeros((T, K))
    rpe_hist = np.zeros((T, K))


    alpha = float(learning_rate)

    for t in range(T - 1):
        x_t = state_matrix[t]
        x_tp1 = state_matrix[t + 1]
        r_tp1 = rewards[t + 1]

        for k in range(K):
            gamma_k = gammas[k]

            w = weights[t, k]
            e = eligibility[t, k]

            v_t = np.dot(w, x_t)
            v_tp1 = np.dot(w, x_tp1) #Future_value

            delta = rpe(r_tp1, v_tp1, v_t, gamma_k)
            rpe_hist[t, k] = delta

            # Dutch eligibility trace uses gamma_k (alpha is constant)
            e_new = (
                gamma_k * lambda_ * e
                + alpha * x_t
                - (alpha * gamma_k * lambda_ * np.dot(e, x_t) * x_t)
            )

            # TRUE-ONLINE TD WEIGHT UPDATE
            w_new = (
                w
                + delta * e_new
                + (alpha * (v_t - np.dot(w, x_t)) * x_t)
            )

            weights[t + 1, k] = w_new
            eligibility[t + 1, k] = e_new
            values[t, k] = v_tp1
            rpe_hist[t, k] = delta


    rpe_hist[-1] = np.nan
    eligibility = eligibility[1:]

    return DistTDLambdaResult(
        values=values,
        rpe=rpe_hist,
        weights=weights,
        eligibility=eligibility,
    )
    


@nb.jit(nopython=True)
def rpe(future_reward, future_value, current_value, gamma):
    """Reward prediction error."""
    return future_reward + gamma * future_value - current_value




