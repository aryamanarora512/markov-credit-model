import numpy as np
from numba import njit  # optional
# If numba not wanted, remove decoration and import.

def vectorized_simulate(initial_state_counts, P, n_steps, rng=None):
    """
    Vectorized simulation using multinomial draws per state.
    initial_state_counts: array of counts per state (size n_states)
    P: transition matrix (n_states x n_states), rows sum to 1
    n_steps: number of time steps to simulate
    Returns: history: (n_steps+1, n_states) counts
    """
    if rng is None:
        rng = np.random.default_rng()
    n_states = P.shape[0]
    counts = np.array(initial_state_counts, dtype=int)
    history = np.zeros((n_steps+1, n_states), dtype=int)
    history[0] = counts.copy()
    for t in range(1, n_steps+1):
        new_counts = np.zeros_like(counts)
        # For each state i, sample how many transition to each j
        for i in range(n_states):
            if counts[i] == 0:
                continue
            # multinomial: how many of counts[i] go to each j
            draws = rng.multinomial(counts[i], P[i])
            new_counts += draws
        counts = new_counts
        history[t] = counts
    return history

# Numba accelerated version for large sims
@njit
def _multinomial_draws_numba(counts, P, n_steps, out):
    # very simple deterministic multinomial using inverse-CDF sampling
    rng = np.random
    n_states = P.shape[0]
    counts_local = counts.copy()
    out[0, :] = counts_local
    for t in range(1, n_steps+1):
        new_counts = np.zeros(n_states, dtype=np.int64)
        for i in range(n_states):
            ci = counts_local[i]
            if ci == 0:
                continue
            # draw ci independent categorical draws (slow if ci big) - this can be improved
            for draw in range(ci):
                u = rng.rand()
                cum = 0.0
                for j in range(n_states):
                    cum += P[i, j]
                    if u <= cum:
                        new_counts[j] += 1
                        break
        counts_local = new_counts
        out[t, :] = counts_local
