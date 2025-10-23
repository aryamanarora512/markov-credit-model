import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm as sparse_expm

class DiscreteMarkov:
    def __init__(self, P: np.ndarray, states: list):
        """
        P: (n_states, n_states) row-stochastic matrix (rows sum to 1)
        states: list of state names in order corresponding to P
        """
        self.P = np.array(P, dtype=float)
        self.states = list(states)
        self.n = self.P.shape[0]

    def n_step(self, n):
        """Return P^n (n-step transition matrix)."""
        return np.linalg.matrix_power(self.P, n)

    def stationary_distribution(self):
        """Return stationary dist pi s.t. pi = pi P (left eigenvector).
           Solve (P^T - I)^T x = 0  with normalization sum(x)=1.
        """
        # solve left eigenvector
        w, v = linalg.eig(self.P.T)
        # find eigenvector for eigenvalue 1
        idx = np.argmin(np.abs(w - 1.0))
        vec = np.real(v[:, idx])
        pi = vec / vec.sum()
        pi = np.maximum(pi, 0)
        pi = pi / pi.sum()
        return pi

    def absorb_probabilities(self, absorbing_states):
        """
        Compute absorbing probabilities for absorbing Markov chains.
        absorbing_states: list of state indices that are absorbing.
        Returns fundamental matrix N = (I-Q)^-1 etc.
        """
        transient = [i for i in range(self.n) if i not in absorbing_states]
        Q = self.P[np.ix_(transient, transient)]
        I = np.eye(Q.shape[0])
        N = np.linalg.inv(I - Q)
        # B = N * R where R is transitions from transient to absorbing
        R = self.P[np.ix_(transient, absorbing_states)]
        B = N @ R
        return {
            'transient_indices': transient,
            'absorbing_indices': absorbing_states,
            'N': N,
            'B': B
        }

    def expected_time_to_absorb(self, absorbing_states):
        trans = self.absorb_probabilities(absorbing_states)
        N = trans['N']
        # expected number of steps until absorption from each transient state
        t = N.sum(axis=1)
        res = np.zeros(self.n)
        for idx, tr in enumerate(trans['transient_indices']):
            res[tr] = t[idx]
        for a in trans['absorbing_indices']:
            res[a] = 0.0
        return res
