import numpy as np
from src.markov_model import DiscreteMarkov
from src.simulate_portfolio import vectorized_simulate

# quick example states
states = ['performing', 'delinq30', 'delinq60', 'NPE', 'recovered', 'default']
# toy transition matrix (rows sum to 1)
P = np.array([
    [0.88, 0.08, 0.02, 0.01, 0.00, 0.01],  # performing
    [0.30, 0.50, 0.10, 0.05, 0.00, 0.05],  # delinq30
    [0.10, 0.40, 0.30, 0.10, 0.05, 0.05],  # delinq60
    [0.00, 0.00, 0.05, 0.70, 0.20, 0.05],  # NPE
    [0.00, 0.00, 0.00, 0.00, 1.0, 0.0],    # recovered (absorbing)
    [0.00, 0.00, 0.00, 0.00, 0.0, 1.0],    # default (absorbing)
])
model = DiscreteMarkov(P, states)

# compute 12-month transition probabilities
P12 = model.n_step(12)
print("12-month transition prob from performing to NPE:", P12[0, states.index('NPE')])

# expected time to absorption (recovered or default)
abs_idx = [states.index('recovered'), states.index('default')]
print("Expected time to absorb (months):", model.expected_time_to_absorb(abs_idx))

# simulate a portfolio of 10k loans
initial_counts = np.zeros(len(states), dtype=int)
initial_counts[0] = 10000  # all performing
hist = vectorized_simulate(initial_counts, P, n_steps=24)
print("Counts after 24 months:", hist[-1])
