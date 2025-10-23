markov-credit-model/
├ README.md

├ requirements.txt

├ data/

│  └ sample_loan_transitions.csv   # small example

├ notebooks/

│  └ demo_notebook.ipynb

├ src/

│  ├ __init__.py

│  ├ data_processing.py

│  ├ markov_model.py

│  ├ simulate_portfolio.py

│  └ utils.py

└ examples/

   └ run_demo.py
Estimate transition matrix by counting transitions between discrete monthly states (vectorized pandas / numpy).
n-step probabilities via matrix power (np.linalg.matrix_power) or sparse exponentiation for large state spaces.
Simulate many loan paths vectorized using multinomial draws or using numba.jit for very large Monte-Carlo.


Usage:
python examples/run_demo.py` shows sample outputs.
Place your historical loan-level panel (loan_id, date, state) into `data/` and run `src/data_processing.build_transition_counts` to estimate a monthly transition matrix.
 Use `src/markov_model.DiscreteMarkov` for n-step probabilities, absorption analysis, and expected time to recovery/default.
 The `src/simulate_portfolio.py` module demonstrates scalable Monte Carlo via vectorized multinomial draws. For heavy simulations use the Numba-jitted routine.

Potential extensions:
 Fit separate P matrices by loan vintage, collateral type, or macro regime.
 Condition transition probabilities on macro variables (regime switching) with a Hidden Markov or time-inhomogeneous model.
 Replace discrete time with continuous-time Markov and use generator matrices if required.

