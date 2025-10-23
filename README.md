markov-credit-model/
├─ README.md

├─ requirements.txt

├─ data/

│  └─ sample_loan_transitions.csv   # small example

├─ notebooks/

│  └─ demo_notebook.ipynb

├─ src/

│  ├─ __init__.py

│  ├─ data_processing.py

│  ├─ markov_model.py

│  ├─ simulate_portfolio.py

│  └─ utils.py

└─ examples/

   └─ run_demo.py
Estimate transition matrix by counting transitions between discrete monthly states (vectorized pandas / numpy).
n-step probabilities via matrix power (np.linalg.matrix_power) or sparse exponentiation for large state spaces.
Simulate many loan paths vectorized using multinomial draws or using numba.jit for very large Monte-Carlo.
