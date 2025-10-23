import pandas as pd
import numpy as np

def build_transition_counts(df, id_col='loan_id', time_col='date', state_col='state', freq='M'):
    """
    df: DataFrame with columns [loan_id, date, state]; date should be datetime-like
    returns: transition_counts: dict[state_from][state_to] = count
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([id_col, time_col])
    # shift per loan to get transitions
    df['next_state'] = df.groupby(id_col)[state_col].shift(-1)
    counts = df.dropna(subset=['next_state']).groupby([state_col, 'next_state']).size().unstack(fill_value=0)
    return counts

def counts_to_transition_matrix(counts_df, states=None):
    """
    Convert counts DataFrame (rows = from, cols = to) to transition matrix (numpy array).
    states: optional ordered list of states to fix ordering
    """
    if states is None:
        states = list(counts_df.index)
    counts_df = counts_df.reindex(index=states, columns=states, fill_value=0)
    counts = counts_df.values.astype(float)
    row_sums = counts.sum(axis=1, keepdims=True)
    # avoid division by zero: if row sum == 0, make self-loop
    zero_rows = (row_sums.squeeze() == 0)
    row_sums[zero_rows, 0] = 1.0
    P = counts / row_sums
    # for zero rows we put self-loop
    for i, z in enumerate(zero_rows):
        if z:
            P[i, :] = 0.0
            P[i, i] = 1.0
    return np.array(P), states
