import numpy as np
import pandas as pd


def generate_padded_data(feature_df: pd.DataFrame, df_window_sizes: pd.DataFrame,target_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create padded sliding windows for each feature, using the per-feature window sizes.
    Returns:
        padded_data: shape (max_window, num_features, num_samples)
        y_tgt_adj:   target vector aligned to the padded windows
    """
    data_cols = feature_df.columns.tolist()
    num_features = len(data_cols)

    # Convert df_window_sizes → dict: {feature: window_size}
    window_sizes = df_window_sizes.set_index('feature')['window_size'].to_dict()

    # Find maximum sliding window size among all features
    max_window_size = max(window_sizes.values())

    # Number of samples that can be produced
    num_samples = feature_df.shape[0] - max_window_size + 1

    # Initialize padded array (zeros everywhere)
    padded_data = np.zeros((max_window_size, num_features, num_samples))

    # Fill padded_data for each sample + feature
    for ii in range(num_samples):
        for jj, feature in enumerate(data_cols):

            win_size = window_sizes[feature]

            # Fill last w entries with real data
            padded_data[max_window_size - win_size:, jj, ii] = \
                feature_df[feature].iloc[ii:ii + win_size].to_numpy()

    # Align target vector
    y_tgt_adj = target_vec[(max_window_size - 1):]

    return padded_data, y_tgt_adj



def create_masking_vector(feature_df: pd.DataFrame, df_window_sizes: pd.DataFrame) -> np.ndarray:
    """
    Create a masking matrix of shape (max_window, num_features),
    where each column contains 1's in the valid window portion
    and 0's otherwise.
    """
    # remove "All Features" entry if present
    df_ws = df_window_sizes[df_window_sizes['feature'] != 'All Features']

    window_sizes = df_ws.set_index('feature')['window_size'].to_dict()
    data_cols = feature_df.columns.tolist()

    num_features = len(data_cols)
    max_window_size = max(window_sizes.values())

    mask = np.zeros((max_window_size, num_features))

    for jj, feature in enumerate(data_cols):
        win_size = window_sizes[feature]

        # valid region (bottom w rows)
        mask[max_window_size - win_size:, jj] = 1

    return mask