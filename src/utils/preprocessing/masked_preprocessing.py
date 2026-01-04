import pandas as pd 
import numpy as np 


def generate_padded_data(feature_df: pd.DataFrame, df_window_sizes: pd.DataFrame, 
                         target_vec:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data_cols : list[str] = feature_df.columns.tolist()
    num_features : int = len(data_cols)
    window_sizes : dict[str, int] = df_window_sizes.set_index('feature')['window_size'].to_dict()
    max_window_size : int = max(window_sizes.values())
    num_samples : int = feature_df.shape[0] - max_window_size + 1; # how many feature vectors I will have as an np.ndarray
    padded_data : np.ndarray = np.zeros((max_window_size, num_features, num_samples))
    
    # Now I need to fill in padded data... For each row in the df_data dataframe, I need to 
    # fill the padded_data array with the appropriate feature vector size. For example, some of the 
    # features have a window size of 12, so I will take all 12 data points. In other features, if the
    # window size is 1, I will have 11 zeros and 1 datapoint at the end. 
    
    for i in range(num_samples):
        for j, feature in enumerate(data_cols):
            window_size:int = window_sizes[feature]
            # Fill in the last 'window_size' entries with actual data
            padded_data[max_window_size - window_size:, j, i] = feature_df[feature].iloc[i:i + window_size].to_numpy()
            # The rest are already zeros due to initialization
    
    y_tgt_adj: np.ndarray = target_vec[(max_window_size-1):]
    return padded_data, y_tgt_adj
    
def create_masking_vector(feature_df: pd.DataFrame, df_window_sizes:pd.DataFrame) -> np.ndarray:
    window_sizes_filtered : pd.DataFrame = df_window_sizes[df_window_sizes['feature'] != 'All Features']
    window_sizes: dict[str, int] = window_sizes_filtered.set_index('feature')['window_size'].to_dict()
    max_window_size : int = max(window_sizes.values())
    num_features : int = window_sizes_filtered.shape[0]
    masking_matrix: np.ndarray = np.zeros((max_window_size, num_features))
    data_cols: list[str] = feature_df.columns.tolist()
    # now that we have the masking matrix of zeros, I need to replace each of the columns 
    # with the necessary 1's based off the window length

    for j, feature in enumerate(data_cols):
        # print(f"index {j} relates to feature {feature}")
        window_size : int = window_sizes[feature]
        if window_size == max_window_size:
            masking_matrix[:,j] = np.ones((max_window_size, ))
        else:
            masking_matrix[(max_window_size - window_size):, j] = np.ones((window_size, ))

    # print(masking_matrix)
    return masking_matrix