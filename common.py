# Some constants/functions that are used in multiple files.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Tickers to use for building datasets and training.
tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN',
           '^GSPC', '^DJI', '^RUT', 'CL=F', 'GC=F']
# tickers = ['^GSPC']

# Number of time points to use in defining sequence data.
WINDOW_LENGTH = 30

# Columns from CSV files to keep out of the training data.
ignore_cols = ['Year', 'Month', 'Day', 'Ticker']


def prepare_model_data(data: pd.DataFrame, label: str, col: str):
    """
    Prepare input instances and ground truth labels (X and y) given raw CSV data, using the defined
        WINDOW_LENGTH as the sequence length, and normalizing each sequence with a MinMaxScaler.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing the raw CSV data.
        label (str): String indicating what value to use as the labels:
            "price": Use the price of the given column.
            "percent-change": Use the percent change in values of the given column.
        col (str): Column name to use in creating the labels.

    Returns:
        numpy.array, numpy.array: Two numpy arrays X and y containing the training instances and ground
            truth labels, respectively. X will have shape (len(data) - WINDOW_LENGTH, WINDOW_LENGTH, NUM_FEATURES),
            while y will have shape (len(data) - WINDOW_LENGTH).
            Each training instance of X is normalized
    """
    local_data = data.drop(columns=ignore_cols, errors='ignore')

    # Define the label function based on the label
    if label == 'price':
        def labeller(i): return data.iloc[i][col]

    # Init return instances, labels, and scaler values
    scaler = MinMaxScaler()
    X, y, scaler_mins, scaler_scales = [], [], [], []

    # Iterate through every sequence (sliding window) in the data
    for i in range(len(data) - WINDOW_LENGTH):
        sequence = local_data.iloc[i:i+WINDOW_LENGTH]
        sequence = scaler.fit_transform(sequence)
        gt_label = labeller(i+WINDOW_LENGTH)
        X.append(sequence)
        # Get the scaler values needed to scale/revert
        col_index = np.where(scaler.get_feature_names_out() == col)[0][0]
        mi, scale = scaler.data_min_[col_index], scaler.scale_[col_index]
        y.append((gt_label - mi) * scale)
        scaler_mins.append(mi)
        scaler_scales.append(scale)

    X = np.array(X)
    y = np.array(y)
    scaler_mins = np.array(scaler_mins)
    scaler_scales = np.array(scaler_scales)

    return X, y, scaler_mins, scaler_scales
