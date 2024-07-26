# Some constants/functions that are used in multiple files.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Tickers to use for building datasets and training.
tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', '^GSPC', '^DJI', '^RUT', 'CL=F', 'GC=F']

# Number of days to use in defining sequence data.
WINDOW_LENGTH = 30

def percent_change_label(data: pd.DataFrame, i: int, col: str):
    """
    Calculates the percent change of the column's value in question, from index i-1 to index i.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing the data.
        i (int): The index of data to calculate the percent change for (from i-1 to i).
        col (str): The name of the column in data to use values.
    
    Returns:
        float: The percent change from data[col][i-1] to data[col][i].
    """
    last_val = data.iloc[i-1][col]
    this_val = data.iloc[i][col]
    percent_change = (this_val - last_val) / last_val

    return percent_change

# Columns to not normalize when normalizing the input data.
preserve_cols = ['MACD', 'Stochastic_Oscillator']

def normalize(data: pd.DataFrame, scaler: StandardScaler):
    """
    Normalize all columns in the given DataFrame, except for preserve_cols, using the given StandardScaler.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containin the data.
        scaler (sklearn.preprocessing.StandardScaler): Instance of a StandardScaler.
    
    Returns:
        pandas.DataFrame: New Pandas DataFrame containing the normalized data.
    """
    # Columns to normalize
    norm_cols = [col for col in data.columns if col not in preserve_cols]
    normalized_data = scaler.fit_transform(data[norm_cols])

    # Add back non-normalized data
    normalized_data = np.hstack([normalized_data, data[preserve_cols]])

    return normalized_data

# Columns from CSV files to keep out of the training data.
ignore_cols = ['Year', 'Month', 'Day']

def prepare_model_data(data: pd.DataFrame, norm: bool, label: str, col: str):
    """
    Prepare input instances and ground truth labels (X and y) given raw CSV data, using the defined
        WINDOW_LENGTH as the sequence length.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing the raw CSV data.
        norm (bool): Boolean indicating whether to normalize the data or not.
        label (str): String indicating what value to use as the labels:
            "price": Use the price of the given column.
            "percent-change": Use the percent change in values of the given column.
        col (str): Column name to use in creating the labels.
    
    Returns:
        numpy.array, numpy.array: Two numpy arrays X and y containing the training instances and ground
            truth labels, respectively. X will have shape (len(data) - WINDOW_LENGTH, WINDOW_LENGTH, NUM_FEATURES),
            while y will have shape (len(data) - WINDOW_LENGTH).
    """
    # Define normalization transformation (or just conversion to numpy otherwise)
    if norm:
        scaler = StandardScaler()
        transform = lambda seq: normalize(seq, scaler)
    else:
        transform = lambda seq: seq.to_numpy()

    # Define the label function based on the label
    if label == 'price':
        labeller = lambda i: data.iloc[i][col]
    elif label == 'percent-change':
        labeller = lambda i: percent_change_label(data, i, col)

    # Init return instances and labels
    X, y = [], []

    # Iterate through every sequence (sliding window) in the data
    for i in range(len(data) - WINDOW_LENGTH):
        sequence = data[i:i+WINDOW_LENGTH].drop(columns=ignore_cols)
        sequence = transform(sequence)
        gt_label = labeller(i+WINDOW_LENGTH)
        X.append(sequence)
        y.append(gt_label)
        
    X = np.array(X)
    y = np.array(y)

    return X, y