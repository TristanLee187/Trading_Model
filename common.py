# Some constants/functions that are used in multiple files.

import numpy as np
import pandas as pd

# Tickers to use for building datasets and training.
tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', '^GSPC', '^DJI', '^RUT', 'CL=F', 'GC=F']

# Number of time points to use in defining sequence data.
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


# Columns from CSV files to keep out of the training data.
ignore_cols = ['Year', 'Month', 'Day', 'Ticker']


def prepare_model_data(data: pd.DataFrame, label: str, col: str):
    """
    Prepare input instances and ground truth labels (X and y) given raw CSV data, using the defined
        WINDOW_LENGTH as the sequence length.

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
    """
    # Define the label function based on the label
    if label == 'price':
        def labeller(i): return data.iloc[i][col]
    elif label == 'percent-change':
        def labeller(i): return percent_change_label(data, i, col)

    # Init return instances and labels
    X, y = [], []

    # Iterate through every sequence (sliding window) in the data
    for i in range(len(data) - WINDOW_LENGTH):
        sequence = data[i:i+WINDOW_LENGTH].drop(columns=ignore_cols)
        sequence = sequence.to_numpy()
        gt_label = labeller(i+WINDOW_LENGTH)
        X.append(sequence)
        y.append(gt_label)

    X = np.array(X).astype('float32')
    y = np.array(y).astype('float32')

    return X, y
