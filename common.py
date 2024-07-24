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
    keep_cols = [col for col in data.columns if col not in preserve_cols]

    normalized_data = scaler.fit_transform(data[keep_cols])

    normalized_data = np.hstack([normalized_data, data[preserve_cols]])

    return normalized_data

# Columns from CSV files to keep out of the training data.
ignore_cols = ['Year', 'Month', 'Day']