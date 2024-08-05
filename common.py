# Some constants/functions that are used in multiple files.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# Tickers to use for building datasets and training.
# tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN',
#            '^GSPC', '^DJI', '^RUT', 'CL=F', 'GC=F']
tickers = ['^GSPC','^DJI', '^RUT']

# Number of time points to use in defining sequence data.
WINDOW_LENGTH = 30

# Columns from CSV files to keep out of the training data.
ignore_cols = ['Year', 'Month', 'Day', 'Ticker']

# Version folder to save models and plots to.
VERSION = 'v4'

# Slope to use when classifying buy/sell labels.
buy_sell_slope = 1.1


def buy_sell_label(data: pd.DataFrame, index: int, col: str, mi: float, scale: float):
    """
    Create buy/sell/do nothing labels from the given data.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing (unscaled) data.
        index (int): Index of the starting index of the input sequence.
        col (str): Name of the column for the prices to use.
        mi (float): Minimum value (computed by a MinMaxScaler), to be used in normalization.
        scale (float): Scale value (computed by a MinMaxScaler), to be used in normalization.
    
    Returns:
        numpy.array: One-hot encoded vector for the signal:
            - Do nothing: [1,0,0]
            - Buy: [1,0,0]
            - Sell: [0,1,0]
    """
    # Throw exception if out of bounds
    if index + 2 * WINDOW_LENGTH > len(data):
        raise IndexError(
            f"Index {index} and window length {WINDOW_LENGTH} are out of bounds for data of length {len(data)}")

    # Calculate the parameters of the best fit line (constrained such that it passes through
    # the most recent known price)
    def best_fit_line_through_today_price(today_price, next_prices):
        n = len(next_prices) + 1
        A = np.vstack([np.arange(n), np.ones(n)])

        def loss(x):
            return np.sum(np.square(np.dot(x, A) - np.concatenate([[today_price], next_prices])))
        constraint = ({'type': 'eq', 'fun': lambda x: x[1] - today_price})
        x0 = np.zeros(2)
        res = minimize(loss, x0, method='SLSQP', constraints=constraint)
        return res.x

    # Choose a label depending on the slope of the constrained regression line using the next
    # WINDOW_LENGTH time steps.
    # [1,0,0] for do nothing, [0,1,0] for buy, [0,0,1] for sell.
    today_price = (data[col].iloc[index+WINDOW_LENGTH-1] - mi) * scale
    next_prices = (data[col].iloc[index+WINDOW_LENGTH: index+2*WINDOW_LENGTH] - mi) * scale
    slope, intercept = best_fit_line_through_today_price(
        today_price, next_prices)
    if slope <= -buy_sell_slope/WINDOW_LENGTH:
        return np.array([0, 0, 1])
    elif -buy_sell_slope/WINDOW_LENGTH < slope < buy_sell_slope/WINDOW_LENGTH:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 1, 0])


def prepare_model_data(data: pd.DataFrame, label: str, col: str):
    """
    Prepare input instances and ground truth labels (X and y) given raw CSV data, using the defined
        WINDOW_LENGTH as the sequence length, and normalizing each sequence with a MinMaxScaler.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing the raw CSV data.
        label (str): String indicating what value to use as the labels:
            "price": Use the price of the given column.
            "price-change": Use the change in values of the given column.
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
        def labeller(i): return local_data.iloc[i][col]
    elif label == 'price-change':
        def labeller(i): return local_data.iloc[i][col] - local_data.iloc[i-1][col]
    elif label == 'signal':
        def labeller(i, mi, scale): return buy_sell_label(local_data, i, col, mi, scale)

    # Init return instances, labels, and scaler values
    scaler = MinMaxScaler()
    X, y, scaler_mins, scaler_scales = [], [], [], []

    if label in ['price', 'price-change']:
        offset = 0
    elif label == 'signal':
        offset = WINDOW_LENGTH

    # Iterate through every sequence (sliding window) in the data
    for i in range(len(data) - WINDOW_LENGTH - offset):
        sequence = local_data.iloc[i:i+WINDOW_LENGTH]
        sequence = scaler.fit_transform(sequence)
        X.append(sequence)
        # Get the scaler values needed to scale/revert
        col_index = np.where(scaler.get_feature_names_out() == col)[0][0]
        mi, scale = scaler.data_min_[col_index], scaler.scale_[col_index]
        if label == 'signal':
            gt_label = labeller(i, mi, scale)
        elif label == 'price':
            gt_label = (labeller(i+WINDOW_LENGTH) - mi) * scale
        elif label == 'price-change':
            gt_label = labeller(i+WINDOW_LENGTH) * scale
        y.append(gt_label)
        scaler_mins.append(mi)
        scaler_scales.append(scale)

    X = np.array(X)
    y = np.array(y)
    scaler_mins = np.array(scaler_mins)
    scaler_scales = np.array(scaler_scales)

    return X, y, scaler_mins, scaler_scales
