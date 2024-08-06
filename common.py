# Some constants/functions that are used in multiple files.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from scipy.optimize import minimize

# Tickers to use for building datasets and training.
sp_100_tickers = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN', 'AVGO',
                  'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'C', 'CAT', 'CHTR', 'CL',
                  'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DE', 'DHR', 'DIS',
                  'DOW', 'DUK', 'EMR', 'F', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD',
                  'HON', 'IBM', 'IBM', 'INTC', 'INTU', 'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY', 'LMT',
                  'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT',
                  'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX',
                  'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH',
                  'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM']
mag_7_tickers = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
tickers = mag_7_tickers

# Number of time points to use in defining sequence data.
WINDOW_LENGTH = 30

# Number of time points to use for defining buy/sell labels (for constrained linear regression).
FUTURE_WINDOW_LENGTH = 30

# Columns from CSV files to keep out of the training data.
ignore_cols = ['Year', 'Month', 'Day', 'Ticker']

# Version folder to save models and plots to.
VERSION = 'v5'

# Slope to use when classifying buy/sell labels.
buy_sell_slope = 0.01/FUTURE_WINDOW_LENGTH


def buy_sell_label(data: pd.DataFrame, index: int, col: str, ma: float):
    """
    Create buy/sell/do nothing labels from the given data.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing (unscaled) data.
        index (int): Index of the starting index of the input sequence.
        col (str): Name of the column for the prices to use.
        ma (float): Maximum value (computed by a MaxAbsScaler), to be used in normalization.

    Returns:
        numpy.array: One-hot encoded vector for the signal:
            - Do nothing: [1,0,0]
            - Buy: [1,0,0]
            - Sell: [0,1,0]
    """
    # Throw exception if out of bounds
    if index + WINDOW_LENGTH + FUTURE_WINDOW_LENGTH > len(data):
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
    today_price = data[col].iloc[index+WINDOW_LENGTH-1] / ma
    next_prices = data[col].iloc[index+WINDOW_LENGTH: index + WINDOW_LENGTH+FUTURE_WINDOW_LENGTH] / ma
    slope, intercept = best_fit_line_through_today_price(
        today_price, next_prices)
    if slope <= -buy_sell_slope:
        return np.array([0, 0, 1])
    elif -buy_sell_slope < slope < buy_sell_slope:
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
        def labeller(
            i): return local_data.iloc[i][col] - local_data.iloc[i-1][col]
    elif label == 'signal':
        def labeller(i, ma): return buy_sell_label(
            local_data, i, col, ma)

    # Init return instances, labels, and scaler values
    scaler = MaxAbsScaler()
    X, y, scaler_maxes = [], [], []

    if label in ['price', 'price-change']:
        offset = 0
    elif label == 'signal':
        offset = FUTURE_WINDOW_LENGTH

    # Iterate through every sequence (sliding window) in the data
    for i in range(len(data) - WINDOW_LENGTH - offset):
        sequence = local_data.iloc[i:i+WINDOW_LENGTH]
        sequence = scaler.fit_transform(sequence)
        X.append(sequence)
        # Get the scaler values needed to scale/revert
        col_index = np.where(scaler.get_feature_names_out() == col)[0][0]
        ma = scaler.max_abs_[col_index]
        if label == 'signal':
            gt_label = labeller(i, ma)
        elif label in ['price', 'price-change']:
            gt_label = labeller(i+WINDOW_LENGTH) / ma
        y.append(gt_label)
        scaler_maxes.append(ma)

    X = np.array(X)
    y = np.array(y)
    scaler_maxes = np.array(scaler_maxes)

    return X, y, scaler_maxes
