# Some constants/functions that are used in multiple files.

import numpy as np
import pandas as pd
from scipy.optimize import minimize

pd.options.mode.chained_assignment = None

# Tickers to use for building datasets and training.
sp_100_tickers = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN', 'AVGO',
                  'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'C', 'CAT', 'CHTR', 'CL',
                  'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DE', 'DHR', 'DIS',
                  'DOW', 'DUK', 'EMR', 'F', 'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD',
                  'HON', 'IBM', 'INTC', 'INTU', 'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY', 'LMT',
                  'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT',
                  'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX',
                  'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UNH',
                  'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM']
tickers = sp_100_tickers

# Sector to vector mapping
# One-hot
sec_to_vec = np.eye(11)

# Version folder to save models and plots to.
VERSION = 'preliminary'

# Number of time points to use in defining sequence data.
WINDOW_LENGTH = 30

# Number of time points to use for defining buy/sell labels (for constrained linear regression).
FUTURE_WINDOW_LENGTH = 15

# Proportional change to use when classifying buy/sell labels.
percent_change_slope = 0.05

# Stride to take when generating training instances.
train_stride = 15


def buy_sell_label(data: pd.DataFrame, index: int, col: str, scale: float):
    """
    Create buy/sell/do nothing labels from the given data.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing (unscaled) data.
        index (int): Index of the starting index of the input sequence.
        col (str): Name of the column for the prices to use.
        scale (float): Scale value to be used in normalization.

    Returns:
        numpy.array: One-hot encoded vector for the signal:
            - Sell: [1,0,0]
            - Do nothing: [0,1,0]
            - Buy: [0,0,1]
    """
    wl, fwl = WINDOW_LENGTH, FUTURE_WINDOW_LENGTH
    # Throw exception if out of bounds
    if index + wl + fwl > len(data):
        raise IndexError(
            f"Index {index} and window length {wl} are out of bounds for data of length {len(data)}")

    # Calculate the parameters of the best fit line (constrained such that it passes through
    # the most recent known price)
    def constrained_lin_reg(today_price, next_prices):
        n = len(next_prices) + 1
        A = np.vstack([np.arange(n), np.ones(n)])

        def loss(x):
            return np.sum(np.square(np.dot(x, A) - np.concatenate([[today_price], next_prices])))
        constraint = ({'type': 'eq', 'fun': lambda x: x[1] - today_price})
        x0 = np.zeros(2)
        res = minimize(loss, x0, method='SLSQP', constraints=constraint)
        return res.x

    # Choose a label depending on the slope of the constrained regression line using the next
    # FUTURE_WINDOW_LENGTH time steps.
    today_price = data[col].iloc[index+wl-1]
    next_prices = data[col].iloc[index+wl: index+wl+fwl]
    slope, intercept = constrained_lin_reg(today_price, next_prices)
    delta = FUTURE_WINDOW_LENGTH * slope / today_price

    if delta <= -percent_change_slope:
        return np.array([1, 0, 0])
    elif delta <= percent_change_slope:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


# Columns from CSV files to keep out of the training data.
ignore_cols = ['Open', 'High', 'Low', 'Adj Close', 'Year', 'Month', 'Day', 'Ticker']

# Columns to not normalize (using the closing price) for the training data.
keep_cols = ['Proportional_Change', 'Stochastic_Oscillator', 'RSI', 'Volume',
             'estimate_EPS', 'report_EPS', 'surprise_percent', 'PE', 'Sector_ID',
             'sector_avg_prop_change']


def prepare_model_data(data: pd.DataFrame, label: str, col: str, is_train: bool):
    """
    Prepare input instances and ground truth labels (X and y) given raw CSV data, using the defined
        WINDOW_LENGTH as the sequence length, and using certain normalization methods.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing the raw CSV data (for a specific ticker).
        label (str): String indicating what value to use as the labels:
            "price": Use the price of the given column.
            "signal": Use regression to indicate upward/downward/neither movement.
        col (str): Column name to base creating the labels off of.
        is_train (bool): Boolean for if the prepared data is training or eval data.
            If true, takes a certain step size (stride) in the sliding window.

    Returns:
        numpy.array, numpy.array: Two numpy arrays X and y containing the input instances and ground
            truth labels, respectively.
    """
    wl, fwl = WINDOW_LENGTH, FUTURE_WINDOW_LENGTH
    # Drop ignored columns
    local_data = data.drop(columns=ignore_cols, errors='ignore')

    # Define the label function based on the label
    if label == 'price':
        def labeller(i, scale):
            return local_data.iloc[i+wl][col] * scale
    elif label == 'signal':
        def labeller(i, scale):
            return buy_sell_label(local_data, i, col, scale)

    # Init sequences, metadata, labels, and scaler values
    X, x_meta, y, scaler_scales = [], [], [], []

    # Define right boundary for regression based signals
    if (not is_train) or (label == 'price'):
        right_offset = 0
    elif label == 'signal':
        right_offset = fwl

    # Rolling mins/maxes for normalization
    maxes = local_data[col].rolling(wl).max()
    
    # Stride
    if is_train:
        stride = train_stride
    else:
        stride = 1

    for i in range(0, len(data) - wl - right_offset, stride):
        sequence = local_data.iloc[i:i+wl]

        scale = 1 / (maxes.iloc[i+wl-1])

        # Normalize by scaling
        sequence_1 = sequence.drop(columns=keep_cols) * scale

        # Custom normalization and handling
        sequence_2 = sequence[keep_cols]
        # Scale volume by max
        sequence_2['Volume'] /= sequence_2['Volume'].max()
        # Sigmoid PE
        sequence_2['PE'] = 1/(1 + np.exp(sequence_2['PE']/100))
        # Get the sector vector
        sector_id = sequence_2['Sector_ID'].iloc[0]
        sector_vector = sec_to_vec[sector_id]
        sequence_2.drop(columns=['Sector_ID'], inplace=True)

        sequence = pd.concat([sequence_1, sequence_2], axis=1).to_numpy()

        X.append(sequence)

        x_meta.append(sector_vector)

        # Add the gt label IF in bounds
        if i + wl + fwl <= len(local_data):
            gt_label = labeller(i, scale)
            y.append(gt_label)

        scaler_scales.append(scale)

    X = np.array(X)
    x_meta = np.array(x_meta)
    y = np.array(y)
    scaler_scales = np.array(scaler_scales)

    return X, x_meta, y, scaler_scales
