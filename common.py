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
VERSION = 'v9'

# Number of time points to use in defining sequence data.
WINDOW_LENGTH = 30

# Number of time points to use for defining buy/sell labels (for constrained linear regression).
FUTURE_WINDOW_LENGTH = 15

# Proportional change to use when classifying buy/sell labels.
percent_change_slope = 0.05

# Stride to take when generating training instances.
train_stride = 5


def buy_sell_label(data: pd.DataFrame, index: int, col: str, mi: float, scale: float):
    """
    Create buy/sell/do nothing labels from the given data.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing (unscaled) data.
        index (int): Index of the starting index of the input sequence.
        col (str): Name of the column for the prices to use.
        mi (float): Minimum value to be used in normalization.
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
        def labeller(i, mi, scale):
            return (local_data.iloc[i+wl][col] - mi) * scale
    elif label == 'signal':
        def labeller(i, mi, scale):
            return buy_sell_label(local_data, i, col, mi, scale)

    # Init sequences, metadata, labels, and scaler values
    X, x_meta, y, scaler_mins, scaler_scales = [], [], [], [], []

    # Define right boundary for regression based signals
    if label == 'price':
        right_offset = 0
    elif label == 'signal':
        right_offset = fwl

    # Rolling mins/maxes for normalization
    mins = local_data[col].rolling(wl).min()
    maxes = local_data[col].rolling(wl).max()
    
    # Stride
    if is_train:
        stride = train_stride
    else:
        stride = 1

    for i in range(0, len(data) - wl - right_offset, stride):
        sequence = local_data.iloc[i:i+wl]

        mi = mins.iloc[i+wl-1]
        scale = 1 / (maxes.iloc[i+wl-1] - mi)

        # Get columns we don't want to translate (crosses, MACD)
        no_translate_cols = [col for col in sequence.columns if 'Cross' in col] + ['MACD']

        # Normalize by translation and scaling (moving averages)
        sequence_1 = (sequence.drop(columns=keep_cols + no_translate_cols) - mi) * scale

        # Normalize by just scaling (divergence)
        sequence_2 = sequence[no_translate_cols] * scale

        # Custom normalization and handling
        sequence_3 = sequence[keep_cols]
        # Scale volume by max
        sequence_3['Volume'] /= sequence_3['Volume'].max()
        # Sigmoid PE
        sequence_3['PE'] = 1/(1 + np.exp(sequence_3['PE']/100))
        # Get the sector vector
        sector_id = sequence_3['Sector_ID'].iloc[0]
        sector_vector = sec_to_vec[sector_id]
        sequence_3.drop(columns=['Sector_ID'], inplace=True)

        sequence = pd.concat([sequence_1, sequence_2, sequence_3], axis=1).to_numpy()

        X.append(sequence)

        x_meta.append(sector_vector)

        gt_label = labeller(i, mi, scale)
        y.append(gt_label)

        scaler_mins.append(mi)
        scaler_scales.append(scale)

    X = np.array(X)
    x_meta = np.array(x_meta)
    y = np.array(y)
    scaler_mins = np.array(scaler_mins)
    scaler_scales = np.array(scaler_scales)

    return X, x_meta, y, scaler_mins, scaler_scales
