# Price indicators (percent change, SMA, EMA, cross, MACD, stochastic oscillator, and RSI)

import pandas as pd
import numpy as np


def proportional_change(series: pd.Series):
    """
    Computes the change from the previous to the current point in time, as a proportion of the previous point.

    Args:
        series (pandas.Series): Series of numerical data to calculate the changes on.
    
    Returns:
        pandas.Series: Series containing the changes for each point in time.
    """
    return series.diff() / series

def sma(series: pd.Series, size: int):
    """
    Computes the Simple Moving Average (SMA) for each applicable point in time.

    Args:
        series (pandas.Series): Series of numerical data to base the SMA on.
        size (int): Size of the window to use for the SMA (number of previous points of time to consider).

    Returns:
        pandas.Series: Series containing the SMA at each applicable point in time.
    """
    return series.rolling(size).mean()


def cross(series: pd.Series, size1: int, size2: int):
    """
    Computes a crossover, using the difference in 2 SMAs.

    Args:
        series (pandas.Series): Series of numerical data to base the crossover on.
        size1 (int): Size of the window for the first SMA.
        size2 (int): Size of the window for the second SMA.

    Returns:
        pandas.Series: Series containing the crossover, computed as size1-period SMA - size2-period SMA,
            for each applicable point in time.
    """
    return sma(series, size1) - sma(series, size2)


def ema(series: pd.Series, size: int):
    """
    Computes the Exponential Moving Average (EMA) for each applicable point in time.

    Args:
        series (pandas.Series): Series of numerical data to base the EMA on.
        size (int): Size of the window to use for the EMA (number of previous points of time to consider).

    Returns:
        pandas.Series: Series containing the EMA at each applicable point in time.
    """

    # Use 2 as the smoothing factor, or 2 / (size + 1) as alpha
    return series.ewm(alpha=2/(size+1), min_periods=size, adjust=False).mean()


def macd(series: pd.Series, size1: int, size2: int):
    """
    Computes the Moving Average Convergence/Divergence (MACD), using the difference in 2 EMAs.

    Args:
        series (pandas.Series): Series of numerical data to base the MACD on.
        size1 (int): Size of the window for the first EMA.
        size2 (int): Size of the window for the second EMA.

    Returns:
        pandas.Series: Series containing the MACD, computed as size1-period EMA - size2-period EMA,
            for each applicable point in time.
    """
    return ema(series, size1) - ema(series, size2)


def stochastic_oscillator(series: pd.Series, size: int):
    """
    Computes the Stochastic Oscillator for each applicable point in time.

    Args:
        series (pandas.Series): Series of numerical data to base the Stochastic Oscillator on.
        size (int): Size of the window to get the high/low price from.

    Returns:
        pandas.Series: Series containing the Stochastic Oscillator at each applicable point in time.
    """
    highs = series.rolling(size).max()
    lows = series.rolling(size).min()
    return (series - lows) / (highs - lows)


def rsi(series: pd.Series, size: int):
    """
    Computes the Relative Strength Index (RSI) for each applicable point in time.

    Args:
        series (pandas.Series): Series of numerical data to base the RSI on.
        size (int): Number of days to use in when counting up/down days.

    Returns:
        pandas.Series: Series containing the RSI at each applicable point in time.
    """
    change = proportional_change(series)
    gains, losses = np.maximum(change, 0), np.minimum(change, 0)
    avg_gains, avg_losses = sma(gains, size), sma(losses, size)
    rs = avg_gains / np.maximum(np.abs(avg_losses), 1e-7)
    raw_rsi = 100 - 100 / (1 + rs)
    smooth_rsi = sma(raw_rsi, size)
    return smooth_rsi
