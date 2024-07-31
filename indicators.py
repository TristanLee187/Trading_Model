# Price indicators (SMA, EMA, cross, MACD, and stochastic oscillator)

import pandas as pd


def sma(series: pd.Series, size: int):
    """
    Computes the Simple Moving Average (SMA) for each applicable point in time.

    Args:
        series (pandas.Series): Series of numerical data to base the SMA on.
        size (int): Size of the window to use for the SMA (number of previous points of time to consider).

    Returns:
        pandas.Series: Series containing the SMA at each applicable point in time.
    """
    return series.rolling(size).sum() / size


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
