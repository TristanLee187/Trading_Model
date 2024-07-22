# Produce a CSV file of market data for multiple tickers

import yfinance as yf
import indicators as ind
from datetime import date, timedelta

def build_dataset(ticker: str, start_date: date, end_date: date):
    """
    Build a DataFrame containing daily market data of a ticker between 2 dates.

    Args:
        ticker (str): Ticker of the company.
        start_date (datetime.datetime): First day of data to include in the file.
            If the market is not open this day, uses the first open day after this day instead.
        end_date (datetime.datetime): Last day of data to include in the file.
            If the market is not open this day, uses the last open day before this day instead.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the ticker's market data for the given date range.
    """

    # Define the first day to fetch data from as 1 year before the start_date, so that 200 SMA/EMA
    # can be computed for start_date.
    earlier_start_date = start_date - timedelta(days=365)

    # Fetch daily data from Yahoo Finance.
    data = yf.Ticker(ticker=ticker).history(start=earlier_start_date, end=end_date, 
                                            interval='1d', actions=False)

    # Extract the year, month, and day.
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

    # Use certain amounts of time to define SMAs and EMAs
    time_periods = [5, 20, 50, 200]

    # Calculate various SMAs and EMAs using the closing price.
    for time_period in time_periods:
        data[f'{time_period}_SMA'] = ind.sma(data['Close'], time_period)
        data[f'{time_period}_EMA'] = ind.ema(data['Close'], time_period)

    # Calculate a golden/death cross using 50- and 200-SMA.
    data['Cross'] = ind.cross(data['Close'], 50, 200)

    # Calculate the MACD using 12- and 26-EMA.
    data['MACD'] = ind.macd(data['Close'], 12, 26)

    # Calculate the stochastic oscillator.
    data['Stochastic_Oscillator'] = ind.stochastic_oscillator(data['Close'], 14)

    # Filter out rows with null values and whose dates are before the requested start_date
    data = data.dropna()
    data = data[data.index.date >= start_date]

    # Remove the "Date" index.
    data.reset_index(drop=True, inplace=True)

    # Return the DataFrame
    return data


if __name__ == '__main__':
    # Define certain tickers
    tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', '^GSPC', '^DJI', '^RUT', 'CL=F', 'GC=F']

    # Start from 2000-01-01 and end on 2023-12-31 (leaving out 2024 for testing)
    s = date(2000, 1, 1)
    e = date(2023, 12, 31)

    for ticker in tickers:
        data = build_dataset(ticker, s, e)
        
        # Export the data to a CSV file.
        data.to_csv(f'./market_data/{ticker}.csv', index=False)