# Produce a CSV file of market data for multiple tickers by fetching from Yahoo Finance.

import yfinance as yf
import indicators as ind
from datetime import date, timedelta
from common import tickers
import argparse
import pandas as pd

# Sector names to unique ids
sec_to_id = {
    'Basic Materials': 1,
    'Industrials': 2,
    'Consumer Defensive': 3,
    'Consumer Cyclical': 4,
    'Technology': 5,
    'Communication Services': 6,
    'Financial Services': 7,
    'Energy': 8,
    'Utilities': 9,
    'Healthcare': 10,
    'Real Estate': 11
}

def build_daily_dataset(ticker: str, start_date: date, end_date: date):
    """
    Build a DataFrame containing daily market data of a ticker between 2 dates.

    Args:
        ticker (str): Ticker of the company, index, etc..
        start_date (datetime.date): First day of data to include in the file.
            If the market is not open this day, uses the first open day after this day instead.
        end_date (datetime.date): Last day of data to include in the file.
            If the market is not open this day, uses the last open day before this day instead.

    Returns:
        pandas.DataFrame: A DataFrame containing the ticker's market data for the given date range.
    """

    # Define the first day to fetch data from as 1 year before the start_date, so that 200 SMA/EMA
    # can be computed for start_date.
    earlier_start_date = start_date - timedelta(days=365)

    # Fetch daily data from Yahoo Finance.
    data = yf.Ticker(ticker=ticker).history(start=earlier_start_date, end=end_date,
                                            interval='1d', actions=False, auto_adjust=False)

    # Return empty dataframe if the data couldn't be retrieved.
    if data.empty:
        return data

    # Extract the year, month, and day.
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

    # Calculate percent changes.
    data['Proportional_Change'] = ind.proportional_change(data['Close'])

    # Use certain amounts of time to define SMAs and EMAs.
    time_periods = [5, 20, 50, 200]

    # Calculate various SMAs and EMAs using the closing price.
    for time_period in time_periods:
        data[f'{time_period}_SMA'] = ind.sma(data['Close'], time_period)
        data[f'{time_period}_EMA'] = ind.ema(data['Close'], time_period)

    # Calculate crosses using different pairs of SMAs.
    data['Cross_1'] = ind.cross(data['Close'], 5, 20)
    data['Cross_2'] = ind.cross(data['Close'], 20, 50)
    data['Cross_3'] = ind.cross(data['Close'], 50, 200)

    # Calculate the MACD using 12- and 26-EMA.
    data['MACD'] = ind.macd(data['Close'], 12, 26)

    # Calculate the stochastic oscillator.
    data['Stochastic_Oscillator'] = ind.stochastic_oscillator(
        data['Close'], 14)

    # Calculate the RSI.
    data['RSI'] = ind.rsi(data['Close'], 14)

    # Add the sector ID.
    data['Sector_ID'] = sec_to_id[yf.Ticker(ticker).info['sector']]

    # Join with fundamentals data
    fund_data = pd.read_csv("daily_market_data/quarterly_earnings.csv")
    ticker_fund_data = fund_data[fund_data['symbol'] == ticker].drop(columns=['symbol'])
    # Resolve same day reports by taking the average
    ticker_fund_data = ticker_fund_data.groupby(['Year', 'Month', 'Day'], as_index=False).mean()
    # Fix some values
    ticker_fund_data['estimate_EPS'] = ticker_fund_data['estimate_EPS'].replace(0, None).ffill()
    ticker_fund_data['report_EPS'] = ticker_fund_data['report_EPS'].replace(0, None).ffill()
    ticker_fund_data['surprise_percent'] = (ticker_fund_data['report_EPS'] - ticker_fund_data['estimate_EPS'])/ticker_fund_data['estimate_EPS']
    # Extract the dates to be used for filtering later
    dates = data.index.date.copy()
    data = data.merge(ticker_fund_data, how="left")
    # P/E ratio
    data['PE'] = data['Close'] / data['report_EPS'].ffill()
    # Smooth EPS data slightly
    data['estimate_EPS'] = ind.ema(data['estimate_EPS'], size=4)
    data['report_EPS'] = ind.ema(data['report_EPS'], size=4)
    data['surprise_percent'] = ind.ema(data['surprise_percent'], size=4)

    data.fillna(0, inplace=True)

    # Filter out rows with null values and whose dates are before the requested start_date
    data = data[dates >= start_date]
    data = data.dropna()

    return data


def build_minute_dataset(ticker: str, day: date):
    """
    Build a DataFrame containing 1-minute interval market data of a ticker on a given day.

    Args:
        ticker (str): Ticker of the company, index, etc..
        day (datetime.date): Date to retrive the data for.

    Returns:
        pandas.DataFrame: A DataFrame containing the ticker's market data for the given date.
    """

    # Fetch 1-minute data from Yahoo Finance.
    data = yf.Ticker(ticker=ticker).history(start=day, end=day + timedelta(days=1),
                                            interval='1m', actions=False, auto_adjust=False)

    # Return empty dataframe if the data couldn't be retrieved.
    if data.empty:
        return data

    # Extract time information (date, minutes).
    data['Year'] = day.year
    data['Month'] = day.month
    data['Day'] = day.day
    data['Minute'] = 60 * data.index.hour + data.index.minute

    # Calculate percent changes.
    data['Proportional_Change'] = ind.proportional_change(data['Close'])

    # Use certain amounts of time to define SMAs and EMAs
    time_periods = [5, 20, 50]

    # Calculate various SMAs and EMAs using the closing price.
    for time_period in time_periods:
        data[f'{time_period}_SMA'] = ind.sma(data['Close'], time_period)
        data[f'{time_period}_EMA'] = ind.ema(data['Close'], time_period)

    # Calculate a golden/death cross using 20- and 50-SMA.
    data['Cross'] = ind.cross(data['Close'], 20, 50)

    # Calculate the MACD using 12- and 26-EMA.
    data['MACD'] = ind.macd(data['Close'], 12, 26)

    # Calculate the stochastic oscillator.
    data['Stochastic_Oscillator'] = ind.stochastic_oscillator(
        data['Close'], 14)

    # Calculate the RSI.
    data['RSI'] = ind.rsi(data['Close'], 14)

    # Filter out rows with null values (times where the SMA/EMA aren't defined yet because it's too early in the day).
    data = data.dropna()

    # Remove the "Date" index.
    data.reset_index(drop=True, inplace=True)

    return data


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        description="Export Raw Data"
    )
    parser.add_argument('-t', '--time_interval', type=str, help='time interval data to train on',
                        choices=['1m', '1d'], required=True)
    args = parser.parse_args()

    # Init single dataframe to hold all data.
    tickers_df = []

    if args.time_interval == "1m":
        for ticker in tickers:
            # Init list of DataFrames to merge together
            dfs = []

            # Start from 2024-07-08 and end on 2024-08-02
            s = date(2024, 7, 8)
            e = date(2024, 8, 2)

            while s <= e:
                # Don't try to fetch weekend data
                if s.isoweekday() not in [6, 7]:
                    dfs.append(build_minute_dataset(ticker, s))
                s += timedelta(days=1)

            # Concat the data for the ticker
            data = pd.concat(dfs)
            data['Ticker'] = ticker

            # Add to master list
            tickers_df.append(data)

            print(f'{ticker} is done')

        tickers_df = pd.concat(tickers_df)
        tickers_df.to_csv('./minute_market_data/all_tickers.csv', index=False)

    elif args.time_interval == "1d":
        # Start from 2000-01-01 and end on 2023-12-31 (leaving out 2024 for testing)
        s = date(2000, 1, 1)
        e = date(2023, 12, 31)

        for ticker in tickers:
            data = build_daily_dataset(ticker, s, e)
            data['Ticker'] = ticker

            # Add to master list
            tickers_df.append(data)

            print(f'{ticker} is done')

        tickers_df = pd.concat(tickers_df)
        tickers_df.to_csv('./daily_market_data/all_tickers.csv', index=False)
