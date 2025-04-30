# Produce a CSV file of market data for multiple tickers by fetching from Yahoo Finance.

import yfinance as yf
import indicators as ind
from datetime import date, timedelta
from common import tickers
import pandas as pd
import argparse

# Sector names to unique ids
sec_to_id = {
    'Basic Materials': 0,
    'Industrials': 1,
    'Consumer Defensive': 2,
    'Consumer Cyclical': 3,
    'Technology': 4,
    'Communication Services': 5,
    'Financial Services': 6,
    'Energy': 7,
    'Utilities': 8,
    'Healthcare': 9,
    'Real Estate': 10
}

def build_daily_dataset_helper(ticker: str, start_date: date, end_date: date):
    """
    Build a DataFrame containing daily market data of a ticker between 2 dates, BUT
        excluding the sector average proportional changes.

    Args:
        ticker (str): Ticker of the company, index, etc..
        start_date (datetime.date): First day of data to include in the file.
            If the market is not open this day, uses the first open day after this day instead.
        end_date (datetime.date): Last day of data to include in the file.
            If the market is not open this day, uses the last open day before this day instead.

    Returns:
        pandas.DataFrame: A DataFrame containing the ticker's market data for the given date range, BUT
            excluding the sector average proportional changes.
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
    data['Stochastic_Oscillator'] = ind.stochastic_oscillator(data['Close'], 14)

    # Calculate the RSI.
    data['RSI'] = ind.rsi(data['Close'], 14)

    # Add the sector ID.
    data['Sector_ID'] = sec_to_id[yf.Ticker(ticker).info['sector']]

    # Join with fundamentals data
    fund_data = pd.read_csv("daily_market_data/quarterly_earnings.csv")
    ticker_fund_data = fund_data[fund_data['symbol'] == ticker].drop(columns=['symbol'])

    # Resolve same day reports by taking the average
    ticker_fund_data = ticker_fund_data.groupby(['Year', 'Month', 'Day'], as_index=False).mean()

    # Add a day to all earnings dates to guarentee they are available on its day
    ticker_fund_data['date'] = pd.to_datetime(ticker_fund_data[['Year', 'Month', 'Day']])
    ticker_fund_data['date'] += timedelta(days=1)
    ticker_fund_data['Year'] = ticker_fund_data['date'].dt.year
    ticker_fund_data['Month'] = ticker_fund_data['date'].dt.month
    ticker_fund_data['Day'] = ticker_fund_data['date'].dt.day
    ticker_fund_data.drop(columns=['date'], inplace=True)

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


def build_daily_dataset_full(start_date: date, end_date: date):
    """
    Build a DataFrame containing daily market data of all tickers between 2 dates, BUT
        excluding the sector average proportional changes.

    Args:
        start_date (datetime.date): First day of data to include in the file.
            If the market is not open this day, uses the first open day after this day instead.
        end_date (datetime.date): Last day of data to include in the file.
            If the market is not open this day, uses the last open day before this day instead.

    Returns:
        pandas.DataFrame: A DataFrame containing all tickers' market data for the given date range, BUT
            excluding the sector average proportional changes.
    """
    # Init single dataframe to hold all data.
    tickers_df = []

    for ticker in tickers:
        data = build_daily_dataset_helper(ticker, start_date, end_date)
        data['Ticker'] = ticker

        # Add to master list
        tickers_df.append(data)

        print(f'{ticker} is done')

    tickers_df = pd.concat(tickers_df)
    return tickers_df


def add_sector_prop_change(data: pd.DataFrame):
    """
    Given a fill daily dataset (returned by "build_daily_dataset_full), add a column containing 
        the average proportional change for each sector across each day.

    Args:
        data (pandas.DataFrame): Full daily dataset, returned by "buid_daily_dataset_full."

    Returns:
        pandas.DataFrame: New Dataframe, the input dataframe with a new column "sector_avg_prop_change."
    """
    grouped_prop_changes = data.groupby(by=['Sector_ID', 'Year', 'Month', 'Day']).agg({'Proportional_Change': 'mean'}).reset_index()
    grouped_prop_changes.rename(columns={"Proportional_Change": "sector_avg_prop_change"}, inplace=True)
    
    return data.merge(grouped_prop_changes, how='left')


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        description="Build CSV files for Training and Evaluation Data"
    )
    parser.add_argument('-d', '--dataset', type=str, help='train or eval data', 
                        choices=['train', 'eval'], required=True)
    args = parser.parse_args()
    
    if args.dataset == 'train':
        # Start from 2000-01-01 and end on 2023-12-31 (leaving out 2024 for testing)
        s = date(2000, 1, 1)
        e = date(2023, 12, 31)

    elif args.dataset == 'eval':
        # All of 2024 plus some of 2025
        s = date(2024, 1, 1)
        e = date(2025, 1, 1)

    tickers_df = build_daily_dataset_full(s, e)
    tickers_df = add_sector_prop_change(tickers_df)
    tickers_df.to_csv(f'./daily_market_data/all_tickers_{args.dataset}.csv', index=False)
