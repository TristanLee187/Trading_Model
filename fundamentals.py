# File to update the quarterly earnings data
# Assumes "daily_market_data/quarterly_earnings.csv" ALREADY EXISTS with old data,
# and needs to be updated.

import requests
from alpha_vantage_api_key import ALPHA_VANTAGE_API_KEY
import pandas as pd
from datetime import date, timedelta
from common import tickers
import yfinance as yf


RATE_LIMIT = 25
EARNINGS_API_ENDPOINT = "https://www.alphavantage.co/query?function=EARNINGS"


# Get earnings data for a ticker from Alpha Vantage
# (should only be needed for newly added tickers or tickers that haven't been updated in like 2 years)
def get_av_earnings_data(ticker: str, disk_date: date):
    params = {
        'symbol' : ticker,
        'apikey' : ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(EARNINGS_API_ENDPOINT, params=params)

    # Success!
    if response.status_code == 200:
        # Check if API limit was reached (inside the response)
        if 'Information' in response.json():
            print("Alpha Vantage API limit reached!")
            return -1
        
        print("Alpha Vantage API called...")

        # Convert the JSON into a pd Dataframe
        quarterly_earnings = {
            'symbol': [],
            'Year': [],
            'Month': [],
            'Day': [],
            'estimate_EPS': [],
            'report_EPS': [],
            'surprise_percent': []
        }

        av_quarterly_earnings = response.json()["quarterlyEarnings"]

        # Collect into a dictionary
        for quarter in av_quarterly_earnings:
            report_date = date.fromisoformat(quarter['reportedDate'])
            # Only add if more recent than the disk date
            if report_date > disk_date:
                quarterly_earnings['symbol'].append(ticker)
                quarterly_earnings['Year'].append(report_date.year)
                quarterly_earnings['Month'].append(report_date.month)
                quarterly_earnings['Day'].append(report_date.day)
                quarterly_earnings['estimate_EPS'].append(quarter['estimatedEPS'])
                quarterly_earnings['report_EPS'].append(quarter['reportedEPS'])
                quarterly_earnings['surprise_percent'].append(quarter['surprisePercentage'])

        # Convert dict to Dataframe
        return pd.DataFrame(quarterly_earnings)
        
    # Error!
    print(response.json())
    raise Exception(f"Error in getting {ticker} earnings")


# Update earnings data
def update_earnings_data():
    # Load the most recent earnings data on disk
    disk_earnings = pd.read_csv('daily_market_data/quarterly_earnings.csv')

    # Init new earnings data
    new_earnings_data = []

    # While maintaining the rate limit, for each ticker, update earnings data if the current
    # data is out of date
    rate_count = 0
    ticker_index = 0
    while rate_count < RATE_LIMIT and ticker_index < len(tickers):
        ticker = tickers[ticker_index]
        ticker_index += 1

        # Most recent earnings from yf
        yf_earnings = yf.Ticker(ticker).earnings_dates.dropna().sort_index().reset_index()
        # Add one day because yf seems to use a different date system...
        yf_earnings['Earnings Date'] = yf_earnings['Earnings Date'].apply(
            lambda t: (t+timedelta(days=1)).date())
        recent_earnings_date = yf_earnings['Earnings Date'].iloc[-1]

        # Most recent earnings from disk
        ticker_last_earnings = disk_earnings.groupby('symbol').get_group(ticker)[['Year', 'Month', 'Day']]
        last_date = date(*tuple(ticker_last_earnings.sort_values(by=['Year', 'Month', 'Day']).iloc[-1]))
        
        # If out of date, update
        if last_date < recent_earnings_date:
            # Check if we can just use yfinance for the data
            if last_date >= yf_earnings['Earnings Date'].iloc[0]:
                yf_earnings = yf_earnings[yf_earnings['Earnings Date'] > last_date]
                yf_earnings['symbol'] = ticker
                yf_earnings['Year'] = yf_earnings['Earnings Date'].apply(lambda t: t.year)
                yf_earnings['Month'] = yf_earnings['Earnings Date'].apply(lambda t: t.month)
                yf_earnings['Day'] = yf_earnings['Earnings Date'].apply(lambda t: t.day)
                yf_earnings.rename(columns={'EPS Estimate': 'estimate_EPS', 
                                            'Reported EPS': 'report_EPS', 
                                            'Surprise(%)': 'surprise_percent'}, inplace=True)
                yf_earnings.drop(columns=['Earnings Date'], inplace=True)

                print(f"{ticker} retrieved from yfinance")
                new_earnings_data.append(yf_earnings)
            
            # Call Alpha Vantage
            else:
                rate_count += 1
                new_earnings = get_av_earnings_data(ticker, last_date)
                if type(new_earnings) == int and new_earnings == -1:
                    pass
                else:
                    print(f"{ticker} retrieved from Alpha Vantage")
                    new_earnings_data.append(new_earnings)
        
        # Already up to date
        else:
            print(f"{ticker} already up to date")

    # Combine with disk earnings
    new_earnings_data = pd.concat([disk_earnings] + new_earnings_data)
    return new_earnings_data


if __name__ == '__main__':
    # Update the fundamental data
    new_earnings_data = update_earnings_data()
    new_earnings_data = new_earnings_data.astype({"Year": int, "Month": int, "Day": int})
    new_earnings_data.to_csv('./daily_market_data/quarterly_earnings.csv', index=False)
