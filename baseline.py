# Baseline performance measurements

from common import *
import pandas as pd
import numpy as np
from datetime import date
import argparse


np.random.seed(42)


def build_eval_data(ticker: str):
    """
    Prepare data needed for evaluation by reading from disk.

    Args:
        ticker (str): String with the ticker to evaluate the model on.

    Returns:
        pandas.DataFrame, pandas.Series: Market data, and a time column to use for plotting.
    """
    data = pd.read_csv('daily_market_data/all_tickers_eval.csv')
    data = data[data['Ticker'] == ticker]
    time_col = pd.to_datetime(data[['Year', 'Month', 'Day']]).dt.date

    return data, time_col


def buy_and_hold_strategy(data: pd.DataFrame):
    """
    Execute a buy and hold strategy, buying once at the beginning of the time window and selling
    once at the end of the time window.

    Args:
        data (pandas.DataFrame): DataFrame holding the data associated with a single ticker.

    Returns:
        numpy.array: List of predicted actions: 0 for sell, 1 for do nothing, or 2 for buy.
            Will have the same length as the given data.
    """
    predictions = np.ones(len(data))
    # Buy
    predictions[0] = 2
    # Sell
    predictions[-1] = 0
    return predictions


def buy_strategy(data: pd.DataFrame):
    """
    Execute a buy strategy, buying on every day of the time window.

    Args:
        data (pandas.DataFrame): DataFrame holding the data associated with a single ticker.

    Returns:
        numpy.array: List of predicted actions: 0 for sell, 1 for do nothing, or 2 for buy.
            Will have the same length as the given data.
    """
    predictions = np.zeros(len(data))
    predictions.fill(2)
    return predictions


def random_strategy(data: pd.DataFrame):
    """
    Execute random buy, sells, and do-nothings for every day in the time window.

    Args:
        data (pandas.DataFrame): DataFrame holding the data associated with a single ticker.

    Returns:
        numpy.array: List of predicted actions: 0 for sell, 1 for do nothing, or 2 for buy.
            Will have the same length as the given data.
    """
    predictions = np.random.randint(0, 3, len(data))
    return predictions


def momentum_strategy(data: pd.DataFrame):
    """
    Execute buys and sells following the momentum (MACD).

    Args:
        data (pandas.DataFrame): DataFrame holding the data associated with a single ticker.

    Returns:
        numpy.array: List of predicted actions: 0 for sell, 1 for do nothing, or 2 for buy.
            Will have the same length as the given data.
    """
    # If MACD switches from negative to positive, buy.
    # If other way around, sell.
    macd = np.array(data['MACD'])
    predictions = np.ones(len(data))
    for i in range(1, len(data)):
        if macd[i]>=0 and macd[i-1]<0:
            predictions[i] = 2
        elif macd[i]<0 and macd[i-1]>=0:
            predictions[i] = 0
    return predictions


def swing_strategy(data: pd.DataFrame):
    """
    Execute buys and sells following an overbought/oversold indicator (RSI).

    Args:
        data (pandas.DataFrame): DataFrame holding the data associated with a single ticker.

    Returns:
        numpy.array: List of predicted actions: 0 for sell, 1 for do nothing, or 2 for buy.
            Will have the same length as the given data.
    """
    # If RSI is 0.7 or above, sell.
    # If RSI is 0.3 or less, buy.
    rsi = np.array(data['RSI'])
    predictions = np.ones(len(data))
    for i in range(len(data)):
        if rsi[i]<=0.3:
            predictions[i] = 2
        elif rsi[i]>=0.7:
            predictions[i] = 0
    return predictions


def all_tickers_baseline_eval(strategy: str):
    """
    Evaluate a given strategy for all tickers on a given time range.

    Args:
        strategy (str): String indicating what strategy to use.
            - buy_and_hold: buy once at the beginning, hold until the end.
            - buy: buy every day.
            - random: random action every day.

    Returns:
        None

    Side Effects:
        For each ticker, simulate buying/selling according to the model's predictions, and print out the
        percentage gain/loss. Also, print out the average gain/loss for all tickers. weighted by the number
        of actions (buy/sell) made for that ticker. Export these results to a text file as well.
    """

    total_cost = 0
    total_revenue = 0
    performance_output = ''

    strategy_mapper = {
        'buy_and_hold': buy_and_hold_strategy,
        'buy': buy_strategy,
        'random': random_strategy,
        'momentum': momentum_strategy,
        'swing': swing_strategy
    }
    strategy_func = strategy_mapper[strategy]

    def sign_blank_or_negative(x):
        return "" if x>=0 else "-"

    for ticker in tickers:
        # Get data from Yahoo Finance and (on disk) quarterly earnings
        data, time_col = build_eval_data(ticker)
        # Python scoping is weird...
        start_date, end_date = time_col.iloc[0], time_col.iloc[-1]
        prices = np.array(data['Close'])
        # Normalize by the first day's price
        prices /= prices[0]

        predictions = strategy_func(data)
        cost = 0
        revenue = 0
        count = 0
        # Execute the buy/sells
        # Buy: buy 1, sell: sell all
        for i in range(len(predictions)):
            if predictions[i] == 0:
                revenue += count * prices[i]
                count = 0
            elif predictions[i] == 2:
                cost += prices[i]
                count += 1
            # Close out the last position
            if i == len(predictions)-1:
                revenue += count * prices[i]
                count = 0

        total_cost += cost
        total_revenue += revenue

        # Record the performance as strings for printing and export
        profit = revenue - cost
        performance_string = f'{ticker}: {sign_blank_or_negative(profit)}${abs(round(profit, 2))} profit from ${round(cost, 2)} cost'
        print(performance_string)
        performance_output += performance_string + '\n'

    # Record the total profit and return
    total_profit = total_revenue - total_cost
    profit_string = f'{sign_blank_or_negative(total_profit)}${abs(round(total_profit, 2))} total profit from ${round(total_cost, 2)} total cost'
    return_string = f'Total return for all tickers: {round(100*total_profit/total_cost, 2)}%'
    print(profit_string)
    print(return_string)
    performance_output += '\n' + profit_string + '\n' + return_string + '\n'

    # Export to a text file
    date_string = str(start_date) + '_' + str(end_date)
    file_name = './plots/{}/baselines/baseline_{}_{}_performance.txt'.format(
        VERSION, strategy, date_string
    )
    with open(file_name, 'w') as file:
        file.write(performance_output)
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test a Baseline Strategy"
    )
    parser.add_argument('-s', '--strategy', type=str,
                        choices=['buy_and_hold', 'buy', 'random', 'momentum', 'swing'],
                        help='strategy to test for all tickers on the evaluation data', required=True)
    args = parser.parse_args()

    all_tickers_baseline_eval(args.strategy)
