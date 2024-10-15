# Evaluate a model based on recent market data

import numpy as np
import pandas as pd
from common import *
from build_data_set import build_daily_dataset, build_minute_dataset
from keras.api.models import load_model
from keras_nlp.api.layers import SinePositionEncoding
from keras.api.utils import custom_object_scope
from datetime import date, timedelta
import matplotlib.pyplot as plt
import joblib
import argparse


def build_eval_data(ticker: str, time_interval: str, start_date: date, end_date: date = None):
    """
    Prepare data needed for evaluation.

    Args:
        ticker (str): String with the ticker to evaluate the model on.
        time_interval (str): String with the time interval of the model. 
            Should be "1m" or "1d".
        start_date (datetime.date): Date for the first date of the time window for the evaluation.
            If time_interval is "1m", then this defines the day to get the 1-minute chart for.
        end_date (datetime.date): Date for the last date of the time window for the evaluation.
            If time_interval is "1m", then this is ignored (since only the start_date is used).

    Returns:
        pandas.DataFrame, pandas.Series: Market data, and a time column to use for plotting.
            If time_interval is "1d", then the data contains enough previous data to make sequences
                ending before start_date, thus the model can make predictions for start_date.
    """
    if time_interval == '1d':
        # Include enough previous data so that predictions can be made for start_date
        data = build_daily_dataset(
            ticker, start_date - timedelta(days=2 * WINDOW_LENGTH), end_date)
        time_col = pd.to_datetime(data[['Year', 'Month', 'Day']]).dt.date
        # Cut off dates that are too early
        for i in range(len(data)):
            if time_col.iloc[i+WINDOW_LENGTH] >= start_date:
                data = data[i:]
                break
        time_col = time_col[time_col >= start_date]
    elif time_interval == '1m':
        data = build_minute_dataset(ticker, start_date)
        time_col = data['Minute'][WINDOW_LENGTH:]

    return data, time_col


def reg_model_eval(model_path: str, model_arch: str, ticker: str, time_interval: str,
                   label: str, error: str, start_date: date, end_date: date = None):
    """
    Evaluate a regression model's predictions for the given ticker's price over some time period.

    Args:
        model_path (str): String with the path to the model (which should be generated by "train.py").
        model_arch (str): String containing the architecture used.
        ticker (str): String with the ticker to evaluate the model on.
        time_interval (str): String with the time interval of the model. 
            Should be "1m" or "1d".
        label (str): String with the label used for the model. 
            Should be "price".
        error (str): String with the error (loss) function used by the model.
        start_date (datetime.date): First date of the time window for the evaluation.
            If time_interval is "1m", then this defines the day to get the 1-minute chart for.
        end_date (datetime.date): Last date of the time window for the evaluation.
            If time_interval is "1m", then this is ignored (since only the start_date is used).

    Returns:
        None

    Side Effects:
        Prints to standard output 1) the mean absolute error, 2) the mean squared error,
            and 3) the percentage of days correctly classified as higher/lower.
        Plots the ground truth labels vs. the model's predictions.
    """
    # Fetch Yahoo Finance data and init a time column for plotting
    data, time_col = build_eval_data(
        ticker, time_interval, start_date, end_date)

    # Prepare test data and scalers to plot the real values
    X, y_gt, scaler_mins, scaler_scales = prepare_model_data(
        data, label, 'Close')

    # Predict
    if model_arch in ['LSTM', 'transformer']:
        with custom_object_scope({'SinePositionEncoding': SinePositionEncoding}):
            model = load_model(model_path, compile=False)
    elif model_arch == 'forest':
        model = joblib.load(model_path)
        X = X.reshape(X.shape[0], -1)
    y_predictions = model.predict(X).reshape(len(y_gt))

    # Scale the normalized ground truth and predictions back to their original values
    y_gt = y_gt / scaler_scales + scaler_mins
    y_predictions = y_predictions / scaler_scales + scaler_mins

    # Calculate various metrics.
    caption = ''

    # MAE
    mae = np.mean(np.abs(y_gt - y_predictions))
    caption += f"Mean Absolute Error (MAE): {round(mae, 10)}\n"

    # MSE
    mse = np.mean(np.power(y_gt - y_predictions, 2))
    caption += f"Mean Squared Error (MSE): {round(mse, 10)}\n"

    # Percentage of days correctly classified as positive/negative returns.
    X_prices = data['Close'].iloc[WINDOW_LENGTH-1:-1]
    correct_incorrect = np.sign(
        (y_gt - X_prices) * (y_predictions - X_prices))
    percent_correct = len(
        [sign for sign in correct_incorrect if sign == 1]) / len(y_gt)
    caption += f"Percent Correctly Classified: {round(percent_correct, 10) * 100}%"
    print(caption)

    # Plot the ground truth vs. predictions.
    fig = plt.figure(figsize=(10, 8))
    plt.plot(time_col, y_gt, "k", label="Ground Truth")
    plt.plot(time_col, y_predictions, "b", label="Predictions")
    plt.legend()

    x_mapper = {
        '1d': "Date", '1m': "Time (in minutes)"
    }
    plt.xlabel(x_mapper[time_interval])

    plt.ylabel('Price')

    plt.title(f"{ticker} Ground Truth vs. Predicted Price")

    fig.tight_layout(pad=5)
    fig.text(0.125, 0.01, caption, ha='left')

    # Save the plot
    date_string = str(start_date)
    if time_interval == '1d':
        date_string += '_' + str(end_date)

    plot_name = './plots/{}/{}_{}_{}_{}_close-{}_{}.jpeg'.format(
        VERSION, model_arch, ticker, time_interval, date_string, label, error
    )

    plt.savefig(plot_name)
    plt.show()


def all_tickers_class_model_eval(model_path: str, model_arch: str, time_interval: str,
                                 start_date: date, end_date: date = None):
    """
    Evaluate a classification model's predictions for all tickers over some time period. 

    Args:
        model_path (str): String with the path to the model (which should be generated by "train.py").
        model_arch (str): String containing the architecture used.
        time_interval (str): String with the time interval of the model. 
            Should be "1m" or "1d".
        start_date (datetime.date): First date of the time window for the evaluation.
            If time_interval is "1m", then this defines the day to get the 1-minute chart for.
        end_date (datetime.date): Last date of the time window for the evaluation.
            If time_interval is "1m", then this is ignored (since only the start_date is used).

    Returns:
        None

    Side Effects:
        For each ticker, simulate buying/selling according to the model's predictions, and print out the
        percentage gain/loss. Also, print out the average gain/loss for all tickers. weighted by the number
        of actions (buy/sell) made for that ticker. Export these results to a text file as well.
    """
    # Compute the profit/loss of predictions given ground truth prices, as well as
    # the number of total actions to use as a "confidence" weight.
    def ticker_class_buy_sell_eval(predicted_actions, prices):
        cost = 0
        revenue = 0
        count = 0
        confidence = 1
        for i in range(len(prices)):
            # Check if there are potential actions
            if i < len(predicted_actions):
                action = predicted_actions[i]
                # Buy
                if action == 1:
                    cost += prices[i]
                    count += 1
                    confidence += 1
                # Sell
                elif action == 2:
                    revenue += count * prices[i]
                    count = 0
                    confidence += 1

            # Close out the last position
            elif i == len(prices) - 1:
                revenue += count * prices[i]

        # Return percentage performance (positive for gain, negative for loss, 0 if no actions were taken)
        if cost == 0:
            return 0, confidence
        delta = revenue - cost
        return delta / cost, confidence

    average_performance = 0
    total_confidence = 0
    performance_output = ''

    if model_arch in ['LSTM', 'transformer']:
        with custom_object_scope({'SinePositionEncoding': SinePositionEncoding}):
            model = load_model(model_path, compile=False)
    elif model_arch == 'forest':
        model = joblib.load(model_path)

    for ticker in tickers:
        # Fetch Yahoo Finance data
        data, time_col = build_eval_data(
            ticker, time_interval, start_date, end_date)

        # Prepare test data
        X, y_gt, scaler_mins, scaler_scales = prepare_model_data(
            data, 'signal', 'Close')
        if model_arch == 'forest':
            X = X.reshape(X.shape[0], -1)

        # Predict
        y_predictions = model.predict(X)

        # Convert one-hot predictions to classes (0 for do nothing, 1 for buy, 2 for sell)
        if model_arch != 'forest':
            y_predictions = np.argmax(y_predictions, axis=1)

        # Evaluate the decisions against the actual prices
        prices = data['Close'].iloc[WINDOW_LENGTH:].to_numpy()
        performance, confidence = ticker_class_buy_sell_eval(
            y_predictions, prices)
        average_performance += confidence * performance
        total_confidence += confidence

        # Record the performance as strings for printing and export
        performance_string = f'{ticker}: {round(100 * performance, 2)}% return with {confidence-1} actions'
        print(performance_string)
        performance_output += performance_string + '\n'

    # Record the average performance as strings
    average_performance /= total_confidence
    average_performance_string = f'Average Return for all Tickers: {round(100 * average_performance, 2)}%'
    print(average_performance_string)
    performance_output += '\n' + average_performance_string

    # Export to a text file
    date_string = str(start_date)
    if time_interval == '1d':
        date_string += '_' + str(end_date)
    file_name = './plots/{}/{}_{}_{}_buy_sell_performance.txt'.format(
        VERSION, model_arch, time_interval, date_string
    )
    with open(file_name, 'w') as file:
        file.write(performance_output)
        file.close()


def ticker_class_model_eval(model_path: str, model_arch: str, ticker: str, time_interval: str,
                            start_date: date, end_date: date = None):
    """
    Evaluate a classification model's predictions for the given ticker's price over some time period.

    Args:
        model_path (str): String with the path to the model (which should be generated by "train.py").
        model_arch (str): String containing the architecture used.
        ticker (str): String with the ticker to evaluate the model on.
        time_interval (str): String with the time interval of the model. 
            Should be "1m" or "1d".
        start_date (datetime.date): First date of the time window for the evaluation.
            If time_interval is "1m", then this defines the day to get the 1-minute chart for.
        end_date (datetime.date): Last date of the time window for the evaluation.
            If time_interval is "1m", then this is ignored (since only the start_date is used).

    Returns:
        None

    Side Effects:
        Plots the ground prices with that day's predicted action.
    """
    # Fetch Yahoo Finance data and init a time column for plotting
    data, time_col = build_eval_data(
        ticker, time_interval, start_date, end_date)

    # Prepare test data and scalers to plot the real values
    X, y_gt, scaler_mins, scaler_scales = prepare_model_data(
        data, 'signal', 'Close')

    # Predict
    if model_arch in ['LSTM', 'transformer']:
        with custom_object_scope({'SinePositionEncoding': SinePositionEncoding}):
            model = load_model(model_path, compile=False)
    elif model_arch == 'forest':
        model = joblib.load(model_path)
        X = X.reshape(X.shape[0], -1)
    y_predictions = model.predict(X)

    # Convert one-hot predictions to classes (0 for do nothing, 1 for buy, 2 for sell)
    y_gt = np.argmax(y_gt, axis=1)
    if model_arch != 'forest':
        y_predictions = np.argmax(y_predictions, axis=1)

    # Plot the ground truth vs. predictions.
    gt_buy_mask = y_gt == 1
    gt_sell_mask = y_gt == 2
    buy_mask = y_predictions == 1
    sell_mask = y_predictions == 2
    prices = data['Close'].iloc[WINDOW_LENGTH:]

    # Plot ground truth prices
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(18, 8))
    ax1.plot(time_col, prices, "k", label="Close")
    ax2.plot(time_col, prices, "k", label="Close")

    # Plot buy/sell signals
    time_col = time_col.iloc[:-FUTURE_WINDOW_LENGTH]
    prices = prices.iloc[:-FUTURE_WINDOW_LENGTH]

    ax1.scatter(time_col[gt_buy_mask], prices[gt_buy_mask],
                s=50, c='g', label="Ground Truth Buy")
    ax1.scatter(time_col[gt_sell_mask], prices[gt_sell_mask],
                s=50, c='darkred', label="Ground Truth Sell")
    ax1.legend()

    ax2.scatter(time_col[buy_mask], prices[buy_mask], s=40,
                c='lime', alpha=1, label="Predicted Buy")
    ax2.scatter(time_col[sell_mask], prices[sell_mask],
                s=40, c='red', alpha=1, label="Predicted Sell")
    ax2.legend()

    x_mapper = {
        '1d': "Date", '1m': "Time (in minutes)"
    }
    fig.supxlabel(x_mapper[time_interval])

    fig.supylabel('Price')
    fig.suptitle(f"{ticker} Predicted Buy/Sell Signals")

    # Save the plot
    date_string = str(start_date)
    if time_interval == '1d':
        date_string += '_' + str(end_date)

    plot_name = './plots/{}/{}_{}_{}_{}_close-signal.jpeg'.format(
        VERSION, model_arch, ticker, time_interval, date_string
    )

    plt.savefig(plot_name)
    plt.show()


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        description="Train a Model"
    )
    parser.add_argument('-k', '--ticker', type=str,
                        help='ticker to evaluate the model on, or "all" for evaluating buy/sell signals', required=True)
    parser.add_argument('-m', '--model', type=str, help='model architecture to use',
                        choices=['LSTM', 'transformer', 'forest'], required=True)
    parser.add_argument('-t', '--time_interval', type=str, help='time interval data to train on',
                        choices=['1m', '1d'], required=True)
    parser.add_argument('-l', '--label', type=str, help='labels to use for each instance',
                        choices=['price', 'signal'], required=True)
    parser.add_argument('-e', '--error', type=str,
                        help='error (loss) function to use (required if regression, ignored if classification)')
    args = parser.parse_args()

    # Get the model location for NNs
    if args.model in ['LSTM', 'transformer']:
        if args.label == 'price':
            loss_func_str = args.error
        elif args.label == 'signal':
            loss_func_str = 'cce'

        tag = './models/{}/{}_{}_close-{}_{}'.format(
            VERSION, args.model, args.time_interval, args.label, loss_func_str
        )

        model_path = f'{tag}_model.keras'
    # Get the model location for Random Forest
    elif args.model == 'forest':
        tag = './models/{}/{}_{}_close-{}'.format(
            VERSION, args.model, args.time_interval, args.label
        )

        model_path = f'{tag}_model.pkl'

    if args.time_interval == '1d':
        start, end = date(2024, 1, 1), date(2024, 9, 30)
    elif args.time_interval == '1m':
        start, end = date(2024, 8, 5), None

    if args.label == 'price':
        if args.ticker == 'all':
            raise ValueError(
                '"all" ticker choice is only supported for buy/sell signals')
        reg_model_eval(model_path, args.model, args.ticker, args.time_interval,
                       args.label, args.error, start, end)
    elif args.label == 'signal':
        if args.ticker == 'all':
            all_tickers_class_model_eval(model_path, args.model,
                                         args.time_interval, start, end)
        else:
            ticker_class_model_eval(model_path, args.model, args.ticker,
                                    args.time_interval, start, end)
