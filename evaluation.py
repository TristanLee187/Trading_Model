# Evaluate a model based on recent market data

import numpy as np
import pandas as pd
from common import *
from build_data_set import build_daily_dataset, build_minute_dataset
from keras.models import load_model
from datetime import date, timedelta
import matplotlib.pyplot as plt
import argparse


def reg_model_eval(model_path: str, model_arch: str, ticker: str, time_interval: str,
                   label: str, error: str, start_date: date, end_date: date = None):
    """
    Evaluate a regression model's predictions for the given ticker's price over some time period.

    Args:
        model_path (str): String with the path to the LSTM model (which should be generated by "train.py").
        model_arch (str): String containing the architecture used.
        ticker (str): String with the ticker to evaluate the model on.
        time_interval (str): String with the time interval of the model. 
            Should be "1m" or "1d".
        label (str): String with the label used for the model. 
            Should be "price" or "price-change".
        error (str): String with the error (loss) function used by the model.
        start_date (datetime.date): Date for the first date of the time window for the evaluation.
            If time_interval is "1m", then this defines the day to get the 1-minute chart for.
        end_date (datetime.date): Date for the last date of the time window for the evaluation.
            If time_interval is "1m", then this is ignored (since only the start_date is used).

    Returns:
        None

    Side Effects:
        Prints to standard output 1) the mean absolute error, 2) the mean squared error,
            and 3) the percentage of days correctly classified as higher/lower.
        Plots the ground truth labels vs. the model's predictions.
    """
    # Fetch Yahoo Finance data and init a time column for plotting
    if time_interval == '1d':
        data = build_daily_dataset(
            ticker, start_date - timedelta(days=2 * WINDOW_LENGTH), end_date)
        time_col = pd.to_datetime(data[['Year', 'Month', 'Day']]).dt.date
        # Cutoff dates that are too early
        for i in range(len(data)):
            if time_col[i+WINDOW_LENGTH] >= start_date:
                data = data[i:]
                break
        time_col = time_col[time_col >= start_date]
    else:
        data = build_minute_dataset(ticker, start_date)
        time_col = data['Minute'][WINDOW_LENGTH:]

    # Prepare test data and scalers to plot the real values
    X, y_gt, scaler_mins, scaler_scales = prepare_model_data(
        data, label, 'Close')

    # Predict
    model = load_model(model_path)
    y_predictions = model.predict(X).reshape(len(y_gt))

    # Scale the ground truth and predictions back to their original scales
    if label == 'price':
        y_gt = y_gt / scaler_scales + scaler_mins
        y_predictions = y_predictions / scaler_scales + scaler_mins
    elif label == 'price-change':
        y_gt = y_gt / scaler_scales
        y_predictions = y_predictions / scaler_scales

    # Calculate various metrics.
    # MAE
    mae = np.mean(np.abs(y_gt - y_predictions))
    print(f"Mean Absolute Error (MAE): {round(mae, 10)}")

    # MSE
    mse = np.mean(np.power(y_gt - y_predictions, 2))
    print(f"Mean Squared Error (MSE): {round(mse, 10)}")

    # Percentage of days correctly classified as positive/negative returns.
    if label == 'price':
        X_prices = data['Close'].iloc[WINDOW_LENGTH-1:-1]
        correct_incorrect = np.sign(
            (y_gt - X_prices) * (y_predictions - X_prices))
    elif label == 'price-change':
        correct_incorrect = np.sign(y_gt * y_predictions)

    percent_correct = len(
        [sign for sign in correct_incorrect if sign == 1]) / len(y_gt)
    print(f"Percent Correctly Classified: {round(percent_correct, 10) * 100}%")

    # Plot the ground truth vs. predictions.
    fig = plt.figure(figsize=(10, 8))
    if label == 'price':
        plt.plot(time_col, y_gt, "k", label="Ground Truth")
        plt.plot(time_col, y_predictions, "b", label="Predictions")
    elif label == 'price-change':
        plt.scatter(time_col, y_gt, c="k", label="Ground Truth")
        plt.scatter(time_col, y_predictions, c="b", label="Predictions")
    plt.legend()

    x_mapper = {
        '1d': "Date", '1m': "Time (in minutes)"
    }
    plt.xlabel(x_mapper[time_interval])

    y_mapper = {
        'price': "Price", 'price-change': "Price Change"
    }
    plt.ylabel(y_mapper[label])

    plt.title(f"{ticker} Ground Truth vs. Predicted {y_mapper[label]}")

    date_string = str(start_date)
    if time_interval == '1d':
        date_string += '_' + str(end_date)

    plot_name = './plots/v3/{}_{}_{}_{}_close-{}_{}.jpeg'.format(
        model_arch, ticker, time_interval, date_string, label, error
    )

    plt.savefig(plot_name)
    plt.show()


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        prog="Train a Model"
    )
    parser.add_argument('-k', '--ticker', type=str,
                        help='ticker to evaluate the model on', required=True)
    parser.add_argument('-m', '--model', type=str, help='model architecture to use',
                        choices=['LSTM', 'transformer'], required=True)
    parser.add_argument('-t', '--time_interval', type=str, help='time interval data to train on',
                        choices=['1m', '1d'], required=True)
    parser.add_argument('-l', '--label', type=str, help='labels to use for each instance',
                        choices=['price', 'price-change'], required=True)
    parser.add_argument('-e', '--error', type=str,
                        help='error (loss) function to use', required=True)
    args = parser.parse_args()

    if args.model in ['LSTM', 'transformer']:
        tag = './models/v3/{}_{}_close-{}_{}'.format(
            args.model, args.time_interval, args.label, args.error
        )

        model_path = f'{tag}_model.keras'

        if args.time_interval == '1d':
            start, end = date(2024, 1, 1), date(2024, 6, 30)
        elif args.time_interval == '1m':
            start, end = date(2024, 7, 26), None

        reg_model_eval(model_path, args.model, args.ticker, args.time_interval,
                       args.label, args.error, start, end)
