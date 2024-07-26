# Evaluate a model based on recent market data

import numpy as np
import pandas as pd
from common import *
from build_data_set import build_daily_dataset, build_minute_dataset
from keras.models import load_model
from datetime import date, timedelta
import matplotlib.pyplot as plt
import argparse

def lstm_model_eval(model_path: str, ticker: str, time_interval: str, label: str, norm: bool,
                    start_date: date, end_date: date=None):
    """
    Evaluate an LSTM model's predictions for the given ticker's price over some time period.

    Args:
        model_path (str): String with the path to the LSTM model (which should be generated by "train.py").
        ticker (str): String with the ticker to evaluate the model on.
        time_interval (str): String with the time interval of the model. 
            Should be "1m" or "1d".
        label (str): String with the label used for the model. 
            Should be "price" or "percent-change".
        norm (bool): Boolean for if the model used normalization.
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
        data = build_daily_dataset(ticker, start_date - timedelta(days=2 * WINDOW_LENGTH), end_date)
        time_col = pd.to_datetime(data[['Year', 'Month', 'Day']]).dt.date
        # Cutoff dates that are too early
        for i in range(len(data)):
            if time_col[i+WINDOW_LENGTH] >= start_date:
                data = data[i:]
                break
        time_col = time_col[time_col >= start_date]
    else:
        data = build_minute_dataset(ticker, start_date)
        time_col = pd.to_datetime(data['Minute'], unit='m').dt.time

    # Transform the data into the sequence format the model wants
    X, y_gt = prepare_model_data(data, norm, label, 'Close')

    # Predict
    model = load_model(model_path)
    y_predictions = model.predict(X).reshape(len(y_gt))

    # Calculate various metrics.
    # MAE
    mae = np.mean(np.abs(y_gt - y_predictions))
    print(f"Mean Absolute Error (MAE): {round(mae, 4)}")

    # MSE
    mse = np.mean(np.power(y_gt - y_predictions, 2))
    print(f"Mean Squared Error (MSE): {round(mse, 4)}")

    # Percentage of days correctly classified as positive/negative returns.
    correct_incorrect = np.sign(y_gt * y_predictions)
    percent_correct = len([sign for sign in correct_incorrect if sign == 1]) / len(y_gt)
    print(f"Percent Correctly Classified: {round(percent_correct, 4) * 100}%")

    # Plot the ground truth vs. predictions.
    fig = plt.figure(figsize=(10, 8))
    plt.plot(time_col, y_gt, "k", label="Ground Truth")
    plt.plot(time_col, y_predictions, "b", label="Predictions")
    plt.legend()

    x_mapper = {
        '1d': "Date", '1m': "Time"
    }
    plt.xlabel(x_mapper[time_interval])

    y_mapper = {
        'price': "Price", 'percent-change': "Daily Return"
    }
    plt.ylabel(y_mapper[label])

    plt.title(f"{ticker} Ground Truth vs. Predicted {y_mapper[label]}")

    date_string = str(start_date)
    if time_interval == '1d':
        date_string += '_' + str(end_date)

    plot_name = './plots/v2/LSTM_{}_{}_{}_close-{}_{}normed.jpeg'.format(
            ticker, time_interval, date_string, label, "" if norm else "not-", 
        )

    plt.savefig(plot_name)
    plt.show()


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        prog="Train a Model"
    )
    parser.add_argument('-k', '--ticker', type=str, help='ticker to evaluate the model on', required=True)
    parser.add_argument('-m', '--model', type=str, help='model architecture to use',
                        choices=['LSTM'], required=True)
    parser.add_argument('-t', '--time_interval', type=str, help='time interval data to train on',
                        choices=['1m', '1d'], required=True)
    parser.add_argument('-l', '--label', type=str, help='labels to use for each instance',
                        choices=['price', 'percent-change'], required=True)
    parser.add_argument('-n', '--norm', type=int, help='whether or not to normalize data where applicable',
                        choices=[0, 1], required=True)
    parser.add_argument('-e', '--error', type=str, help='error (loss) function to use', required=True)
    args = parser.parse_args()

    if args.model == 'LSTM':
        model_path = './models/LSTM_{}_close-{}_{}normed_{}_model.keras'.format(
                args.time_interval, args.label, "" if args.norm else "not-", args.error
            )
        
        if args.time_interval == '1d':
            start, end = date(2024,1,1), date(2024,6,30)
        elif args.time_interval == '1m':
            start, end = date(2024,7,26), None

        lstm_model_eval(model_path, args.ticker, args.time_interval, args.label, args.norm, start, end)
