# Evaluate a model based on recent market data

import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import date, timedelta
from build_data_set import build_dataset
import matplotlib.pyplot as plt

def regression_model_eval(model_name: str, ticker: str, start_date: date, end_date: date):
    """
    Evaluate a regression model's predictions for the ticker's prices over the given dates.

    Args:
        model_name (str): Name of the file containing the Keras model to load.
            It should be in the models/ directory.
        ticker (str): Ticker of the company, index, etc..
        start_date (datetime.date): First day to include in the evaluation.
        end_date (datetime.date): Last day to include in the evaluation.

    Returns:
        None
    
    Side Effects:
        Prints to standard output 1) the mean absolute error, 2) the mean squared error,
            and 3) the percentage of days correctly classified as higher/lower.
        Plots the ground truth daily returns vs. the model's predictions.
    """

    window_length = 30

    # Fetch and transform the market data in question.
    data = build_dataset(ticker, start_date - timedelta(days=2 * window_length), end_date)
    
    # Create a date column for plotting.
    date_col = pd.to_datetime(data[['Year', 'Month', 'Day']]).dt.date

    # Construct an array of inputs to the model, and record the grounth truths.
    X, y_gt = [], []

    for i in range(len(data) - window_length):
        # Make sure the date in question is in range.
        if date_col[i+window_length] < start_date:
            continue

        sequence = data[i:i+window_length]

        last_close = sequence.iloc[window_length-1]['Close']
        this_close = data.iloc[i+window_length]['Close']
        percent_change = (this_close - last_close) / last_close

        X.append(sequence.to_numpy())
        y_gt.append(percent_change)

    X = np.array(X)
    y_gt = np.array(y_gt)

    # Use the model to make predictions.
    model = load_model(model_name)
    y_predictions = model.predict(X).reshape(len(y_gt))

    # Calculate various metrics.
    # MAE
    mae = np.mean(np.abs(y_gt - y_predictions))
    print(f"Mean Absolute Error (MAE): {round(mae, 4) * 100}%")

    # MSE
    mse = np.mean(np.power(y_gt - y_predictions, 2))
    print(f"Mean Squared Error (MSE): {round(mse, 4) * 10000}%^2")

    # Percentage of days correctly classified as positive/negative returns.
    correct_incorrect = np.sign(y_gt * y_predictions)
    percent_correct = len([sign for sign in correct_incorrect if sign == 1]) / len(y_gt)
    print(f"Percent Correctly Classified: {round(percent_correct, 4) * 100}%")

    # Plot the ground truth vs. predictions.
    date_col = date_col[date_col >= start_date]

    fig = plt.figure(figsize=(10, 8))
    plt.plot(date_col, 100 * y_gt, "k", label="Ground Truth")
    plt.plot(date_col, 100 * y_predictions, "b", label="Predictions")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")
    plt.title(f"{ticker} {str(start_date)}-{str(end_date)} Ground Truth vs. Predicted Daily Returns")
    plt.savefig(f"plots/regression_{ticker}_{str(start_date)}_{str(end_date)}.jpeg")
    plt.show()


if __name__ == '__main__':
    regression_model_eval("./models/regression_model.keras", "AAPL", 
                          start_date=date(2024,1,1), end_date=date(2024,6,30))
