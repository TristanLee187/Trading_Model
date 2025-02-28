# Evaluate a model based on recent market data

import numpy as np
import pandas as pd
from common import *
from build_data_set import build_daily_dataset
import keras
from keras.api.models import load_model
from keras.api.utils import custom_object_scope
from train import custom_categorical_crossentropy, Expert, MoETopKLayer
from datetime import date, timedelta
import matplotlib.pyplot as plt
import argparse


CUSTOM_OBJECTS = {
    'custom_categorical_crossentropy': custom_categorical_crossentropy, 
    'Expert': Expert, 
    'MoETopKLayer': MoETopKLayer
}


keras.config.enable_unsafe_deserialization()


def build_eval_data(ticker: str, start_date: date, end_date: date = None):
    """
    Prepare data needed for evaluation.

    Args:
        ticker (str): String with the ticker to evaluate the model on.
        start_date (datetime.date): Date for the first date of the time window for the evaluation.
            If time_interval is "1m", then this defines the day to get the 1-minute chart for.
        end_date (datetime.date): Date for the last date of the time window for the evaluation.
            If time_interval is "1m", then this is ignored (since only the start_date is used).

    Returns:
        pandas.DataFrame, pandas.Series: Market data, and a time column to use for plotting.
            If time_interval is "1d", then the data contains enough previous data to make sequences
                ending before start_date, thus the model can make predictions for start_date.
    """
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

    return data, time_col


def reg_model_eval(model_path: str, model_arch: str, ticker: str,
                   label: str, error: str, start_date: date, end_date: date = None):
    """
    Evaluate a regression model's predictions for the given ticker's price over some time period.

    Args:
        model_path (str): String with the path to the model (which should be generated by "train.py").
        model_arch (str): String containing the architecture used.
        ticker (str): String with the ticker to evaluate the model on.
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
    data, time_col = build_eval_data(ticker, start_date, end_date)

    # Prepare test data and scalers to plot the real values
    X, x_meta, y_gt, scaler_mins, scaler_scales = prepare_model_data(
        data, label, 'Close')

    # Predict
    if model_arch == 'transformer':
        with custom_object_scope(CUSTOM_OBJECTS):
            model = load_model(model_path, compile=False)
    y_predictions = model.predict([X, x_meta]).reshape(len(y_gt))

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

    plt.xlabel("Date")

    plt.ylabel('Price')

    plt.title(f"{ticker} Ground Truth vs. Predicted Price")

    fig.tight_layout(pad=5)
    fig.text(0.125, 0.01, caption, ha='left')

    # Save the plot
    date_string = str(start_date) + '_' + str(end_date)

    plot_name = './plots/{}/{}_{}_{}_close-{}_{}.jpeg'.format(
        VERSION, model_arch, ticker, date_string, label, error
    )

    plt.savefig(plot_name)
    plt.show()


def all_tickers_class_model_eval(model_path: str, model_arch: str,
                                 start_date: date, end_date: date = None):
    """
    Evaluate a classification model's predictions for all tickers over some time period. 

    Args:
        model_path (str): String with the path to the model (which should be generated by "train.py").
        model_arch (str): String containing the architecture used.
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
    # Compute the profit/loss of predictions given ground truth prices.
    def ticker_class_buy_sell_eval(predicted_actions, prices):
        cost = 0
        cost_basis = 0
        revenue = 0
        count = 0
        for i in range(len(prices)):
            # Check if there are potential actions
            if i < len(predicted_actions):
                action = predicted_actions[i]
                # Buy
                if action == 2:
                    cost += prices[i]
                    cost_basis += prices[i]
                    count += 1
                # Sell
                elif action == 0:
                    revenue += count * prices[i]
                    count = 0
                    cost_basis = 0

            # Close out the last position
            elif i == len(prices) - 1:
                revenue += count * prices[i]

        # Return cost and revenue
        return cost, revenue
    
    def sign_blank_or_negative(x):
        return "" if x>=0 else "-"

    total_cost = 0
    total_revenue = 0
    performance_output = ''
    total_loss = 0

    if model_arch == 'transformer':
        with custom_object_scope(CUSTOM_OBJECTS):
            model = load_model(model_path)

    for ticker in tickers:
        # Fetch Yahoo Finance data
        data, time_col = build_eval_data(ticker, start_date, end_date)

        # Prepare test data
        X, x_meta, y_gt, scaler_mins, scaler_scales = prepare_model_data(
            data, 'signal', 'Close')

        # Predict
        y_predictions = model.predict([X, x_meta])
        loss, *metrics = model.evaluate([X, x_meta], y_gt)
        total_loss += loss

        # Convert one-hot predictions to classes (0 for do nothing, 1 for buy, 2 for sell)
        y_actions = np.argmax(y_predictions, axis=1)

        # Evaluate the decisions against the actual prices (standardize all prices so that each
        # ticker starts with price $1)
        prices = data['Close'].iloc[WINDOW_LENGTH:].to_numpy()
        prices /= prices[0]
        cost, revenue = ticker_class_buy_sell_eval(
            y_actions, prices)
        total_cost += cost
        total_revenue += revenue

        # Record the performance as strings for printing and export
        profit = revenue - cost
        performance_string = f'{ticker}: {sign_blank_or_negative(profit)}${abs(round(profit, 2))} profit from ${round(cost, 2)} cost'
        print(performance_string)
        performance_output += performance_string + '\n'

    # Record the total profit and return
    total_profit = total_revenue - total_cost
    profit_string = f'{sign_blank_or_negative(total_profit)}${round(total_profit, 2)} total profit from ${round(total_cost, 2)} total cost'
    return_string = f'Total return for all tickers: {round(100*total_profit/total_cost, 2)}%'
    print(profit_string)
    print(return_string)
    performance_output += '\n' + profit_string + '\n' + return_string + '\n'
    print(f"Total loss: {total_loss/len(tickers)}")

    # Export to a text file
    date_string = str(start_date) + '_' + str(end_date)
    file_name = './plots/{}/{}_{}_buy_sell_performance.txt'.format(
        VERSION, model_arch, date_string
    )
    with open(file_name, 'w') as file:
        file.write(performance_output)
        file.close()


def ticker_class_model_eval(model_path: str, model_arch: str, ticker: str,
                            start_date: date, end_date: date = None):
    """
    Evaluate a classification model's predictions for the given ticker's price over some time period.

    Args:
        model_path (str): String with the path to the model (which should be generated by "train.py").
        model_arch (str): String containing the architecture used.
        ticker (str): String with the ticker to evaluate the model on.
        start_date (datetime.date): First date of the time window for the evaluation.
        end_date (datetime.date): Last date of the time window for the evaluation.

    Returns:
        None

    Side Effects:
        Plots the ground prices with that day's predicted action.
    """
    # Fetch Yahoo Finance data and init a time column for plotting
    data, time_col = build_eval_data(ticker, start_date, end_date)

    # Prepare test data and scalers to plot the real values
    X, x_meta, y_gt, scaler_mins, scaler_scales = prepare_model_data(
        data, 'signal', 'Close')

    # Predict
    if model_arch == 'transformer':
        with custom_object_scope(CUSTOM_OBJECTS):
            model = load_model(model_path, compile=False)
    y_predictions = model.predict([X, x_meta])

    # Convert one-hot predictions to classes (0 for do nothing, 1 for buy, 2 for sell)
    y_gt = np.argmax(y_gt, axis=1)
    y_actions = np.argmax(y_predictions, axis=1)

    # Plot the ground truth vs. predictions.
    gt_buy_mask = y_gt == 2
    gt_sell_mask = y_gt == 0
    buy_mask = y_actions == 2
    sell_mask = y_actions == 0
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

    fig.supxlabel("Date")

    fig.supylabel('Price')
    fig.suptitle(f"{ticker} Predicted Buy/Sell Signals")

    # Save the plot
    date_string = str(start_date) + '_' + str(end_date)

    plot_name = './plots/{}/{}_{}_{}_close-signal.jpeg'.format(
        VERSION, model_arch, ticker, date_string
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
                        choices=['transformer'], required=True)
    parser.add_argument('-l', '--label', type=str, help='labels to use for each instance',
                        choices=['price', 'signal'], required=True)
    parser.add_argument('-e', '--error', type=str,
                        help='error (loss) function to use (required if regression, ignored if classification)')
    args = parser.parse_args()

    # Get the model location for NNs
    if args.model == 'transformer':
        if args.label == 'price':
            loss_func_str = args.error
        elif args.label == 'signal':
            loss_func_str = 'cce'

        tag = './models/{}/{}_close-{}_{}'.format(
            VERSION, args.model, args.label, loss_func_str
        )

        model_path = f'{tag}_model.keras'

    start, end = date(2024, 1, 1), date(2024, 12, 31)

    if args.label == 'price':
        if args.ticker == 'all':
            raise ValueError(
                '"all" ticker choice is only supported for buy/sell signals')
        reg_model_eval(model_path, args.model, args.ticker,
                       args.label, args.error, start, end)
    elif args.label == 'signal':
        if args.ticker == 'all':
            all_tickers_class_model_eval(model_path, args.model, start, end)
        else:
            ticker_class_model_eval(model_path, args.model, args.ticker, start, end)
