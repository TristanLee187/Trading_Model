# Evaluate a model based on recent market data

import numpy as np
import pandas as pd
from common import *
import keras
from keras.api.models import load_model
from keras.api.utils import custom_object_scope
from model import CUSTOM_OBJECTS
from datetime import date, timedelta
import matplotlib.pyplot as plt
import argparse


keras.config.enable_unsafe_deserialization()


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


def reg_model_eval(model_tag: str, ticker: str):
    """
    Evaluate a regression model's predictions for the given ticker's price.

    Args:
        model_tag (str): String of the tag of the model file (which should be generated by "train.py").
        ticker (str): String with the ticker to evaluate the model on.

    Returns:
        None

    Side Effects:
        Prints to standard output 1) the mean absolute error, 2) the mean squared error,
            and 3) the percentage of days correctly classified as higher/lower.
        Plots the ground truth labels vs. the model's predictions.
    """
    # Fetch Yahoo Finance data and init a time column for plotting
    data, time_col = build_eval_data(ticker)
    start_date, end_date = time_col.iloc[0], time_col.iloc[-1]

    # Prepare test data and scalers to plot the real values
    X, x_meta, y_gt, scaler_scales = prepare_model_data(data, 'price', 'Close', False)

    # Predict
    with custom_object_scope(CUSTOM_OBJECTS):
        model_path = f'models/{VERSION}/{model_tag}_model.keras'
        model = load_model(model_path, compile=False)
    y_predictions = model.predict([X, x_meta]).reshape(len(y_gt))

    # Scale the normalized ground truth and predictions back to their original values
    y_gt = y_gt / scaler_scales
    y_predictions = y_predictions / scaler_scales

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
    correct_incorrect = np.sign((y_gt - X_prices) * (y_predictions - X_prices))
    percent_correct = len([sign for sign in correct_incorrect if sign == 1]) / len(y_gt)
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

    plot_name = './plots/{}/{}_{}.jpeg'.format(
        VERSION, model_tag, date_string
    )

    plt.savefig(plot_name)
    plt.show()


def all_tickers_class_model_eval(model_tag: str):
    """
    Evaluate a classification model's predictions for all tickers, as well as the ground truth labels.

    Args:
        model_tag (str): String of the tag of the model file (which should be generated by "train.py").

    Returns:
        None

    Side Effects:
        For each ticker, simulate buying/selling according to the model's predictions, and print out the
        percentage gain/loss. Also, print out the average gain/loss for all tickers. weighted by the number
        of actions (buy/sell) made for that ticker. Export these results to a text file as well.
        Output the same file but using the ground truth labels as the predictions.
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
            if i == len(prices) - 1:
                revenue += count * prices[i]

        # Return cost and revenue
        return cost, revenue
    
    def sign_blank_or_negative(x):
        return "" if x>=0 else "-"

    total_cost = 0
    total_revenue = 0
    performance_output = ''
    total_loss = 0

    gt_output = ''
    gt_total_cost = 0
    gt_total_revenue = 0

    with custom_object_scope(CUSTOM_OBJECTS):
        model_path = f'models/{VERSION}/{model_tag}_model.keras'
        model = load_model(model_path)

    for ticker in tickers:
        # Fetch Yahoo Finance data
        data, time_col = build_eval_data(ticker)
        # Python scoping is weird...
        start_date, end_date = time_col.iloc[0], time_col.iloc[-1]

        # Prepare test data
        X, x_meta, y_gt, scaler_scales = prepare_model_data(data, 'signal', 'Close', False)

        # Predict
        y_predictions = model.predict([X, x_meta])
        loss, *metrics = model.evaluate([X[:len(y_gt)], x_meta[:len(y_gt)]], y_gt)
        total_loss += loss

        # Convert one-hot predictions to classes (0 for do nothing, 1 for buy, 2 for sell)
        y_actions = np.argmax(y_predictions, axis=1)

        # Evaluate the decisions against the actual prices (standardize all prices so that each
        # ticker starts with price $1)
        prices = data['Close'].iloc[WINDOW_LENGTH-1:].to_numpy()
        prices /= prices[0]
        cost, revenue = ticker_class_buy_sell_eval(y_actions, prices)
        total_cost += cost
        total_revenue += revenue

        # Record the performance as strings for printing and export
        profit = revenue - cost
        performance_string = f'{ticker}: {sign_blank_or_negative(profit)}${abs(round(profit, 2))} profit from ${round(cost, 2)} cost'
        print(performance_string)
        performance_output += performance_string + '\n'

        # Do the same but for ground truth labels
        gt_actions = np.argmax(y_gt, axis=1)
        cost, revenue = ticker_class_buy_sell_eval(gt_actions, prices)
        gt_total_cost += cost
        gt_total_revenue += revenue
        profit = revenue - cost
        performance_string = f'{ticker}: {sign_blank_or_negative(profit)}${abs(round(profit, 2))} profit from ${round(cost, 2)} cost'
        gt_output += performance_string + '\n'


    # Record the total profit and return
    total_profit = total_revenue - total_cost
    profit_string = f'{sign_blank_or_negative(total_profit)}${abs(round(total_profit, 2))} total profit from ${round(total_cost, 2)} total cost'
    return_string = f'Total return for all tickers: {round(100*total_profit/total_cost, 2)}%'
    print(profit_string)
    print(return_string)
    performance_output += '\n' + profit_string + '\n' + return_string + '\n'
    print(f"Total loss: {total_loss/len(tickers)}")

    # Do the same for ground truth
    gt_total_profit = gt_total_revenue - gt_total_cost
    profit_string = f'{sign_blank_or_negative(gt_total_profit)}${abs(round(gt_total_profit, 2))} total profit from ${round(gt_total_cost, 2)} total cost'
    return_string = f'Total return for all tickers: {round(100*gt_total_profit/gt_total_cost, 2)}%'
    gt_output += '\n' + profit_string + '\n' + return_string + '\n'

    # Export to a text file
    date_string = str(start_date) + '_' + str(end_date)
    file_name = './plots/{}/model_performance/{}_{}_buy_sell_performance.txt'.format(
        VERSION, model_tag, date_string
    )
    with open(file_name, 'w') as file:
        file.write(performance_output)
        file.close()
    
    # Same for ground truth
    file_name = './plots/{}/model_performance/ground_truth_{}_buy_sell_performance.txt'.format(
        VERSION, date_string
    )
    with open(file_name, 'w') as file:
        file.write(gt_output)
        file.close()


def ticker_class_model_eval(model_tag: str, ticker: str):
    """
    Evaluate a classification model's predictions for the given ticker's price.

    Args:
        model_tag (str): String of the tag of the model file (which should be generated by "train.py").
        ticker (str): String with the ticker to evaluate the model on.

    Returns:
        None

    Side Effects:
        Plots the ground prices with that day's predicted action.
    """
    # Fetch Yahoo Finance data and init a time column for plotting
    data, time_col = build_eval_data(ticker)
    start_date, end_date = time_col.iloc[0], time_col.iloc[-1]

    # Prepare test data and scalers to plot the real values
    X, x_meta, y_gt, scaler_scales = prepare_model_data(data, 'signal', 'Close', False)

    # Predict
    with custom_object_scope(CUSTOM_OBJECTS):
        model_path = f'models/{VERSION}/{model_tag}_model.keras'
        model = load_model(model_path, compile=False)
    y_predictions = model.predict([X, x_meta])

    # Convert one-hot predictions to classes (0 for do nothing, 1 for buy, 2 for sell)
    y_gt = np.argmax(y_gt, axis=1)
    y_actions = np.argmax(y_predictions, axis=1)
    num_gt, num_pred = len(y_gt), len(y_actions)

    # Plot the ground truth vs. predictions.
    gt_buy_mask = y_gt == 2
    gt_sell_mask = y_gt == 0
    buy_mask = y_actions == 2
    sell_mask = y_actions == 0
    prices = data['Close']

    # Plot ground truth prices
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(18, 8))
    ax1.plot(time_col, prices, "k", label="Close")
    ax2.plot(time_col, prices, "k", label="Close")

    # Plot buy/sell signals
    time_col = time_col.iloc[WINDOW_LENGTH-1:]
    prices = prices.iloc[WINDOW_LENGTH-1:]

    ax1.scatter(time_col[:num_gt][gt_buy_mask], prices[:num_gt][gt_buy_mask],
                s=50, c='g', label="Ground Truth Buy")
    ax1.scatter(time_col[:num_gt][gt_sell_mask], prices[:num_gt][gt_sell_mask],
                s=50, c='darkred', label="Ground Truth Sell")
    ax1.legend()

    ax2.scatter(time_col[:num_pred][buy_mask], prices[:num_pred][buy_mask], s=40,
                c='lime', alpha=1, label="Predicted Buy")
    ax2.scatter(time_col[:num_pred][sell_mask], prices[:num_pred][sell_mask],
                s=40, c='red', alpha=1, label="Predicted Sell")
    ax2.legend()

    fig.supxlabel("Date")

    fig.supylabel('Price')
    fig.suptitle(f"{ticker} Predicted Buy/Sell Signals")

    # Save the plot
    date_string = str(start_date) + '_' + str(end_date)

    plot_name = './plots/{}/model_plots/{}_{}_{}_close-signal.jpeg'.format(
        VERSION, model_tag, ticker, date_string
    )

    plt.savefig(plot_name)
    plt.show()


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        description="Evaluate a Model"
    )
    parser.add_argument('-k', '--ticker', type=str,
                        help='ticker to evaluate the model on, or "all" for evaluating buy/sell signals', 
                        required=True)
    parser.add_argument('-m', '--model_tag', type=str, 
                        help='tag of the model file to use (not including "models/" path or "_model.keras" suffix)', 
                        required=True)
    parser.add_argument('-l', '--label', type=str, help='labels to use for each instance',
                        choices=['price', 'signal'], required=True)
    args = parser.parse_args()

    if args.label == 'price':
        if args.ticker == 'all':
            raise ValueError(
                '"all" ticker choice is only supported for buy/sell signals')
        reg_model_eval(args.model_tag, args.ticker)
    elif args.label == 'signal':
        if args.ticker == 'all':
            all_tickers_class_model_eval(args.model_tag)
        else:
            ticker_class_model_eval(args.model_tag, args.ticker)
