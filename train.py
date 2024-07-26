# Train a model based on the market data

import numpy as np
import pandas as pd
from common import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping
import argparse

def prepare_training_data(time_interval: str, norm: bool, label: str):
    """
    Prepare training data (inputs and ground truth labels).

    Args:
        time_interval (str): String defining the time interval data to use in training:
            "1m": Use the "miniute_market_data" data. Sequences are limited to within a day
                (they do not span multiple days).
            "1d": Use the "daily_market_data" data. Sequences span any gaps days.
        normalize (bool): Boolean indicating whether to normalize the data or not.
        label (str): String indicating what value to use as the labels:
            "price": Use the price of the given column.
            "percent-change": Use the percent change in values of the given column.
    
    Returns:
        numpy.array, numpy.array: Two numpy arrays X and y containing the training instances and ground
            truth labels, respectively.
    """
    # Init training instances and labels
    X, y = [], []

    # Distinguish which directory to read from based on the time interval
    dir_prefix = 'minute' if time_interval == '1m' else 'daily'

    for ticker in tickers:
        data = pd.read_csv(f'./{dir_prefix}_market_data/{ticker}.csv')
        
        if time_interval == '1m':
            # Break down each file into its component days
            daily_data = data.groupby(by=['Year', 'Month', 'Day'])
            days = daily_data.groups.keys()
            for day in days:
                day_data = daily_data.get_group(day)
                ticker_X, ticker_y = prepare_model_data(day_data, norm, label, 'Close')

                X.append(ticker_X)
                y.append(ticker_y)
        
        else:
            # Just use the whole file as the training set
            ticker_X, ticker_y = prepare_model_data(data, norm, label, 'Close')

            X.append(ticker_X)
            y.append(ticker_y)
        
        print(f'{ticker} done!')

    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y

def get_lstm_model(shape: tuple[int, int]):
    """
    Define an LSTM model based on the given architecture string.

    Args:
        shape (tuple[int, int]): shape of each input instance.
    
    Returns:
        keras.models.Sequential: Sequential model with an LSTM architecture.
    """
    # Define the LSTM model
    window_length, num_features = shape
    model = Sequential([
        Input(shape=(window_length, num_features)),
        LSTM(units=num_features**2, return_sequences=True),
        LSTM(units=50),
        Dense(units=1)
    ])
    return model

if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        prog="Train a Model"
    )
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

    # Prepare training data
    X, y = prepare_training_data(args.time_interval, args.norm, args.label)

    # Prepare validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.model == 'LSTM':
        model = get_lstm_model(X[0].shape)

        # Compile with early stopping
        model.compile(optimizer='adam', loss=args.error)
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

        # Train!
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Save the model
        model.save('./models/LSTM_{}_close-{}_{}normed_{}_model.keras'.format(
            args.time_interval, args.label, "" if args.norm else "not-", args.error
        ))
