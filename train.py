# Train a model based on the market data

import numpy as np
import pandas as pd
from common import *
from sklearn.model_selection import train_test_split
from keras import Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, MultiHeadAttention, Add, LayerNormalization
from keras.callbacks import EarlyStopping
import argparse


def prepare_training_data(time_interval: str, label: str):
    """
    Prepare training data (inputs and ground truth labels).

    Args:
        time_interval (str): String defining the time interval data to use in training:
            "1m": Use the "miniute_market_data" data. Sequences are limited to within a day
                (they do not span multiple days).
            "1d": Use the "daily_market_data" data. Sequences span any gaps days.
        label (str): String indicating what value to use as the labels:
            "price": Use the price of the given column.

    Returns:
        numpy.array, numpy.array: Two numpy arrays X and y containing the training instances and ground
            truth labels, respectively.
    """
    # Init training instances and labels
    X, y = [], []

    # Distinguish which directory to read from based on the time interval
    dir_prefix = 'minute' if time_interval == '1m' else 'daily'

    # Read the master list
    tickers_df = pd.read_csv(f'./{dir_prefix}_market_data/all_tickers.csv')

    tickers_df_grouped = tickers_df.groupby(by=['Ticker'])

    for ticker in tickers:
        data = tickers_df_grouped.get_group(ticker)

        if time_interval == '1m':
            # Break down each file into its component days
            daily_data = data.groupby(by=['Year', 'Month', 'Day'])
            days = daily_data.groups.keys()
            for day in days:
                day_data = daily_data.get_group(day)
                ticker_X, ticker_y, mins, scales = prepare_model_data(
                    day_data, label, 'Close')

                X.append(ticker_X)
                y.append(ticker_y)

        else:
            # Just use the whole file as the training set
            ticker_X, ticker_y, mins, scales = prepare_model_data(
                data, label, 'Close')

            X.append(ticker_X)
            y.append(ticker_y)

        print(f'{ticker} done!')

    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y


def get_lstm_model(shape: tuple[int, int]):
    """
    Define an LSTM model.

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
        LSTM(units=100),
        Dense(units=1)
    ])
    return model


def get_transformer_model(shape: tuple[int, int]):
    """
    Define an LSTM and attention-based model..

    Args:
        shape (tuple[int, int]): shape of each input instance.

    Returns:
        keras.models.Sequential: Sequential model with an LSTM and attention architecture.
    """
    # Define Transformer block
    def transformer_block(x, num_heads, key_dim, ff_dim_1, ff_dim_2):
        attn_layer = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim)(x, x)
        x = Add()([x, attn_layer])
        x = LayerNormalization(epsilon=1e-6)(x)
        ff = Dense(ff_dim_2)(Dense(ff_dim_1)(x))
        x = Add()([x, ff])
        x = LayerNormalization(epsilon=1e-6)(x)
        return x

    # Define the Transformer model
    input_layer = Input(shape=shape)
    transformer_layer_1 = transformer_block(
        input_layer, num_heads=4, key_dim=64, ff_dim_1=256, ff_dim_2=shape[1])
    transformer_layer_2 = transformer_block(
        transformer_layer_1, num_heads=4, key_dim=32, ff_dim_1=64, ff_dim_2=shape[1])
    lstm_pooling_layer = LSTM(units=32)(transformer_layer_2)
    dense_layer = Dense(units=32)(lstm_pooling_layer)
    output_layer = Dense(units=1)(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        prog="Train a Model"
    )
    parser.add_argument('-m', '--model', type=str, help='model architecture to use',
                        choices=['LSTM', 'transformer'], required=True)
    parser.add_argument('-t', '--time_interval', type=str, help='time interval data to train on',
                        choices=['1m', '1d'], required=True)
    parser.add_argument('-l', '--label', type=str, help='labels to use for each instance',
                        choices=['price'], required=True)
    parser.add_argument('-e', '--error', type=str,
                        help='error (loss) function to use', required=True)
    args = parser.parse_args()

    # Prepare training data
    X, y = prepare_training_data(
        args.time_interval, args.label)

    # Prepare validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    if args.model in ['LSTM', 'transformer']:
        if args.model == 'LSTM':
            model = get_lstm_model(X[0].shape)
        else:
            model = get_transformer_model(X[0].shape)

        # Compile with early stopping
        model.compile(optimizer='adam', loss=args.error)
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

        # Train!
        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_data=(X_val, y_val), callbacks=[early_stopping])

        tag = './models/v3/{}_{}_close-{}_{}'.format(
            args.model, args.time_interval, args.label, args.error
        )

        # Save the model
        model.save(f'{tag}_model.keras')
