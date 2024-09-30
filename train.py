# Train a model based on the market data

import numpy as np
import pandas as pd
from common import *
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import Model
from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Dense, Input, MultiHeadAttention, Add, LayerNormalization, Permute, Concatenate, Flatten
from keras_nlp.api.layers import SinePositionEncoding
from keras.api.initializers import HeNormal
from keras.api.optimizers import RMSprop
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.metrics import F1Score
from keras.api.utils import custom_object_scope
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse

def prepare_training_data(time_interval: str, label: str):
    """
    Prepare training data (inputs and ground truth labels).

    Args:
        time_interval (str): String defining the time interval data to use in training:
            "1m": Use the "miniute_market_data" data. Sequences are limited to within a day
                (they do not span multiple days).
            "1d": Use the "daily_market_data" data. Sequences span any gaps in days (weekends, holidays, etc.).
        label (str): String indicating what value to use as the labels:
            "price": Use the price of the given column.
            "signal": Use regression to indicate upward/downward/neither movement in following time points.

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

    # Generate data for each ticker
    for ticker in tickers:
        data = tickers_df_grouped.get_group((ticker,))

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

        elif time_interval == '1d':
            # Just use the ticker's whole file data as a contiguous training set
            ticker_X, ticker_y, mins, scales = prepare_model_data(
                data, label, 'Close')

            X.append(ticker_X)
            y.append(ticker_y)

        print(f'{ticker} is done')

    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y


def custom_categorical_crossentropy(y_true, y_pred):
    """
    Customer categorical-crossentropy loss function on 3 classes that uses weights to
    penalize different classificiations differently.

    Args:
        y_true (np.array): Ground truth one-hot encoding vector.
        y_pred (np.array): Prediction one-hot encoding vector.

    Returns:
        function with arguments (np.array, np.array): Function that computes the loss w.r.t.
            ground truth and prediction inputs.
    """
    # weights[i][j]: penalty for if the ground truth was i but the predicted was j.
    weights = tf.constant([
        [0.0, 2.0, 2.0],
        [3.0, 0.0, 10.0],
        [3.0, 10.0, 0.0]
    ])

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    weights_tensor = tf.reduce_sum(tf.expand_dims(
        weights, axis=0) * tf.expand_dims(y_true, axis=-1), axis=-2)
    weighted_loss = ce_loss * tf.reduce_sum(weights_tensor, axis=-1)
    return weighted_loss


def last_layer(label: str):
    """
    Define the last layer of the architectrue depending on the task (given by the label).

    Args:
        label (str): The label type to train on.

    Returns:
        keras.layers.Layer: Keras layer to use for the model's output:
            - "price" (regression): A Dense layer with 1 unit and sigmoid activiation.
            - "signal" (classification): A Dense layer with 3 units and softmax activation.
    """
    if label == 'price':
        return Dense(units=1, activation='sigmoid')
    elif label == 'signal':
        return Dense(units=3, activation='softmax')


def get_lstm_model(shape: tuple, label: str):
    """
    Define an LSTM model.

    Args:
        shape (tuple[int, int]): shape of each input instance.
        label (str):  The label type to train on.

    Returns:
        keras.models.Sequential: Sequential model with an LSTM architecture.
    """
    # Define the LSTM model
    window_length, num_features = shape
    model = Sequential([
        Input(shape=(window_length, num_features)),
        LSTM(units=num_features**2, return_sequences=True),
        LSTM(units=100),
        last_layer(label)
    ])
    return model


def get_transformer_model(shape: tuple, label: str):
    """
    Define a transformer and LSTM based architecture.

    Args:
        shape (tuple[int, int]): shape of each input instance.
        label (str):  The label type to train on.

    Returns:
        keras.Model: Model with an transformer and LSTM architecture.
    """
    # Transformer block
    def transformer_block(x, num_heads, key_dim, ff_dim_1, ff_dim_2):
        x = Add()([x, SinePositionEncoding()(x)])
        attn_layer = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            dropout=0.1)(x, x)
        x = Add()([x, attn_layer])
        x = LayerNormalization(epsilon=1e-6)(x)
        ff = Dense(ff_dim_2, activation='sigmoid')(
            Dense(ff_dim_1, activation='sigmoid')(x))
        x = Add()([x, ff])
        x = LayerNormalization(epsilon=1e-6)(x)
        return x

    # Stack of Transformer blocks
    def transformer_stack(x, num_heads, key_dim, ff_dim_1, ff_dim_2, num_blocks):
        for _ in range(num_blocks):
            x = transformer_block(x, num_heads=num_heads, key_dim=key_dim, 
                                  ff_dim_1=ff_dim_1, ff_dim_2=ff_dim_2)
        return x

    # Define the Transformer model
    # Get inputs as both temporal and feature sequences
    input_layer = Input(shape=shape)
    # transposed_input_layer = Permute((2, 1))(input_layer)
    # Apply transformer stacks to both of them
    temporal_transformer_layer = transformer_stack(
        input_layer, num_heads=4, key_dim=8, ff_dim_1=64, ff_dim_2=shape[1], num_blocks=1)
    # feature_transformer_layer = transformer_stack(
    #     transposed_input_layer, num_heads=6, key_dim=6, ff_dim_1=128, ff_dim_2=shape[0], num_blocks=8)
    # Concatenate them together
    # concated_layer = Concatenate()([
    #     temporal_transformer_layer, Permute((2, 1))(feature_transformer_layer)
    # ])
    # Apply transformer stacks to the concatenation
    # combined_transformer_layer = transformer_stack(
    #     concated_layer, num_heads=6, key_dim=6, ff_dim_1=128, ff_dim_2=2*shape[1], num_blocks=8)
    # Pool
    pooling_layer = LSTM(units=64)(temporal_transformer_layer)
    # Output
    dense_layer_1 = Dense(units=256, activation='sigmoid')(pooling_layer)
    dense_layer_2 = Dense(units=64, activation='sigmoid')(dense_layer_1)
    output_layer = last_layer(label)(dense_layer_2)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def get_random_forest_model():
    """
    Define a Random Forest with its hyperparameters.

    Args:
        None

    Returns:
        sklearn.ensemble.RandomForestClassifier: Random Forest model with
            set hyperparameters.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_samples=0.1,
        max_features=None,
        class_weight='balanced',
        criterion="entropy",
        min_samples_leaf=10,
        oob_score=True,
        random_state=42,
    )

    return model


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        description="Train a Model"
    )
    parser.add_argument('-m', '--model', type=str, help='model type/architecture to use',
                        choices=['LSTM', 'transformer', 'forest'], required=True)
    parser.add_argument('-t', '--time_interval', type=str, help='time interval data to train on',
                        choices=['1m', '1d'], required=True)
    parser.add_argument('-d', '--train_data', type=str, help='if set, path to file containing X and y sequence data')
    parser.add_argument('-l', '--label', type=str, help='labels to use for each instance',
                        choices=['price', 'signal'], required=True)
    parser.add_argument('-e', '--error', type=str,
                        help='error (loss) function to use (required for regression, ignored if classification)')
    parser.add_argument('-r', '--resume', type=str,
                        help='if set, path to a model to resume training on (only works for NNs)')
    parser.add_argument('-p', '--epochs', type=int,
                        help='number of training epochs for the NN models (defaults to 20)')
    args = parser.parse_args()

    # Prepare training data
    if args.train_data:
        npzfile = np.load(args.train_data)
        X, y = npzfile['X'], npzfile['y']
    else:
        X, y = prepare_training_data(
            args.time_interval, args.label)
        np.savez(f'./models/{VERSION}/{args.label}_X_and_y.npz', X=X, y=y)

    if args.model in ['LSTM', 'transformer']:
        # Prepare validation data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=0)
        # Get appropriate NN architecture
        if args.resume is not None:
            with custom_object_scope({'SinePositionEncoding': SinePositionEncoding}):
                model = load_model(args.resume, compile=False)
        elif args.model == 'LSTM':
            model = get_lstm_model(X[0].shape, args.label)
        else:
            model = get_transformer_model(X[0].shape, args.label)

        # Compile
        if args.label == 'price':
            model.compile(optimizer='adam', loss=args.error)
        elif args.label == 'signal':
            model.compile(
                optimizer=RMSprop(learning_rate=0.002),
                loss="categorical_crossentropy", 
                metrics=[F1Score()])

        # Train!
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr = 1e-7)
        class_proportions = Counter(y_train.argmax(axis=1))
        class_weights = {i: X.shape[0]/class_proportions[i] for i in class_proportions}
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=32,
                  class_weight=class_weights, validation_data=(X_val, y_val), 
                  callbacks=[lr_scheduler])
        
        # Save the model
        if args.label == 'price':
            loss_func_str = args.error
        elif args.label == 'signal':
            loss_func_str = 'cce'

        tag = './models/{}/{}_{}_close-{}_{}'.format(
            VERSION, args.model, args.time_interval, args.label, loss_func_str
        )

        model.save(f'{tag}_model.keras')

    elif args.model == "forest":
        model = get_random_forest_model()

        # Flatten data to feed into random forest
        X = X.reshape(X.shape[0], -1)

        # Change to class numbers if classifying
        if args.label == 'signal':
            y = np.argmax(y, axis=1)

        # Train!
        model = model.fit(X, y)
        
        # Save the model
        tag = './models/{}/{}_{}_close-{}'.format(
            VERSION, args.model, args.time_interval, args.label
        )

        joblib.dump(model, f'{tag}_model.pkl')
