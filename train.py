# Train a model based on the market data

import numpy as np
import pandas as pd
from common import *
from model import custom_categorical_crossentropy, get_transformer_model, CUSTOM_OBJECTS
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.api.models import load_model
from keras.api.optimizers import RMSprop
from keras.api.utils import custom_object_scope
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.metrics import F1Score
import argparse


SEED = 42
tf.keras.config.enable_unsafe_deserialization()


def prepare_training_data(label: str):
    """
    Prepare training data (inputs and ground truth labels).

    Args:
        label (str): String indicating what value to use as the labels:
            "price": Use the price of the given column.
            "signal": Use regression to indicate upward/downward/neither movement in following time points.

    Returns:
        numpy.array, numpy.array: Two numpy arrays X and y containing the training instances and ground
            truth labels, respectively.
    """
    # Init training instances, metadata, and labels
    X, x_meta, y = [], [], []

    # Read the master list
    tickers_df = pd.read_csv(f'./daily_market_data/all_tickers_train.csv')

    tickers_df_grouped = tickers_df.groupby(by=['Ticker'])

    # Generate data for each ticker
    for ticker in tickers:
        data = tickers_df_grouped.get_group((ticker,))

        # Just use the ticker's whole file data as a contiguous training set
        ticker_X, ticker_x_meta, ticker_y, mins, scales = prepare_model_data(data, label, 'Close')

        X.append(ticker_X)
        x_meta.append(ticker_x_meta)
        y.append(ticker_y)

        print(f'{ticker} is done')

    X = np.concatenate(X)
    x_meta = np.concatenate(x_meta)
    y = np.concatenate(y)

    return X, x_meta, y


if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(
        description="Train a Model"
    )
    parser.add_argument('-m', '--model', type=str, help='model type/architecture to use',
                        choices=['transformer'], required=True)
    parser.add_argument('-d', '--train_data', type=str, help='if set, path to file containing X and y sequence data')
    parser.add_argument('-l', '--label', type=str, help='labels to use for each instance',
                        choices=['price', 'signal'], required=True)
    parser.add_argument('-e', '--error', type=str,
                        help='error (loss) function to use (required for regression, ignored if classification)')
    parser.add_argument('-r', '--resume', type=str,
                        help='if set, path to a model to resume training on (only works for NNs)')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='batch size (defaults to 64)')
    parser.add_argument('-s', '--learning_rate', type=float,
                        help='learning rate (defaults to 0.001)')
    parser.add_argument('-p', '--epochs', type=int,
                        help='number of training epochs for the NN models (defaults to 20)')
    args = parser.parse_args()

    # Prepare training data
    if args.train_data:
        npzfile = np.load(args.train_data)
        X, x_meta, y = npzfile['X'], npzfile['x_meta'], npzfile['y']
    else:
        X, x_meta, y = prepare_training_data(args.label)
        np.savez(f'./models/{VERSION}/{args.label}_X_x_meta_and_y.npz', X=X, x_meta=x_meta, y=y)

    # Prepare validation data
    X_train, X_val, x_meta_train, x_meta_val, y_train, y_val = train_test_split(X, x_meta, y, test_size=0.2, random_state=SEED)
    
    # Get appropriate NN model
    if args.resume is not None:
        with custom_object_scope(CUSTOM_OBJECTS):
            model = load_model(args.resume, compile=False)
    else:
        model = get_transformer_model(X[0].shape, x_meta[0].shape, args.label)

    # Compile
    if args.label == 'price':
        model.compile(optimizer='adam', loss=args.error)
    elif args.label == 'signal':
        model.compile(
            optimizer=RMSprop(learning_rate=(args.learning_rate if args.learning_rate is not None else 0.001)),
            loss=custom_categorical_crossentropy, 
            metrics=[F1Score()])
    
    model.summary()
    
    # Train!
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr = 1e-7)
    model.fit([X_train, x_meta_train], y_train, epochs=(args.epochs if args.epochs is not None else 20), 
                batch_size=(args.batch_size if args.batch_size is not None else 64),
                validation_data=([X_val, x_meta_val], y_val), 
                callbacks=[lr_scheduler])
    
    # Save the model
    if args.label == 'price':
        loss_func_str = args.error
    elif args.label == 'signal':
        loss_func_str = 'cce'

    tag = './models/{}/{}_close-{}_{}'.format(
        VERSION, args.model, args.label, loss_func_str
    )

    model.save(f'{tag}_model.keras')
