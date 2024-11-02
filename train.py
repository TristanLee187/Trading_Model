# Train a model based on the market data

import numpy as np
import pandas as pd
from common import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import Model
from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Dense, Input, MultiHeadAttention, TimeDistributed, Add, LayerNormalization, Flatten, Layer
from keras.api.regularizers import L1L2
from keras.api.optimizers import RMSprop
from keras.api.utils import custom_object_scope
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.metrics import F1Score
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
        [0.0, 2.0, 10.0],
        [3.0, 0.0, 3.0],
        [10.0, 2.0, 0.0],
    ])

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
            - "signal" (classification): A Dense layer with 5 units and softmax activation.
    """
    if label == 'price':
        return Dense(units=1)
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


# Single expert
class Expert(Layer):
    def __init__(self, units1, units2):
        super(Expert, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):
        self.dense1 = Dense(self.units1, activation='gelu')
        self.dense2 = Dense(self.units2, activation='gelu')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    

# Mixture of experts top k layer
class MoETopKLayer(Layer):
    def __init__(self, num_experts, expert_units_1, expert_units_2, top_k, **kwargs):
        super(MoETopKLayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_units_1 = expert_units_1
        self.expert_units_2 = expert_units_2
        self.top_k = top_k

        self.experts = None
        self.gating_network = None

    def build(self, input_shape):
        self.experts = [TimeDistributed(Expert(self.expert_units_1, self.expert_units_2)) for _ in range(self.num_experts)]
        self.gating_network = TimeDistributed(Dense(self.num_experts, activation='softmax'))

    def call(self, inputs):
        # Get probs for each expert
        gate_outputs = self.gating_network(inputs)
        # Get indices of top k experts
        top_k_values, top_k_indices = tf.nn.top_k(gate_outputs, k=self.top_k)
        # Mask bottom experts; normalize
        mask = tf.reduce_sum(tf.one_hot(top_k_indices, depth=self.num_experts), axis=-2)
        gated_outputs = gate_outputs * mask
        gated_outputs /= tf.reduce_sum(gated_outputs, axis=-1, keepdims=True) + tf.constant(1e-9)
        # Weighted avg of top k
        expert_outputs = tf.stack([expert(inputs) for expert in self.experts], axis=-1)
        weighted_expert_outputs = tf.reduce_sum(expert_outputs * tf.expand_dims(gated_outputs, axis=2), axis=-1)

        return weighted_expert_outputs
    

def get_transformer_model(shape: tuple, label: str):
    """
    Define a transformer and LSTM based architecture.

    Args:
        shape (tuple[int, int]): shape of each input instance.
        label (str):  The label type to train on.

    Returns:
        keras.Model: Model with a transformer-LSTM architecture.
    """
    # Transformer block with LSTM position encoding and MoE
    def transformer_block(x, num_heads, key_dim, ff_dim_1, ff_dim_2):
        x = Add()([x, LSTM(units=x.shape[2], return_sequences=True)(x)])
        attn_layer = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, 
            dropout=0.2, kernal_regularizer=L1L2(1e-4, 1e-4))(x, x)
        x = Add()([x, attn_layer])
        x = LayerNormalization(epsilon=1e-8)(x)
        moe = MoETopKLayer(num_experts=5, expert_units_1=ff_dim_1, expert_units_2=ff_dim_2, top_k=2)(x)
        x = Add()([x, moe])
        x = LayerNormalization(epsilon=1e-8)(x)
        return x

    # Stack of Transformer blocks
    def transformer_stack(x, num_heads, key_dim, ff_dim_1, ff_dim_2, num_blocks):
        for _ in range(num_blocks):
            x = transformer_block(x, num_heads=num_heads, key_dim=key_dim, 
                                  ff_dim_1=ff_dim_1, ff_dim_2=ff_dim_2)
        return x

    # Define the Transformer model
    input_layer = Input(shape=shape)
    # Encoder
    encoder = transformer_stack(
        input_layer, num_heads=4, key_dim=8, ff_dim_1=shape[1], ff_dim_2=shape[1], num_blocks=2)
    encoder = Dense(8, activation='gelu')(encoder)
    # Decoder
    decoder = Dense(shape[1], activation='gelu')(encoder)
    decoder = transformer_stack(
        decoder, num_heads=4, key_dim=8, ff_dim_1=shape[1], ff_dim_2=shape[1], num_blocks=2)
    # Pool
    pooling_layer = Flatten()(decoder)
    # Output
    dense_layer_1 = Dense(units=256, activation='gelu')(pooling_layer)
    dense_layer_2 = Dense(units=64, activation='gelu')(dense_layer_1)
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
    parser.add_argument('-b', '--batch_size', type=int,
                        help='batch size (defaults to 32)')
    parser.add_argument('-s', '--learning_rate', type=float,
                        help='learning rate (defaults to 0.001)')
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
                X, y, test_size=0.2, random_state=42)
        # Get appropriate NN architecture
        if args.resume is not None:
            with custom_object_scope({'Expert': Expert, 'MoETopKLayer': MoETopKLayer}):
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
                optimizer=RMSprop(learning_rate=(args.learning_rate if args.learning_rate is not None else 0.001)),
                loss=custom_categorical_crossentropy, 
                metrics=[F1Score()])

        # Train!
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr = 1e-7)
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=(args.batch_size if args.batch_size is not None else 32),
                  validation_data=(X_val, y_val), 
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
