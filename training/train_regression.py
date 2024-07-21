# Train a model based on the market data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout, Activation
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

# Initialize the training data/labels
X, y = [], []

# Use a sequence of the previous 30 days' data as input
WINDOW_LENGTH = 30

# Read each ticker's data
tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', '^GSPC', '^DJI', '^RUT', 'CL=F', 'GC=F']
for ticker in tickers:
    data = pd.read_csv(f'../data/market_data/{ticker}.csv')

    # Read sequences of length WINDOW_LENGTH
    for i in range(len(data) - WINDOW_LENGTH):
        sequence = data[i:i+WINDOW_LENGTH]

        # Use the percentage change from the sequence's last day to the next day as the label.
        last_close = sequence.iloc[WINDOW_LENGTH-1]['Close']
        this_close = data.iloc[i+WINDOW_LENGTH]['Close']
        percent_change = (this_close - last_close) / last_close

        # Add to the training data
        X.append(sequence.to_numpy())
        y.append(percent_change)

# Prepare and normalize the data
X = np.array(X)
y = np.array(y)

# Prepare validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
NUM_FEATURES = X.shape[2]
model = Sequential([
    Input(shape=(WINDOW_LENGTH, NUM_FEATURES)),
    LSTM(units=50, return_sequences=True),
    Dropout(rate=0.4, seed=42),
    LSTM(units=50),
    Dropout(rate=0.2, seed=42),
    Dense(units=25, activation=Activation('sigmoid'), activity_regularizer=l2(1e-5)),
    Dense(units=1)
])

# Compile with early stopping
model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train!
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save('regression_model.keras')