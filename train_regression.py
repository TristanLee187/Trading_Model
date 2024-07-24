# Train a model based on the market data

import numpy as np
import pandas as pd
from common import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping

# Initialize the training data/labels
X, y = [], []

# Init the standard scaler
std_scaler = StandardScaler()

# Read each ticker's data
for ticker in tickers:
    data = pd.read_csv(f'./market_data/{ticker}.csv')

    # Read sequences of length WINDOW_LENGTH
    for i in range(len(data) - WINDOW_LENGTH):
        sequence = data[i:i+WINDOW_LENGTH].drop(columns=ignore_cols)

        # Normalize
        sequence_norm = normalize(sequence, std_scaler)

        # Use the percentage change from the sequence's last day to the next day as the label.
        percent_change = percent_change_label(data, i+WINDOW_LENGTH, 'Close')

        # Add to the training data
        X.append(sequence_norm)
        y.append(percent_change)
    
    print(f'{ticker} done!')

X = np.array(X)
y = np.array(y)

# Prepare validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
NUM_FEATURES = X.shape[2]
model = Sequential([
    Input(shape=(WINDOW_LENGTH, NUM_FEATURES)),
    LSTM(units=NUM_FEATURES**2, return_sequences=True),
    LSTM(units=50),
    Dense(units=1)
])

# Compile with early stopping
model.compile(optimizer='adam', loss='mae')
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train!
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save('./models/regression_model.keras')