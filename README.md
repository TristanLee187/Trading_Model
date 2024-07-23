# Trading_Model

## Notes
First working model: regression
- LSTM, 1e-5 L2 regularization.
- For AAPL, NVDA, and PTON, the model predicts a daily return of around 0.024107% for all days!
- Can be interpreted as "just buy the stock, regardless of previous prices."
- Even after removing regularization and increasing LSTM units, this result stays the same: constant values very close to 0 (though sometimes positive or negative, depending only on the model training).

