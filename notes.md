First working model (v1 and v2): regression
- LSTM, 1e-5 L2 regularization.
- For AAPL, NVDA, and PTON, the model predicts a daily return of around 0.024107% for all days!
- Can be interpreted as "just buy the stock, regardless of previous prices."
- Even after removing regularization and increasing LSTM units, this result stays the same: constant values very close to 0 (though sometimes positive or negative, depending only on the model training).
- After normalizing each sequence's data (standard normalization) and trying a different loss function (MAE), the predictions are more varied but still close to 0.
- Trying to use the next closing price as the label instead of the percent return, the result is similar: the model predicts the same value across the entire range.
- Finally, after forcing normalization, the model predicts prices somewhat well (not just staying at a constant, following the ups and downs). Need to investigate certain behavior of seeming offset by an amount.
- It seems for stocks with lower absolute prices, the models performs worse. This is probably because of all values of all stocks are normalized according to the same min-max.
- Even after using MAPE (mean absolute percent error) as the loss, the problem persits.
- For the percent error, the expected result is very close to 0%.

Second working model (v3): regression (just price), Transformers
- Transformer + LSTM pooling, and normalizing each input sequence separately.
- Now the behavior is much closer to expected (very sensitive to the previous day's prices).
- Even when trying a price change label, these predictions are very close to 0.

Third working model (v4): classification (buy, sell, or do nothing), Transformers
- Same architecture and normalizing as before, but with classification and softmax activation at the end.
- Make ground truth labels as buy/sell/do nothing using constrained linear regression: for a some day, compute the best fit line for the next WINDOW_LENGTH days, but constrained such that the line goes through the day's price. Label based on that line's slope.
- The model predicts mostly "do nothing" for the 1 day chart, but predicts more buys/sells for the 1 minute chart. Looks like it's very sensitive to the slope thresholds.
- Finish training on the S&P 100 companies daily data, results are mixed, the average return is 3.33% using a simple strategy following the signals.