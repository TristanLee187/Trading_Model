# Trading Model
## About
This repository contains a Python and TensorFlow/Keras machine learning pipeline for predicting stock price movements and buy/sell signals. The pipeline can be broken into 3 parts:
- Fetch market data from Yahoo Finance, with my own added indicators like SMAs, EMAs, crosses, and the stochastic oscillator.
- Train a time series model on this data, creating fixed length input sequences as input instances and computing appropriate values or classes as ground truth labels, depending on the task.
- Backtest the trained model on unseen data, producing appropriate plots and/or executing a simple trading strategy to produce a profit/loss.

For modularity, each of these parts has a dedicated source file that can be run from the command line and produces some output used later in the pipeline.

## Usage
Typical usage of the pipeline will go as follows:
- In the `common.py` file, set hyperparameters such as the length of the sequence and the list of tickers to use when building the dataset and training.
- Run `build_data_set.py` to export a single CSV file with market data for all tickers, using either day-to-day or minute-to-minute data. This will create the appropriate file at `./daily_market_data/` or `./minute_market_data/` (not in this repository for space's sake). For daily data, the years 2000-2023 are used, and for minute-to-minute data, the latter 3 or so weeks of July 2024 are used (see the source code for exact details).
- Run `train.py` to train a model depending on, among other things, the desired architecture and labels. I chose to include LSTM and Transformer based architectures since they focus on sequence data, appropriate for time series. The generated model will be saved to `./models/VERSION/`, where `VERSION` is set in the `common.py` file. None of the models are included in this repository, again for space's sake.
- Run `evaluate.py` to evaluate the trained model on unseen data. For daily data, this is 2024 data up to and including July, and for minute data, this is August 5th, 2024. Produce plots for specific tickers to evaluate with the human eye, or for buy/sell signals specifically, also simulate a simple strategy using the signals across all tickers to evaluate the average profit/loss. These outputs will be saved to `./plots/VERSON/`. I've included most of my own output across different versions.

Each of these files use `argparse` to be run from the command line with arguments, so you can use `-h` to see exaclty how to run each file. For example:

```console
$ python train.py -h
usage: train.py [-h] -m {LSTM,transformer} -t {1m,1d} -l {price,price-change,signal} -e ERROR

Train a Model

optional arguments:
  -h, --help            show this help message and exit
  -m {LSTM,transformer}, --model {LSTM,transformer}
                        model architecture to use
  -t {1m,1d}, --time_interval {1m,1d}
                        time interval data to train on
  -l {price,price-change,signal}, --label {price,price-change,signal}
                        labels to use for each instance
  -e ERROR, --error ERROR
                        error (loss) function to use (ignored if classification)
```
Note that though the output says the arguments are optional, they are actually required.

## My Results
My most successful regression model (predicting the next day's price or price change) mostly predicted a price very close to the previous day's price, or a change very close to 0, respectively. `./plots/v3` has some example plots for both daily and minute data of ground truth versus predicted prices. Visually, this type of prediction manifects as a small gap between the ground truth and predictions, with the predictions chasing the previous day's price (in general, anyway). Outside of this, the model appears to give importance to a "momentum" of the previous days' prices; for instance, if price increased for 3 consecutive days, the model "adds" some positive amount to its prediction.

My most successful classification model (predicting buy, sell, or do nothing signals) had mixed performance across different tickers. `./plots/final` has example plots of some successful (AAPL and NVDA) and unsuccessful (INTC) predictions. I used a simple strategy that soley depends on the predicted signals and consists of just buying/selling the stock (not options): buy a stock whenever there's a buy signal, or sell all stock whenever there's a sell signal, and do nothing otherwise. Using this strategy and the model produced a 3.33% profit averaged across all tickers, in my case the S&P 100 companies.