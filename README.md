# Trading Model
## About
This repository contains a Python and TensorFlow/Keras/Scikit-learn machine learning pipeline for predicting stock price movements and buy/sell signals. The pipeline can be broken into 3 parts:
- Fetch market data from Yahoo Finance, with my own added indicators like SMAs, EMAs, crosses, and the stochastic oscillator, as well as fundamental data from Yahoo Finance and/or Alpha Vantage like earnings, PE, and the ticker's sector.
- Train a time series model on this data, creating fixed length input sequences as input instances and computing appropriate values or classes as ground truth labels, depending on the task.
- Backtest the trained model on unseen data, producing appropriate plots and/or executing a simple trading strategy to produce a profit/loss.

For modularity, each of these parts has a dedicated source file that can be run from the command line and produces some output used later in the pipeline.

## Usage
Typical usage of the pipeline will go as follows:
- In the `common.py` file, set hyperparameters such as the length of the sequence and the list of tickers to use when building the dataset and training.
- Run `build_data_set.py` to export a single CSV file with daily market data for all tickers. This will create the appropriate file at `./daily_market_data/` (not in this repository for space's sake). The years 2000-2023 are used.
- Run `train.py` to train a model depending on, among other things, the desired architecture and labels. I chose to use a transformer and LSTM based architecture since they focus on sequence data, appropriate for time series, along with mixture of experts in the transformer blocks. The generated model will be saved to `./models/VERSION/`, where `VERSION` is set in the `common.py` file. None of the models are included in this repository, again for space's sake.
- Run `evaluate.py` to evaluate the trained model on unseen data. This is all 2024 data up to and including December. Produce plots for specific tickers to evaluate with the human eye, or for buy/sell signals specifically, also simulate a simple strategy using the signals across all tickers to evaluate the average profit/loss. These outputs will be saved to `./plots/VERSON/`. I've included most of my own output across different versions.

Each of these files use `argparse` to be run from the command line with arguments, so you can use `-h` to see exaclty how to run each file. For example:

```console
$ python train.py -h
usage: train.py [-h] -m {transformer} -t {1m,1d} [-d TRAIN_DATA] -l {price,signal} [-e ERROR] [-r RESUME] [-b BATCH_SIZE] [-s LEARNING_RATE] [-p EPOCHS]

Train a Model

optional arguments:
  -h, --help            show this help message and exit
  -m {transformer}, --model {transformer}
                        model type/architecture to use
  -d TRAIN_DATA, --train_data TRAIN_DATA
                        if set, path to file containing X and y sequence data
  -l {price,signal}, --label {price,signal}
                        labels to use for each instance
  -e ERROR, --error ERROR
                        error (loss) function to use (required for regression, ignored if classification)
  -r RESUME, --resume RESUME
                        if set, path to a model to resume training on (only works for NNs)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (defaults to 32)
  -s LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (defaults to 0.001)
  -p EPOCHS, --epochs EPOCHS
                        number of training epochs for the NN models (defaults to 20)
```

## My Results
My most successful regression model (predicting the next day's price) mostly predicted a price pretty close to the previous day's price. Visually, this type of prediction manifects as a small gap between the ground truth and predictions, with the predictions chasing the previous day's price (in general, anyway). Outside of this, the model appears to give importance to a "momentum" of the previous days' prices; for instance, if price increased for 3 consecutive days, the model "adds" some positive amount to its prediction. `./plots/final/prices` includes additional plots.

My most successful classification models (predicting buy, sell, or do nothing signals) had mixed performance across different tickers. `./plots/final/signals` has example plots of some successful (AAPL, NVDA, TSLA) and unsuccessful (INTC) predictions. I used a simple strategy that soley depends on the predicted signals and consists of just buying/selling the stock (not options): buy a stock whenever there's a buy signal, or sell all stock whenever there's a sell signal, and do nothing otherwise. Using this strategy produced a 12.92% when averaged across all tickers (in my case the S&P 100 companies). The model seems to predict no buy actions for many companies, with most of its profit coming from choosing a relatively small number of tickers. It also very seldom predicts sell signals.