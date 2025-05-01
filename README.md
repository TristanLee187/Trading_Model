# Trading Model
## About
This repository contains a Python and TensorFlow/Keras/machine learning pipeline for predicting stock price movements and buy/sell signals. The pipeline can be broken into 3 parts:
- Fetch market data from Yahoo Finance, with my own added indicators like SMAs, EMAs, crosses, and the stochastic oscillator, as well as fundamental data from Yahoo Finance and/or Alpha Vantage like earnings and the P/E ratio.
- Train a time series model on this data, creating fixed length input sequences as input instances and computing appropriate classes as ground truth labels.
- Backtest the trained model on unseen data, producing appropriate plots and/or executing a simple trading strategy to produce a profit/loss.

For modularity, each of these parts has a dedicated source file that can be run from the command line and produces some output used later in the pipeline.

## Usage
Typical usage of the pipeline should go as follows:
- In the `common.py` file, set hyperparameters such as the length of the sequence and the list of tickers to use when building the dataset and training.
- Run `fundamentals.py` to update a single CSV file with fundamental data for all tickers. This will update the appropriate file at `./daily_market_data/`.
- Run `build_data_set.py` to export CSV files with daily market data for all tickers. This will create the appropriate files at `./daily_market_data/` (not in this repository for space's sake). The years 2000-2023 are used for training and 2024-2025 for testing.
- Run `train.py` to train a model depending on. I chose to use a transformer and LSTM based architecture since they focus on sequence data, appropriate for time series, along with mixture of experts in the transformer blocks. The generated model will be saved to `./models/VERSION/`, where `VERSION` is set in the `common.py` file. My final trained models can be found `deploy_dir`.
- Run `evaluate.py` to evaluate the trained model on the test data produced by `build_data_set.py`. Produce plots for specific tickers to evaluate with the human eye, and simulate a simple strategy using the signals across all tickers to evaluate the average profit/loss. These outputs will be saved to `./plots/VERSON/`. I've included my own output across the preliminary and final versions.

You can also use `baseline.py` to execute various simple baseline strategies.

Each of these files use `argparse` to be run from the command line with arguments, so you can use `-h` to see exaclty how to run each file. For example:

```console
$ python train.py -h
usage: train.py [-h] [-d TRAIN_DATA] [-e ERROR] [-r RESUME] [-b BATCH_SIZE] [-s LEARNING_RATE] [-p EPOCHS] [-t TAG]

Train a Model

optional arguments:
  -h, --help            show this help message and exit
  -d TRAIN_DATA, --train_data TRAIN_DATA
                        if set, path to file containing X and y sequence data
  -e ERROR, --error ERROR
                        error (loss) function to use (required for regression, ignored if classification)
  -r RESUME, --resume RESUME
                        if set, path to a model to resume training on (only works for NNs)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (defaults to 64)
  -s LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (defaults to 0.001)
  -p EPOCHS, --epochs EPOCHS
                        number of training epochs for the NN models (defaults to 20)
  -t TAG, --tag TAG     if set, tag to save the model file as (do not include "models/" path or "_model.keras" suffix)
```

## My Results
Check out `report.pdf`.

## Other notes
- References to a label argument that can be `"price"` and `"signal"` refer to a partially removed functionality for performing different tasks (regression for prices or classification for buy/sell signals). Only `"signal"` should be used.