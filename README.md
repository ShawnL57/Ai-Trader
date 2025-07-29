# AI Trader

This project is an algorithmic trading system that uses machine learning to predict stock price movements. It includes a data pipeline for fetching, processing, and splitting financial data, and a training script to build a predictive model.

## Project Structure

```
Ai-Trader/
├── data/
│   ├── Processed/
│   │   ├── processed_data.csv
│   │   ├── scaler.joblib
│   │   ├── train_data.csv
│   │   └── test_data.csv
│   └── Raw/
│       └── raw_data.csv
├── src/
│   ├── data_pipeline/
│   │   ├── fetch_data.py
│   │   ├── populate_data.py
│   │   ├── process_data.py
│   │   └── nasdaq_tickers.txt
│   └── train/
│       ├── split_data.py
│       └── train.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Prerequisites

- Python 3.8 or higher
- Pip

### 2. Installation

Clone the repository and install the required packages:

```bash
git clone <https://github.com/shawnl57/ai-trader>
cd Ai-Trader
pip install -r requirements.txt
```

## Usage

The project is designed to be run as a sequence of steps.

### Step 1: Populate the Data

This step fetches raw financial data, processes it by adding technical indicators, and saves the result.

```bash
python3 src/data_pipeline/populate_data.py
```

This script will:
1.  Read the ticker symbols from `src/data_pipeline/nasdaq_tickers.txt`.
2.  Download historical data for each ticker into `data/Raw/raw_data.csv`.
3.  Process the raw data, calculate features, and save the result to `data/Processed/processed_data.csv`.
4.  Save the feature scaler to `data/Processed/scaler.joblib`.

### Step 2: Split the Data

This step splits the processed data into training and testing sets for model training.

```bash
python3 src/train/split_data.py
```

This will create `train_data.csv` and `test_data.csv` in the `data/Processed/` directory, using a time-series-aware split (75% for training, 25% for testing).

### Step 3: Train the Model

Once the data is split, you can run the training script:

```bash
python3 src/train/train.py
```

This will load the training data, train a model, and (eventually) save the trained model for future use. (Note: `train.py` is currently a placeholder).

## Data Pipeline Explained

1.  **`populate_data.py`**: The main script to orchestrate data fetching and processing.
2.  **`fetch_data.py`**: Uses `yfinance` to download data for a list of tickers.
3.  **`process_data.py`**:
    -   Cleans the raw data.
    -   Engineers features by calculating technical indicators (RSI, SMA, EMA, MACD, etc.).
    -   Creates a binary target variable `y` (1 if the next day's close is higher, 0 otherwise).
    -   Scales the features using `StandardScaler`.
    -   Supports incremental updates to the processed data.
4.  **`split_data.py`**:
    -   Loads the processed data.
    -   Uses `sklearn.model_selection.TimeSeriesSplit` to ensure that the training data always comes before the testing data, which is crucial for financial time-series models. 
