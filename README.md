# Financial-Prediction-Pipeline

This project is an end-to-end algorithmic trading system that uses machine learning to predict daily stock price movements. It includes a complete data pipeline for fetching, processing, and splitting financial data, and a training pipeline to build, tune, and evaluate a predictive model.

## Project Structure

```
Financial-Prediction-Pipeline/
├── data/
│   ├── Processed/
│   │   ├── processed_data.csv
│   │   ├── scaler.joblib
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── model.joblib
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

Clone the repository and install the required packages. The `requirements.txt` file includes a pinned version of `xgboost` to ensure reproducibility.

```bash
git clone https://github.com/shawnl57/Financial-Prediction-Pipeline
cd Financial-Prediction-Pipeline
pip install -r requirements.txt
```

## Usage

The project is designed to be run as a sequence of steps.

### Step 1: Populate the Data

This step fetches raw financial data, processes it by adding technical indicators, and saves the result.

```bash
python3 src/data_pipeline/populate_data.py
```

### Step 2: Split the Data

This step splits the processed data into training and testing sets for model training using a time-series-aware split.

```bash
python3 src/train/split_data.py
```

### Step 3: Train the Model

This step trains the XGBoost model. It automatically handles class imbalance and performs a hyperparameter search to find the best model, which is then saved to disk.

```bash
python3 src/train/train.py
```

## The Pipeline Explained

### Data Pipeline
1.  **`populate_data.py`**: The main script to orchestrate data fetching and processing.
2.  **`fetch_data.py`**: Uses `yfinance` to download data for the 3,900+ tickers listed in `nasdaq_tickers.txt`.
3.  **`process_data.py`**:
    -   Cleans the raw data.
    -   Engineers over 15 features by calculating technical indicators (RSI, SMA, EMA, MACD, etc.).
    -   Creates a binary target variable `y` (1 if the next day's close is higher, 0 otherwise).
    -   Scales the features using `StandardScaler`.
    -   Supports incremental updates to the processed data.
4.  **`split_data.py`**:
    -   Loads the processed data.
    -   Uses `sklearn.model_selection.TimeSeriesSplit` to ensure that the training data always comes before the testing data, which is crucial for financial time-series models.

### Model Training Pipeline
1.  **`train.py`**: This script orchestrates the entire modeling workflow:
    -   **Handles Class Imbalance**: It first calculates the ratio of positive to negative classes and uses the `scale_pos_weight` parameter in XGBoost to prevent the model from being biased towards the majority class.
    -   **Hyperparameter Tuning**: It uses `GridSearchCV` to systematically search for the best model hyperparameters, optimizing for **precision**.
    -   **Evaluation**: It evaluates the best model found during the grid search on the unseen test set, printing a full classification report and confusion matrix.
    -   **Saves the Model**: The final, best-performing model is saved to `data/Processed/model.joblib` for future use.

## Results

The hyperparameter search identified the following best parameters:
`{'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000, 'subsample': 0.8}`

The final model achieved the following performance on the test set:

**Classification Report:**
```
              precision    recall  f1-score   support

        Down       0.59      0.49      0.54    197406
          Up       0.48      0.57      0.52    157781

    accuracy                           0.53    355187
   macro avg       0.53      0.53      0.53    355187
weighted avg       0.54      0.53      0.53    355187
```

**Confusion Matrix:**
```
[[97424 99982]
 [67131 90650]]
```

While the model is not yet profitable (precision < 50%), the pipeline successfully balanced the class predictions and established a strong baseline for future work.

## Future Work

The current limitation is not the model, but the features. The next step to improve precision is to engineer more advanced features in `process_data.py`, such as:
-   **Lagged Features**: The value of indicators from previous days (e.g., `rsi_lag_1`, `macd_lag_1`).
-   **Rolling Statistics**: The mean, standard deviation, or other stats of indicators over a rolling window (e.g., `rsi_7d_mean`, `volume_30d_std`).
-   **Interaction Features**: Combinations of existing features that might capture more complex relationships.
