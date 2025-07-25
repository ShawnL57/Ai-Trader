import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

def generate_sample_df(days=252):
    """
    Generates a sample DataFrame simulating financial time series data.
    
    This function creates a DataFrame with common financial columns and
    engineered features, including some with intentional issues (NaNs, leakage)
    to demonstrate the validation checks.

    Args:
        days (int): The number of trading days to simulate.

    Returns:
        pd.DataFrame: A sample DataFrame with a DatetimeIndex.
    """
    print(f"Generating a sample DataFrame with {days} days of data...")
    
    # 1. Create a date range
    dates = pd.to_datetime(pd.date_range('2023-01-01', periods=days, freq='B'))
    
    # 2. Simulate stock prices
    close_prices = 100 + np.random.randn(days).cumsum()
    high_prices = close_prices + np.random.uniform(0, 2, size=days)
    low_prices = close_prices - np.random.uniform(0, 2, size=days)
    open_prices = (close_prices + np.roll(close_prices, 1)) / 2
    open_prices[0] = close_prices[0] - np.random.uniform(0, 1)
    volume = np.random.randint(1_000_000, 5_000_000, size=days)

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Adj Close': close_prices,
        'Volume': volume
    }, index=dates)

    # 3. Engineer some features
    df['SMA_20'] = df['Close'].rolling(window=20).mean() # Will have NaNs at the start
    df['RSI_14'] = np.random.uniform(30, 70, size=days) # Dummy RSI
    df['Lag_Return_1'] = df['Close'].pct_change(1).shift(1) # Correctly lagged
    
    # Intentionally create a leaky feature for demonstration
    df['Leaky_SMA_5'] = df['Close'].rolling(window=5).mean()
    
    # Create log returns which might produce -inf if price hits 0 (unlikely here)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 4. Create the target variable
    # y=1 if next day's return is positive, else 0
    df['y'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Replace the last y value with a common practice (e.g., forward-fill or drop)
    # as it will be NaN. For this example, we'll see it in the NaN check.
    
    print("Sample DataFrame created.\n")
    return df

def validate_data_pipeline(df: pd.DataFrame, target_col: str = 'y'):
    """
    Runs a series of checks to validate a financial time series DataFrame.

    This function checks for NaNs, infinities, class balance, and ensures
    compatibility with TimeSeriesSplit, including a demonstration of correct
    scaler usage.

    Args:
        df (pd.DataFrame): The DataFrame to validate. Assumed to have a DatetimeIndex.
        target_col (str): The name of the target variable column.
    
    Returns:
        None: Prints a detailed report to the console.
    """
    print("--- Starting Data Pipeline Validation ---")
    issues_found = []
    
    # --- 1. Check for common data issues ---
    print("\n[Step 1/5] Checking for Missing and Infinite Values...")
    
    # Check for NaNs
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        issues_found.append(f"NaNs found in columns: {list(nan_cols.index)}.")
        print(f"-> Found {nan_cols.sum()} total NaN values. Columns affected:")
        print(nan_cols)
    else:
        print("-> No missing values (NaNs) found.")

    # Check for Infinite values
    inf_counts = df.isin([np.inf, -np.inf]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if not inf_cols.empty:
        issues_found.append(f"Infinite values found in columns: {list(inf_cols.index)}.")
        print(f"-> Found {inf_cols.sum()} total infinite values. Columns affected:")
        print(inf_cols)
    else:
        print("-> No infinite values found.")

    # Clean the data
    if not nan_cols.empty or not inf_cols.empty:
        print("\nCleaning data by replacing infinities with NaNs and dropping all rows with NaNs...")
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    else:
        df_clean = df.copy()
        print("\nData is clean. No NaNs or infinities to remove.")
    
    # --- 2. Check for Feature Leakage (Lookahead Bias) ---
    print("\n[Step 2/5] Checking for Feature Leakage...")
    # This is a conceptual check. A common mistake is using current day's data
    # in a feature to predict the same day's target. All features for a given
    # day should be computable using data available *before* that day.
    # Example: A simple moving average should use a shifted series.
    # Correct: df['Close'].shift(1).rolling(5).mean()
    # Incorrect (leaky): df['Close'].rolling(5).mean()
    if 'Leaky_SMA_5' in df_clean.columns:
        # We demonstrate the check by creating the correctly lagged version
        correct_sma = df_clean['Close'].shift(1).rolling(window=5).mean()
        if df_clean['Leaky_SMA_5'].equals(correct_sma):
             print("-> Leaky_SMA_5 check: Feature appears correctly lagged.")
        else:
             issues_found.append("Potential lookahead bias in 'Leaky_SMA_5'.")
             print("-> WARNING: 'Leaky_SMA_5' does not match a correctly lagged calculation.")
             print("   This suggests it might be using current-day data, causing leakage.")
    else:
        print("-> No example leaky feature found to check. Ensure your rolling features use .shift(1).")
        
    # --- 3. Check Class Balance ---
    print("\n[Step 3/5] Checking Class Balance...")
    if target_col in df_clean.columns:
        balance = df_clean[target_col].value_counts(normalize=True)
        print(f"-> Target variable '{target_col}' distribution:")
        print(balance)
        if abs(balance.get(0, 0) - balance.get(1, 0)) > 0.4:
            issues_found.append("Significant class imbalance detected.")
            print("-> WARNING: Significant class imbalance. Consider using stratisfied sampling or techniques like SMOTE.")
    else:
        issues_found.append(f"Target column '{target_col}' not in DataFrame.")
        print(f"-> ERROR: Target column '{target_col}' not found!")

    # --- 4. Validate TimeSeriesSplit Compatibility ---
    print("\n[Step 4/5] Validating TimeSeriesSplit...")
    features = [col for col in df_clean.columns if col != target_col and 'Date' not in col]
    X = df_clean[features]
    y = df_clean[target_col]

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    print(f"-> Created TimeSeriesSplit with {n_splits} splits.")
    
    overlap_found = False
    for i, (train_index, val_index) in enumerate(tscv.split(X)):
        train_start, train_end = df_clean.index[train_index[0]], df_clean.index[train_index[-1]]
        val_start, val_end = df_clean.index[val_index[0]], df_clean.index[val_index[-1]]
        
        print(f"  Fold {i+1}:")
        print(f"    Train: {len(train_index)} obs from {train_start.date()} to {train_end.date()}")
        print(f"    Valid: {len(val_index)} obs from {val_start.date()} to {val_end.date()}")
        
        # Critical check: The last training index must be before the first validation index
        if train_index[-1] >= val_index[0]:
            overlap_found = True
            print("    -> FATAL: Overlap detected! Validation data is leaking into training data.")

    if overlap_found:
        issues_found.append("TimeSeriesSplit validation failed: Train/validation sets overlap.")
    else:
        print("-> TimeSeriesSplit check passed. No overlap between folds.")
        
    # --- 5. Validate Scaling/Normalization ---
    print("\n[Step 5/5] Validating Scaler Fitting (Leakage Prevention)...")
    # To prevent leakage, the scaler must be fit ONLY on the training data for each fold.
    scaler = MinMaxScaler()
    
    # We'll check the last fold as an example
    train_index, val_index = list(tscv.split(X))[-1]
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    
    # Fit on train, transform both train and val
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("-> Scaler was fit on training data from the last fold.")
    
    min_val, max_val = X_val_scaled.min(), X_val_scaled.max()
    print(f"-> Validation set feature range after scaling: [{min_val:.4f}, {max_val:.4f}]")
    
    # It's okay if validation data goes outside [0, 1], but it's a good sanity check.
    if min_val < -0.1 or max_val > 1.1:
        issues_found.append("Scaled validation data is significantly outside the [0, 1] range.")
        print("-> WARNING: Scaled validation features fall significantly outside the [0,1] range. This is not an error but may indicate different data distributions between train/validation sets.")
    else:
        print("-> Scaler check passed. Validation data was transformed correctly without fitting.")


    # --- Final Report ---
    print("\n--- Validation Summary ---")
    print(f"Total rows and columns after cleaning: {df_clean.shape}")
    print(f"Remaining NaN count: {df_clean.isnull().sum().sum()}")
    print("Preview of the first 5 rows of the cleaned data:")
    print(df_clean.head())
    
    print("\n--- Final Verdict ---")
    if not issues_found:
        print("✅ Data is clean and safe for TimeSeriesSplit.")
    else:
        print("❌ Issues found. Please review the following:")
        for issue in issues_found:
            print(f"  - {issue}")
    print("-----------------------\n")


if __name__ == '__main__':
    # Generate a sample DataFrame to run the validation on.
    # In your workflow, you would load your actual 'df' here instead.
    # e.g., df = pd.read_csv('my_processed_data.csv', index_col='Date', parse_dates=True)
    sample_df = generate_sample_df()
    
    # Run the validation pipeline
    validate_data_pipeline(sample_df) 