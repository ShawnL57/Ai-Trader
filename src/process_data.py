import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

def process_data(raw_data_path: str, processed_data_path: str):
    """

    Loads raw stock data, normalizes it, and saves the processed data to a csv file. 
    if tikcer and date already exists in processed data, skip processing for that ticker and date. 
    if the file already exists, skip processing. 
    if the ticker column is not found in the data, return None. 
    if the raw data file is not found, return None. 


        raw_data_path: str, path to csv file that contains the raw data  
        processed_data_path: str, to path to csv that contains processed data   

    Function: 
        -Reads the raw data from teh csv file : raw_data_path
        -instead of setting the date column as the index, we set the ticker column as the index
        -handles any missing values using forward-fill
        -normalizes [open,high,low,close,volume] using MinMaxScaler: 
        -saves the process data to a csv file: processed_data_path
    """

    if not os.path.exists(raw_data_path):
        print(f"Raw data file not found at {raw_data_path}")
        return None
    if os.path.exists(processed_data_path):
        print(f"Processed data already exists at {processed_data_path}")
        return None

    data = pd.read_csv(raw_data_path) #read raw data from csv 
    all_processed_data = []
    if 'Ticker' not in data.columns:
        print("No ticker column found in data")
        return None
    #group by ticker, then process each group
    for ticker, group in data.groupby('Ticker'):

        processed_group = group.copy() #create a copy to avoid SettingWithCopyWarning

        # Set the Date as the index for time-series operations
        processed_group['Date'] = pd.to_datetime(processed_group['Date'])
        processed_group.set_index('Date', inplace=True)

        # --- Feature Engineering ---
        # Create features that might introduce NaNs
        processed_group['SMA_20'] = processed_group['Close'].rolling(window=20).mean()
        processed_group['Lag_Return_1'] = processed_group['Close'].pct_change(1).shift(1)
        processed_group['Log_Return'] = np.log(processed_group['Close'] / processed_group['Close'].shift(1))

        # --- Data Cleaning ---
        # Forward-fill to handle intermittent NaNs, then drop any remaining NaNs (like at the start)
        processed_group.ffill(inplace=True)
        processed_group.dropna(inplace=True)
        
        #skip if group is empty after cleaning:
        if processed_group.empty:
            print(f"Skipping {ticker} due to no data after cleaning.")
            continue

        # --- Target Variable ---
        # y=1 if next day's return is positive, else 0. Drop last row where y is NaN.
        processed_group['y'] = (processed_group['Close'].shift(-1) > processed_group['Close']).astype(int)
        processed_group.dropna(inplace=True)

        # --- Normalization ---
        scaler = MinMaxScaler()
        columns_to_normalize = ["Open", "High", "Low", "Close","Adj Close","Volume", "SMA_20", "Lag_Return_1", "Log_Return"]

        # Remove columns that don't exist in the group
        # (This is good practice in case some tickers lack certain data)
        valid_columns_to_normalize = [col for col in columns_to_normalize if col in processed_group.columns]
        
        # Now only normalize the columns that exist in the group
        if not valid_columns_to_normalize:
            print(f"No valid columns to normalize for {ticker}")
            continue
        
        processed_group[valid_columns_to_normalize] = scaler.fit_transform(processed_group[valid_columns_to_normalize])
        all_processed_data.append(processed_group)

    if not all_processed_data:
        print("No data was processed.")
        return None

    final_df = pd.concat(all_processed_data)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    final_df.to_csv(processed_data_path)
    print(f"Processed data saved to {processed_data_path}")
    return final_df