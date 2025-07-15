import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def process_data(raw_data_path: str, processed_data_path: str):
    """
    Loads raw multi-ticker stock data, normalizes it on a per-ticker basis,
    and saves the processed data.

    This function performs the following steps:
    1. Reads the raw data, which may contain multiple tickers.
    2. Groups the data by the 'Ticker' column.
    3. For each ticker, it normalizes 'Open', 'High', 'Low', 'Close', 'Volume'.
    4. Combines the processed data back into a single DataFrame.
    5. Saves the processed data to a new CSV file.

    Args:
        raw_data_path (str): The file path of the raw CSV data.
        processed_data_path (str): The file path for the processed data.

    Returns:
        pd.DataFrame: A DataFrame of the processed data, or None on failure.
    """
    if os.path.exists(processed_data_path):
        print(f"Processed data already exists at {processed_data_path}. Skipping.")
        return None
    if not os.path.exists(raw_data_path):
        print(f"Raw data file not found at {raw_data_path}.")
        return None

    data = pd.read_csv(raw_data_path)
    
    if 'Ticker' not in data.columns:
        print("Error: 'Ticker' column not found in raw data. Cannot process.")
        return None

    all_processed_data = []
    
    for ticker, group in data.groupby('Ticker'):
        print(f"Processing data for {ticker}...")
        
        processed_group = group.copy()
        
        processed_group['Date'] = pd.to_datetime(processed_group['Date'])
        processed_group.set_index('Date', inplace=True)
        
        processed_group.fillna(method="ffill", inplace=True)
        processed_group.dropna(inplace=True)
        
        if processed_group.empty:
            print(f"Skipping {ticker} due to no data after cleaning.")
            continue
            
        scaler = MinMaxScaler()
        columns_to_scale = ["Open", "High", "Low", "Close", "Volume"]
        existing_columns = [col for col in columns_to_scale if col in processed_group.columns]
        
        if existing_columns:
            processed_group[existing_columns] = scaler.fit_transform(processed_group[existing_columns])
        
        all_processed_data.append(processed_group)

    if not all_processed_data:
        print("No data was processed.")
        return None

    final_df = pd.concat(all_processed_data)
    
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    final_df.to_csv(processed_data_path)
    
    print(f"Processed data for all tickers saved to {processed_data_path}")
    return final_df