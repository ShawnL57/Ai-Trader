import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

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

        #front fill missing values, drop any remaining missing values
        processed_group.ffill(inplace=True)
        processed_group.dropna(inplace=True)
        
        #skip if group is empty after cleaning:
        if processed_group.empty:
            print(f"Skipping {ticker} due to no data after cleaning.")
            continue
        #normalize columns:
        scaler = MinMaxScaler()
        columns_to_normalize = ["Open", "High", "Low", "Close","Adj Close","Volume"]

        #remove columns that don't exist in the group
        i = 0
        while i < len(columns_to_normalize):
            if columns_to_normalize[i] not in processed_group.columns:
                columns_to_normalize.pop(i)
            else:
                i+=1
        
        #now only normalize the columns that exist in the group
        if not columns_to_normalize:
            print(f"No valid columns to normalize for {ticker}")
            continue
        processed_group[columns_to_normalize] = scaler.fit_transform(processed_group[columns_to_normalize])
        all_processed_data.append(processed_group)

    final_df = pd.concat(all_processed_data)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    final_df.to_csv(processed_data_path)
    print(f"Processed data saved to {processed_data_path}")
    return final_df