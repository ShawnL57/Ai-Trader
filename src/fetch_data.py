import yfinance as yf
import os
import pandas as pd




def download_data(ticker : str, start_date: str, end_date:str, save_path:str):
    """
    Downloads historical stock data and appends it to a single CSV file.

    A "Ticker" column is added to distinguish data from different stocks. If the
    target file does not exist, it is created with a header. Otherwise, the new
    data is appended without a header.


    Returns:
        pd.DataFrame: A DataFrame of the newly downloaded data, or None on failure.
    """
    try:
        data = yf.download(ticker,start = start_date, end = end_date)
        if data.empty: #check if there is no data available 
            print(f"No data found for ticker {ticker} from {start_date} to {end_date}.")
            return None

        # add the ticker column to identify the stock
        data['Ticker'] = ticker
        
        # reset the index to make 'Date' a column
        data.reset_index(inplace=True)
        
        # check if file exists to determine if we should write headers
        output_dir = os.path.dirname(save_path) 
        if output_dir:
            os.makedirs(output_dir, exist_ok=True) #creates raw directory if it doesn't exist 
        
        file_exists = os.path.exists(save_path) 

        data.to_csv(save_path, mode='a', header= not file_exists, index=False) 
        
        if not file_exists: #if the file does not exist 
            print(f"Created file and saved data for {ticker} to {save_path}") 
        else: 
            print(f"Appended data for {ticker} to {save_path}")

        return data
    except Exception as e: #if there is an error: print the error message
        print(f"An error occured while downloading data for {ticker}: {e}")
        return None
    



