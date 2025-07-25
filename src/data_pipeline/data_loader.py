
import pandas as pd
import os
from typing import List, Optional
from fetch_data import download_data

def load_data(file_path_tickers:str, start_date:str, end_date:str, save_path:str):
    """
    loads stock data from yfinace and saves it to a csv file:
    if the file already exists, emptys the file and downloads the data again. 
    centralized caller for the download_data function. 

    file_path_tickers: str, the path to the txt file that contains the tickers to load
    start_date: str, the start date to load data from
    end_date: str, the end date to load data to
    save_path: str, the path to the csv file to save the data to

    returns:
    pd.DataFrame, the loaded data
    """

    
    try: 
        with open(file_path_tickers, 'r') as file:  #open the file and read tickers
            tickers = []
            for line in file:
                ticker = line.strip() 
                if ticker:
                    tickers.append(ticker)
        if tickers == []:
            print(f"Warning: Ticker file is empty or dosen't have readable characters: {file_path_tickers}")
            return None
    except FileNotFoundError:
        print(f"Error: Ticker file not found at {file_path_tickers}")
        return None

    all_downloaded_data = [] #list to hold all downloaded data
    
    for ticker in tickers: #go through all tickers 
        print(f"--- Downloading data for {ticker} ---")
        downloaded_df = download_data(ticker, start_date, end_date, save_path) #just download data
        if downloaded_df is not None: 
            all_downloaded_data.append(downloaded_df) 

    if not all_downloaded_data:
        print("No data was downloaded for any tickers.")
        return None
 
    final_df = pd.concat(all_downloaded_data, ignore_index=True)
    
    print("\nBatch download complete.")
    print(f"All data saved in: {save_path}")
    return final_df





