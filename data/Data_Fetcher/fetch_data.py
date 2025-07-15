import yfinance as yf
import os
import pandas as pd


def download_data(ticker : str, start_date: str, end_date:str, save_path:str):
    """
    downloads historical Open, High, Low, Close, and Volume data for a given ticker between start and end dates
    downloaded from yfinace 
    save the data to a csv file: save_path
    if the file already exists, skip the download
    if data is not available, skip the download or raise an error
    if the data is available, save it to the csv file
    return: Cleaned DataFrame with columns: Date, Open, High, Low, Close, Volume
    """



    if os.path.exists(save_path): #check if the file already exists
        print(f"File already exists at {save_path}. Skipping download.")
        return None
    try:
        data = yf.download(ticker,start = start_date, end = end_date)
        if data.empty: #check if there is no data available 
            print(f"No data found for ticker {ticker} from {start_date} to {end_date}.")
            return None
        data.to_csv(save_path) #chagne data into a csv file 
        print(f"Data for {ticker} successfully saved to {save_path}")
        return data
    except Exception as e: #if there is an error: print the error message
        print(f"An error occured while downloading data for {ticker}: {e}")
        return None


    # data.to_csv(save_path) #change data into a csv file 
    # print(f"Data for {ticker} successfully saved to {save_path}")
    # return data #
    # except Exception as e:
    #     print(f"An error occurred while downloading data for {ticker}: {e}")
    #     return None


