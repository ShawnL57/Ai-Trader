import pandas as pd
import os
import numpy as np
from fetch_data import download_data
from process_data import process_data
from data_loader import load_data


def populate_data(populate_path:str, tickers_file_path:str, start_date:str, end_date:str):

    """
    centralized caller for all data functions to populate ../Data
    """
    
    #no checks are needed in this function, all checks are done in other functions

    #download data
    downloaded_data = download_data(populate_path, tickers_file_path, start_date, end_date)

    #process data
    processed_data = process_data(populate_path, downloaded_data)

    return processed_data

def call_populate_data(populate_path:str, tickers_file_path:str, start_date: str, end_date:str): #function is just for readablility 
    return populate_data(populate_path, tickers_file_path, start_date, end_date)

