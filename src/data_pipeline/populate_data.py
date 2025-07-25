import pandas as pd
import os
from .data_loader import load_data
from .process_data import process_data


def populate_data(tickers_file_path: str, raw_data_path: str, processed_data_path: str, start_date: str, end_date: str):
    """
    Centralized caller to fetch and process data.
    1. Loads raw data from Yahoo Finance using a list of tickers.
    2. Processes the raw data and saves it.
    """
    # Step 1: Download data using the data_loader
    print("--- Starting Data Population ---")
    print(f"Loading raw data based on tickers from {tickers_file_path}...")
    load_data(
        file_path_tickers=tickers_file_path,
        start_date=start_date,
        end_date=end_date,
        save_path=raw_data_path
    )
    
    print("\nRaw data download complete.")

    # Step 2: Process the downloaded raw data
    print(f"\nProcessing raw data from {raw_data_path}...")
    processed_df = process_data(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path
    )

    if processed_df is not None:
        print(f"\nProcessing complete. Processed data saved to {processed_data_path}")
        print("--- Data Population Finished ---")
        return processed_df
    else:
        print("\nProcessing failed.")
        print("--- Data Population Finished with Errors ---")
        return None

def call_populate_data(tickers_file_path: str, raw_data_path: str, processed_data_path: str, start_date: str, end_date: str):
    """Wrapper function for populate_data for easier calling."""
    return populate_data(tickers_file_path, raw_data_path, processed_data_path, start_date, end_date)
if __name__ == "__main__":
    # Define paths and parameters for the script
    TICKERS_FILE = 'src/data_pipeline/nasdaq_tickers.txt'
    RAW_DATA_PATH = 'data/Raw/raw_data.csv'
    PROCESSED_DATA_PATH = 'data/Processed/processed_data.csv'
    START_DATE = '2020-01-01'
    END_DATE = '2023-01-01'

    # Ensure the output directories exist
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    call_populate_data(
        tickers_file_path=TICKERS_FILE,
        raw_data_path=RAW_DATA_PATH,
        processed_data_path=PROCESSED_DATA_PATH,
        start_date=START_DATE,
        end_date=END_DATE
    )
