from data.Data_Fetcher.fetch_data import download_data
from src.data_processor import process_data
import os

def main():
    """
    Main function to run the data fetching and processing pipeline.
    """
    # Parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    # Define paths
    raw_data_dir = "data/Raw"
    processed_data_dir = "data/Processed"
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    raw_data_path = os.path.join(raw_data_dir, f"{ticker}_raw.csv")
    processed_data_path = os.path.join(processed_data_dir, f"{ticker}_processed.csv")

    # --- Step 1: Download Data ---
    print(f"--- Starting data download for {ticker} ---")
    raw_df = download_data(ticker, start_date, end_date, raw_data_path)

    # --- Step 2: Process Data ---
    if raw_df is not None:
        print(f"\n--- Starting data processing for {ticker} ---")
        
        processed_df = process_data(raw_df)

        if processed_df is not None:
            # Save the processed data
            processed_df.to_csv(processed_data_path)
            
            print("\n--- Pipeline Complete ---")
            print("Raw data is located at:", raw_data_path)
            print("Processed data is located at:", processed_data_path)
            print("\nHead of processed data:")
            print(processed_df.head())
    else:
        print("Halting pipeline because data download failed.")

if __name__ == "__main__":
    main() 