import unittest
import os
import shutil
import pandas as pd

from data_loader import load_data
from process_data import process_data

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for testing."""
        self.temp_dir = "temp_test_data"
        self.raw_data_dir = os.path.join(self.temp_dir, "raw")
        self.processed_data_dir = os.path.join(self.temp_dir, "processed")
        self.tickers_file = os.path.join(self.temp_dir, "tickers.txt")
        self.raw_data_path = os.path.join(self.raw_data_dir, "raw_data.csv")
        self.processed_data_path = os.path.join(self.processed_data_dir, "processed_data.csv")

        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

        with open(self.tickers_file, "w") as f:
            f.write("AAPL\n")
            f.write("GOOG\n")

    def tearDown(self):
        """Clean up the temporary environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_pipeline(self):
        """Test the full data loading and processing pipeline."""
        # Step 1: Load data
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        loaded_df = load_data(self.tickers_file, start_date, end_date, self.raw_data_path)
        
        self.assertIsNotNone(loaded_df, "load_data should return a DataFrame.")
        self.assertTrue(os.path.exists(self.raw_data_path), "Raw data file should be created.")
        
        raw_df = pd.read_csv(self.raw_data_path)
        self.assertFalse(raw_df.empty, "Raw data file should not be empty.")
        self.assertIn("AAPL", raw_df["Ticker"].unique())
        self.assertIn("GOOG", raw_df["Ticker"].unique())

        # Step 2: Process data
        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        
        self.assertIsNotNone(processed_df, "process_data should return a DataFrame.")
        self.assertTrue(os.path.exists(self.processed_data_path), "Processed data file should be created.")
        
        # Check if data is normalized (values between 0 and 1)
        self.assertTrue((processed_df["Open"] >= 0).all() and (processed_df["Open"] <= 1).all())
        self.assertTrue((processed_df["High"] >= 0).all() and (processed_df["High"] <= 1).all())
        self.assertTrue((processed_df["Low"] >= 0).all() and (processed_df["Low"] <= 1).all())
        self.assertTrue((processed_df["Close"] >= 0).all() and (processed_df["Close"] <= 1).all())

if __name__ == '__main__':
    unittest.main() 