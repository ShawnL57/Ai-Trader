import unittest
import os
import pandas as pd
from process_data import process_data
import shutil
import numpy as np

class TestProcessDataSafely(unittest.TestCase):

    def setUp(self):
        """
        This method runs BEFORE each test.
        It creates a dedicated, temporary directory structure for testing.
        """
        self.test_dir = "test_artefacts"  # Main test folder
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.processed_dir = os.path.join(self.test_dir, "processed")

        # Create the entire directory structure
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Define paths for test files
        self.raw_data_path = os.path.join(self.raw_dir, "test_raw_data.csv")
        self.processed_data_path = os.path.join(self.processed_dir, "test_processed_data.csv")
        
        # Create a sample raw data file to be used by the tests
        sample_data = {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]),
            "Ticker": ["AAPL", "AAPL", "GOOG", "GOOG"],
            "Open": [100, 110, 2000, 2020],
            "Close": [110, 108, 2025, 2015]
        }
        pd.DataFrame(sample_data).to_csv(self.raw_data_path, index=False)

    def tearDown(self):
        """
        This method runs AFTER each test.
        It completely removes the temporary directory and all its contents.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_successful_processing(self):
        """Test the successful processing of a valid multi-ticker data file."""
        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        
        self.assertIsNotNone(processed_df)
        self.assertTrue(os.path.exists(self.processed_data_path))
        
        aapl_data = processed_df[processed_df['Ticker'] == "AAPL"]
        goog_data = processed_df[processed_df['Ticker'] == "GOOG"]
        
        self.assertAlmostEqual(aapl_data["Open"].min(), 0.0)
        self.assertAlmostEqual(aapl_data["Open"].max(), 1.0)
        self.assertAlmostEqual(goog_data["Open"].min(), 0.0)
        self.assertAlmostEqual(goog_data["Open"].max(), 1.0)

    def test_returns_none_if_raw_file_missing(self):
        """Test that the function returns None when the raw data file does not exist."""
        os.remove(self.raw_data_path)
        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        self.assertIsNone(processed_df)

    def test_returns_none_if_processed_file_exists(self):
        """Test that the function skips and returns None if the processed file already exists."""
        # Create a dummy processed file to trigger the skip logic
        with open(self.processed_data_path, 'w') as f:
            f.write("dummy")
            
        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        self.assertIsNone(processed_df)

    def test_returns_none_if_ticker_column_missing(self):
        """Test that the function returns None if the 'Ticker' column is missing."""
        bad_df = pd.DataFrame({"Date": ["2023-01-01"], "Open": [100]})
        bad_df.to_csv(self.raw_data_path, index=False, mode='w')
        
        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        self.assertIsNone(processed_df)
        
    def test_handles_missing_values_within_groups(self):
        """Test that NaNs are correctly filled or dropped within each ticker group."""
        # Create a more complex raw file for this specific test
        tricky_data = {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]),
            "Ticker": ["NOK", "NOK", "MSFT", "MSFT"],
            "Open": [5, np.nan, np.nan, 300],  # NaN in NOK, leading NaN in MSFT
            "Close": [5.1, 5.2, np.nan, 302]
        }
        pd.DataFrame(tricky_data).to_csv(self.raw_data_path, index=False, mode='w')

        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        
        # The MSFT group should have its first row dropped, leaving only one row
        self.assertEqual(len(processed_df[processed_df['Ticker'] == 'MSFT']), 1)
        
        # The NOK group should have its NaN value forward-filled
        nok_open = processed_df[processed_df['Ticker'] == 'NOK']['Open']
        self.assertFalse(nok_open.isnull().any())
        # After ffill, Open values are [5, 5]. After scaling, both should be 0.
        self.assertTrue((nok_open == 0).all())

if __name__ == "__main__":
    unittest.main()
