import unittest
import pandas as pd
import numpy as np
import os
import joblib
import sys

# Add the project root to the Python path
# This allows us to import modules from the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.data_pipeline.process_data import process_data

class TestProcessData(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        # Create dummy directories for test data
        self.raw_data_dir = 'test_raw_data'
        self.processed_data_dir = 'test_processed_data'
        os.makedirs(self.raw_data_dir, exist_ok=True) 
        os.makedirs(self.processed_data_dir, exist_ok=True) 

        # Define file paths
        self.raw_data_path = os.path.join(self.raw_data_dir, 'raw_data.csv')
        self.processed_data_path = os.path.join(self.processed_data_dir, 'processed_data.csv')
        self.scaler_path = os.path.join(self.processed_data_dir, 'scaler.joblib')

        # Clean up any files from previous runs
        self._cleanup_files()

    def tearDown(self):
        """Clean up test environment after each test."""
        self._cleanup_files()
        os.rmdir(self.raw_data_dir)
        os.rmdir(self.processed_data_dir)

    def _cleanup_files(self):
        """Helper to remove generated files."""
        if os.path.exists(self.raw_data_path):
            os.remove(self.raw_data_path)
        if os.path.exists(self.processed_data_path):
            os.remove(self.processed_data_path)
        if os.path.exists(self.scaler_path):
            os.remove(self.scaler_path)

    def _create_dummy_raw_data(self, num_days, ticker):
        """Creates a dummy raw data CSV file."""
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_days, freq='D'))
        data = {
            'Date': dates,
            'Open': np.random.uniform(100, 102, size=num_days),
            'High': np.random.uniform(102, 104, size=num_days),
            'Low': np.random.uniform(98, 100, size=num_days),
            'Close': np.random.uniform(100, 103, size=num_days),
            'Volume': np.random.randint(1_000_000, 5_000_000, size=num_days),
            'Ticker': ticker
        }
        df = pd.DataFrame(data)
        df.to_csv(self.raw_data_path, index=False)
        return df

    def test_full_data_processing(self):
        """Test processing from scratch (full run)."""
        print("\n--- Running test_full_data_processing ---")
        # 1. Setup: Create initial raw data
        num_days = 150
        self._create_dummy_raw_data(num_days=num_days, ticker='AAPL')

        # 2. Action: Run the processing function
        processed_df = process_data(self.raw_data_path, self.processed_data_path)

        # 3. Assertions
        self.assertIsNotNone(processed_df, "Processing should return a DataFrame.")
        self.assertTrue(os.path.exists(self.processed_data_path), "Processed data file should be created.")
        self.assertTrue(os.path.exists(self.scaler_path), "Scaler object should be saved.")

        # Check data integrity
        self.assertFalse(processed_df.isnull().values.any(), "Processed data should not contain NaN values.")
        self.assertIn('y', processed_df.columns, "Target column 'y' should be present.")
        self.assertIn('RSI_14', processed_df.columns, "Feature 'RSI_14' should be present.")

        # Verify scaler was applied (scaled data should have mean ~0 and std dev ~1)
        self.assertAlmostEqual(processed_df['Close'].mean(), 0, delta=0.1, msg="Scaled 'Close' mean should be near 0.")
        self.assertAlmostEqual(processed_df['Close'].std(), 1, delta=0.1, msg="Scaled 'Close' std dev should be near 1.")

    def test_incremental_data_processing(self):
        """Test processing new data incrementally."""
        print("\n--- Running test_incremental_data_processing ---")
        # 1. Setup: First run (full processing)
        initial_days = 150
        self._create_dummy_raw_data(num_days=initial_days, ticker='AAPL')
        initial_processed_df = process_data(self.raw_data_path, self.processed_data_path)
        initial_rows = len(initial_processed_df)

        # Check that the first run was successful
        self.assertTrue(os.path.exists(self.processed_data_path))
        self.assertTrue(os.path.exists(self.scaler_path))

        # 2. Setup: Add new data to the raw file
        additional_days = 20
        self._create_dummy_raw_data(num_days=initial_days + additional_days, ticker='AAPL')

        # 3. Action: Run processing again (incremental run)
        incremental_processed_df = process_data(self.raw_data_path, self.processed_data_path)
        
        # 4. Assertions
        self.assertIsNotNone(incremental_processed_df, "Incremental processing should return a DataFrame.")
        
        # Verify that only new data was added
        final_df = pd.read_csv(self.processed_data_path)
        self.assertGreater(len(final_df), initial_rows, "Total rows should increase after incremental run.")
        
        # The returned dataframe from process_data should only contain the new rows
        # The number might not be exactly additional_days due to lookahead shifts and NaN dropping
        self.assertLess(len(incremental_processed_df), initial_rows, "Returned DF should only contain new data.")
        self.assertTrue(np.isclose(len(incremental_processed_df), additional_days, atol=2), "Should process approximately the number of additional days.")

        # Verify the scaler was not refit (by loading and checking its properties, if possible)
        # For simplicity, we trust the log messages and file modification times,
        # but a more rigorous test could inspect the scaler object itself.

if __name__ == '__main__':
    unittest.main() 