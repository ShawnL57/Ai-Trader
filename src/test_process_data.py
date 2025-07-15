import unittest
import os
import pandas as pd
from process_data import process_data
import shutil

class TestProcessData(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and sample data for testing."""
        self.test_dir = "temp_test_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.raw_data_path = os.path.join(self.test_dir, "raw_data.csv")
        self.processed_data_path = os.path.join(self.test_dir, "processed_data.csv")
        
        # Create a sample raw data file
        sample_data = {
            "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            "Open": [100, 102, 101, 103],
            "High": [103, 104, 103, 105],
            "Low": [99, 101, 100, 102],
            "Close": [102, 103, 102, 104],
            "Volume": [1000, 1100, 1050, 1200]
        }
        self.raw_df = pd.DataFrame(sample_data)
        self.raw_df.to_csv(self.raw_data_path, index=False)

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def test_successful_processing(self):
        """Test the successful processing of a valid raw data file."""
        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        
        self.assertIsNotNone(processed_df)
        self.assertTrue(os.path.exists(self.processed_data_path))
        
        # Check that data is normalized (between 0 and 1)
        self.assertTrue((processed_df["Open"] >= 0).all() and (processed_df["Open"] <= 1).all())
        self.assertTrue((processed_df["Volume"] >= 0).all() and (processed_df["Volume"] <= 1).all())

    def test_raw_file_not_found(self):
        """Test that the function returns None when the raw data file does not exist."""
        os.remove(self.raw_data_path)  # Delete the raw file
        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        self.assertIsNone(processed_df)

    def test_processed_file_already_exists(self):
        """Test that the function returns None if the processed file already exists."""
        # Create a dummy processed file
        with open(self.processed_data_path, 'w') as f:
            f.write("dummy content")
            
        processed_df = process_data(self.raw_data_path, self.processed_data_path)
        self.assertIsNone(processed_df)

    def test_missing_value_handling(self):
        """Test that missing values are correctly forward-filled."""
        # Create a file with a missing value
        data_with_nan = self.raw_df.copy()
        data_with_nan.loc[1, "Open"] = None
        nan_raw_path = os.path.join(self.test_dir, "nan_raw.csv")
        nan_processed_path = os.path.join(self.test_dir, "nan_processed.csv")
        data_with_nan.to_csv(nan_raw_path, index=False)

        processed_df = process_data(nan_raw_path, nan_processed_path)
        
        # The original value at index 0 for "Open" was 100
        # The ffill should propagate this to index 1
        # After scaling, it won't be 100, but it shouldn't be NaN
        self.assertFalse(processed_df["Open"].isnull().any())
        
        # In this specific case, the min is 100 and max is 103 for "Open"
        # After ffill, row 1 "Open" is 100.
        # So the first two scaled "Open" values should be the same and equal to 0.
        self.assertEqual(processed_df["Open"].iloc[0], 0)
        self.assertEqual(processed_df["Open"].iloc[1], 0)


if __name__ == "__main__":
    unittest.main() 