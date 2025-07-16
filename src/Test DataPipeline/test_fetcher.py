import os
import pandas as pd
from fetch_data import download_data
import shutil

# Create a directory for test outputs
TEST_OUTPUT_DIR = "test_output"
if not os.path.exists(TEST_OUTPUT_DIR):
    os.makedirs(TEST_OUTPUT_DIR)

def test_successful_download():
    """
    Test case 1: Successful data download for a valid ticker.
    """
    print("--- Running Test 1: Successful Download ---")
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    save_path = os.path.join(TEST_OUTPUT_DIR, "aapl_test.csv")

    # Clean up previous test file if it exists
    if os.path.exists(save_path):
        os.remove(save_path)

    df = download_data(ticker, start_date, end_date, save_path)
    assert df is not None, "DataFrame should not be None"
    assert not df.empty, "DataFrame should not be empty"
    assert os.path.exists(save_path), "CSV file should be created"
    print("Test 1 Passed: Data downloaded and saved successfully.\n")
    return save_path

def test_existing_file():
    """
    Test case 2: Function should skip download if the file already exists.
    """
    print("--- Running Test 2: Existing File Check ---")
    ticker = "GOOGL"
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    save_path = os.path.join(TEST_OUTPUT_DIR, "googl_test.csv")

    # Create a dummy file to simulate an existing file
    with open(save_path, "w") as f:
        f.write("dummy content")

    df = download_data(ticker, start_date, end_date, save_path)
    assert df is None, "Function should return None for existing file"
    print("Test 2 Passed: Download skipped for existing file.\n")

def test_invalid_ticker():
    """
    Test case 3: Function should handle invalid tickers gracefully.
    """
    print("--- Running Test 3: Invalid Ticker ---")
    ticker = "INVALIDTICKER"
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    save_path = os.path.join(TEST_OUTPUT_DIR, "invalid_ticker_test.csv")

    # Clean up previous test file if it exists
    if os.path.exists(save_path):
        os.remove(save_path)

    df = download_data(ticker, start_date, end_date, save_path)
    assert df is None, "Function should return None for an invalid ticker"
    assert not os.path.exists(save_path), "No file should be created for an invalid ticker"
    print("Test 3 Passed: Invalid ticker handled correctly.\n")

def test_no_data_for_date_range():
    """
    Test case 4: Function should handle valid ticker with no data in the date range.
    """
    print("--- Running Test 4: No Data for Date Range ---")
    ticker = "TSLA"
    start_date = "1990-01-01"
    end_date = "1990-01-31"
    save_path = os.path.join(TEST_OUTPUT_DIR, "no_data_test.csv")

    # Clean up previous test file if it exists
    if os.path.exists(save_path):
        os.remove(save_path)

    df = download_data(ticker, start_date, end_date, save_path)
    assert df is None, "Function should return None when no data is found"
    assert not os.path.exists(save_path), "No file should be created when no data is found"
    print("Test 4 Passed: No data scenario handled correctly.\n")


if __name__ == "__main__":
    test_successful_download()
    test_existing_file()
    test_invalid_ticker()
    test_no_data_for_date_range()

    # Clean up the test directory
    shutil.rmtree(TEST_OUTPUT_DIR)
    print("Test output directory cleaned up.") 