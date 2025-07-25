# Commands for AI-Trader Data Pipeline

Here are the primary commands used to run the data pipeline for this project. These should be run from the root directory of the project.

---

### 1. To Run the Full Data Pipeline (Download and Process)

This command executes the main `populate_data.py` script. It will first download all the raw stock data for the tickers listed in `nasdaq_tickers.txt` and then immediately process that raw data, saving the final, normalized output to the `data/Processed` directory.

```bash
python -m src.data_pipeline.populate_data
```

**Note:** This can be a memory-intensive process if the ticker list is very large.

---

### 2. To Run Only the Data Processing Step

Use this command if you have already downloaded the raw data (e.g., to `data/Raw/raw_data.csv`) and only need to run the processing and normalization steps. This is useful for debugging the feature engineering logic or for reprocessing existing raw data.

```bash
python -c "from src.data_pipeline.process_data import process_data; process_data('data/Raw/raw_data.csv', 'data/Processed/processed_data.csv')"
```

---

### 3. To Run the Unit Tests for the Data Pipeline

This command will execute the test suite located in `src/test_pipeline/test_data_processing.py`. It's good practice to run this after making changes to the data processing logic to ensure everything still works as expected.

```bash
python -m unittest src/test_pipeline/test_data_processing.py
``` 