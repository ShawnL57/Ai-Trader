import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import os 

def time_series_split(processed_data_path: str):
    """
    splits data frame into trainign and testing sets.
    split each year by : 9 months trainign , 3 months testing
    this leads to a 75% training split and 25% testing split.
    
    creates two new csv files: 
    -train_data.csv
    -test_data.csv
    """ 
    
    df = pd.read_csv(processed_data_path, parse_dates= ['Date'], index_col= 'Date')
    df.sort_index(inplace=True)

    #for a single 75/25 split, n_splits=4 means, 
    # last split will use the first 3 chunks for training, and the 4th chunk for testing. 
    tscv = TimeSeriesSplit(n_splits=4)

    #we only need the last split for testing
    train_index, test_index = list(tscv.split(df))[-1] #this is the last split, which is the testing set

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    output_dir = os.path.dirname(processed_data_path)
    train_data_path = os.path.join(output_dir, 'train_data.csv')
    test_data_path = os.path.join(output_dir, 'test_data.csv')

    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)

    print("Data split into training and testing sets and saved successfully.")
    print(f"Training data has {len(train_df)} rows.")
    print(f"Testing data has {len(test_df)} rows.")
    print(f"Training data saved to {train_data_path}")
    print(f"Testing data saved to {test_data_path}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))    
    data_path = os.path.join(project_root, 'data', 'Processed', 'processed_data.csv')
    
    if os.path.exists(data_path):
        time_series_split(data_path)
    else:
        print(f"Error: Data file not found at {data_path}")
