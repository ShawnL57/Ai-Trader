#plan for training the model. 


"""
step 1: Load the traingin and testing data
step 2 : prepare the data for the model
    to train th model, we need to distinguish ebtween our features and our target
    - features: x_train and x_test are created by dropping the target colum ('y') and any identifier colums (date) columns Date and Ticker that are not predictive 
    - target: y_train and y_test are created by selecting only the 'y' column, which represents whetehr the stoc price goes up(1) or goes down (0)

step 3:  initalize XGboost classifier 
-create an instance of 'xgboost.xgbclassifer' which is our predictive model. 
- it's configured with a set of hyper perameters to control its training prodess
-  'objective' : 'binary:logistic' : specifies that this is a binary (two-classification) problem. 
-  'n_estimators' : 1000 : the maximum number of descision trees to build. 
- 'max_depth' : 5 : the maximum depth of each individual tree
- 'eval_metric' : 'logloss' : the metric used to evaluate the model's performance during training. 


4: 
- train the model with early stopping to prevent overfitting
- the model is trianged by calling the 'fit' method on the traingin data (x_train and y_train)
- Early stopping: This is a key technique to prevent overfitting. The models performance is monitored on the test set (eval_    
- contd. stopping_rounds = 50, the training stops automatically, ensuring we get the best version of the model without it memoriziing the training data

"""




#step 1 : load the data

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
def train_model(train_data_path:str, test_data_path: str, model_save_path: str):
    #step 1: load data
    try:
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    #step 2: prep data

    print("Preparing data...")

    target_col = 'y' #target variable 

    #drop non-feature colums
    #'Date' and 'Ticker' are non-predictive
    features_to_drop = ['Date', 'Ticker', target_col]
    

    x_train = train_data.drop(columns = features_to_drop, errors='ignore')
    y_train = train_data[target_col]

    x_test = test_data.drop(columns = features_to_drop, errors='ignore')
    y_test = test_data[target_col]


    print("Data prepared successfully")
    print(f"X_train shape :{x_train.shape}")
    print(f"y_train shape :{y_train.shape}")
    print(f"X_test shape :{x_test.shape}")
    print(f"y_test shape :{y_test.shape}")

    # Calculate scale_pos_weight for handling class imbalance
    # This is a common technique to help the model pay more attention to the minority class.
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")

    #step 3: initialize model with hyper parameters
    xgb_model = xgb.XGBClassifier(
        objective = 'binary:logistic',
        n_estimators = 1000,
        max_depth = 5,
        subsample = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder = False,
        eval_metric = 'logloss',
        scale_pos_weight=scale_pos_weight
    )
    
    # Hyperparameter Tuning with GridSearchCV
    print("\nStarting hyperparameter tuning with GridSearchCV...")

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [500, 1000],
        'subsample': [0.7, 0.8]
    }

    # Set up GridSearchCV
    # We are optimizing for 'precision' for the positive class (Up)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='precision',
        n_jobs=-1,  # Use all available CPU cores
        cv=3,       # 3-fold cross-validation
        verbose=2
    )

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)

    print("\nGrid search complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Get the best model from the grid search
    best_model = grid_search.best_estimator_



 # Step 4: Evaluate the Model
    print("\nEvaluating model performance...")
    
    # Make predictions on the test set
    y_pred = best_model.predict(x_test)
    y_pred_proba = best_model.predict_proba(x_test)[:, 1] # get probabilities for the positive class

    #calculate and print performance metrics: 
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) #roc_auc means area under the roc curve and that gives the model's ability to distinguish between the two classes

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    #step 5: save the model 
    print(f"\nSaving model to {model_save_path}...")
    try:
        joblib.dump(best_model, model_save_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TRAIN_DATA_PATH = os.path.join(project_root, 'data', 'Processed', 'train_data.csv')
    TEST_DATA_PATH = os.path.join(project_root, 'data', 'Processed', 'test_data.csv')
    MODEL_SAVE_PATH = os.path.join(project_root, 'data', 'model.joblib')

    # Create directory for the model if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    train_model(TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH)


