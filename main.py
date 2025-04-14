# project/main.py

import argparse
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Import functions from the src package.
from src.preprocessing import load_data, build_preprocessor, remove_outliers
from src.models import build_stacked_model, prepare_data

def train_and_evaluate(data_path: str, target_column: str, remove_outliers_flag: bool):
    """
    Loads data, builds a pipeline with preprocessing and a stacked regressor,
    trains the model, and prints evaluation metrics.
    """
    # Prepare the data (load CSV, optionally remove outliers, and split into train/test sets)
    X_train, X_test, y_train, y_test = prepare_data(data_path, target_column, outlier_flag=remove_outliers_flag)
    
    # Build preprocessor using the training data (attach target column temporarily)
    train_df = X_train.copy()
    train_df[target_column] = y_train
    preprocessor = build_preprocessor(train_df, target_column, remove_outlier_flag=False)
    
    # Build the stacked model pipeline
    model_pipeline = build_stacked_model(preprocessor)
    
    # Train the pipeline
    model_pipeline.fit(X_train, y_train)
    
    # Generate predictions and evaluate the model
    predictions = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("Stacked Model Performance:")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")

def main(args):
    data_path = args.data
    target_column = args.target
    remove_outliers_flag = not args.no_outliers
    train_and_evaluate(data_path, target_column, remove_outliers_flag)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Education & Career Success Predictor")
    parser.add_argument(
        '--data',
        type=str,
        default='project/data/education_career_success.csv',
        help='Path to the CSV dataset file'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='CareerSuccess',
        help='Name of the target column in the dataset'
    )
    parser.add_argument(
        '--no_outliers',
        action='store_true',
        help='Disable outlier removal during preprocessing'
    )
    args = parser.parse_args()
    main(args)
