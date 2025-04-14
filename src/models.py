# src/models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

def prepare_data(filepath: str, target_column: str, outlier_flag: bool = True):
    """
    Load the dataset, optionally remove outliers, and then split into training and testing sets.
    """
    # Import helper functions from preprocessing.
    from src.preprocessing import load_data, remove_outliers

    # Load the data.
    df = load_data(filepath)
    
    # Identify numeric columns (excluding the target)
    numeric_cols = df.drop(columns=[target_column]).select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Optionally remove outliers from numeric features.
    if outlier_flag:
        df = remove_outliers(df, numeric_cols)
    
    # Separate the features and target.
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Perform train/test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def build_stacked_model(preprocessor):
    """
    Build and return a pipeline with a given preprocessor and a stacked regressor.
    The stacked model combines RandomForest and XGBoost, with a Ridge regressor for blending.
    """
    # Define base learners.
    estimators = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
    ]
    
    # Define the final estimator using Ridge regression.
    stacked_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1
    )
    
    # Build the pipeline with the preprocessor and stacked regressor.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('stacked_model', stacked_regressor)
    ])
    
    return model_pipeline
