# src/models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

def prepare_data(filepath: str, target_column: str, outlier_flag: bool = True):
    """
    Load the dataset, optionally remove outliers, then split into training and testing sets.
    """
    # Import helper functions from preprocessing
    from src.preprocessing import load_data, remove_outliers

    # Load the dataset from CSV
    df = load_data(filepath)
    
    # Identify numeric columns (excluding the target)
    numeric_cols = df.drop(columns=[target_column]).select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Optionally remove outliers for numeric features
    if outlier_flag:
        df = remove_outliers(df, numeric_cols)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Perform a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def build_stacked_model(preprocessor):
    """
    Build a pipeline with the given preprocessor and a stacked regressor
    that combines a RandomForest and an XGBoost model, with Ridge as the final estimator.
    """
    # Define base estimators for stacking
    estimators = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
    ]
    
    # Create the stacked regressor with a Ridge regressor as the final estimator
    stacked_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1
    )
    
    # Build the complete pipeline by chaining the preprocessor and the stacked regressor
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('stacked_model', stacked_regressor)
    ])
    
    return model_pipeline
