# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    
    Parameters:
        filepath (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise FileNotFoundError(f"Could not load file from {filepath}: {e}")

def remove_outliers(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from the specified numeric columns using the IQR method.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numeric column names.
        factor (float): The multiplier for the IQR (default is 1.5).
    
    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    df_clean = df.copy()
    for col in columns:
        # Only apply on numeric columns
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def build_preprocessor(df: pd.DataFrame, target_column: str, remove_outlier_flag: bool = True) -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that preprocesses numeric and categorical features.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        remove_outlier_flag (bool): Whether to remove outliers on numeric data.
    
    Returns:
        ColumnTransformer: A transformer that can be used to fit/transform features.
    """
    # Separate features and drop the target column
    X = df.drop(columns=[target_column])
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Optionally remove outliers based on numeric features
    if remove_outlier_flag:
        df = remove_outliers(df, numeric_features)
        X = df.drop(columns=[target_column])
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create a numeric data pipeline: impute missing values and scale features
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create a categorical data pipeline: impute missing values and one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine both pipelines into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )
    
    return preprocessor

def preprocess_data(df: pd.DataFrame, target_column: str, remove_outlier_flag: bool = True):
    """
    Preprocess the features and target data.
    
    This function builds the preprocessor, optionally removes outliers, and fits and transforms the features.
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame.
        target_column (str): The target variable column.
        remove_outlier_flag (bool): Flag to remove outliers (default True).
    
    Returns:
        X_transformed: The transformed features (sparse matrix or numpy array).
        y: The target variable.
        preprocessor: The fitted ColumnTransformer.
    """
    # If outlier removal is enabled, remove them first
    if remove_outlier_flag:
        numeric_cols = df.drop(columns=[target_column]).select_dtypes(include=['int64', 'float64']).columns.tolist()
        df = remove_outliers(df, numeric_cols)
    
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Build and fit the preprocessor on the feature data
    preprocessor = build_preprocessor(df, target_column, remove_outlier_flag)
    X_transformed = preprocessor.fit_transform(X)
    
    return X_transformed, y, preprocessor

# Example usage (this block can be called from your main script)
if __name__ == '__main__':
    # Update the file path and target column name according to your dataset
    file_path = '/mnt/data/education_career_success.csv'
    
    # IMPORTANT: Replace 'CareerSuccess' with the actual target column in your dataset.
    target_column = 'CareerSuccess'  
    
    # Load the data
    df = load_data(file_path)
    print("Dataset columns:", df.columns.tolist())
    
    # Preprocess data
    X_transformed, y, preprocessor = preprocess_data(df, target_column)
    print("Preprocessed feature shape:", X_transformed.shape)
    print("Target shape:", y.shape)
