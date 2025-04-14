# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise FileNotFoundError(f"Could not load file from {filepath}: {e}")

def remove_outliers(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from numeric columns using the IQR method.
    """
    df_clean = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def build_preprocessor(df: pd.DataFrame, target_column: str, remove_outlier_flag: bool = True):
    """
    Build a ColumnTransformer that preprocesses numeric and categorical features:
      - Numeric features are imputed (with median) and scaled.
      - Categorical features are imputed (using a constant) and one-hot encoded.
    """
    # Drop the target column.
    X = df.drop(columns=[target_column])
    
    # Identify numeric and categorical features.
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Optionally remove outliers.
    if remove_outlier_flag:
        df = remove_outliers(df, numeric_features)
        X = df.drop(columns=[target_column])
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create a pipeline for numeric features.
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create a pipeline for categorical features.
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine both pipelines into a ColumnTransformer.
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor

