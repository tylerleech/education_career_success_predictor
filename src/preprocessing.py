# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the CSV file."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise FileNotFoundError(f"Error loading file at {filepath}: {e}")

def remove_outliers(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Removes outliers using the IQR method for a list of numeric columns."""
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

def build_preprocessor(df: pd.DataFrame, target_column: str, remove_outlier_flag: bool = True) -> ColumnTransformer:
    """
    Constructs the ColumnTransformer that handles:
       - Numeric imputation with median and scaling.
       - Categorical imputation with a constant value and one-hot encoding.
    """
    X = df.drop(columns=[target_column])
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Optionally remove outliers for numeric features
    if remove_outlier_flag:
        df = remove_outliers(df, numeric_features)
        X = df.drop(columns=[target_column])
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )
    return preprocessor

def preprocess_data(df: pd.DataFrame, target_column: str, remove_outlier_flag: bool = True):
    """Returns the transformed features and target after fitting the preprocessor."""
    if remove_outlier_flag:
        numeric_cols = df.drop(columns=[target_column]).select_dtypes(include=['int64', 'float64']).columns.tolist()
        df = remove_outliers(df, numeric_cols)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    preprocessor = build_preprocessor(df, target_column, remove_outlier_flag)
    X_transformed = preprocessor.fit_transform(X)
    
    return X_transformed, y, preprocessor

if __name__ == '__main__':
    file_path = '/mnt/data/education_career_success.csv'
    # Adjust this target column name based on your dataset (for instance, 'CareerSuccess')
    target_column = 'CareerSuccess'  
    df = load_data(file_path)
    print("Columns:", df.columns.tolist())
    X_transformed, y, _ = preprocess_data(df, target_column)
    print("Transformed shape:", X_transformed.shape)
