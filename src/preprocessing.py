# src/preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler

def inspect_data(df):
    """
    Inspect the data and print out key diagnostics:
      - Data info and descriptive statistics
      - Missing values count per column
      - Skewness of numeric features
      - Correlation matrix
    Also displays histograms for numeric columns.
    """
    print("===== Data Information =====")
    print(df.info())
    print("\n===== Descriptive Statistics =====")
    print(df.describe())
    print("\n===== Missing Values =====")
    print(df.isnull().sum())
    print("\n===== Skewness of Numeric Features =====")
    print(df.select_dtypes(include=[np.number]).skew())
    print("\n===== Correlation Matrix =====")
    print(df.corr())
    
    # Plot histograms for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

def validate_data(df):
    """
    Perform a series of diagnostic checks to detect potential data issues.
    - Checks for missing values and duplicates.
    - Detects outliers using the IQR method.
    - Reports summary statistics for all numeric features.
    
    This information helps inform further cleaning steps.
    """
    print("===== Validating Data =====")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Warning: Missing values detected:")
        print(missing[missing > 0])
    else:
        print("No missing values detected.")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates:
        print(f"Warning: {duplicates} duplicate rows detected.")
    else:
        print("No duplicate rows detected.")
    
    # Outlier detection using IQR method for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        n_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        print(f"{col}: {n_outliers} outlier(s) detected (IQR method)")
    
    print("===== Data Validation Completed =====")

def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the DataFrame.
      - 'drop': Remove rows with any missing values.
      - 'fill': Fill missing numeric values with the median.
    """
    if strategy == 'drop':
        df_clean = df.dropna()
    elif strategy == 'fill':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        # For non-numeric columns, you might consider mode imputation
        non_numeric = df_clean.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    return df_clean

def remove_outliers(df, column, method='IQR'):
    """
    Remove outliers from the specified column.
      - Uses the IQR method by default.
    """
    if method == 'IQR':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    else:
        raise NotImplementedError("Only the IQR method is currently implemented for outlier removal.")
    return df

def scale_features(df, numerical_columns, scaler_type='robust'):
    """
    Scale numerical features using either StandardScaler or RobustScaler.
    RobustScaler is recommended when outliers are present.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'robust'")
    
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def encode_categorical(df):
    """
    Encode categorical variables using one-hot encoding.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded


