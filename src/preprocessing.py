# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the DataFrame.
    
    Parameters:
    - df (DataFrame): Input dataset.
    - strategy (str): 'drop' to remove missing rows; 'fill' to impute missing numeric values
                      using the column mean. (You can modify this to use 'median' if preferred.)
    
    Returns:
    - df (DataFrame): Dataset with missing values handled.
    """
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        # Use mean imputation; you might change this to median imputation.
        for col in df.select_dtypes(include=['float64','int64']).columns:
            df[col] = df[col].fillna(df[col].mean())
    return df

def encode_categorical(df):
    """
    Convert categorical columns into dummy/indicator variables.
    
    Parameters:
    - df (DataFrame): Dataset with categorical columns.
    
    Returns:
    - df_encoded (DataFrame): Dataset with one-hot encoded variables.
    """
    categorical_features = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_encoded

def scale_features(df, numerical_columns):
    """
    Scale numerical features using StandardScaler.
    
    Parameters:
    - df (DataFrame): Input dataset.
    - numerical_columns (list): List of numerical column names to scale.
    
    Returns:
    - df (DataFrame): Dataset with scaled numerical columns.
    """
    scaler = StandardScaler()  # You could swap in RobustScaler if outliers remain problematic
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def remove_outliers(df, column, factor=1.5):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    - df (DataFrame): Input dataset.
    - column (str): Column from which to remove outliers.
    - factor (float): Multiplier for the IQR; default 1.5.
    
    Returns:
    - df (DataFrame): DataFrame with outliers removed for that column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[column] >= Q1 - factor * IQR) & (df[column] <= Q3 + factor * IQR)]
    return df
