import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the DataFrame.
    
    Args:
    df (pandas.DataFrame): The DataFrame to handle missing values in.
    strategy (str): The strategy to use for handling missing values.
        'drop': Drop rows with missing values.
        'fill': Replace/fill missing values with the column mean.
    
    Returns:
    pandas.DataFrame: The DataFrame with missing values handled.
    """
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(df[col].mean())
    else:
        raise ValueError("Invalid strategy. Use 'drop', 'mean', or 'median'.")
    return df

def encode_categorical(df):
    """
    Convert categorical columns to dummy variables

    ARgs:
    - df: pandas DataFrame
        the dataset containing categorical data
    
    Returns:
    - df: pandas DataFrame
        the dataset with categorical columns encoded as numeric features
    """
    # Find columns with categorical data 
    categorical_features = df.select_dtypes(include=['object']).columns
    # Create dummy variables and drop the first category to avoid multicollinearity 
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_encoded

def scale_features(df, numerical_columns):
    """
    Scale numerical features using StandardScaler

    Args:
    - df: pandas DataFrame
    - numerical_columns: list of str. List of column names that need scaling

    Returns:
    -df: pandas DataFrame
        DataFrame with specified numerical columns scaled
    """

    scaler = StandardScaler()
    # Fit and transform the specific numerical columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def remove_outliers(df, column, factor=1.5):
    """
    Remove outliers from a specific column

    Args:
    - df: pandas DataFrame
    - column: str. Name of the column to remove outliers from
    - factor: float. Multiplier for IQR to identify outliers

    Returns:
    - df: pandas DataFrame
        DataFrame with specified column removed of outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # Only keep rows within the acceptable range
    df = df[(df[column] >= Q1 - factor * IQR) & (df[column] <= Q3 + factor * IQR)]
    return df
