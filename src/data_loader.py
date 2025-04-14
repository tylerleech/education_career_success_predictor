import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
    file_path (str): The path to the CSV file.
    
    Returns:
    pandas.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df
