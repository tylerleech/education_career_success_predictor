# src/data_loader.py
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def inspect_data(df):
    print("Head of dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())
