# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(df, column, bins=30):
    """
    Plot a histogram for a given column.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=bins, edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_scatter(df, x_col, y_col):
    """
    Plot a scatter chart comparing two numerical features.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_col], df[y_col], alpha=0.6)
    plt.title(f'{x_col} vs. {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def plot_boxplot(df, column):
    """
    Create a boxplot for a specified column.
    """
    plt.figure(figsize=(8, 5))
    plt.boxplot(df[column].dropna())
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.show()

def plot_bar_chart(models, rmse_values):
    """
    Plot a bar chart to compare RMSE values of various regression models.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(models, rmse_values, edgecolor='black')
    plt.title('Regression Model RMSE Comparison')
    plt.ylabel('RMSE')
    plt.show()

def plot_correlation_heatmap(df):
    """
    Plot a heatmap of the correlation matrix for numerical features.
    
    This function filters out non-numeric columns before computing the
    correlation matrix.
    
    Parameters:
    - df (DataFrame): The input dataset.
    """
    # Filter DataFrame to include only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(corr, cmap='viridis', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.title("Correlation Heatmap")
    plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(np.arange(len(corr.columns)), corr.columns)
    plt.show()
