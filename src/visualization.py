# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(df, column, bins=30):
    """
    Plot a histogram for a given column.
    
    Parameters:
    - df (DataFrame): Input dataset.
    - column (str): Column name.
    - bins (int): Number of bins in the histogram.
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
    
    Parameters:
    - df (DataFrame): Input dataset.
    - x_col (str): Column for the x-axis.
    - y_col (str): Column for the y-axis.
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
    
    Parameters:
    - df (DataFrame): Input dataset.
    - column (str): Column name to plot.
    """
    plt.figure(figsize=(8, 5))
    plt.boxplot(df[column].dropna())
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.show()

def plot_bar_chart(models, rmse_values):
    """
    Plot a bar chart to compare RMSE values of various regression models.
    
    Parameters:
    - models (list): Names of the models.
    - rmse_values (list): RMSE values corresponding to each model.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(models, rmse_values, edgecolor='black')
    plt.title('Regression Model RMSE Comparison')
    plt.ylabel('RMSE')
    plt.show()

def plot_correlation_heatmap(df):
    """
    Plot a heatmap of the correlation matrix using matplotlib.
    
    Parameters:
    - df (DataFrame): Input dataset.
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
