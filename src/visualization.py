import matplotlib.pyplot as plt

def plot_histogram(df, column, bins=30):
    """
    Plot a histogram of a specific column

    Args:
    - df: pandas DataFrame
    - column: str. Name of the column to plot
    - bins: int. Number of bins for the histogram

    Returns:
    None
    """
    plt.figure(figsize=(8,5))
    plt.hist(df[column], bins=bins, edgecolor='black')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_scatter(df, x_col, y_col):
    """
    Create a scatter plot to visualize the relationship between two columns.

    Args:
    - df: pandas DataFrame
    - x_col: str. Name of the column for the x-axis
    - y_col: str. Name of the column for the y-axis
    """

    plt.figure(figsize=(8,5))
    plt.scatter(df[x_col], df[y_col], alpha=0.6)
    plt.title(f'{x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def plot_boxplot(df, column):
    """
    Generate a boxplot for a column to inspect its distribution and detect outliers.
    
    Parameters:
    - df: pandas DataFrame
        The dataset.
    - column: str
        Name of the column to plot.
    
    Use this function to visually identify outliers and the spread of a numerical feature.
    """
    plt.figure(figsize=(8, 5))
    plt.boxplot(df[column].dropna())
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.show()

def plot_bar_chart(models, rmse_values):
    """
    Plot a bar chart to compare RMSE values of different regression models.
    
    Parameters:
    - models: list of str
        Names of the models you are comparing.
    - rmse_values: list of float
        Corresponding RMSE values for each model.
    
    This visualization helps you compare the performance of various models.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(models, rmse_values, edgecolor='black')
    plt.title('Regression Model RMSE Comparison')
    plt.ylabel('RMSE')
    plt.show()