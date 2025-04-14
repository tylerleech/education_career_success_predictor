# src/evaluation.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_classification(y_test, y_pred):
    """
    Evaluate a classification model.
    
    Parameters:
    - y_test: array-like or Series
        True labels.
    - y_pred: array-like or Series
        Model-predicted labels.
    
    Prints the accuracy, confusion matrix, and classification report.
    """
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def evaluate_regression(y_test, y_pred):
    """
    Evaluate a regression model.
    
    Parameters:
    - y_test: array-like or Series
        True target values.
    - y_pred: array-like or Series
        Model-predicted values.
    
    Calculates and prints the RMSE and R-squared metrics.
    
    Returns:
    - rmse: float
        The root mean squared error.
    - r2: float
        The R-squared value.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("RMSE:", rmse)
    print("R^2:", r2)
    return rmse, r2
