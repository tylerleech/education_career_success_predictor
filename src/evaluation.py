# src/evaluation.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_classification(y_test, y_pred):
    """
    Evaluate a classification model by printing its accuracy, confusion matrix,
    and classification report.
    """
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def evaluate_regression(y_test, y_pred):
    """
    Evaluate a regression model by printing RMSE and R-squared metrics.
    
    Returns:
    - rmse (float): Root Mean Squared Error.
    - r2 (float): R-squared value.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("RMSE:", rmse)
    print("R^2:", r2)
    return rmse, r2
