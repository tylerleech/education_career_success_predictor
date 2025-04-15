# src/evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(y_test, y_pred, class_labels=None, save_path=None):
    """
    Evaluate a classification model by printing its accuracy, confusion matrix,
    and classification report. Visualizes the confusion matrix as a heatmap.
    
    Parameters:
        y_test : array-like
            The true labels.
        y_pred : array-like
            The predicted labels.
        class_labels : list, optional
            List of class labels for the plot. Defaults to numeric labels.
        save_path : str, optional
            If provided, saves the confusion matrix plot to the given path.
    """
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    
    plt.figure(figsize=(8, 6))
    if class_labels is None:
        class_labels = [str(i) for i in range(cm.shape[0])]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def evaluate_regression(y_test, y_pred):
    """
    Evaluate a regression model by printing RMSE and R-squared metrics.
    
    Returns:
        rmse (float): Root Mean Squared Error.
        r2 (float): R-squared value.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("RMSE:", rmse)
    print("R^2:", r2)
    return rmse, r2

