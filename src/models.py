# src/models.py
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.
    
    Parameters:
    - X_train: array-like or DataFrame
        Training features for classification.
    - y_train: array-like or Series
        Training labels.
    
    Returns:
    - model: GaussianNB
        The trained Naive Bayes classifier.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Parameters:
    - X_train: array-like or DataFrame
        Training features for regression.
    - y_train: array-like or Series
        Training target values.
    
    Returns:
    - model: LinearRegression
        The trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train, y_train):
    """
    Train a Random Forest classifier.
    
    Parameters:
    - X_train: array-like or DataFrame
        Training features.
    - y_train: array-like or Series
        Training labels.
    
    Returns:
    - model: RandomForestClassifier
        The trained Random Forest classification model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svc_classifier(X_train, y_train):
    """
    Train a Support Vector Classifier (SVC).
    
    Parameters:
    - X_train: array-like or DataFrame
        Training features.
    - y_train: array-like or Series
        Training labels.
    
    Returns:
    - model: SVC
        The trained SVC model.
    """
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest_regressor(X_train, y_train):
    """
    Train a Random Forest Regressor.
    
    Parameters:
    - X_train: array-like or DataFrame
        Training features.
    - y_train: array-like or Series
        Training target values.
    
    Returns:
    - model: RandomForestRegressor
        The trained Random Forest regression model.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train):
    """
    Train a Support Vector Regressor (SVR).
    
    Parameters:
    - X_train: array-like or DataFrame
        Training features.
    - y_train: array-like or Series
        Training target values.
    
    Returns:
    - model: SVR
        The trained SVR model.
    """
    model = SVR()
    model.fit(X_train, y_train)
    return model
