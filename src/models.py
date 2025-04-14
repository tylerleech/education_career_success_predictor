# src/models.py

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train, y_train):
    """
    Train a Random Forest classifier with default hyperparameters.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_random_forest_classifier(X_train, y_train):
    """
    Tune a Random Forest classifier using GridSearchCV.
    Returns the best estimator after hyperparameter search.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters (Classifier):", grid.best_params_)
    return grid.best_estimator_

def train_svc_classifier(X_train, y_train):
    """
    Train a Support Vector Classifier.
    """
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest_regressor(X_train, y_train):
    """
    Train a Random Forest regressor.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_random_forest_regressor(X_train, y_train):
    """
    Tune a Random Forest regressor using GridSearchCV.
    Returns the best estimator.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters (Regressor):", grid.best_params_)
    return grid.best_estimator_

def train_svr(X_train, y_train):
    """
    Train a Support Vector Regressor.
    """
    model = SVR()
    model.fit(X_train, y_train)
    return model

def tune_svr(X_train, y_train):
    """
    Tune a Support Vector Regressor using GridSearchCV.
    Returns the best estimator.
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(SVR(), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters (SVR):", grid.best_params_)
    return grid.best_estimator_

def train_xgboost_classifier(X_train, y_train, params=None):
    """
    Train an XGBoost classifier on the provided data.
    Parameters:
        X_train : array-like, training features.
        y_train : array-like, training labels.
        params : dict, optional
            Parameters for XGBClassifier. If None, default parameters are used.
    Returns:
        model : XGBClassifier, the trained classifier.
    """
    if params is None:
        params = {}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_xgboost_regressor(X_train, y_train, params=None):
    """
    Train an XGBoost regressor on the provided data.
    Parameters:
        X_train : array-like, training features.
        y_train : array-like, training targets.
        params : dict, optional
            Parameters for XGBRegressor. If None, default parameters are used.
    Returns:
        model : XGBRegressor, the trained regressor.
    """
    if params is None:
        params = {}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

# --------------------------------------------------------------------
# Sophisticated Ensemble Methods: Stacking Ensembles
# --------------------------------------------------------------------

def stacking_classifier(X_train, y_train):
    """
    Train a stacking classifier ensemble using multiple base classifiers and a meta-classifier.
    
    Base models used:
      - RandomForestClassifier
      - XGBClassifier
      - GaussianNB
    Meta-model used:
      - LogisticRegression
    
    Parameters:
      X_train : array-like
          Training features.
      y_train : array-like
          Training labels.
    
    Returns:
      model : StackingClassifier
          The trained stacking classifier model.
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.naive_bayes import GaussianNB

    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('nb', GaussianNB())
    ]
    
    meta_estimator = LogisticRegression()
    
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_estimator,
        cv=5,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def stacking_regressor(X_train, y_train):
    """
    Train a stacking regressor ensemble using multiple base regressors and a meta-regressor.
    
    Base models used:
      - RandomForestRegressor
      - XGBRegressor
      - LinearRegression
    Meta-model used:
      - Ridge Regression
    
    Parameters:
      X_train : array-like
          Training features.
      y_train : array-like
          Training targets.
    
    Returns:
      model : StackingRegressor
          The trained stacking regressor model.
    """
    from sklearn.ensemble import StackingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    estimators = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lr', LinearRegression())
    ]
    
    meta_estimator = Ridge()
    
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_estimator,
        cv=5,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model
