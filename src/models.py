# src/models.py

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.dummy import DummyRegressor

def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting_classifier(X_train, y_train, params=None):
    """
    Train a Gradient Boosting Classifier using scikit-learn's implementation.

    Parameters:
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        params : dict, optional
            Dictionary of hyperparameters. Defaults to:
                n_estimators: 100,
                max_depth: 3,
                learning_rate: 0.1,
                random_state: 42

    Returns:
        model : GradientBoostingClassifier
            The trained gradient boosting classifier.
    """
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42
        }
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model

def baseline_mean_regressor(X_train, y_train):
    """
    Train a baseline regressor that predicts the mean of the target variable for every instance.
    
    Parameters:
        X_train: array-like (not used by the regressor)
        y_train: array-like, target values
    
    Returns:
        model: DummyRegressor
    """
    baseline_model = DummyRegressor(strategy='mean')
    baseline_model.fit(X_train, y_train)
    return baseline_model

# Additional functions (for regression or other models) can be added below if needed:

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest_classifier(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_random_forest_classifier(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters (Classifier):", grid.best_params_)
    return grid.best_estimator_

def train_random_forest_regressor(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_random_forest_regressor(X_train, y_train):
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
    model = SVR()
    model.fit(X_train, y_train)
    return model

def tune_svr(X_train, y_train):
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
    if params is None:
        params = {}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_xgboost_regressor(X_train, y_train, params=None):
    if params is None:
        params = {}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

# Sophisticated Ensemble Methods: Stacking Ensembles can also be added if needed:
def stacking_classifier(X_train, y_train):
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
