# src/models.py

# Import necessary models and tools from scikit-learn
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV

# Import XGBoost (make sure to install with: pip install xgboost)
import xgboost as xgb

#############################
# Classification Models
#############################
def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_xgboost_classifier(X_train, y_train):
    """
    Train an XGBoost classifier.
    """
    # Set some default parameters; these can be tuned later.
    model = xgb.XGBClassifier(objective='binary:logistic',
                              eval_metric='logloss',
                              use_label_encoder=False,
                              random_state=42)
    model.fit(X_train, y_train)
    return model

def train_mlp_classifier(X_train, y_train):
    """
    Train a Multi-Layer Perceptron classifier.
    """
    model = MLPClassifier(hidden_layer_sizes=(100,),
                          max_iter=300,
                          random_state=42)
    model.fit(X_train, y_train)
    return model

def tune_random_forest_classifier(X_train, y_train):
    """
    Tune a Random Forest classifier using GridSearchCV.
    
    Returns the best estimator based on cross-validation.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42),
                        param_grid,
                        cv=5,
                        n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters (Classifier):", grid.best_params_)
    return grid.best_estimator_

def stacking_classifier(X_train, y_train):
    """
    Create a stacked classifier that combines predictions from multiple models.
    """
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('mlp', MLPClassifier(max_iter=300, random_state=42))
    ]
    # Use RidgeClassifier as the final estimator for stacking.
    model = StackingClassifier(estimators=estimators, final_estimator=RidgeClassifier())
    model.fit(X_train, y_train)
    return model

#############################
# Regression Models
#############################
def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_xgboost_regressor(X_train, y_train):
    """
    Train an XGBoost regressor.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_mlp_regressor(X_train, y_train):
    """
    Train a Multi-Layer Perceptron regressor.
    """
    model = MLPRegressor(hidden_layer_sizes=(100,),
                         max_iter=300,
                         random_state=42)
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
    grid = GridSearchCV(RandomForestRegressor(random_state=42),
                        param_grid,
                        cv=5,
                        n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters (Regressor):", grid.best_params_)
    return grid.best_estimator_

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

def stacking_regressor(X_train, y_train):
    """
    Create a stacked regressor that blends predictions from multiple base models.
    """
    estimators = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('svr', SVR())
    ]
    model = StackingRegressor(estimators=estimators,
                              final_estimator=MLPRegressor(random_state=42))
    model.fit(X_train, y_train)
    return model
