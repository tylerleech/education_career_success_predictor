# src/__init__.py

# Data loading functions
from .data_loader import load_data, inspect_data

# Preprocessing functions
from .preprocessing import handle_missing_values, encode_categorical, scale_features, remove_outliers

# Visualization functions
from .visualization import plot_histogram, plot_scatter, plot_boxplot, plot_bar_chart, plot_correlation_heatmap

# Modeling functions
from .models import (
    train_naive_bayes,
    train_xgboost_classifier,
    train_mlp_classifier,
    tune_random_forest_classifier,
    stacking_classifier,
    train_linear_regression,
    train_xgboost_regressor,
    train_mlp_regressor,
    tune_random_forest_regressor,
    tune_svr,
    stacking_regressor
)

# Evaluation functions
from .evaluation import evaluate_classification, evaluate_regression