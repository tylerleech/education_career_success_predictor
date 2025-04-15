# src/__init__.py

from .data_loader import load_data
from .preprocessing import (
    inspect_data,
    validate_data,
    handle_missing_values,
    encode_categorical,
    scale_features,
    remove_outliers
)
from .visualization import (
    plot_histogram,
    plot_scatter,
    plot_boxplot,
    plot_bar_chart,
    plot_correlation_heatmap
)
from .models import (
    train_naive_bayes,
    train_gradient_boosting_classifier,
    baseline_mean_regressor,
    train_linear_regression,
    train_random_forest_classifier,
    tune_random_forest_classifier,
    train_random_forest_regressor,
    tune_random_forest_regressor,
    train_svr,
    tune_svr,
    train_xgboost_classifier,
    train_xgboost_regressor,
    stacking_classifier,
    stacking_regressor
)
from .evaluation import evaluate_classification, evaluate_regression
