# src/__init__.py

# Import functions from the data_loader module.
from .data_loader import load_data, inspect_data

# Import functions from the preprocessing module.
from .preprocessing import (
    handle_missing_values,
    encode_categorical,
    scale_features,
    remove_outliers
)

# Import functions from the visualization module.
from .visualization import (
    plot_histogram,
    plot_scatter,
    plot_boxplot,
    plot_bar_chart
)

# Import functions from the models module.
from .models import (
    train_naive_bayes,
    train_linear_regression,
    train_random_forest_classifier,
    train_svc_classifier,
    train_random_forest_regressor,
    train_svr
)

# Import functions from the evaluation module.
from .evaluation import evaluate_classification, evaluate_regression

# You can add additional package-level initialization code here if needed.
