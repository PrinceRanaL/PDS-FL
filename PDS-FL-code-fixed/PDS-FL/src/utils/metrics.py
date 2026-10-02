import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def bias_variance_gap(train_metrics: dict, test_metrics: dict) -> dict:
    """Simple train/test gap used as the bias-variance indicator in the
    manuscript's Phase-1 model-selection discussion."""
    return {
        f"gap_{k}": train_metrics[k] - test_metrics[k]
        for k in train_metrics
    }
