"""Factory functions for baseline multi-output regressors."""

from .linear_regression import build_model as build_linear_regression
from .random_forest import build_model as build_random_forest
from .gradient_boosting import build_model as build_gradient_boosting
from .knn import build_model as build_knn
from .ridge import build_model as build_ridge
from .decision_tree import build_model as build_decision_tree

__all__ = [
    "build_linear_regression",
    "build_random_forest",
    "build_gradient_boosting",
    "build_knn",
    "build_ridge",
    "build_decision_tree",
]
