# ============================================================
# MODEL TUNING — Insurance Premium Prediction ML System
# ============================================================

import numpy as np
import logging

from typing import Dict, List, Tuple

from sklearn.base              import BaseEstimator
from sklearn.compose           import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection   import RandomizedSearchCV, KFold

from sklearn.linear_model  import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import (RandomForestRegressor, GradientBoostingRegressor,
                                   AdaBoostRegressor, ExtraTreesRegressor)
from xgboost                import XGBRegressor

from sklearn.pipeline      import Pipeline

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from src.config        import RANDOM_STATE, N_JOBS, CV_FOLDS, RANDOM_SEARCH_ITERS, SELECT_K
from src.preprocessing import safe_k

logger = logging.getLogger(__name__)


# ============================================================
# HELPER — compute safe n_iter for RandomizedSearchCV
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# MODEL GRIDS
# ============================================================

# Linear / distance models  →  need scaled input
scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "Ridge": (
        Ridge(random_state=RANDOM_STATE),
        {
            "regressor__alpha":    [0.01, 0.1, 1.0, 10.0, 50.0],
            "regressor__solver":   ["auto", "svd", "cholesky"],
        }
    ),

    "Lasso": (
        Lasso(random_state=RANDOM_STATE, max_iter=5000),
        {
            "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 5.0],
        }
    ),

    "ElasticNet": (
        ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
        {
            "regressor__alpha":    [0.001, 0.01, 0.1, 1.0],
            "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    ),

    "KNN": (
        KNeighborsRegressor(),
        {
            "regressor__n_neighbors": [3, 5, 7, 9, 15],
            "regressor__weights":     ["uniform", "distance"],
        }
    ),

    "SGD": (
        SGDRegressor(random_state=RANDOM_STATE, max_iter=3000, tol=1e-3),
        {
            "regressor__loss":     ["squared_error", "huber"],
            "regressor__alpha":    [1e-4, 1e-3, 1e-2],
            "regressor__penalty":  ["l2", "elasticnet"],
        }
    ),
}

# Tree-based models  →  work on raw / clipped input
unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "DecisionTree": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {
            "regressor__max_depth":        [3, 5, 10, 20, None],
            "regressor__min_samples_leaf": [1, 2, 4, 8],
        }
    ),

    "RandomForest": (
        RandomForestRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE),
        {
            "regressor__n_estimators":     [100, 150, 200],
            "regressor__max_depth":        [6, 10, 20],
            "regressor__min_samples_leaf": [1, 2, 4],
        }
    ),

    "ExtraTrees": (
        ExtraTreesRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE),
        {
            "regressor__n_estimators": [100, 150, 200],
            "regressor__max_depth":    [10, 20],
        }
    ),

    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "regressor__n_estimators":  [100, 150],
            "regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__max_depth":     [2, 3, 5],
            "regressor__subsample":     [0.8, 1.0],
        }
    ),

    "AdaBoost": (
        AdaBoostRegressor(random_state=RANDOM_STATE),
        {
            "regressor__n_estimators":  [50, 100, 200],
            "regressor__learning_rate": [0.01, 0.1, 1.0],
        }
    ),

    "XGBoost": (
        XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"),
        {
            "regressor__n_estimators":  [100, 150, 200],
            "regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__max_depth":     [2, 3, 5],
            "regressor__subsample":     [0.8, 1.0],
        }
    ),
}

if LGBMRegressor is not None:
    unscaled_models["LightGBM"] = (
        LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        {
            "regressor__n_estimators":  [100, 150, 200],
            "regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__max_depth":     [-1, 5, 10],
            "regressor__num_leaves":    [15, 31, 63],
        }
    )

if CatBoostRegressor is not None:
    unscaled_models["CatBoost"] = (
        CatBoostRegressor(
            verbose=0,
            random_state=RANDOM_STATE,
            allow_writing_files=False,   # prevents catboost_info dir creation (fixes Windows path/permission crash)
        ),
        {
            "regressor__iterations":    [100, 150, 200],
            "regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__depth":         [4, 6, 8],
        }
    )


# ============================================================
# TUNE MODELS
# ============================================================

def tune_models(
    models:         Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor:   ColumnTransformer,
    cat_indices:    List[int],
    X_train:        "pd.DataFrame",
    y_train:        "pd.Series",
    selector_k:     int  = SELECT_K
) -> Dict[str, Pipeline]:
    """
    For each model:
      preprocessor → SelectKBest → regressor
    Tuned with RandomizedSearchCV (scoring = neg_root_mean_squared_error).
    Returns dict of {model_name: best_pipeline}.
    """

    final_pipelines: Dict[str, Pipeline] = {}

    k_safe = safe_k(selector_k, preprocessor, X_train)
    logger.info("Selector k set to %d (requested %d)", k_safe, selector_k)

    for name, (reg, param_dist) in models.items():

        logger.info("Tuning: %s", name)

        steps = [
            ("preprocessor", preprocessor),
            # f_regression (vectorized ANOVA F-test) instead of mutual_info_regression:
            # mutual_info_regression uses a k-NN estimator that is roughly O(n^2) per
            # feature, which is fine for a ~1K-row dataset but far too slow to refit
            # inside every RandomizedSearchCV fold on a 100K+ row dataset like this one.
            ("selector",     SelectKBest(f_regression, k=k_safe)),
            ("regressor",    reg),
        ]

        pipe = Pipeline(steps)

        n_iter = _compute_n_iter(param_dist, RANDOM_SEARCH_ITERS)

        search = RandomizedSearchCV(
            pipe,
            param_distributions = param_dist,
            n_iter              = n_iter,
            scoring             = "neg_root_mean_squared_error",
            cv                  = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs              = N_JOBS,
            random_state        = RANDOM_STATE,
            verbose             = 0
        )

        search.fit(X_train, y_train)

        logger.info("%s best params: %s", name, search.best_params_)

        final_pipelines[name] = search.best_estimator_

    return final_pipelines


# ============================================================
# NEURAL NETWORK — trained separately (no CV search)
# ============================================================

def train_mlp_pipeline(X_train, y_train, preprocessor, cat_indices: List[int]):
    """
    MLPRegressor trained separately outside RandomizedSearchCV.
    Reason: MLP training time makes CV search impractical.
    """
    from sklearn.neural_network import MLPRegressor

    logger.info("Training Neural Network (MLP Regressor) ...")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    MLPRegressor(
            hidden_layer_sizes   = (64, 32),
            activation           = "relu",
            solver               = "adam",
            alpha                = 0.001,
            batch_size           = 64,
            learning_rate        = "adaptive",
            max_iter             = 300,
            early_stopping       = True,
            validation_fraction  = 0.15,
            n_iter_no_change     = 15,
            random_state         = RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)

    logger.info("MLP training done")

    return pipe
