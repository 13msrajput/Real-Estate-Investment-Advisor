"""Model factories, evaluation helpers, tuning and artifact persistence."""

# Type hints for readable function signatures
from __future__ import annotations

# JSON for metadata files
import json

# Numerical helpers
import numpy as np

# Tabular data handling
import pandas as pd

# Model persistence
import joblib

# Gradient boosting library
from xgboost import XGBClassifier, XGBRegressor

# Scikit-learn building blocks
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project configuration and logger
from src import config as cfg
from src.logger import get_logger

# Module-level logger
logger = get_logger(__name__)

# ---------------------------------------------------------------------------------------------
# Feature lists used by every model
# ---------------------------------------------------------------------------------------------
# Numeric model inputs (all engineered in src/features.py)
NUMERIC_FEATURES = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt_Capped",
    "Age_of_Property",
    "Total_Floors",
    "Floor_Ratio",
    "Size_per_BHK",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "School_Density_Score",
    "Transport_Score",
    "Parking_Flag",
    "Security_Flag",
    "Amenity_Count",
    "Has_Gym",
    "Has_Pool",
    "Has_Clubhouse",
    "Has_Garden",
    "Has_Playground",
    "Infra_Score",
    "PPS_vs_City_Median",
    "Price_vs_City_Median",
    "Is_Under_Construction",
    "City_Demand_Index",
]
# Categorical model inputs (one-hot encoded)
CATEGORICAL_FEATURES = [
    "City",
    "Property_Type",
    "Furnished_Status",
    "Facing",
    "Owner_Type",
]
# All model inputs in a single list
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Columns that carry price information (removed in the leakage ablation experiment)
PRICE_FEATURES = [
    "Price_in_Lakhs",
    "Price_per_SqFt_Capped",
    "PPS_vs_City_Median",
    "Price_vs_City_Median",
]


# ---------------------------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------------------------
def build_preprocessor(
    numeric: list[str] | None = None, categorical: list[str] | None = None
) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns."""
    # Default to the full feature lists
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    # Standardise numbers, encode categories (unknown categories become all-zero rows instead of errors)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        # Drop any column not listed above
        remainder="drop",
    )


def make_pipeline(
    model, numeric: list[str] | None = None, categorical: list[str] | None = None
) -> Pipeline:
    """Chain the preprocessor and a model into one deployable object."""
    # Preprocessing + estimator in a single pipeline
    return Pipeline(
        [("preprocess", build_preprocessor(numeric, categorical)), ("model", model)]
    )


def classification_models(pos_weight: float = 1.0) -> dict:
    """Candidate classifiers requested in the brief."""
    return {
        # Linear baseline
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=cfg.RANDOM_STATE
        ),
        # Bagged trees
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=5,
            max_samples=0.6,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        ),
        # Gradient boosting
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        ),
    }


def regression_models() -> dict:
    """Candidate regressors requested in the brief."""
    return {
        # Linear baseline
        "Linear Regression": LinearRegression(),
        # Bagged trees
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=5,
            max_samples=0.6,
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        ),
        # Gradient boosting
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        ),
    }


# Hyper-parameter search space for XGBoost (shared by both tasks)
XGB_SEARCH_SPACE = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [4, 6, 8],
    "model__learning_rate": [0.05, 0.1, 0.15],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 5, 10],
}


def tune_xgboost(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    n_iter: int = 6,
    sample_size: int = 40_000,
) -> tuple[dict, float]:
    """Randomised search on a subsample to keep tuning fast; returns the best parameters and CV score."""
    try:
        # Use a reproducible subsample so tuning stays quick on a single CPU
        idx = X.sample(n=min(sample_size, len(X)), random_state=cfg.RANDOM_STATE).index
        # Randomised search with 3-fold cross validation
        search = RandomizedSearchCV(
            pipeline,
            XGB_SEARCH_SPACE,
            n_iter=n_iter,
            scoring=scoring,
            cv=3,
            random_state=cfg.RANDOM_STATE,
            n_jobs=1,
            refit=False,
        )
        # Run the search
        search.fit(X.loc[idx], y.loc[idx])
        # Strip the pipeline prefix so parameters can be applied directly to the estimator
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        # Log the winner
        logger.info(
            "Best XGBoost parameters (%s=%.4f): %s", scoring, search.best_score_, best
        )
        # Return parameters and score
        return best, float(search.best_score_)
    except (
        Exception
    ) as exc:  # tuning is optional, so fall back to defaults instead of failing the run
        logger.warning(
            "Hyper-parameter tuning failed (%s); using default parameters.", exc
        )
        return {}, float("nan")


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
def evaluate_classifier(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
) -> dict:
    """Return accuracy, precision, recall, F1, ROC-AUC and the confusion matrix."""
    # Probability of the positive class
    proba = pipeline.predict_proba(X)[:, 1]
    # Hard predictions at the chosen threshold
    pred = (proba >= threshold).astype(int)
    # Collect the metrics
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def evaluate_regressor(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, R2 and MAPE."""
    # Root mean squared error (same unit as the target, in lakhs)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Mean absolute percentage error with a safe denominator
    mape = float(
        np.mean(
            np.abs((np.asarray(y_true) - y_pred) / np.maximum(np.asarray(y_true), 1e-9))
        )
        * 100
    )
    # Collect the metrics
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


def get_feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Importance per original feature (one-hot columns are summed back to their parent column)."""
    # Final estimator and the preprocessing step
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    # Tree models expose feature_importances_, linear models expose coefficients
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        raise ValueError("The model does not expose feature importance.")
    # Remove the 'num__' / 'cat__' prefixes
    clean = [n.split("__", 1)[1] for n in names]
    # Map one-hot columns (e.g. City_Pune) back to the parent column (City)
    parents = []
    for name in clean:
        # Find the categorical parent whose name starts the encoded column
        parent = next(
            (c for c in CATEGORICAL_FEATURES if name.startswith(c + "_")), name
        )
        parents.append(parent)
    # Sum importance per parent and normalise to 100%
    table = (
        pd.DataFrame({"Feature": parents, "Importance": values})
        .groupby("Feature", as_index=False)
        .sum()
    )
    table["Importance"] = 100 * table["Importance"] / table["Importance"].sum()
    # Return the strongest features first
    return (
        table.sort_values("Importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------------
def save_json(obj: dict, path) -> None:
    """Write a dictionary to disk as pretty JSON."""
    try:
        # Make sure the folder exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write the file
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=float)
    except OSError as exc:
        # Surface a clear message for disk problems
        raise OSError(f"Could not write {path}: {exc}") from exc


def load_json(path) -> dict:
    """Read a JSON file into a dictionary."""
    try:
        # Read and parse the file
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        # Tell the user how to create the missing artifact
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.train` first."
        ) from exc


def save_model(pipeline: Pipeline, path) -> None:
    """Persist a fitted pipeline with joblib compression."""
    try:
        # Make sure the folder exists
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise the model
        joblib.dump(pipeline, path, compress=3)
        # Log the size so oversized artifacts are noticed
        logger.info("Saved %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    except OSError as exc:
        # Surface a clear message for disk problems
        raise OSError(f"Could not save model to {path}: {exc}") from exc


def load_model(path) -> Pipeline:
    """Load a fitted pipeline from disk."""
    try:
        # Deserialise the model
        return joblib.load(path)
    except FileNotFoundError as exc:
        # Tell the user how to create the missing artifact
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.train` first."
        ) from exc
