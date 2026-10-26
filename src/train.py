"""End-to-end training pipeline: clean -> engineer -> train -> evaluate -> track (MLflow) -> save.

Run from the project root:
    python -m src.train                 # full run with tuning and MLflow tracking
    python -m src.train --skip-tuning   # faster run with default XGBoost parameters
    python -m src.train --no-mlflow     # disable experiment tracking
"""

# Type hints for readable function signatures
from __future__ import annotations

# Command line parsing
import argparse

# Timestamps for metadata
from datetime import datetime, timezone

# Numerical helpers
import numpy as np

# Tabular data handling
import pandas as pd

# Headless plotting backend (no display needed)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Train / test splitting
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split

# Project modules
from src import config as cfg
from src import modeling as mdl
from src.data_loader import clean_data, load_raw_data
from src.features import build_training_frame
from src.logger import get_logger

# Module-level logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------------------------
# MLflow helpers (every call is wrapped so tracking problems never crash training)
# ---------------------------------------------------------------------------------------------
def init_mlflow(experiment: str):
    """Point MLflow at the local SQLite store and select (or create) the experiment; returns the module or None."""
    try:
        # Import lazily so the project still runs when MLflow is not installed
        import mlflow

        # Configure the tracking backend
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        # Create the experiment with an explicit artifact folder on first use
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(
                experiment, artifact_location=cfg.MLFLOW_ARTIFACT_DIR.as_uri()
            )
        # Activate the experiment
        mlflow.set_experiment(experiment)
        # Hand back the configured module
        return mlflow
    except Exception as exc:  # tracking is optional
        logger.warning("MLflow unavailable (%s). Continuing without tracking.", exc)
        return None


def log_run(
    mlflow,
    run_name: str,
    params: dict,
    metrics: dict,
    pipeline,
    X_example: pd.DataFrame,
    tags: dict,
    figures: list | None = None,
) -> str | None:
    """Log one model run (parameters, metrics, figures, fitted pipeline) and return its run id."""
    # Skip silently when tracking is disabled
    if mlflow is None:
        return None
    try:
        # Open a new run
        with mlflow.start_run(run_name=run_name) as run:
            # Log the hyper-parameters (strings keep the UI tidy)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            # Log numeric metrics only (matrices are stored as figures)
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and np.isfinite(v)
                }
            )
            # Tag the run for easy filtering
            mlflow.set_tags(tags)
            # Attach any saved figures
            for fig_path in figures or []:
                mlflow.log_artifact(str(fig_path), artifact_path="figures")
            # Infer the input/output schema from a few rows
            from mlflow.models import infer_signature

            sample = X_example.head(5)
            signature = infer_signature(sample, pipeline.predict(sample))
            # Store the fitted pipeline as an MLflow model (cloudpickle avoids the skops trusted-type restriction)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                serialization_format="cloudpickle",
            )
            # Return the identifier so the champion can be registered later
            return run.info.run_id
    except Exception as exc:  # never let tracking break training
        logger.warning("Could not log run '%s' to MLflow: %s", run_name, exc)
        return None


def register_champion(mlflow, run_id: str | None, registry_name: str) -> None:
    """Register the winning run in the MLflow model registry and tag it with the 'champion' alias."""
    # Nothing to register without a run
    if mlflow is None or run_id is None:
        return
    try:
        # Create a new registry version from the run's model artifact
        version = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
        # Point the 'champion' alias at that version
        mlflow.MlflowClient().set_registered_model_alias(
            registry_name, "champion", version.version
        )
        # Confirm in the log
        logger.info("Registered %s v%s as champion", registry_name, version.version)
    except Exception as exc:  # registry problems are non-fatal
        logger.warning("Could not register %s: %s", registry_name, exc)


# ---------------------------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------------------------
def save_fig(fig, name: str):
    """Save a matplotlib figure into reports/figures and return its path."""
    # Make sure the folder exists
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Build the output path
    path = cfg.FIGURES_DIR / f"{name}.png"
    # Save with tight margins
    fig.savefig(path, dpi=130, bbox_inches="tight")
    # Free memory
    plt.close(fig)
    # Return the location
    return path


def plot_classification(pipe, X_test, y_test, importance: pd.DataFrame) -> list:
    """Confusion matrix, ROC curve and feature importance of the champion classifier."""
    # Paths of the created figures
    paths = []
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, display_labels=["Not Good", "Good"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix - Champion Classifier")
    paths.append(save_fig(fig, "clf_confusion_matrix"))
    # ROC curve
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve - Champion Classifier",
    )
    ax.legend()
    paths.append(save_fig(fig, "clf_roc_curve"))
    # Feature importance
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        importance["Feature"][::-1], importance["Importance"][::-1], color="#2a6f97"
    )
    ax.set(xlabel="Importance (%)", title="Top Features - Good Investment Classifier")
    paths.append(save_fig(fig, "clf_feature_importance"))
    # Return all paths
    return paths


def plot_regression(y_test, pred, importance: pd.DataFrame) -> list:
    """Actual vs predicted, residual histogram and feature importance of the champion regressor."""
    # Paths of the created figures
    paths = []
    # Random sample keeps scatter plots light
    idx = np.random.default_rng(cfg.RANDOM_STATE).choice(
        len(y_test), size=min(4000, len(y_test)), replace=False
    )
    # Actual vs predicted scatter
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(np.asarray(y_test)[idx], pred[idx], s=4, alpha=0.4, color="#2a6f97")
    lims = [float(np.min(y_test)), float(np.max(y_test))]
    ax.plot(lims, lims, "--", color="red")
    ax.set(
        xlabel="Actual future price (Lakhs)",
        ylabel="Predicted (Lakhs)",
        title="Actual vs Predicted - 5-Year Price",
    )
    paths.append(save_fig(fig, "reg_actual_vs_predicted"))
    # Residual histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(y_test) - pred, bins=60, color="#468faf")
    ax.set(xlabel="Residual (Lakhs)", ylabel="Count", title="Residual Distribution")
    paths.append(save_fig(fig, "reg_residuals"))
    # Feature importance
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        importance["Feature"][::-1], importance["Importance"][::-1], color="#2a6f97"
    )
    ax.set(xlabel="Importance (%)", title="Top Features - Future Price Regressor")
    paths.append(save_fig(fig, "reg_feature_importance"))
    # Return all paths
    return paths


# ---------------------------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------------------------
def prepare_data() -> tuple[pd.DataFrame, dict, dict]:
    """Load, clean and engineer the dataset, then persist processed data and benchmark tables."""
    # Load and validate the raw file
    raw = load_raw_data()
    # Clean it and collect the audit report
    clean, report = clean_data(raw)
    # Engineer features and targets
    df, ref = build_training_frame(clean)

    # Persist the processed dataset (deliverable: cleaned and processed dataset)
    cfg.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.PROCESSED_DATA_PATH, index=False, compression="gzip")
    # Persist benchmark tables used by the app for new properties
    mdl.save_json(ref, cfg.REFERENCE_PATH)

    # City-level summary used by dashboard heatmaps
    city = (
        df.groupby(["State", "City"])
        .agg(
            Properties=("ID", "count"),
            Avg_Price_Lakhs=("Price_in_Lakhs", "mean"),
            Avg_Price_per_SqFt=("Price_per_SqFt_INR", "mean"),
            Avg_Infra_Score=("Infra_Score", "mean"),
            Avg_Growth_Rate=("Growth_Rate", "mean"),
            Good_Investment_Rate=("Good_Investment", "mean"),
            Avg_Future_Price_Lakhs=("Future_Price_5Y", "mean"),
        )
        .reset_index()
        .round(4)
    )
    city.to_csv(cfg.CITY_SUMMARY_PATH, index=False)
    # Trend by construction year used by the dashboard line chart
    trend = (
        df.groupby("Year_Built")
        .agg(
            Avg_Price_Lakhs=("Price_in_Lakhs", "mean"),
            Avg_Price_per_SqFt=("Price_per_SqFt_INR", "mean"),
            Properties=("ID", "count"),
        )
        .reset_index()
        .round(3)
    )
    trend.to_csv(cfg.YEAR_TREND_PATH, index=False)
    # Return everything the training stage needs
    return df, ref, report


# ---------------------------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------------------------
def run_training(skip_tuning: bool = False, use_mlflow: bool = True) -> dict:
    """Train, compare and persist the classification and regression models."""
    # Data preparation
    df, ref, report = prepare_data()
    # Feature matrix and both targets
    X = df[mdl.ALL_FEATURES]
    y_clf = df["Good_Investment"]
    y_reg = df["Future_Price_5Y"]
    # One split shared by both tasks (stratified on the classification label)
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X,
        y_clf,
        y_reg,
        test_size=cfg.TEST_SIZE,
        stratify=y_clf,
        random_state=cfg.RANDOM_STATE,
    )
    logger.info(
        "Train rows: %s | Test rows: %s", f"{len(X_train):,}", f"{len(X_test):,}"
    )

    # Metadata that will be shown in the app and README
    meta: dict = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "rows": int(len(df)),
            "good_investment_share": float(y_clf.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "cleaning_report": report,
        "features": {
            "numeric": mdl.NUMERIC_FEATURES,
            "categorical": mdl.CATEGORICAL_FEATURES,
        },
    }

    # ============================================================ classification
    logger.info("=== Classification: Good Investment ===")
    # Tracking handle for the classification experiment
    mlf = init_mlflow(cfg.MLFLOW_CLF_EXPERIMENT) if use_mlflow else None
    # Class imbalance ratio used by XGBoost
    pos_weight = float((yc_train == 0).sum() / max((yc_train == 1).sum(), 1))
    # Tune XGBoost once, then reuse the parameters
    xgb_params: dict = {}
    if not skip_tuning:
        xgb_params, _ = mdl.tune_xgboost(
            mdl.make_pipeline(mdl.classification_models(pos_weight)["XGBoost"]),
            X_train,
            yc_train,
            scoring="f1",
        )
    # Containers for results
    clf_results, clf_pipes, clf_runs = {}, {}, {}
    # Fit every candidate
    for name, model in mdl.classification_models(pos_weight).items():
        try:
            # Apply tuned parameters to XGBoost
            if name == "XGBoost" and xgb_params:
                model.set_params(**xgb_params)
            # Build and fit the pipeline
            pipe = mdl.make_pipeline(model).fit(X_train, yc_train)
            # Evaluate on unseen data and on the training data (overfitting check)
            test_m = mdl.evaluate_classifier(pipe, X_test, yc_test)
            sample_idx = X_train.sample(30_000, random_state=cfg.RANDOM_STATE).index
            train_m = mdl.evaluate_classifier(
                pipe, X_train.loc[sample_idx], yc_train.loc[sample_idx]
            )
            test_m["train_f1"] = train_m["f1"]
            # Store results
            clf_results[name], clf_pipes[name] = test_m, pipe
            logger.info(
                "%s | acc %.4f | F1 %.4f | AUC %.4f",
                name,
                test_m["accuracy"],
                test_m["f1"],
                test_m["roc_auc"],
            )
            # Log to MLflow
            clf_runs[name] = log_run(
                mlf,
                name,
                model.get_params(),
                test_m,
                pipe,
                X_test,
                {"task": "classification", "target": "Good_Investment"},
            )
        except Exception as exc:  # one failing model must not stop the others
            logger.error("Model '%s' failed: %s", name, exc)
    # Fail loudly if nothing trained
    if not clf_pipes:
        raise RuntimeError("No classification model could be trained.")

    # Pick the champion by F1 (balances precision and recall on the minority class)
    best_clf = max(clf_results, key=lambda n: clf_results[n]["f1"])
    logger.info("Champion classifier: %s", best_clf)
    # Importance, figures and registry entry for the champion
    clf_imp = mdl.get_feature_importance(clf_pipes[best_clf])
    clf_figs = plot_classification(clf_pipes[best_clf], X_test, yc_test, clf_imp)
    if mlf is not None:
        # Attach figures to a dedicated run so they are visible in the MLflow UI
        log_run(
            mlf,
            f"{best_clf} - champion report",
            {},
            clf_results[best_clf],
            clf_pipes[best_clf],
            X_test,
            {"task": "classification", "stage": "champion"},
            clf_figs,
        )
    register_champion(mlf, clf_runs.get(best_clf), cfg.CLF_REGISTRY_NAME)

    # Leakage ablation: remove every price column to see whether property attributes alone carry signal
    keep_num = [c for c in mdl.NUMERIC_FEATURES if c not in mdl.PRICE_FEATURES]
    abl_model = mdl.classification_models(pos_weight)["XGBoost"]
    abl_pipe = mdl.make_pipeline(abl_model, numeric=keep_num).fit(X_train, yc_train)
    abl_metrics = mdl.evaluate_classifier(abl_pipe, X_test, yc_test)
    logger.info(
        "Ablation (no price features) | acc %.4f | F1 %.4f | AUC %.4f",
        abl_metrics["accuracy"],
        abl_metrics["f1"],
        abl_metrics["roc_auc"],
    )

    # Save champion classifier and its metadata
    mdl.save_model(clf_pipes[best_clf], cfg.CLASSIFIER_PATH)
    meta["classification"] = {
        "results": clf_results,
        "best_model": best_clf,
        "majority_baseline_accuracy": float(1 - yc_test.mean()),
        "ablation_no_price_features": abl_metrics,
        "importance": clf_imp.round(3).to_dict("records"),
        "threshold": 0.5,
    }

    # ============================================================ regression
    logger.info("=== Regression: Future price after 5 years ===")
    # Tracking handle for the regression experiment
    mlf = init_mlflow(cfg.MLFLOW_REG_EXPERIMENT) if use_mlflow else None
    # Tune XGBoost regressor
    xgb_params = {}
    if not skip_tuning:
        xgb_params, _ = mdl.tune_xgboost(
            mdl.make_pipeline(mdl.regression_models()["XGBoost"]),
            X_train,
            yr_train,
            scoring="neg_root_mean_squared_error",
        )
    # Containers for results
    reg_results, reg_pipes, reg_runs = {}, {}, {}
    # Fit every candidate
    for name, model in mdl.regression_models().items():
        try:
            # Apply tuned parameters to XGBoost
            if name == "XGBoost" and xgb_params:
                model.set_params(**xgb_params)
            # Build and fit the pipeline
            pipe = mdl.make_pipeline(model).fit(X_train, yr_train)
            # Evaluate on unseen data
            test_m = mdl.evaluate_regressor(yr_test, pipe.predict(X_test))
            # Store results
            reg_results[name], reg_pipes[name] = test_m, pipe
            logger.info(
                "%s | RMSE %.3f | MAE %.3f | R2 %.4f",
                name,
                test_m["rmse"],
                test_m["mae"],
                test_m["r2"],
            )
            # Log to MLflow
            reg_runs[name] = log_run(
                mlf,
                name,
                model.get_params(),
                test_m,
                pipe,
                X_test,
                {"task": "regression", "target": "Future_Price_5Y"},
            )
        except Exception as exc:  # one failing model must not stop the others
            logger.error("Model '%s' failed: %s", name, exc)
    # Fail loudly if nothing trained
    if not reg_pipes:
        raise RuntimeError("No regression model could be trained.")

    # Pick the champion by lowest RMSE
    best_reg = min(reg_results, key=lambda n: reg_results[n]["rmse"])
    logger.info("Champion regressor: %s", best_reg)
    # Champion predictions on the test set
    reg_pred = reg_pipes[best_reg].predict(X_test)
    # Importance and figures
    reg_imp = mdl.get_feature_importance(reg_pipes[best_reg])
    reg_figs = plot_regression(yr_test, reg_pred, reg_imp)
    if mlf is not None:
        log_run(
            mlf,
            f"{best_reg} - champion report",
            {},
            reg_results[best_reg],
            reg_pipes[best_reg],
            X_test,
            {"task": "regression", "stage": "champion"},
            reg_figs,
        )
    register_champion(mlf, reg_runs.get(best_reg), cfg.REG_REGISTRY_NAME)

    # Naive benchmark: every property grows at a fixed 8% a year
    fixed_pred = (
        X_test["Price_in_Lakhs"].to_numpy()
        * (1 + cfg.BASELINE_GROWTH_RATE) ** cfg.HORIZON_YEARS
    )
    fixed_metrics = mdl.evaluate_regressor(yr_test, fixed_pred)
    # Prediction interval: 5th and 95th percentile of actual / predicted on the test set
    ratio = np.asarray(yr_test) / np.maximum(reg_pred, 1e-9)
    interval = {
        "ratio_low": float(np.quantile(ratio, 0.05)),
        "ratio_high": float(np.quantile(ratio, 0.95)),
    }

    # Save champion regressor and its metadata
    mdl.save_model(reg_pipes[best_reg], cfg.REGRESSOR_PATH)
    meta["regression"] = {
        "results": reg_results,
        "best_model": best_reg,
        "fixed_rate_baseline": fixed_metrics,
        "interval_90pct": interval,
        "importance": reg_imp.round(3).to_dict("records"),
    }
    # Assumptions are stored so the app can display them
    meta["assumptions"] = {
        "horizon_years": cfg.HORIZON_YEARS,
        "good_investment_min_score": cfg.GOOD_INVESTMENT_MIN_SCORE,
        "growth_base": cfg.GROWTH_BASE,
        "growth_min": cfg.GROWTH_MIN,
        "growth_max": cfg.GROWTH_MAX,
    }
    # Write metadata
    mdl.save_json(meta, cfg.METADATA_PATH)
    logger.info("Training complete. Artifacts saved in %s", cfg.MODELS_DIR)
    # Return the metadata for programmatic use
    return meta


def main() -> None:
    """Command line entry point."""
    # Define the command line options
    parser = argparse.ArgumentParser(
        description="Train the Real Estate Investment Advisor models."
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip XGBoost hyper-parameter search.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow tracking."
    )
    args = parser.parse_args()
    try:
        # Run the pipeline
        run_training(skip_tuning=args.skip_tuning, use_mlflow=not args.no_mlflow)
    except Exception as exc:  # log the failure and exit with a non-zero code for CI use
        logger.exception("Training failed: %s", exc)
        raise SystemExit(1) from exc


# Run only when executed as a script
if __name__ == "__main__":
    main()
