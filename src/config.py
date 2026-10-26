"""Central configuration for the Real Estate Investment Advisor project."""

# Import pathlib for OS-independent file paths
from pathlib import Path

# Project root directory (one level above the src folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- paths
# Raw dataset provided with the project brief
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "india_housing_prices.csv"
# Cleaned and feature-engineered dataset (compressed CSV keeps the repo small)
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "housing_processed.csv.gz"
# Pre-aggregated tables used by the Streamlit dashboard
CITY_SUMMARY_PATH = BASE_DIR / "data" / "processed" / "city_summary.csv"
# Average price metrics by year of construction (used for trend charts)
YEAR_TREND_PATH = BASE_DIR / "data" / "processed" / "year_trend.csv"
# Folder that stores trained models and metadata
MODELS_DIR = BASE_DIR / "models"
# Trained classification pipeline (Good Investment)
CLASSIFIER_PATH = MODELS_DIR / "classifier_pipeline.joblib"
# Trained regression pipeline (Future price after 5 years)
REGRESSOR_PATH = MODELS_DIR / "regressor_pipeline.joblib"
# Market benchmark tables needed to engineer features for new properties
REFERENCE_PATH = MODELS_DIR / "reference_tables.json"
# Metrics, feature importance and other metadata shown in the app
METADATA_PATH = MODELS_DIR / "model_metadata.json"
# Folder for saved figures
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# ---------------------------------------------------------------- MLflow
# SQLite backend is required for the MLflow model registry
MLFLOW_TRACKING_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
# Folder where MLflow stores run artifacts
MLFLOW_ARTIFACT_DIR = BASE_DIR / "mlartifacts"
# Experiment name for the classification runs
MLFLOW_CLF_EXPERIMENT = "RealEstate_GoodInvestment_Classification"
# Experiment name for the regression runs
MLFLOW_REG_EXPERIMENT = "RealEstate_FuturePrice_Regression"
# Registry name for the champion classifier
CLF_REGISTRY_NAME = "RealEstate_GoodInvestment_Classifier"
# Registry name for the champion regressor
REG_REGISTRY_NAME = "RealEstate_FuturePrice_Regressor"

# ---------------------------------------------------------------- modelling
# Seed used everywhere for reproducibility
RANDOM_STATE = 42
# Share of data held out for testing
TEST_SIZE = 0.20
# Reference year used to compute property age (matches Age_of_Property in the data)
CURRENT_YEAR = 2025
# Forecast horizon in years
HORIZON_YEARS = 5
# Minimum points (out of 4) for a property to be labelled a Good Investment
GOOD_INVESTMENT_MIN_SCORE = 3
# Fixed-rate benchmark growth used as a naive baseline (brief example: r = 8%)
BASELINE_GROWTH_RATE = 0.08

# ---------------------------------------------------------------- growth-rate assumptions
# Base annual appreciation for any property
GROWTH_BASE = 0.045
# Extra appreciation for the most in-demand cities (scaled by demand index 0-1)
GROWTH_CITY_WEIGHT = 0.030
# Extra appreciation for the best-connected / best-equipped localities (scaled by infra 0-1)
GROWTH_INFRA_WEIGHT = 0.020
# Extra appreciation for under-construction projects (value unlocks on completion)
GROWTH_UNDER_CONSTRUCTION = 0.008
# Annual growth lost per year of age beyond 10 years
GROWTH_AGE_PENALTY = 0.0005
# Standard deviation of random market noise added to realised growth
GROWTH_NOISE_STD = 0.004
# Lower hard limit keeping growth rates realistic
GROWTH_MIN = 0.03
# Upper hard limit keeping growth rates realistic
GROWTH_MAX = 0.12

# ---------------------------------------------------------------- infrastructure score weights (sum = 1)
INFRA_WEIGHTS = {
    "schools": 0.25,  # nearby schools
    "hospitals": 0.25,  # nearby hospitals
    "transport": 0.25,  # public transport accessibility
    "parking": 0.10,  # parking availability
    "security": 0.05,  # security availability
    "amenities": 0.10,  # in-society amenities
}
# Amenities that can be selected for a property
AMENITY_LIST = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]
# Ordinal mapping for public transport accessibility
TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}
