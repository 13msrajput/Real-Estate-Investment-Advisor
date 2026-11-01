"""Inference service used by the Streamlit app (and usable from any Python script)."""

# Type hints for readable function signatures
from __future__ import annotations

# Tabular data handling
import pandas as pd

# Project modules
from src import config as cfg
from src import modeling as mdl
from src.features import engineer_features, investment_factors
from src.logger import get_logger

# Module-level logger
logger = get_logger(__name__)

# Fields the caller must provide for every property
REQUIRED_INPUTS = [
    "City",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]
# Allowed numeric ranges (min, max) used for input validation
NUMERIC_LIMITS = {
    "BHK": (1, 10),
    "Size_in_SqFt": (100, 20_000),
    "Price_in_Lakhs": (1, 10_000),
    "Year_Built": (1900, cfg.CURRENT_YEAR + 5),
    "Floor_No": (0, 200),
    "Total_Floors": (1, 200),
    "Nearby_Schools": (0, 10),
    "Nearby_Hospitals": (0, 10),
}


class AdvisorService:
    """Loads the trained artifacts once and turns raw property details into predictions."""

    def __init__(self) -> None:
        # Load both trained pipelines
        self.classifier = mdl.load_model(cfg.CLASSIFIER_PATH)
        self.regressor = mdl.load_model(cfg.REGRESSOR_PATH)
        # Load benchmark tables and model metadata
        self.reference = mdl.load_json(cfg.REFERENCE_PATH)
        self.metadata = mdl.load_json(cfg.METADATA_PATH)
        # Cities the benchmark tables know about (used to warn about unseen cities)
        self.known_cities = set(self.reference["city_median_pps"])

    # ------------------------------------------------------------------ validation
    def validate_inputs(self, inputs: dict) -> dict:
        """Check that all fields exist and are in range; return a cleaned copy."""
        # Find missing fields
        missing = [
            k for k in REQUIRED_INPUTS if k not in inputs or inputs[k] in (None, "")
        ]
        if missing:
            raise ValueError(f"Missing property details: {', '.join(missing)}")
        # Copy so the caller's dictionary is not changed
        clean = dict(inputs)
        # Validate every numeric field
        for field, (low, high) in NUMERIC_LIMITS.items():
            try:
                # Convert to a number
                value = float(clean[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"'{field}' must be a number.") from exc
            # Reject out-of-range values
            if not low <= value <= high:
                raise ValueError(f"'{field}' must be between {low} and {high}.")
            # Store the numeric value
            clean[field] = value
        # A flat cannot be above the top floor, so cap it
        clean["Floor_No"] = min(clean["Floor_No"], clean["Total_Floors"])
        # Amenities may arrive as a list; convert to the comma-separated format of the dataset
        if isinstance(clean["Amenities"], (list, tuple, set)):
            clean["Amenities"] = ", ".join(clean["Amenities"])
        # Derived age
        clean["Age_of_Property"] = max(cfg.CURRENT_YEAR - clean["Year_Built"], 0)
        # Return the validated inputs
        return clean

    def build_features(self, inputs: dict) -> pd.DataFrame:
        """Validate raw inputs and engineer the exact features the models were trained on."""
        # Validate first
        clean = self.validate_inputs(inputs)
        # One-row frame
        frame = pd.DataFrame([clean])
        # Engineer features with the stored market benchmarks
        return engineer_features(frame, self.reference)

    # ------------------------------------------------------------------ prediction
    def predict(self, inputs: dict) -> dict:
        """Return the investment verdict, 5-year price forecast and supporting numbers."""
        try:
            # Engineered features for one property
            feats = self.build_features(inputs)
            # Model input matrix
            X = feats[mdl.ALL_FEATURES]
            # Probability that the property is a good investment
            proba = float(self.classifier.predict_proba(X)[0, 1])
            # Forecast price after five years
            future = float(self.regressor.predict(X)[0])
            # Current price in lakhs
            price = float(feats["Price_in_Lakhs"].iloc[0])
            # Prediction interval from test-set residuals
            band = self.metadata["regression"]["interval_90pct"]
            # Rule-based factor results (explains the verdict)
            factors = investment_factors(feats, self.reference).iloc[0].to_dict()
            # Assemble the response
            return {
                "good_investment": bool(
                    proba >= self.metadata["classification"]["threshold"]
                ),
                "probability": proba,
                "current_price": price,
                "future_price": future,
                "future_price_low": future * band["ratio_low"],
                "future_price_high": future * band["ratio_high"],
                "appreciation_pct": 100 * (future / price - 1),
                "implied_cagr_pct": 100
                * ((future / price) ** (1 / cfg.HORIZON_YEARS) - 1),
                "expected_growth_pct": 100
                * float(feats["Expected_Growth_Rate"].iloc[0]),
                "infra_score": float(feats["Infra_Score"].iloc[0]),
                "price_vs_city_median": float(feats["Price_vs_City_Median"].iloc[0]),
                "pps_vs_city_median": float(feats["PPS_vs_City_Median"].iloc[0]),
                "price_per_sqft": float(feats["Price_per_SqFt_INR"].iloc[0]),
                "factors": {k: bool(v) for k, v in factors.items()},
                "city_known": inputs["City"] in self.known_cities,
            }
        except ValueError:
            # Validation errors are meant for the user, so pass them through unchanged
            raise
        except Exception as exc:  # anything else is unexpected
            logger.exception("Prediction failed")
            raise RuntimeError(f"Prediction failed: {exc}") from exc
