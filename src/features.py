"""Feature engineering, market benchmarks and target creation.

Targets
-------
* Good_Investment   : binary label from a transparent, rule-based, four-factor score.
* Future_Price_5Y   : current price compounded for five years at a feature-based growth rate.

The dataset has no time dimension (one snapshot per property), so real historical appreciation cannot be
measured. The growth rate is therefore a documented, simulated assumption - see src/config.py.
"""

# Type hints for readable function signatures
from __future__ import annotations

# Numerical helpers
import numpy as np

# Tabular data handling
import pandas as pd

# Project configuration
from src import config as cfg

# Shared logger
from src.logger import get_logger

# Module-level logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------------------------
# Row-wise features (work for a full dataset or a single property entered in the app)
# ---------------------------------------------------------------------------------------------
def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features that depend only on the property's own columns."""
    # Work on a copy so the caller's frame is not modified
    out = df.copy()

    # Price per sq ft in rupees (lakhs -> rupees, divided by area)
    out["Price_per_SqFt_INR"] = out["Price_in_Lakhs"] * 100_000 / out["Size_in_SqFt"]
    # Ordinal public-transport score (Low=1, Medium=2, High=3)
    out["Transport_Score"] = (
        out["Public_Transport_Accessibility"]
        .map(cfg.TRANSPORT_MAP)
        .fillna(2)
        .astype(int)
    )
    # Binary flags for parking and security
    out["Parking_Flag"] = (
        out["Parking_Space"].astype(str).str.lower() == "yes"
    ).astype(int)
    out["Security_Flag"] = (out["Security"].astype(str).str.lower() == "yes").astype(
        int
    )
    # Binary flag for under-construction projects
    out["Is_Under_Construction"] = (
        out["Availability_Status"].astype(str) == "Under_Construction"
    ).astype(int)

    # One flag per amenity parsed from the comma-separated text column
    amenity_text = out["Amenities"].fillna("").astype(str)
    for amenity in cfg.AMENITY_LIST:
        # 1 when the amenity name appears in the text
        out[f"Has_{amenity}"] = amenity_text.str.contains(
            amenity, case=False, regex=False
        ).astype(int)
    # Total number of amenities
    out["Amenity_Count"] = out[[f"Has_{a}" for a in cfg.AMENITY_LIST]].sum(axis=1)

    # Position of the flat inside the building (0 = ground, 1 = top floor)
    out["Floor_Ratio"] = (out["Floor_No"] / out["Total_Floors"].clip(lower=1)).clip(
        0, 1
    )
    # Average room size, a proxy for spaciousness
    out["Size_per_BHK"] = out["Size_in_SqFt"] / out["BHK"].clip(lower=1)

    # Weighted 0-100 infrastructure score built from schools, hospitals, transport, parking, security, amenities
    w = cfg.INFRA_WEIGHTS
    infra = (
        w["schools"] * (out["Nearby_Schools"] / 10).clip(0, 1)
        + w["hospitals"] * (out["Nearby_Hospitals"] / 10).clip(0, 1)
        + w["transport"] * ((out["Transport_Score"] - 1) / 2)
        + w["parking"] * out["Parking_Flag"]
        + w["security"] * out["Security_Flag"]
        + w["amenities"] * (out["Amenity_Count"] / len(cfg.AMENITY_LIST))
    )
    # Scale to 0-100 for readability
    out["Infra_Score"] = (infra * 100).round(2)
    # Return the enriched frame
    return out


# ---------------------------------------------------------------------------------------------
# Market benchmark tables (fitted once on the full dataset, reused for every new property)
# ---------------------------------------------------------------------------------------------
def fit_reference_tables(df: pd.DataFrame) -> dict:
    """Compute market benchmarks (city medians, demand index, thresholds) from the engineered dataset."""
    # Median price per sq ft for each city
    city_pps = df.groupby("City")["Price_per_SqFt_INR"].median()
    # Median price for each city + property type combination
    city_type_price = df.groupby(["City", "Property_Type"])["Price_in_Lakhs"].median()
    # Average number of nearby schools per city (used for School_Density_Score)
    city_schools = df.groupby("City")["Nearby_Schools"].mean()
    # Demand index: percentile rank of the city's median price per sq ft (0 = cheapest, 1 = dearest)
    demand = city_pps.rank(pct=True)
    # Demand index re-scaled so the cheapest city is exactly 0 and the dearest exactly 1
    demand = (demand - demand.min()) / (demand.max() - demand.min())

    # Assemble the benchmark dictionary (JSON-serialisable)
    ref = {
        "city_median_pps": city_pps.round(2).to_dict(),
        "city_type_median_price": {
            f"{c}|{t}": round(float(v), 3) for (c, t), v in city_type_price.items()
        },
        "city_avg_schools": city_schools.round(3).to_dict(),
        "city_demand_index": demand.round(4).to_dict(),
        "global_median_pps": round(float(df["Price_per_SqFt_INR"].median()), 2),
        "global_median_price": round(float(df["Price_in_Lakhs"].median()), 3),
        "global_avg_schools": round(float(df["Nearby_Schools"].mean()), 3),
        "pps_clip_low": round(float(df["Price_per_SqFt_INR"].quantile(0.01)), 2),
        "pps_clip_high": round(float(df["Price_per_SqFt_INR"].quantile(0.99)), 2),
        # Infrastructure threshold: top 40% of properties count as "well equipped"
        "infra_threshold": round(float(df["Infra_Score"].quantile(0.60)), 2),
    }
    # Expected growth needs the tables above, so compute it now and store its median as a threshold
    expected_growth = compute_growth_rate(df, ref)
    # Properties at or above the median growth earn the growth point
    ref["growth_median"] = round(float(expected_growth.median()), 5)
    # Remember how many rows the benchmarks were built from
    ref["n_rows"] = int(len(df))
    # Log the key thresholds
    logger.info(
        "Reference tables fitted on %s rows (infra threshold %.1f, median growth %.2f%%)",
        f"{len(df):,}",
        ref["infra_threshold"],
        100 * ref["growth_median"],
    )
    # Return the dictionary
    return ref


def add_relative_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Add features that compare a property with its local market benchmarks."""
    # Work on a copy
    out = df.copy()
    # City median price per sq ft (fall back to the national median for unseen cities)
    city_pps = out["City"].map(ref["city_median_pps"]).fillna(ref["global_median_pps"])
    # City + type median price (fall back to the national median)
    key = out["City"].astype(str) + "|" + out["Property_Type"].astype(str)
    type_price = key.map(ref["city_type_median_price"]).fillna(
        ref["global_median_price"]
    )
    # Average schools in the city
    city_schools = (
        out["City"].map(ref["city_avg_schools"]).fillna(ref["global_avg_schools"])
    )

    # Ratio of the property's price per sq ft to its city median (<1 means cheaper than the market)
    out["PPS_vs_City_Median"] = out["Price_per_SqFt_INR"] / city_pps
    # Ratio of the property's price to the median for the same city and type
    out["Price_vs_City_Median"] = out["Price_in_Lakhs"] / type_price
    # School density relative to the city average (>1 means better served than typical)
    out["School_Density_Score"] = out["Nearby_Schools"] / city_schools.clip(lower=0.1)
    # City demand index (0-1); unseen cities get a neutral 0.5
    out["City_Demand_Index"] = out["City"].map(ref["city_demand_index"]).fillna(0.5)
    # Winsorised price per sq ft for scale-sensitive models (raw value is kept for label rules)
    out["Price_per_SqFt_Capped"] = out["Price_per_SqFt_INR"].clip(
        ref["pps_clip_low"], ref["pps_clip_high"]
    )
    # Return the enriched frame
    return out


def compute_growth_rate(
    df: pd.DataFrame, ref: dict, rng: np.random.Generator | None = None
) -> pd.Series:
    """Annual appreciation rate from city demand, infrastructure, status and age (optionally with noise)."""
    # City demand index (0-1)
    demand = df["City"].map(ref["city_demand_index"]).fillna(0.5)
    # Linear growth formula built from the documented assumptions
    growth = (
        cfg.GROWTH_BASE
        + cfg.GROWTH_CITY_WEIGHT * demand
        + cfg.GROWTH_INFRA_WEIGHT * (df["Infra_Score"] / 100)
        + cfg.GROWTH_UNDER_CONSTRUCTION * df["Is_Under_Construction"]
        - cfg.GROWTH_AGE_PENALTY * (df["Age_of_Property"] - 10).clip(lower=0)
    )
    # Add reproducible market noise when a generator is supplied
    if rng is not None:
        growth = growth + rng.normal(0, cfg.GROWTH_NOISE_STD, size=len(df))
    # Keep growth inside realistic bounds
    return growth.clip(cfg.GROWTH_MIN, cfg.GROWTH_MAX)


def investment_factors(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Evaluate the four investment factors (True / False) for every property."""
    # Factor 1 - value: price per sq ft at or below the city median
    value = df["PPS_vs_City_Median"] <= 1.0
    # Factor 2 - affordability: total price at or below the median for the same city and type
    affordable = df["Price_vs_City_Median"] <= 1.0
    # Factor 3 - infrastructure: infrastructure score in the top 40%
    infra = df["Infra_Score"] >= ref["infra_threshold"]
    # Factor 4 - growth: expected annual appreciation at or above the median
    growth = df["Expected_Growth_Rate"] >= ref["growth_median"]
    # Combine into one tidy frame
    return pd.DataFrame(
        {
            "Value_PPS": value,
            "Affordable_Price": affordable,
            "Strong_Infrastructure": infra,
            "High_Growth": growth,
        }
    )


# ---------------------------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Full feature pipeline for inference: base + relative features + expected growth rate."""
    # Row-wise features
    out = add_base_features(df)
    # Market-relative features
    out = add_relative_features(out, ref)
    # Expected (noise-free) growth, used by the label rules and shown in the app
    out["Expected_Growth_Rate"] = compute_growth_rate(out, ref)
    # Return the engineered frame
    return out


def build_training_frame(
    clean_df: pd.DataFrame, seed: int = cfg.RANDOM_STATE
) -> tuple[pd.DataFrame, dict]:
    """Engineer features and create both targets for the whole dataset."""
    # Base features are needed before the benchmarks can be fitted
    base = add_base_features(clean_df)
    # Fit market benchmarks on the full dataset
    ref = fit_reference_tables(base)
    # Add relative features and expected growth
    out = engineer_features(clean_df, ref)

    # Reproducible random generator for market noise
    rng = np.random.default_rng(seed)
    # Realised growth = expected growth + noise
    out["Growth_Rate"] = compute_growth_rate(out, ref, rng)
    # Regression target: price after five years of compounding
    out["Future_Price_5Y"] = (
        out["Price_in_Lakhs"] * (1 + out["Growth_Rate"]) ** cfg.HORIZON_YEARS
    ).round(3)

    # Classification target: sum of the four factor points
    factors = investment_factors(out, ref)
    out["Investment_Score"] = factors.sum(axis=1).astype(int)
    # Keep each factor visible for explainability
    out = pd.concat([out, factors.astype(int)], axis=1)
    # Good investment when enough factors are satisfied
    out["Good_Investment"] = (
        out["Investment_Score"] >= cfg.GOOD_INVESTMENT_MIN_SCORE
    ).astype(int)
    # Log the class balance
    logger.info(
        "Good_Investment share: %.1f%% | mean growth %.2f%%",
        100 * out["Good_Investment"].mean(),
        100 * out["Growth_Rate"].mean(),
    )
    # Return the dataset and the benchmark tables
    return out, ref
