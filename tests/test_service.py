"""Tests for the inference service used by the Streamlit app (skipped when models are not trained yet)."""

# Test framework
import pytest

# Project modules
from src import config as cfg

# Skip the whole module when the trained artifacts are missing
pytestmark = pytest.mark.skipif(
    not cfg.CLASSIFIER_PATH.exists(), reason="Run `python -m src.train` first."
)

# A well-priced, well-connected apartment that should pass all four investment factors
GOOD = dict(
    City="Pune",
    Property_Type="Apartment",
    BHK=3,
    Size_in_SqFt=2600,
    Price_in_Lakhs=235,
    Year_Built=2016,
    Furnished_Status="Semi-furnished",
    Floor_No=6,
    Total_Floors=14,
    Nearby_Schools=7,
    Nearby_Hospitals=6,
    Public_Transport_Accessibility="High",
    Parking_Space="Yes",
    Security="Yes",
    Amenities=["Gym", "Pool", "Clubhouse"],
    Facing="East",
    Owner_Type="Builder",
    Availability_Status="Ready_to_Move",
)


@pytest.fixture(scope="module")
def service():
    """Load the prediction service once for all tests."""
    from src.predict import AdvisorService

    return AdvisorService()


def test_good_property_is_recommended(service):
    """The well-priced apartment must be labelled good and forecast to appreciate."""
    result = service.predict(GOOD)
    assert result["good_investment"] is True
    assert result["future_price"] > result["current_price"]
    assert (
        result["future_price_low"]
        < result["future_price"]
        < result["future_price_high"]
    )
    assert all(result["factors"].values())


def test_overpriced_property_is_rejected(service):
    """A small, expensive, poorly connected flat must not be recommended."""
    bad = {
        **GOOD,
        "City": "Mumbai",
        "BHK": 2,
        "Size_in_SqFt": 900,
        "Price_in_Lakhs": 420,
        "Nearby_Schools": 3,
        "Nearby_Hospitals": 2,
        "Public_Transport_Accessibility": "Low",
        "Parking_Space": "No",
        "Security": "No",
        "Amenities": [],
    }
    assert service.predict(bad)["good_investment"] is False


def test_missing_fields_raise_value_error(service):
    """An incomplete request must raise a readable ValueError."""
    with pytest.raises(ValueError, match="Missing property details"):
        service.predict({"City": "Pune"})


def test_out_of_range_value_raises_value_error(service):
    """A size outside the allowed range must be rejected."""
    with pytest.raises(ValueError, match="Size_in_SqFt"):
        service.predict({**GOOD, "Size_in_SqFt": 5})


def test_floor_above_building_is_capped(service):
    """A floor number above the building height is capped instead of crashing."""
    result = service.predict({**GOOD, "Floor_No": 50, "Total_Floors": 10})
    assert 0 <= result["probability"] <= 1


def test_unknown_city_uses_national_benchmarks(service):
    """A city never seen in training must still produce a prediction and be flagged."""
    result = service.predict({**GOOD, "City": "Atlantis"})
    assert result["city_known"] is False
