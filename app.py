"""Real Estate Investment Advisor - premium Streamlit app.

Tabs
----
1. Advisor   : describe a property -> live verdict, 5-year price forecast with a 90% range, factor checklist
2. Explorer  : filter 250,000 properties by state, city, BHK, price, size ...
3. Market    : treemap, heatmap, rankings and trend charts
4. Models    : metrics, confusion matrix, feature importance and an honest reading of the results

Run with:  streamlit run app.py
"""

# Numerical helpers
import numpy as np

# Tabular data handling
import pandas as pd

# Interactive charts
import plotly.express as px
import plotly.graph_objects as go

# Web app framework
import streamlit as st

# Project modules
from src import config as cfg
from src import ui
from src.predict import AdvisorService

# Public repository (shown in the sidebar and footer)
GITHUB_URL = "https://github.com/13msrajput/Real-Estate-Investment-Advisor"

# ------------------------------------------------------------------ page setup (must be the first Streamlit call)
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto",
)
# Active theme ('dark' or 'light') and its palette, used by every custom component and chart
THEME = ui.get_theme()
P = ui.palette(THEME)
# Design-system CSS
ui.inject_css(THEME)

# Columns needed by the explorer and market tabs (limits memory use on small hosts)
EXPLORER_COLUMNS = [
    "ID",
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt_INR",
    "Year_Built",
    "Furnished_Status",
    "Availability_Status",
    "Infra_Score",
    "Growth_Rate",
    "Future_Price_5Y",
    "Good_Investment",
]


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading models...")
def load_service() -> AdvisorService:
    """Load the trained models once per server process."""
    return AdvisorService()


@st.cache_data(show_spinner=False)
def load_city_summary() -> pd.DataFrame:
    """City-level aggregates created during training."""
    return pd.read_csv(cfg.CITY_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_year_trend() -> pd.DataFrame:
    """Average price by construction year created during training."""
    return pd.read_csv(cfg.YEAR_TREND_PATH)


@st.cache_data(show_spinner="Loading 250,000 properties...")
def load_explorer_data() -> pd.DataFrame:
    """Processed property-level data for filtering and charts."""
    return pd.read_csv(cfg.PROCESSED_DATA_PATH, usecols=EXPLORER_COLUMNS)


@st.cache_data(show_spinner=False)
def state_type_heatmap() -> pd.DataFrame:
    """Average price per sq ft by state and property type (cached - the source table is large)."""
    return load_explorer_data().pivot_table(
        index="State",
        columns="Property_Type",
        values="Price_per_SqFt_INR",
        aggfunc="mean",
    )


@st.cache_data(show_spinner=False)
def bhk_price_table(top_n: int = 5) -> pd.DataFrame:
    """Average price by BHK for the most expensive cities."""
    data = load_explorer_data()
    cities = (
        load_city_summary()
        .sort_values("Avg_Price_Lakhs", ascending=False)
        .head(top_n)["City"]
    )
    return (
        data[data["City"].isin(cities)]
        .groupby(["City", "BHK"])["Price_in_Lakhs"]
        .mean()
        .reset_index()
    )


def show(fig, key: str | None = None) -> None:
    """Render a styled Plotly figure with the design-system theme."""
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False})


# ------------------------------------------------------------------ start-up checks
try:
    # Load models and summaries; show a friendly message if artifacts are missing
    service = load_service()
    city_summary = load_city_summary()
except FileNotFoundError as err:
    ui.hero([("Setup", "required")])
    st.error(f"Model files are missing: {err}")
    st.info(
        "Run `python -m src.train` from the project root to create them, then restart the app."
    )
    st.stop()
except Exception as err:  # any other loading failure
    st.error(f"The app could not start: {err}")
    st.stop()

# Convenience handles
meta = service.metadata
ref = service.reference
clf_meta, reg_meta = meta["classification"], meta["regression"]
clf_best = clf_meta["results"][clf_meta["best_model"]]
reg_best = reg_meta["results"][reg_meta["best_model"]]
state_city = city_summary.groupby("State")["City"].apply(sorted).to_dict()
# Share by which the champion regressor beats the fixed 8% growth rule (RMSE reduction)
error_cut = 100 * (1 - reg_best["rmse"] / reg_meta["fixed_rate_baseline"]["rmse"])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    ui.sidebar_brand()
    st.caption(
        "Instantly judge whether a property is worth buying and what it could be worth in five years."
    )
    ui.side_card("Classifier F1 (test)", f"{clf_best['f1']:.3f}")
    ui.side_card("Forecast RMSE (test)", f"{reg_best['rmse']:.1f} lakhs")
    ui.side_card("Training data", f"{meta['dataset']['rows']:,} properties")
    ui.render_html(
        '<div class="mini-label">How it works</div>'
        '<ol class="steps"><li>You describe a property</li>'
        "<li>It is benchmarked against its city market</li>"
        f'<li>{clf_meta["best_model"]} models score the deal and project its price</li>'
        "<li>You get a verdict, a forecast and a range</li></ol>"
    )
    st.divider()
    st.link_button(
        "View source on GitHub", GITHUB_URL, icon=":material/code:", width="stretch"
    )
    st.caption(f"Models trained {meta['trained_at'][:10]} · MLflow-tracked")

# ------------------------------------------------------------------ hero + disclaimer
ui.hero(
    [
        (f"{meta['dataset']['rows']:,}", "properties"),
        (f"{len(ref['city_median_pps'])}", "cities"),
        (clf_meta["best_model"], f"F1 {clf_best['f1']:.3f}"),
        (f"{error_cut:.0f}%", "lower error than a fixed 8% rule"),
        ("MLflow", "tracked"),
    ]
)
ui.notice(
    "<b>Educational project.</b> The dataset is a single snapshot with no price history, so the 5-year forecast "
    "rests on documented growth assumptions rather than observed market appreciation. It is not financial advice."
)

tab_adv, tab_exp, tab_mkt, tab_mod = st.tabs(
    [
        ":material/query_stats: Advisor",
        ":material/travel_explore: Explorer",
        ":material/map: Market insights",
        ":material/model_training: Model insights",
    ]
)

# =============================================================================================
# TAB 1 - ADVISOR (fragment: widget changes re-run only this block, so the app stays fast)
# =============================================================================================
# Default values of every input widget
DEFAULTS = {
    "s_state": "Maharashtra",
    "s_city": "Pune",
    "s_type": "Apartment",
    "s_bhk": 3,
    "s_size": 2600,
    "s_price": 235.0,
    "s_year": 2016,
    "s_furn": "Semi-furnished",
    "s_total": 14,
    "s_floor": 6,
    "s_facing": "East",
    "s_schools": 7,
    "s_hosp": 6,
    "s_transport": "High",
    "s_parking": True,
    "s_security": True,
    "s_amen": ["Gym", "Pool", "Clubhouse"],
    "s_owner": "Builder",
    "s_avail": "Ready_to_Move",
}
# One-click example properties (only the keys that differ from DEFAULTS need to be listed)
PRESETS = {
    "Value pick": {},
    "Overpriced flat": {
        "s_state": "Maharashtra",
        "s_city": "Mumbai",
        "s_bhk": 2,
        "s_size": 900,
        "s_price": 420.0,
        "s_year": 2005,
        "s_schools": 3,
        "s_hosp": 2,
        "s_transport": "Low",
        "s_parking": False,
        "s_security": False,
        "s_amen": [],
        "s_owner": "Broker",
        "s_furn": "Unfurnished",
        "s_total": 8,
        "s_floor": 3,
    },
    "Launch-phase villa": {
        "s_state": "Karnataka",
        "s_city": "Bangalore",
        "s_type": "Villa",
        "s_bhk": 4,
        "s_size": 3600,
        "s_price": 300.0,
        "s_year": 2026,
        "s_avail": "Under_Construction",
        "s_schools": 6,
        "s_hosp": 5,
        "s_transport": "Medium",
        "s_amen": ["Pool", "Garden", "Playground", "Gym"],
        "s_total": 2,
        "s_floor": 1,
    },
    "Starter home": {
        "s_state": "Maharashtra",
        "s_city": "Nagpur",
        "s_bhk": 2,
        "s_size": 1800,
        "s_price": 150.0,
        "s_year": 2012,
        "s_schools": 5,
        "s_hosp": 5,
        "s_transport": "Medium",
        "s_security": False,
        "s_amen": ["Garden"],
        "s_owner": "Owner",
        "s_total": 6,
        "s_floor": 2,
    },
}
# Make sure every widget has a value before it is created
for _key, _val in DEFAULTS.items():
    st.session_state.setdefault(_key, _val)


def apply_preset() -> None:
    """Callback: copy a preset into the widget state (runs before widgets are rebuilt)."""
    choice = st.session_state.get("preset_choice")
    # Selecting the active preset again clears the control; keep the current values in that case
    if choice in PRESETS:
        for k, v in {**DEFAULTS, **PRESETS[choice]}.items():
            st.session_state[k] = v


def reset_city() -> None:
    """Callback: when the state changes, jump to its first city so the pair is always valid."""
    st.session_state["s_city"] = state_city[st.session_state["s_state"]][0]


@st.fragment
def advisor_tab() -> None:
    """Input panel (left) and live results (right)."""
    # One-click examples sit above both columns so they never get clipped
    st.pills(
        "Try an example",
        list(PRESETS),
        key="preset_choice",
        on_change=apply_preset,
        help="Load a ready-made property to see how the verdict changes.",
    )
    left, right = st.columns([5, 6], gap="large")

    # ---------------------------------------------------------------- inputs
    with left:
        ui.section(
            "Step 1",
            "Describe the property",
            "Results update instantly as you change any field.",
        )
        with st.container(border=True, key="panel_inputs"):
            ui.mini_label("Location")
            c1, c2 = st.columns(2)
            c1.selectbox(
                "State", sorted(state_city), key="s_state", on_change=reset_city
            )
            c2.selectbox("City", state_city[st.session_state["s_state"]], key="s_city")

            ui.mini_label("Property")
            st.segmented_control(
                "Property type",
                ["Apartment", "Independent House", "Villa"],
                key="s_type",
                required=True,
            )
            st.segmented_control("BHK", [1, 2, 3, 4, 5], key="s_bhk", required=True)

            ui.mini_label("Size and price")
            st.slider("Size (sq ft)", 300, 6000, key="s_size", step=50)
            st.number_input(
                "Asking price (₹ lakhs)",
                min_value=5.0,
                max_value=5000.0,
                step=5.0,
                format="%.1f",
                key="s_price",
            )

            ui.mini_label("Building")
            c1, c2, c3 = st.columns(3)
            c1.slider("Year built", 1980, 2028, key="s_year")
            c2.number_input(
                "Total floors", min_value=1, max_value=100, step=1, key="s_total"
            )
            # The flat cannot be above the top floor, so clamp before the widget is built
            st.session_state["s_floor"] = min(
                st.session_state["s_floor"], st.session_state["s_total"]
            )
            c3.number_input(
                "Floor number",
                min_value=0,
                max_value=st.session_state["s_total"],
                step=1,
                key="s_floor",
            )
            st.segmented_control(
                "Availability",
                ["Ready_to_Move", "Under_Construction"],
                key="s_avail",
                required=True,
                format_func=lambda v: v.replace("_", " "),
            )

            ui.mini_label("Neighbourhood and features")
            c1, c2 = st.columns(2)
            c1.slider("Nearby schools", 0, 10, key="s_schools")
            c2.slider("Nearby hospitals", 0, 10, key="s_hosp")
            st.segmented_control(
                "Public transport access",
                ["Low", "Medium", "High"],
                key="s_transport",
                required=True,
            )
            c1, c2 = st.columns(2)
            c1.toggle("Parking space", key="s_parking")
            c2.toggle("Security", key="s_security")
            st.pills(
                "Amenities", cfg.AMENITY_LIST, selection_mode="multi", key="s_amen"
            )
            c1, c2 = st.columns(2)
            c1.selectbox(
                "Furnishing",
                ["Unfurnished", "Semi-furnished", "Furnished"],
                key="s_furn",
            )
            c2.selectbox("Facing", ["North", "South", "East", "West"], key="s_facing")
            st.segmented_control(
                "Seller", ["Owner", "Builder", "Broker"], key="s_owner", required=True
            )

    # ---------------------------------------------------------------- build the request
    ss = st.session_state
    payload = {
        "City": ss["s_city"],
        "Property_Type": ss["s_type"],
        "BHK": ss["s_bhk"],
        "Size_in_SqFt": ss["s_size"],
        "Price_in_Lakhs": ss["s_price"],
        "Year_Built": ss["s_year"],
        "Furnished_Status": ss["s_furn"],
        "Floor_No": ss["s_floor"],
        "Total_Floors": ss["s_total"],
        "Nearby_Schools": ss["s_schools"],
        "Nearby_Hospitals": ss["s_hosp"],
        "Public_Transport_Accessibility": ss["s_transport"],
        "Parking_Space": "Yes" if ss["s_parking"] else "No",
        "Security": "Yes" if ss["s_security"] else "No",
        "Amenities": list(ss["s_amen"] or []),
        "Facing": ss["s_facing"],
        "Owner_Type": ss["s_owner"],
        "Availability_Status": ss["s_avail"],
    }

    # ---------------------------------------------------------------- results
    with right:
        ui.section(
            "Step 2",
            "Investment verdict",
            "A classifier scores the deal; a regressor projects the price in five years.",
        )
        try:
            res = service.predict(payload)
        except ValueError as err:  # input problems are shown plainly
            st.error(f"Please check your inputs: {err}")
            return
        except Exception as err:  # unexpected failures must not crash the page
            st.error(f"Something went wrong while predicting: {err}")
            return

        ui.verdict_card(res["good_investment"], res["probability"])
        ui.render_html('<div style="height:.9rem"></div>')
        gain = res["appreciation_pct"]
        ui.kpi_grid(
            [
                {
                    "label": "Asking price",
                    "value": ui.fmt_lakhs(res["current_price"]),
                    "sub": f"₹{res['price_per_sqft']:,.0f} per sq ft",
                },
                {
                    "label": f"Estimated price in {cfg.HORIZON_YEARS} years",
                    "value": ui.fmt_lakhs(res["future_price"]),
                    "sub": f"{gain:+.1f}% total",
                    "tone": "up" if gain >= 0 else "down",
                },
                {
                    "label": "Implied annual growth",
                    "value": f"{res['implied_cagr_pct']:.2f}%",
                    "sub": f"vs {100 * cfg.BASELINE_GROWTH_RATE:.0f}% fixed benchmark",
                },
                {
                    "label": "Infrastructure score",
                    "value": f"{res['infra_score']:.0f}<span style='font-size:1rem;opacity:.6'>/100</span>",
                    "sub": f"top-40% bar: {ref['infra_threshold']:.0f}",
                },
            ]
        )

        # Price path with 90% range and the fixed-rate benchmark
        years = np.arange(0, cfg.HORIZON_YEARS + 1)
        path = lambda target: res["current_price"] * (
            target / res["current_price"]
        ) ** (
            years / cfg.HORIZON_YEARS
        )  # noqa: E731
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=years,
                y=path(res["future_price_high"]),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=path(res["future_price_low"]),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=(
                    "rgba(227,178,90,0.18)"
                    if THEME == "dark"
                    else "rgba(183,121,31,0.15)"
                ),
                name="90% range",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=res["current_price"] * (1 + cfg.BASELINE_GROWTH_RATE) ** years,
                mode="lines",
                line=dict(color=P["muted"], dash="dash", width=2),
                name=f"Fixed {cfg.BASELINE_GROWTH_RATE:.0%} benchmark",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=path(res["future_price"]),
                mode="lines+markers",
                name="Model forecast",
                line=dict(color=P["accent"], width=3.5),
                marker=dict(size=8, color=P["accent"]),
            )
        )
        fig.update_xaxes(title="Years from today", dtick=1)
        fig.update_yaxes(title="₹ lakhs")
        ui.style_fig(fig, THEME, height=320, title="Projected price path")
        show(fig, key="price_path")
        st.caption(
            f"90% prediction range for year {cfg.HORIZON_YEARS}: {ui.fmt_lakhs(res['future_price_low'])} – "
            f"{ui.fmt_lakhs(res['future_price_high'])} (5th–95th percentile of actual ÷ predicted on held-out data)."
        )

        # Why the verdict came out this way
        ui.mini_label("Why this verdict - four investment factors")
        f = res["factors"]
        ui.factor_list(
            [
                (
                    f["Value_PPS"],
                    "Value",
                    f"Price per sq ft is {res['pps_vs_city_median']:.2f}× the city median (needs ≤ 1.00)",
                ),
                (
                    f["Affordable_Price"],
                    "Affordability",
                    f"Price is {res['price_vs_city_median']:.2f}× the median for similar properties (needs ≤ 1.00)",
                ),
                (
                    f["Strong_Infrastructure"],
                    "Infrastructure",
                    f"Score {res['infra_score']:.0f} vs the top-40% threshold of {ref['infra_threshold']:.0f}",
                ),
                (
                    f["High_Growth"],
                    "Growth",
                    f"Expected growth {res['expected_growth_pct']:.2f}% a year vs the median of {100 * ref['growth_median']:.2f}%",
                ),
            ]
        )
        st.caption(
            f"{sum(f.values())} of 4 factors passed - the label needs at least {cfg.GOOD_INVESTMENT_MIN_SCORE}."
        )

        # Input sanity warnings (the models were trained on a limited range)
        notes = []
        if not 1 <= ss["s_price"] <= 500:
            notes.append("price is outside the 1–500 lakh range seen in training")
        if not 500 <= ss["s_size"] <= 5000:
            notes.append("size is outside the 500–5,000 sq ft range seen in training")
        if not res["city_known"]:
            notes.append(
                "this city is not in the training data, so national benchmarks were used"
            )
        if notes:
            ui.callout(
                "<b>Extrapolation warning:</b> "
                + "; ".join(notes)
                + ". Treat the result with extra caution.",
                "warn",
            )


with tab_adv:
    advisor_tab()


# =============================================================================================
# TAB 2 - EXPLORER
# =============================================================================================
@st.fragment
def explorer_tab() -> None:
    """Filterable property table with summary statistics."""
    ui.section(
        "Explore",
        "Filter 250,000 properties",
        "Narrow by area, price, BHK and more, then download the shortlist.",
    )
    try:
        data = load_explorer_data()
    except Exception as err:
        st.error(f"Could not load the property dataset: {err}")
        return

    with st.container(border=True, key="panel_filters"):
        f1, f2, f3 = st.columns(3)
        sel_states = f1.multiselect(
            "State", sorted(data["State"].unique()), placeholder="All states"
        )
        city_options = (
            sorted(data[data["State"].isin(sel_states)]["City"].unique())
            if sel_states
            else sorted(data["City"].unique())
        )
        sel_cities = f2.multiselect("City", city_options, placeholder="All cities")
        sel_types = f3.multiselect(
            "Property type",
            sorted(data["Property_Type"].unique()),
            placeholder="All types",
        )
        g1, g2 = st.columns(2)
        price_range = g1.slider("Price (₹ lakhs)", 0, 500, (0, 500))
        size_range = g2.slider("Size (sq ft)", 500, 5000, (500, 5000), step=50)
        h1, h2, h3 = st.columns([2, 2, 1.4])
        sel_bhk = h1.segmented_control(
            "BHK", sorted(data["BHK"].unique()), selection_mode="multi", key="x_bhk"
        )
        sel_avail = h2.segmented_control(
            "Availability",
            sorted(data["Availability_Status"].unique()),
            selection_mode="multi",
            key="x_avail",
            format_func=lambda v: v.replace("_", " "),
        )
        only_good = h3.toggle("Good investments only", key="x_good")

    # Apply every filter that has a selection
    mask = data["Price_in_Lakhs"].between(*price_range) & data["Size_in_SqFt"].between(
        *size_range
    )
    for column, selected in [
        ("State", sel_states),
        ("City", sel_cities),
        ("Property_Type", sel_types),
        ("BHK", sel_bhk),
        ("Availability_Status", sel_avail),
    ]:
        if selected:
            mask &= data[column].isin(selected)
    if only_good:
        mask &= data["Good_Investment"] == 1
    filtered = data[mask]

    if filtered.empty:
        ui.callout(
            "No properties match these filters. Try widening the price or size range.",
            "warn",
        )
        return

    ui.kpi_grid(
        [
            {
                "label": "Matching properties",
                "value": f"{len(filtered):,}",
                "sub": f"{len(filtered) / len(data):.1%} of all listings",
            },
            {
                "label": "Average price",
                "value": ui.fmt_lakhs(filtered["Price_in_Lakhs"].mean()),
            },
            {
                "label": "Average price / sq ft",
                "value": f"₹{filtered['Price_per_SqFt_INR'].mean():,.0f}",
            },
            {
                "label": "Good-investment share",
                "value": f"{filtered['Good_Investment'].mean():.0%}",
                "sub": "rule-based label",
            },
        ]
    )

    # Best candidates first: good label, then highest assumed growth (raw rows are kept for the CSV export)
    export = filtered.sort_values(
        ["Good_Investment", "Growth_Rate"], ascending=False
    ).head(300)
    # Display copy with friendlier values so every column fits on screen
    top = export.copy()
    top["Growth_Rate"] = top["Growth_Rate"] * 100
    top["Good_Investment"] = top["Good_Investment"].astype(bool)
    top["Availability_Status"] = top["Availability_Status"].replace(
        {"Ready_to_Move": "Ready", "Under_Construction": "Building"}
    )
    top["Property_Type"] = top["Property_Type"].replace({"Independent House": "House"})
    st.dataframe(
        top,
        hide_index=True,
        height=420,
        column_order=[
            "Good_Investment",
            "City",
            "Property_Type",
            "BHK",
            "Size_in_SqFt",
            "Price_in_Lakhs",
            "Price_per_SqFt_INR",
            "Year_Built",
            "Availability_Status",
            "Infra_Score",
            "Growth_Rate",
            "Future_Price_5Y",
        ],
        column_config={
            "Good_Investment": st.column_config.CheckboxColumn(
                "Good",
                help="Rule-based label used to train the classifier",
                width="small",
            ),
            "City": st.column_config.TextColumn("City", width="small"),
            "Property_Type": st.column_config.TextColumn("Type", width="small"),
            "BHK": st.column_config.NumberColumn("BHK", format="%d", width="small"),
            "Size_in_SqFt": st.column_config.NumberColumn(
                "Sq ft", format="%d", width="small"
            ),
            "Price_in_Lakhs": st.column_config.NumberColumn(
                "Price (₹ L)", format="%.1f", width="small"
            ),
            "Price_per_SqFt_INR": st.column_config.NumberColumn(
                "₹ / sq ft", format="%d", width="small"
            ),
            "Year_Built": st.column_config.NumberColumn(
                "Built", format="%d", width="small"
            ),
            "Availability_Status": st.column_config.TextColumn("Status", width="small"),
            "Infra_Score": st.column_config.ProgressColumn(
                "Infrastructure",
                min_value=0,
                max_value=100,
                format="%.0f",
                width="medium",
            ),
            "Growth_Rate": st.column_config.NumberColumn(
                "Growth / yr", format="%.2f%%", width="small"
            ),
            "Future_Price_5Y": st.column_config.NumberColumn(
                "In 5 yrs (₹ L)", format="%.1f", width="small"
            ),
        },
    )
    d1, d2 = st.columns([1, 3])
    d1.download_button(
        "Download shortlist (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name="shortlist.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
    d2.caption(
        "Showing the 300 strongest matches. 'Good' is the rule-based label used to train the classifier."
    )

    # Distribution of the filtered set
    sample = filtered.sample(min(len(filtered), 20000), random_state=1)
    sample = sample.assign(
        Label=np.where(sample["Good_Investment"] == 1, "Good investment", "Other")
    )
    fig = px.histogram(
        sample,
        x="Price_in_Lakhs",
        color="Label",
        nbins=40,
        barmode="stack",
        color_discrete_map={"Good investment": P["good"], "Other": P["info"]},
    )
    fig.update_xaxes(title="Price (₹ lakhs)")
    fig.update_yaxes(title="Properties")
    ui.style_fig(
        fig, THEME, height=300, title="Price distribution of the filtered properties"
    )
    show(fig, key="explorer_hist")


with tab_exp:
    explorer_tab()

# =============================================================================================
# TAB 3 - MARKET INSIGHTS
# =============================================================================================
with tab_mkt:
    ui.section(
        "Market",
        "Where the opportunities are",
        "Location-wise views built from all 250,000 properties.",
    )
    try:
        data = load_explorer_data()
        trend = load_year_trend()
    except Exception as err:
        st.error(f"Could not load insight data: {err}")
        st.stop()

    ui.kpi_grid(
        [
            {"label": "Properties analysed", "value": f"{len(data):,}"},
            {
                "label": "Average price",
                "value": ui.fmt_lakhs(data["Price_in_Lakhs"].mean()),
            },
            {
                "label": "Median price / sq ft",
                "value": f"₹{data['Price_per_SqFt_INR'].median():,.0f}",
            },
            {
                "label": "Good-investment share",
                "value": f"{data['Good_Investment'].mean():.0%}",
            },
        ]
    )

    # Treemap: where the listings are and how fast they are assumed to grow
    fig = px.treemap(
        city_summary,
        path=[px.Constant("India"), "State", "City"],
        values="Properties",
        color="Avg_Growth_Rate",
        color_continuous_scale=P["heat"],
    )
    fig.update_traces(
        marker=dict(cornerradius=6),
        root_color="rgba(0,0,0,0)",
        hovertemplate="<b>%{label}</b><br>%{value:,} properties<br>Avg growth %{color:.2%}<extra></extra>",
    )
    fig.update_coloraxes(
        colorbar=dict(title="Avg growth", tickformat=".1%", thickness=10)
    )
    ui.style_fig(
        fig,
        THEME,
        height=430,
        legend=False,
        title="Listings by state and city (colour = average assumed annual growth)",
    )
    show(fig, key="treemap")

    heat = state_type_heatmap()
    fig = px.imshow(
        heat.round(0), text_auto=",.0f", aspect="auto", color_continuous_scale=P["heat"]
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None)
    ui.style_fig(
        fig,
        THEME,
        height=560,
        legend=False,
        title="Average price per sq ft (₹) — state × property type",
    )
    show(fig, key="heatmap")

    a, b = st.columns(2)
    with a:
        top_cities = city_summary.sort_values(
            "Good_Investment_Rate", ascending=False
        ).head(10)
        fig = px.bar(
            top_cities,
            x="Good_Investment_Rate",
            y="City",
            orientation="h",
            text=top_cities["Good_Investment_Rate"].map("{:.0%}".format),
            color_discrete_sequence=[P["accent"]],
        )
        fig.update_traces(
            marker=dict(cornerradius=6), textposition="outside", cliponaxis=False
        )
        fig.update_yaxes(categoryorder="total ascending", title=None)
        fig.update_xaxes(
            tickformat=".0%",
            title="Share labelled good investment",
            range=[0, top_cities["Good_Investment_Rate"].max() * 1.18],
        )
        ui.style_fig(
            fig,
            THEME,
            height=400,
            legend=False,
            title="Top 10 cities by share of good investments",
        )
        show(fig, key="top_cities")
    with b:
        fig = px.scatter(
            city_summary,
            x="Avg_Infra_Score",
            y="Avg_Growth_Rate",
            size="Properties",
            hover_name="City",
            color="Good_Investment_Rate",
            color_continuous_scale=P["heat"],
            size_max=26,
        )
        fig.update_yaxes(tickformat=".1%", title="Average assumed growth")
        fig.update_xaxes(title="Average infrastructure score")
        fig.update_coloraxes(
            colorbar=dict(title="Good share", tickformat=".0%", thickness=10)
        )
        ui.style_fig(
            fig,
            THEME,
            height=400,
            legend=False,
            title="Infrastructure vs assumed growth by city",
        )
        show(fig, key="infra_growth")

    c, d = st.columns(2)
    with c:
        fig = px.line(
            trend,
            x="Year_Built",
            y="Avg_Price_per_SqFt",
            markers=True,
            color_discrete_sequence=[P["accent"]],
        )
        fig.update_yaxes(
            range=[0, trend["Avg_Price_per_SqFt"].max() * 1.2], title="₹ per sq ft"
        )
        fig.update_xaxes(title="Year built")
        ui.style_fig(
            fig,
            THEME,
            height=340,
            legend=False,
            title="Average price per sq ft by year built",
        )
        show(fig, key="trend_year")
    with d:
        sub = bhk_price_table()
        fig = px.line(sub, x="BHK", y="Price_in_Lakhs", color="City", markers=True)
        fig.update_yaxes(
            range=[0, sub["Price_in_Lakhs"].max() * 1.25],
            title="Average price (₹ lakhs)",
        )
        fig.update_xaxes(dtick=1)
        fig.update_layout(legend_title_text="")
        ui.style_fig(
            fig, THEME, height=340, title="Average price by BHK — five priciest cities"
        )
        show(fig, key="trend_bhk")

    ui.callout(
        "<b>Reading these charts.</b> Prices in this dataset barely differ between cities, BHK counts, sizes or years built "
        "(note the zero-based axes). The visible spread in growth and good-investment share comes mainly from the "
        "<b>City Demand Index</b> and the <b>Infrastructure Score</b> used by the investment rules - not from large price gaps. "
        "That is a property of the data, not a charting error.",
        "warn",
    )

# =============================================================================================
# TAB 4 - MODEL INSIGHTS
# =============================================================================================
with tab_mod:
    ui.section(
        "Models",
        "How well do the models work?",
        "Evaluated on 50,000 held-out properties that were never used for training.",
    )
    ui.kpi_grid(
        [
            {
                "label": f"Classifier · {clf_meta['best_model']}",
                "value": f"{clf_best['f1']:.3f}",
                "sub": "F1 score",
            },
            {
                "label": "Classifier ROC-AUC",
                "value": f"{clf_best['roc_auc']:.4f}",
                "sub": f"accuracy {clf_best['accuracy']:.2%}",
            },
            {
                "label": f"Regressor · {reg_meta['best_model']}",
                "value": f"{reg_best['rmse']:.2f} L",
                "sub": "RMSE (₹ lakhs)",
            },
            {
                "label": "Regressor R²",
                "value": f"{reg_best['r2']:.4f}",
                "sub": f"MAPE {reg_best['mape']:.2f}%",
            },
        ]
    )

    a, b = st.columns(2)
    with a:
        cm = np.array(clf_best["confusion_matrix"])
        fig = px.imshow(
            cm,
            x=["Predicted: Not good", "Predicted: Good"],
            y=["Actual: Not good", "Actual: Good"],
            text_auto=",",
            color_continuous_scale=P["heat"],
            aspect="auto",
        )
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textfont=dict(size=18))
        ui.style_fig(
            fig,
            THEME,
            height=340,
            legend=False,
            title="Confusion matrix — champion classifier",
        )
        show(fig, key="confusion")
    with b:
        names = list(reg_meta["results"]) + ["Fixed 8% growth"]
        rmses = [v["rmse"] for v in reg_meta["results"].values()] + [
            reg_meta["fixed_rate_baseline"]["rmse"]
        ]
        fig = px.bar(x=names, y=rmses, text=[f"{v:.1f}" for v in rmses])
        fig.update_traces(
            marker=dict(
                color=[
                    P["accent"] if n == reg_meta["best_model"] else P["info"]
                    for n in names[:-1]
                ]
                + [P["muted"]],
                cornerradius=8,
            ),
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_yaxes(
            title="RMSE (₹ lakhs) — lower is better", range=[0, max(rmses) * 1.2]
        )
        fig.update_xaxes(title=None)
        ui.style_fig(
            fig,
            THEME,
            height=340,
            legend=False,
            title="Forecast error vs the fixed-rate benchmark",
        )
        show(fig, key="reg_rmse")

    c, d = st.columns(2)
    for col, imp_meta, title, key in [
        (c, clf_meta, "Classifier — top drivers", "imp_clf"),
        (d, reg_meta, "Regressor — top drivers", "imp_reg"),
    ]:
        with col:
            imp = pd.DataFrame(imp_meta["importance"]).head(8)
            fig = px.bar(
                imp,
                x="Importance",
                y="Feature",
                orientation="h",
                text=imp["Importance"].map("{:.1f}%".format),
                color_discrete_sequence=[P["accent"]],
            )
            fig.update_traces(
                marker=dict(cornerradius=6), textposition="outside", cliponaxis=False
            )
            fig.update_yaxes(categoryorder="total ascending", title=None)
            fig.update_xaxes(
                title="Importance (%)", range=[0, imp["Importance"].max() * 1.2]
            )
            ui.style_fig(fig, THEME, height=340, legend=False, title=title)
            show(fig, key=key)

    ui.section("Compare", "All candidate models")
    t1, t2 = st.columns(2)
    clf_table = pd.DataFrame(clf_meta["results"]).T.drop(
        columns=["confusion_matrix", "train_f1"], errors="ignore"
    )
    clf_table.index.name = "Classifier"
    t1.dataframe(clf_table.round(4), width="stretch")
    reg_table = pd.DataFrame(reg_meta["results"]).T
    reg_table.loc["Fixed 8% growth"] = reg_meta["fixed_rate_baseline"]
    reg_table.index.name = "Regressor"
    t2.dataframe(reg_table.round(4), width="stretch")

    abl = clf_meta["ablation_no_price_features"]
    ui.section("Honesty", "Reading these results responsibly")
    ui.callout(
        f"<b>The label is rule-based, so accuracy is largely rule recovery.</b> Two of the four label factors depend on price. "
        f"With every price column removed, the same XGBoost model can only learn the infrastructure and growth factors and falls to "
        f"<b>{abl['accuracy']:.1%}</b> accuracy (ROC-AUC {abl['roc_auc']:.3f}) against a <b>{clf_meta['majority_baseline_accuracy']:.1%}</b> "
        f"majority-class baseline.",
        "warn",
    )
    ui.callout(
        f"<b>The 5-year price is an assumption-driven projection.</b> Annual growth of {meta['assumptions']['growth_min']:.0%}–"
        f"{meta['assumptions']['growth_max']:.0%} is derived from city demand, infrastructure, construction status and age, plus random "
        f"market noise. The regressor learns that function (error {reg_best['rmse']:.1f} lakhs vs {reg_meta['fixed_rate_baseline']['rmse']:.1f} for a fixed 8% rule); "
        f"it does not discover real market appreciation.",
        "warn",
    )
    ui.callout(
        "<b>City demand is a thin signal.</b> Median price per sq ft differs by only about 5% across cities in this dataset, "
        "and the 0–1 demand index stretches those small gaps. Use the city effect as a modelling assumption, not market evidence."
    )

    with st.expander("Model card · data, targets and experiment tracking"):
        st.markdown(f"""
**Data** — {meta['dataset']['rows']:,} listings, 20 states, 42 cities. Cleaning capped {meta['cleaning_report']['floor_values_corrected']:,} impossible floor numbers and rebuilt price per sq ft in rupees.

**Classification target** — `Good_Investment` = at least {cfg.GOOD_INVESTMENT_MIN_SCORE} of 4 points: price per sq ft ≤ city median · price ≤ median for the same city and type · infrastructure score in the top 40% · expected growth ≥ median.

**Regression target** — `Future_Price_5Y = price × (1 + r)^5`, with `r` built from city demand, infrastructure, construction status and age (+ noise), clipped to {meta['assumptions']['growth_min']:.0%}–{meta['assumptions']['growth_max']:.0%}.

**Models compared** — Logistic / Linear Regression, Random Forest, XGBoost (tuned with randomised search). Champions are registered in the MLflow Model Registry with the alias `champion`.

**Inspect the experiments locally**
""")
        st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")

ui.footer(GITHUB_URL)
