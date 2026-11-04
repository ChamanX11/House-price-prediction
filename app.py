
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="EstateIQ | House Price Prediction Studio",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color palette
NAVY = "#14263D"
BLUE = "#3478F6"
TEAL = "#12B8A6"
PURPLE = "#8065E8"
ORANGE = "#F59E0B"
BG = "#F4F7FC"
TEXT = "#24354B"
MUTED = "#738198"


# =========================================================
# 2. CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F4F7FC;
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,252,0.9);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14263D 0%, #203D5C 100%);
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #C1D0E2;
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    .hero {
        background: linear-gradient(120deg, #14263D, #245477);
        padding: 30px 32px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 25px rgba(20,38,61,0.12);
    }

    .hero h1 {
        color: white;
        font-size: 31px;
        font-weight: 800;
        margin: 8px 0;
        letter-spacing: -0.8px;
    }

    .hero p {
        color: #D4E2F0;
        font-size: 14px;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #69E1D0;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .section-title {
        color: #24354B;
        font-size: 20px;
        font-weight: 750;
        margin: 12px 0 4px 0;
    }

    .section-subtitle {
        color: #738198;
        font-size: 12px;
        margin-bottom: 15px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 16px;
        padding: 20px;
        min-height: 135px;
        box-shadow: 0 3px 12px rgba(20,38,61,0.035);
    }

    .metric-label {
        color: #738198;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 13px;
    }

    .metric-value {
        color: #14263D;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.7px;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #8491A5;
        font-size: 11px;
        margin-top: 8px;
    }

    .panel {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }

    .insight-card {
        background: #FFFFFF;
        border-left: 4px solid #12B8A6;
        padding: 16px 18px;
        border-radius: 10px;
        border-top: 1px solid #E5EAF2;
        border-right: 1px solid #E5EAF2;
        border-bottom: 1px solid #E5EAF2;
    }

    .insight-label {
        color: #738198;
        font-size: 11px;
        font-weight: 600;
    }

    .insight-value {
        color: #14263D;
        font-size: 19px;
        font-weight: 750;
        margin-top: 5px;
    }

    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 15px;
        padding: 8px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E5EAF2;
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #8491A5;
        font-size: 11px;
        padding: 20px 0 5px 0;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    @media (max-width: 768px) {
        .hero h1 {
            font-size: 23px;
        }
        .metric-value {
            font-size: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD DATA
# =========================================================
@st.cache_data
def get_data():
    return load_data()


try:
    df = get_data()
except Exception as error:
    st.error(f"Unable to load data: {error}")
    st.info(
        "Check that data/data.csv exists and "
        "src/data_loader.py contains load_data()."
    )
    st.stop()


# Validate required columns
required_columns = ["price", "city", "sqft_living"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# Work with a separate copy
df = df.copy()

# Ensure essential numeric columns are numeric
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["sqft_living"] = pd.to_numeric(
    df["sqft_living"], errors="coerce"
)

df = df.dropna(subset=["price", "sqft_living", "city"])
df = df[df["price"] > 0]
df = df[df["sqft_living"] > 0]

# Create a useful derived feature
df["price_per_sqft"] = df["price"] / df["sqft_living"]

# Format location labels
df["city"] = df["city"].astype(str)


# =========================================================
# 4. SIDEBAR & FILTERS
# =========================================================
with st.sidebar:
    st.markdown("# 🏡 EstateIQ")
    st.caption("PROPERTY ANALYTICS PLATFORM")

    st.divider()

    st.markdown("### Navigation")
    st.markdown("📊  Market Overview")
    st.markdown("🏠  Property Valuation")
    st.markdown("📍  Location Intelligence")
    st.markdown("🧠  Model Performance")
    st.markdown("🗂️  Property Explorer")

    st.divider()

    st.markdown("### Market Filters")

    available_cities = sorted(df["city"].unique().tolist())

    selected_cities = st.multiselect(
        "Select cities",
        options=available_cities,
        default=available_cities,
        help="Choose one or more cities to analyze."
    )

    min_price = float(df["price"].min())
    max_price = float(df["price"].max())

    if min_price < max_price:
        selected_price_range = st.slider(
            "Property price ($)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=max((max_price - min_price) / 500, 1.0),
            format="$%.0f"
        )
    else:
        selected_price_range = (min_price, max_price)

    bedroom_options = []
    if "bedrooms" in df.columns:
        bedroom_options = sorted(
            pd.to_numeric(
                df["bedrooms"], errors="coerce"
            ).dropna().unique().tolist()
        )

    selected_bedrooms = st.multiselect(
        "Bedrooms",
        options=bedroom_options,
        default=bedroom_options,
        help="Filter properties by bedroom count."
    )

    st.divider()

    st.caption("DATA SOURCE")
    st.caption("data/data.csv")
    st.caption("Analytics • Regression • Real Estate")


# =========================================================
# 5. APPLY FILTERS
# =========================================================
filtered_df = df[
    df["city"].isin(selected_cities)
    & df["price"].between(
        selected_price_range[0],
        selected_price_range[1]
    )
].copy()

if "bedrooms" in df.columns:
    bedroom_numeric = pd.to_numeric(
        df["bedrooms"], errors="coerce"
    )

    filtered_df = filtered_df[
        bedroom_numeric.loc[filtered_df.index].isin(
            selected_bedrooms
        )
    ]

if filtered_df.empty:
    st.warning(
        "No properties match your current filters. "
        "Adjust the filters in the sidebar."
    )
    st.stop()


# =========================================================
# 6. HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">REAL ESTATE INTELLIGENCE • ANALYTICS</div>
    <h1>House Price Prediction Studio</h1>
    <p>
        Explore property values, discover location insights,
        and understand the factors shaping housing prices.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 7. KPI CARDS
# =========================================================
total_properties = len(filtered_df)
median_price = filtered_df["price"].median()
average_price = filtered_df["price"].mean()
average_area = filtered_df["sqft_living"].mean()
median_ppsf = filtered_df["price_per_sqft"].median()

k1, k2, k3, k4 = st.columns(4)


def metric_card(label, value, note, accent):
    return f"""
    <div class="metric-card" style="border-top: 4px solid {accent};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


with k1:
    st.markdown(
        metric_card(
            "PROPERTIES ANALYZED",
            f"{total_properties:,}",
            "Records matching your filters",
            BLUE
        ),
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        metric_card(
            "MEDIAN PROPERTY PRICE",
            f"${median_price:,.0f}",
            "Middle value of filtered prices",
            TEAL
        ),
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        metric_card(
            "AVERAGE PROPERTY PRICE",
            f"${average_price:,.0f}",
            "Mean asking/sale price in dataset",
            PURPLE
        ),
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        metric_card(
            "MEDIAN PRICE / SQFT",
            f"${median_ppsf:,.0f}",
            f"Average living area: {average_area:,.0f} sqft",
            ORANGE
        ),
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# 8. MARKET OVERVIEW CHARTS
# =========================================================
st.markdown(
    '<div class="section-title">Market Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Understand property price distribution and compare city markets.'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.05, 1])

with left:
    fig_hist = px.histogram(
        filtered_df,
        x="price",
        nbins=45,
        title="Property Price Distribution",
        labels={"price": "Property Price ($)"},
        color_discrete_sequence=[TEAL],
        opacity=0.9
    )

    fig_hist.update_layout(
        template="plotly_white",
        height=390,
        showlegend=False,
        margin=dict(l=25, r=20, t=65, b=30),
        title_font_size=16,
        font=dict(family="Arial", color=TEXT),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#EDF1F7")
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

with right:
    city_summary = (
        filtered_df.groupby("city", as_index=False)
        .agg(
            median_price=("price", "median"),
            property_count=("price", "size")
        )
        .sort_values("median_price", ascending=False)
        .head(12)
    )

    fig_city = px.bar(
        city_summary,
        x="median_price",
        y="city",
        orientation="h",
        title="Top Cities by Median Price",
        labels={
            "median_price": "Median Price ($)",
            "city": "City"
        },
        color="median_price",
        color_continuous_scale=["#B9E7E2", TEAL, NAVY],
        hover_data={"property_count": True}
    )

    fig_city.update_layout(
        template="plotly_white",
        height=390,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=65, b=30),
        title_font_size=16,
        font=dict(family="Arial", color=TEXT),
        xaxis=dict(gridcolor="#EDF1F7"),
        yaxis_title=""
    )

    st.plotly_chart(
        fig_city,
        use_container_width=True
    )


# =========================================================
# 9. PRICE VS PROPERTY FEATURES
# =========================================================
st.markdown(
    '<div class="section-title">Property Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Explore the relationship between property size, '
    'bedrooms, and market price.'
    '</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.35, 1])

with left:
    scatter_columns = [
        c for c in ["city", "bedrooms", "bathrooms", "condition"]
        if c in filtered_df.columns
    ]

    fig_scatter = px.scatter(
        filtered_df,
        x="sqft_living",
        y="price",
        color="bedrooms" if "bedrooms" in filtered_df.columns else None,
        hover_data=scatter_columns,
        title="Living Area vs Property Price",
        labels={
            "sqft_living": "Living Area (sqft)",
            "price": "Property Price ($)",
            "bedrooms": "Bedrooms"
        },
        color_continuous_scale="Viridis",
        opacity=0.72
    )

    fig_scatter.update_layout(
        template="plotly_white",
        height=410,
        margin=dict(l=25, r=20, t=65, b=30),
        title_font_size=16,
        font=dict(family="Arial", color=TEXT),
        xaxis=dict(gridcolor="#EDF1F7"),
        yaxis=dict(gridcolor="#EDF1F7")
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

with right:
    if "bedrooms" in filtered_df.columns:
        bedroom_summary = (
            filtered_df.groupby("bedrooms", as_index=False)
            .agg(
                median_price=("price", "median"),
                properties=("price", "size")
            )
        )

        bedroom_summary["bedrooms"] = (
            bedroom_summary["bedrooms"].astype(str)
        )

        fig_bedrooms = px.bar(
            bedroom_summary,
            x="bedrooms",
            y="median_price",
            title="Median Price by Bedrooms",
            labels={
                "bedrooms": "Bedrooms",
                "median_price": "Median Price ($)"
            },
            color="median_price",
            color_continuous_scale=["#B9E7E2", TEAL, NAVY],
            hover_data={"properties": True}
        )

        fig_bedrooms.update_layout(
            template="plotly_white",
            height=410,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=65, b=30),
            title_font_size=16,
            font=dict(family="Arial", color=TEXT),
            xaxis=dict(type="category"),
            yaxis=dict(gridcolor="#EDF1F7")
        )

        st.plotly_chart(
            fig_bedrooms,
            use_container_width=True
        )
    else:
        st.info("A bedrooms column is not available in this dataset.")


# =========================================================
# 10. MARKET INSIGHTS
# =========================================================
st.markdown(
    '<div class="section-title">Market Highlights</div>',
    unsafe_allow_html=True
)

insight1, insight2, insight3 = st.columns(3)

highest_city = city_summary.iloc[0]
lowest_city = city_summary.sort_values(
    "median_price", ascending=True
).iloc[0]

with insight1:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">HIGHEST-MEDIAN-PRICE CITY</div>
            <div class="insight-value">{highest_city['city']}</div>
            <div class="metric-note">
                Median price: ${highest_city['median_price']:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with insight2:
    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color: {BLUE};">
            <div class="insight-label">LOWEST-MEDIAN-PRICE CITY</div>
            <div class="insight-value">{lowest_city['city']}</div>
            <div class="metric-note">
                Median price: ${lowest_city['median_price']:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with insight3:
    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color: {PURPLE};">
            <div class="insight-label">FILTERED PRICE RANGE</div>
            <div class="insight-value">
                ${filtered_df['price'].min():,.0f}
                – ${filtered_df['price'].max():,.0f}
            </div>
            <div class="metric-note">
                Lowest to highest property price
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 11. PROPERTY EXPLORER
# =========================================================
st.markdown(
    '<div class="section-title">Property Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Search, inspect, and export records from the filtered dataset.'
    '</div>',
    unsafe_allow_html=True
)

search_col, sort_col = st.columns([2, 1])

with search_col:
    search_city = st.text_input(
        "Search city",
        placeholder="Enter a city name..."
    )

with sort_col:
    sort_order = st.selectbox(
        "Sort properties by",
        [
            "Price: High to Low",
            "Price: Low to High",
            "Living Area: High to Low"
        ]
    )

explorer_df = filtered_df.copy()

if search_city:
    explorer_df = explorer_df[
        explorer_df["city"].str.contains(
            search_city, case=False, na=False
        )
    ]

if sort_order == "Price: High to Low":
    explorer_df = explorer_df.sort_values(
        "price", ascending=False
    )
elif sort_order == "Price: Low to High":
    explorer_df = explorer_df.sort_values(
        "price", ascending=True
    )
else:
    explorer_df = explorer_df.sort_values(
        "sqft_living", ascending=False
    )

st.caption(f"{len(explorer_df):,} properties found")

display_columns = [
    c for c in [
        "city", "statezip", "price", "bedrooms",
        "bathrooms", "sqft_living", "sqft_lot",
        "yr_built", "condition", "price_per_sqft"
    ]
    if c in explorer_df.columns
]

st.dataframe(
    explorer_df[display_columns],
    use_container_width=True,
    hide_index=True,
    height=350,
    column_config={
        "price": st.column_config.NumberColumn(
            "Price ($)", format="$%.0f"
        ),
        "price_per_sqft": st.column_config.NumberColumn(
            "Price / sqft", format="$%.2f"
        ),
        "sqft_living": st.column_config.NumberColumn(
            "Living Area", format="%d sqft"
        )
    }
)

csv_data = explorer_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Export Filtered Properties (CSV)",
    data=csv_data,
    file_name="estateiq_filtered_properties.csv",
    mime="text/csv"
)


# =========================================================
# 12. FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    <b>ESTATEIQ</b> &nbsp;|&nbsp;
    House Price Prediction Studio &nbsp;|&nbsp;
    Built with Streamlit, Pandas and Plotly
    <br><br>
    Market statistics describe the selected dataset and filters.
    They are not machine-learning predictions.
</div>
""", unsafe_allow_html=True)
