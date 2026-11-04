
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------
st.set_page_config(
    page_title="Location Intelligence | Property Studio",
    page_icon="🌎",
    layout="wide",
)

st.title("🌎 Location Intelligence")
st.caption(
    "Explore housing prices, property sizes, and market differences "
    "across locations in your dataset."
)


# -----------------------------------
# LOAD DATA
# -----------------------------------
try:
    dfv = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()


# -----------------------------------
# VALIDATE REQUIRED COLUMNS
# -----------------------------------
required_columns = [
    "city",
    "price",
    "sqft_living",
    "bedrooms",
]

missing_columns = [
    column for column in required_columns
    if column not in dfv.columns
]

if missing_columns:
    st.error(
        f"Missing columns required for this page: {missing_columns}"
    )
    st.stop()


# -----------------------------------
# PREPARE DATA
# -----------------------------------
dfv = dfv.copy()

for column in ["price", "sqft_living", "bedrooms"]:
    dfv[column] = pd.to_numeric(
        dfv[column], errors="coerce"
    )

dfv = dfv.dropna(
    subset=["city", "price", "sqft_living", "bedrooms"]
)

dfv = dfv[
    (dfv["price"] > 0)
    & (dfv["sqft_living"] > 0)
].copy()

if dfv.empty:
    st.warning("No valid property records are available.")
    st.stop()

# Calculate price per square foot
dfv["price_per_sqft"] = (
    dfv["price"] / dfv["sqft_living"]
)


# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------
st.sidebar.header("Location Filters")

cities = sorted(dfv["city"].astype(str).unique())

selected_cities = st.sidebar.multiselect(
    "Select cities",
    options=cities,
    default=cities,
)

min_price = float(dfv["price"].min())
max_price = float(dfv["price"].max())

if min_price < max_price:
    selected_price_range = st.sidebar.slider(
        "Property price range ($)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=max((max_price - min_price) / 100, 1.0),
    )
else:
    selected_price_range = (min_price, max_price)

min_bedrooms = int(dfv["bedrooms"].min())
max_bedrooms = int(dfv["bedrooms"].max())

selected_bedrooms = st.sidebar.slider(
    "Bedroom count",
    min_value=min_bedrooms,
    max_value=max_bedrooms,
    value=(min_bedrooms, max_bedrooms),
)


# -----------------------------------
# APPLY FILTERS
# -----------------------------------
filtered_df = dfv[
    dfv["city"].astype(str).isin(selected_cities)
    & dfv["price"].between(
        selected_price_range[0],
        selected_price_range[1],
    )
    & dfv["bedrooms"].between(
        selected_bedrooms[0],
        selected_bedrooms[1],
    )
].copy()

if filtered_df.empty:
    st.warning(
        "No properties match these filters. "
        "Adjust your selections in the sidebar."
    )
    st.stop()


# -----------------------------------
# MARKET SUMMARY
# -----------------------------------
st.subheader("Market Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Properties Analyzed",
    f"{len(filtered_df):,}",
)

col2.metric(
    "Median Property Price",
    f"${filtered_df['price'].median():,.0f}",
)

col3.metric(
    "Median Price / Sqft",
    f"${filtered_df['price_per_sqft'].median():,.2f}",
)

col4.metric(
    "Cities Selected",
    f"{filtered_df['city'].nunique():,}",
)

st.divider()


# -----------------------------------
# CITY-LEVEL AGGREGATION
# -----------------------------------
city_summary = (
    filtered_df.groupby("city", as_index=False)
    .agg(
        median_price=("price", "median"),
        average_price=("price", "mean"),
        median_price_per_sqft=("price_per_sqft", "median"),
        average_living_area=("sqft_living", "mean"),
        property_count=("price", "size"),
        median_bedrooms=("bedrooms", "median"),
    )
    .sort_values("median_price", ascending=False)
)


# -----------------------------------
# CITY PRICE COMPARISON
# -----------------------------------
st.subheader("1. Property Prices by City")

left, right = st.columns(2)

with left:
    fig_median = px.bar(
        city_summary.sort_values("median_price"),
        x="median_price",
        y="city",
        orientation="h",
        color="median_price",
        color_continuous_scale="Teal",
        title="Median Property Price",
        labels={
            "median_price": "Median Price ($)",
            "city": "City",
        },
        hover_data=["property_count"],
    )

    fig_median.update_layout(
        template="plotly_white",
        height=480,
    )

    st.plotly_chart(
        fig_median,
        use_container_width=True,
    )

with right:
    fig_ppsf = px.bar(
        city_summary.sort_values("median_price_per_sqft"),
        x="median_price_per_sqft",
        y="city",
        orientation="h",
        color="median_price_per_sqft",
        color_continuous_scale="Viridis",
        title="Median Price per Square Foot",
        labels={
            "median_price_per_sqft": "Price / Sqft ($)",
            "city": "City",
        },
    )

    fig_ppsf.update_layout(
        template="plotly_white",
        height=480,
    )

    st.plotly_chart(
        fig_ppsf,
        use_container_width=True,
    )


# -----------------------------------
# PROPERTY SIZE VS PRICE
# -----------------------------------
st.divider()
st.subheader("2. Living Area vs Property Price")

fig_area = px.scatter(
    filtered_df,
    x="sqft_living",
    y="price",
    color="city",
    size="bedrooms",
    hover_data=[
        "price_per_sqft",
        "sqft_lot",
        "bathrooms",
    ],
    title="How Property Size Relates to Price",
    labels={
        "sqft_living": "Living Area (sqft)",
        "price": "Property Price ($)",
        "city": "City",
        "bedrooms": "Bedrooms",
    },
    opacity=0.7,
)

fig_area.update_layout(
    template="plotly_white",
    height=500,
)

st.plotly_chart(
    fig_area,
    use_container_width=True,
)


# -----------------------------------
# BEDROOM ANALYSIS
# -----------------------------------
st.divider()
st.subheader("3. Bedroom and Price Analysis")

bedroom_summary = (
    filtered_df.groupby("bedrooms", as_index=False)
    .agg(
        median_price=("price", "median"),
        property_count=("price", "size"),
    )
    .sort_values("bedrooms")
)

fig_bedrooms = px.bar(
    bedroom_summary,
    x="bedrooms",
    y="median_price",
    text_auto=".2s",
    title="Median Property Price by Bedroom Count",
    labels={
        "bedrooms": "Bedrooms",
        "median_price": "Median Price ($)",
    },
    color="median_price",
    color_continuous_scale="Teal",
    hover_data=["property_count"],
)

fig_bedrooms.update_layout(
    template="plotly_white",
    height=400,
)

st.plotly_chart(
    fig_bedrooms,
    use_container_width=True,
)


# -----------------------------------
# CITY COMPARISON TABLE
# -----------------------------------
st.divider()
st.subheader("4. Detailed City Comparison")

display_summary = city_summary.rename(
    columns={
        "city": "City",
        "median_price": "Median Price ($)",
        "average_price": "Average Price ($)",
        "median_price_per_sqft": "Median Price / Sqft ($)",
        "average_living_area": "Average Living Area (sqft)",
        "property_count": "Property Count",
        "median_bedrooms": "Median Bedrooms",
    }
)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Median Price ($)": st.column_config.NumberColumn(
            format="$%.2f"
        ),
        "Average Price ($)": st.column_config.NumberColumn(
            format="$%.2f"
        ),
        "Median Price / Sqft ($)": st.column_config.NumberColumn(
            format="$%.2f"
        ),
        "Average Living Area (sqft)": st.column_config.NumberColumn(
            format="%.0f sqft"
        ),
    },
)

st.download_button(
    label="Download City Analysis CSV",
    data=display_summary.to_csv(index=False).encode("utf-8"),
    file_name="city_market_analysis.csv",
    mime="text/csv",
)

st.caption(
    "Market comparisons describe the filtered dataset. "
    "Differences between cities do not establish that location "
    "alone causes differences in property prices."
)
