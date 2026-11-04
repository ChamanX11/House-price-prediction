
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------
st.set_page_config(
    page_title="Property Explorer | Property Studio",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Property Explorer")
st.caption(
    "Search, filter, analyze, and export housing records."
)


# --------------------------------
# LOAD DATA
# --------------------------------
try:
    dfv = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load housing data: {error}")
    st.stop()

dfv = dfv.copy()

# Ensure important numerical columns are numeric
numeric_columns = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
]

for column in numeric_columns:
    if column in dfv.columns:
        dfv[column] = pd.to_numeric(
            dfv[column], errors="coerce"
        )

if "price" not in dfv.columns:
    st.error("The dataset must contain a price column.")
    st.stop()

# Keep the original records; invalid values are excluded
# only where necessary for filtering and charts.
valid_prices = dfv["price"].dropna()

if valid_prices.empty:
    st.error("No valid property prices were found.")
    st.stop()


# --------------------------------
# SIDEBAR FILTERS
# --------------------------------
st.sidebar.header("Property Search")

search_text = st.sidebar.text_input(
    "Search property details",
    placeholder="Enter city, street, or ZIP..."
)

filtered_df = dfv.copy()

# City filter
if "city" in filtered_df.columns:
    city_options = sorted(
        filtered_df["city"].dropna().astype(str).unique()
    )

    selected_cities = st.sidebar.multiselect(
        "City",
        options=city_options,
        default=city_options,
    )

    filtered_df = filtered_df[
        filtered_df["city"].astype(str).isin(selected_cities)
    ]

# Price filter
price_values = filtered_df["price"].dropna()

if not price_values.empty:
    min_price = float(price_values.min())
    max_price = float(price_values.max())

    if min_price < max_price:
        price_range = st.sidebar.slider(
            "Price range ($)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            step=max((max_price - min_price) / 100, 1.0),
        )

        filtered_df = filtered_df[
            filtered_df["price"].between(
                price_range[0], price_range[1]
            )
        ]

# Bedrooms filter
if "bedrooms" in filtered_df.columns:
    values = filtered_df["bedrooms"].dropna()

    if not values.empty:
        bedroom_min = int(values.min())
        bedroom_max = int(values.max())

        if bedroom_min < bedroom_max:
            bedroom_range = st.sidebar.slider(
                "Bedrooms",
                min_value=bedroom_min,
                max_value=bedroom_max,
                value=(bedroom_min, bedroom_max),
            )

            filtered_df = filtered_df[
                filtered_df["bedrooms"].between(
                    bedroom_range[0],
                    bedroom_range[1],
                )
            ]

# Bathrooms filter
if "bathrooms" in filtered_df.columns:
    values = filtered_df["bathrooms"].dropna()

    if not values.empty:
        bathroom_min = float(values.min())
        bathroom_max = float(values.max())

        if bathroom_min < bathroom_max:
            bathroom_range = st.sidebar.slider(
                "Bathrooms",
                min_value=bathroom_min,
                max_value=bathroom_max,
                value=(bathroom_min, bathroom_max),
                step=0.5,
            )

            filtered_df = filtered_df[
                filtered_df["bathrooms"].between(
                    bathroom_range[0],
                    bathroom_range[1],
                )
            ]

# Living area filter
if "sqft_living" in filtered_df.columns:
    values = filtered_df["sqft_living"].dropna()

    if not values.empty:
        area_min = float(values.min())
        area_max = float(values.max())

        if area_min < area_max:
            area_range = st.sidebar.slider(
                "Living area (sqft)",
                min_value=area_min,
                max_value=area_max,
                value=(area_min, area_max),
                step=max((area_max - area_min) / 100, 1.0),
            )

            filtered_df = filtered_df[
                filtered_df["sqft_living"].between(
                    area_range[0],
                    area_range[1],
                )
            ]

# Text search across available descriptive fields
if search_text.strip():
    search_columns = [
        column
        for column in ["city", "street", "statezip", "country"]
        if column in filtered_df.columns
    ]

    if search_columns:
        search_mask = pd.Series(
            False, index=filtered_df.index
        )

        for column in search_columns:
            search_mask |= (
                filtered_df[column]
                .astype("string")
                .str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            )

        filtered_df = filtered_df[search_mask]


# --------------------------------
# EMPTY RESULTS CHECK
# --------------------------------
if filtered_df.empty:
    st.warning(
        "No properties match the selected filters. "
        "Adjust your search and try again."
    )
    st.stop()


# --------------------------------
# SUMMARY METRICS
# --------------------------------
st.subheader("Filtered Portfolio")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Properties Found",
    f"{len(filtered_df):,}",
)

col2.metric(
    "Median Price",
    f"${filtered_df['price'].median():,.0f}",
)

col3.metric(
    "Average Price",
    f"${filtered_df['price'].mean():,.0f}",
)

if "sqft_living" in filtered_df.columns:
    col4.metric(
        "Average Living Area",
        f"{filtered_df['sqft_living'].mean():,.0f} sqft",
    )
else:
    col4.metric(
        "Cities Found",
        f"{filtered_df['city'].nunique():,}"
        if "city" in filtered_df.columns else "N/A",
    )

st.divider()


# --------------------------------
# PRICE DISTRIBUTION
# --------------------------------
st.subheader("1. Price Distribution")

fig_price = px.histogram(
    filtered_df,
    x="price",
    nbins=40,
    marginal="box",
    title="Distribution of Property Prices",
    labels={"price": "Property Price ($)"},
    color_discrete_sequence=["#168C8C"],
)

fig_price.update_layout(
    template="plotly_white",
    height=430,
)

st.plotly_chart(
    fig_price,
    use_container_width=True,
)


# --------------------------------
# AREA VS PRICE
# --------------------------------
if {"sqft_living", "price"}.issubset(filtered_df.columns):
    st.subheader("2. Property Size vs Price")

    scatter_columns = [
        column
        for column in [
            "bedrooms",
            "bathrooms",
            "city",
            "condition",
        ]
        if column in filtered_df.columns
    ]

    fig_scatter = px.scatter(
        filtered_df,
        x="sqft_living",
        y="price",
        color="bedrooms" if "bedrooms" in filtered_df.columns else None,
        hover_data=scatter_columns,
        title="Living Area and Property Price",
        labels={
            "sqft_living": "Living Area (sqft)",
            "price": "Property Price ($)",
            "bedrooms": "Bedrooms",
        },
        opacity=0.65,
    )

    fig_scatter.update_layout(
        template="plotly_white",
        height=450,
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
    )


# --------------------------------
# FEATURE CORRELATION
# --------------------------------
st.subheader("3. Numerical Feature Relationships")

numeric_df = filtered_df.select_dtypes(
    include="number"
)

if numeric_df.shape[1] >= 2:
    correlation = numeric_df.corr()

    fig_corr = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Matrix",
    )

    fig_corr.update_layout(
        template="plotly_white",
        height=600,
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True,
    )

    st.caption(
        "Correlation measures linear association between numerical "
        "variables. It does not prove causation."
    )
else:
    st.info(
        "At least two numerical columns are required "
        "to calculate correlations."
    )


# --------------------------------
# PROPERTY TABLE
# --------------------------------
st.divider()
st.subheader("4. Property Records")

st.caption(
    f"Displaying {len(filtered_df):,} matching records."
)

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
)

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Properties",
    data=csv_data,
    file_name="filtered_properties.csv",
    mime="text/csv",
    use_container_width=True,
)


# --------------------------------
# FOOTER
# --------------------------------
st.divider()

st.caption(
    "House Price Prediction Studio | "
    "Property Explorer | Streamlit + Plotly"
)
