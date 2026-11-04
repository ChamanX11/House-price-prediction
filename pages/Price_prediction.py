
import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.predict import load_trained_model, predict_property


st.set_page_config(
    page_title="Price Prediction | Property Studio",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 Price Prediction Studio")
st.caption(
    "Estimate property value using your trained "
    "machine learning model."
)

# Load dataset and trained model
try:
    dfv = load_data()
    model = load_trained_model()
except FileNotFoundError:
    st.error(
        "Dataset or trained model not found. "
        "Check data/data.csv and models/best_model.joblib."
    )
    st.info("Train the models first using: python -m src.train_model")
    st.stop()

st.divider()

st.subheader("Enter Property Details")
st.write(
    "Enter the property specifications below to estimate its price."
)

# Start with a real record so all raw input columns exist.
property_data = dfv.iloc[0].to_dict()

# Location
left, right = st.columns(2)

with left:
    if "city" in dfv.columns:
        cities = sorted(dfv["city"].dropna().unique().tolist())
        property_data["city"] = st.selectbox(
            "City",
            cities,
            index=cities.index(property_data["city"])
            if property_data["city"] in cities else 0,
        )

with right:
    if "statezip" in dfv.columns:
        locations = sorted(
            dfv["statezip"].dropna().unique().tolist()
        )
        property_data["statezip"] = st.selectbox(
            "State / ZIP",
            locations,
            index=locations.index(property_data["statezip"])
            if property_data["statezip"] in locations else 0,
        )

st.markdown("### Property Specifications")

# Only show fields present in the dataset.
numeric_fields = [
    ("bedrooms", "Bedrooms", 1.0),
    ("bathrooms", "Bathrooms", 0.5),
    ("sqft_living", "Living Area (sqft)", 100.0),
    ("sqft_lot", "Lot Area (sqft)", 100.0),
    ("floors", "Floors", 0.5),
    ("waterfront", "Waterfront (0 = No, 1 = Yes)", 1.0),
    ("view", "View Rating", 1.0),
    ("condition", "Condition Rating", 1.0),
    ("sqft_above", "Above-ground Area (sqft)", 100.0),
    ("sqft_basement", "Basement Area (sqft)", 100.0),
    ("yr_built", "Year Built", 1.0),
    ("yr_renovated", "Year Renovated (0 = Never)", 1.0),
]

available_fields = [
    item for item in numeric_fields
    if item[0] in dfv.columns
]

columns = st.columns(3)

for index, (column, label, step) in enumerate(available_fields):
    series = pd.to_numeric(dfv[column], errors="coerce").dropna()

    if series.empty:
        continue

    default_value = pd.to_numeric(
        pd.Series([property_data[column]]),
        errors="coerce"
    ).iloc[0]

    if pd.isna(default_value):
        default_value = series.median()

    minimum = float(series.min())
    maximum = float(series.max())

    # Expand bounds slightly so the starting value is valid.
    default_value = float(default_value)
    minimum = min(minimum, default_value)
    maximum = max(maximum, default_value)

    with columns[index % 3]:
        property_data[column] = st.number_input(
            label,
            min_value=minimum,
            max_value=maximum,
            value=default_value,
            step=step,
            key=f"property_{column}",
        )

st.divider()

if st.button(
    "Estimate Property Price",
    type="primary",
    use_container_width=True,
):
    try:
        predicted_price = predict_property(
            property_data,
            model,
        )

        st.success("Price estimate generated successfully!")

        st.metric(
            "Estimated Property Price",
            f"${predicted_price:,.0f}",
        )

        st.caption(
            "This is a model-generated estimate, not a formal "
            "property appraisal or guaranteed market price."
        )

        with st.expander("Review entered property details"):
            st.json({
                key: str(value)
                for key, value in property_data.items()
                if key != "price"
            })

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.caption(
            "Check that the model was trained with the current "
            "preprocessing code and dataset."
        )
