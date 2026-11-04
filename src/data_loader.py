
import pandas as pd


def load_data():
    """Load the housing dataset and perform basic validation."""

    df = pd.read_csv(
        "C:/Users/Chamanpreet Kaur/OneDrive/Desktop/House Price Prediction/data/data.csv"
    )

    required_columns = ["price", "city", "sqft_living"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The housing dataset is empty.")

    return df


# df = load_data()
# print(df.head())
