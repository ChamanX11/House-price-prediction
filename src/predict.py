
import joblib
import pandas as pd

from src.preprocessing import prepare_data


MODEL_PATH = "models/best_model.joblib"


def load_trained_model():
    """Load the trained machine-learning pipeline."""
    return joblib.load(MODEL_PATH)


def predict_property(property_data, model):
    """Predict a property's price using the trained pipeline."""

    # Convert the property dictionary into a DataFrame.
    input_df = pd.DataFrame([property_data.copy()])

    # prepare_data expects the target column to exist.
    # Its value is only a placeholder and is not used for prediction.
    input_df["price"] = 1.0

    # Apply the same feature-engineering logic as training.
    X_input, _, _ = prepare_data(input_df)

    # Get the exact input columns expected by the fitted preprocessor.
    preprocessor = model.named_steps["preprocessor"]
    expected_columns = list(preprocessor.feature_names_in_)

    # Restore any missing training columns.
    # The fitted imputer will handle their missing values.
    for column in expected_columns:
        if column not in X_input.columns:
            X_input[column] = float("nan")

    # Remove any unexpected columns and preserve training order.
    X_input = X_input[expected_columns]

    # Predict using the complete saved pipeline.
    predicted_price = model.predict(X_input)[0]

    return max(0.0, float(predicted_price))
