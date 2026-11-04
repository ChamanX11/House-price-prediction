
# import pandas as pd
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score,
# )


# def evaluate_model(y_true, y_pred):
#     """Return standard regression evaluation metrics."""

#     return {
#         "MAE": mean_absolute_error(y_true, y_pred),
#         "RMSE": mean_squared_error(
#             y_true, y_pred
#         ) ** 0.5,
#         "R2": r2_score(y_true, y_pred),
#     }


# def load_model_comparison():
#     """Load saved model comparison results."""

#     return pd.read_csv("C:/Users/Chamanpreet Kaur/OneDrive/Desktop/House Price Prediction/data/data.csv"
#         )

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(y_true, y_pred):
    """Return standard regression evaluation metrics."""

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def load_model_comparison():
    """Load saved model comparison results."""

    return pd.read_csv("C:/Users/Chamanpreet Kaur/OneDrive/Desktop/House Price Prediction/data/data.csv")


# Test the functions
print("Evaluation metrics:")

actual = [200000, 300000, 400000]
predicted = [210000, 290000, 390000]

print(evaluate_model(actual, predicted))

print("\nModel comparison:")
print(load_model_comparison())
