
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from src.data_loader import load_data 
from src.preprocessing import prepare_data


def train_models():
    # 1. Load and prepare the dataset
    dfv = load_data()
    X, y, preprocessor = prepare_data(dfv)

    # 2. Split before fitting any preprocessing steps
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # 3. Define the models
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    # 4. Train and evaluate each model
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(
            y_test, predictions
        ) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        trained_models[name] = pipeline

        print(f"\n{name}")
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:   {r2:.4f}")

    # 5. Compare results
    results_df = pd.DataFrame(results).sort_values(
        by="RMSE",
        ascending=True,
    )

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    # 6. Save the best model and evaluation results
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/best_model.joblib")

    results_df.to_csv(
        "models/model_comparison.csv",
        index=False,
    )

    # Save the test predictions for later visualizations
    prediction_df = pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": best_model.predict(X_test),
    })

    prediction_df.to_csv(
        "models/test_predictions.csv",
        index=False,
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest model by RMSE: {best_name}")
    print("Saved: models/best_model.joblib")
    print("Saved: models/model_comparison.csv")
    print("Saved: models/test_predictions.csv")

    return results_df, best_model


if __name__ == "__main__":
    train_models()
