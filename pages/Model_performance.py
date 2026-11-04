
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from pathlib import Path


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Model Performance | Property Studio",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Performance Lab")
st.caption(
    "Evaluate regression models, investigate prediction errors, "
    "and understand the features behind house-price estimates."
)

# -----------------------------
# LOAD RESULTS
# -----------------------------
comparison_path = Path("models/model_comparison.csv")
predictions_path = Path("models/test_predictions.csv")
model_path = Path("models/best_model.joblib")

if not all(
    path.exists()
    for path in [
        comparison_path,
        predictions_path,
        model_path,
    ]
):
    st.error(
        "Model files are missing. Train your models first "
        "using: python -m src.train_model"
    )
    st.stop()

results = pd.read_csv(comparison_path)
predictions = pd.read_csv(predictions_path)
best_model = joblib.load(model_path)

required_metrics = {"Model", "MAE", "RMSE", "R2"}
required_predictions = {"Actual Price", "Predicted Price"}

if not required_metrics.issubset(results.columns):
    st.error("The model comparison file has an unexpected format.")
    st.stop()

if not required_predictions.issubset(predictions.columns):
    st.error("The test predictions file has an unexpected format.")
    st.stop()

if results.empty or predictions.empty:
    st.warning("The saved evaluation results are empty.")
    st.stop()

predictions = predictions.copy()
predictions["Residual"] = (
    predictions["Actual Price"]
    - predictions["Predicted Price"]
)

# -----------------------------
# KPI CARDS
# -----------------------------
best_result = results.sort_values("RMSE").iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Best Model by RMSE",
    best_result["Model"],
)

col2.metric(
    "Test RMSE",
    f"${best_result['RMSE']:,.0f}",
)

col3.metric(
    "Test MAE",
    f"${best_result['MAE']:,.0f}",
)

col4.metric(
    "Test R²",
    f"{best_result['R2']:.3f}",
)

st.divider()

# -----------------------------
# MODEL COMPARISON
# -----------------------------
st.subheader("1. Regression Model Comparison")

metric_choice = st.selectbox(
    "Choose evaluation metric",
    ["RMSE", "MAE", "R2"],
)

ascending = metric_choice != "R2"

comparison_sorted = results.sort_values(
    metric_choice,
    ascending=ascending,
)

fig_comparison = px.bar(
    comparison_sorted,
    x="Model",
    y=metric_choice,
    color="Model",
    text_auto=".3f",
    title=f"Models Compared by {metric_choice}",
    labels={metric_choice: metric_choice},
)

fig_comparison.update_layout(
    template="plotly_white",
    showlegend=False,
    height=400,
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True,
)

st.dataframe(
    results.sort_values("RMSE"),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Model Comparison",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv",
)

st.divider()

# -----------------------------
# ACTUAL VS PREDICTED
# -----------------------------
st.subheader("2. Actual vs Predicted Prices")

fig_actual = px.scatter(
    predictions,
    x="Actual Price",
    y="Predicted Price",
    title="Test Set: Actual Prices vs Predictions",
    opacity=0.65,
    hover_data=["Residual"],
)

# Reference line: perfect predictions
minimum = min(
    predictions["Actual Price"].min(),
    predictions["Predicted Price"].min(),
)
maximum = max(
    predictions["Actual Price"].max(),
    predictions["Predicted Price"].max(),
)

fig_actual.add_shape(
    type="line",
    x0=minimum,
    y0=minimum,
    x1=maximum,
    y1=maximum,
    line=dict(dash="dash", width=2),
)

fig_actual.update_layout(
    template="plotly_white",
    height=480,
    xaxis_title="Actual Price ($)",
    yaxis_title="Predicted Price ($)",
)

st.plotly_chart(
    fig_actual,
    use_container_width=True,
)

st.caption(
    "Points near the diagonal represent accurate predictions. "
    "Points farther from the line have larger prediction errors."
)

st.divider()

# -----------------------------
# RESIDUAL ANALYSIS
# -----------------------------
st.subheader("3. Residual Error Analysis")

left, right = st.columns(2)

with left:
    fig_residual = px.histogram(
        predictions,
        x="Residual",
        nbins=40,
        title="Distribution of Prediction Errors",
        labels={"Residual": "Actual − Predicted ($)"},
    )

    fig_residual.add_vline(
        x=0,
        line_dash="dash",
    )

    fig_residual.update_layout(
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(
        fig_residual,
        use_container_width=True,
    )

with right:
    fig_residual_scatter = px.scatter(
        predictions,
        x="Predicted Price",
        y="Residual",
        title="Residuals vs Predicted Price",
        opacity=0.6,
        labels={
            "Residual": "Actual − Predicted ($)",
            "Predicted Price": "Predicted Price ($)",
        },
    )

    fig_residual_scatter.add_hline(
        y=0,
        line_dash="dash",
    )

    fig_residual_scatter.update_layout(
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(
        fig_residual_scatter,
        use_container_width=True,
    )

st.caption(
    "Positive residuals mean the model underestimated the price. "
    "Negative residuals mean it overestimated the price."
)

st.divider()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("4. Feature Importance")

try:
    preprocessor = best_model.named_steps["preprocessor"]
    estimator = best_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
        importance_label = "Feature Importance"

    elif hasattr(estimator, "coef_"):
        # Coefficient magnitude is a proxy, not tree importance.
        importance_values = np.abs(
            np.asarray(estimator.coef_).ravel()
        )
        importance_label = "Absolute Coefficient Magnitude"

    else:
        st.info(
            "The selected model does not expose feature "
            "importance or coefficients."
        )
        importance_values = None

    if importance_values is not None:
        if len(feature_names) != len(importance_values):
            st.warning(
                "Feature names and importance values do not match."
            )
        else:
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                importance_label: importance_values,
            })

            importance_df = importance_df.sort_values(
                importance_label,
                ascending=False,
            ).head(15)

            fig_importance = px.bar(
                importance_df.sort_values(importance_label),
                x=importance_label,
                y="Feature",
                orientation="h",
                title="Top 15 Features",
            )

            fig_importance.update_layout(
                template="plotly_white",
                height=520,
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True,
            )

            st.caption(
                "Tree-based importance measures model contribution "
                "under the estimator's importance definition. "
                "Absolute linear coefficients show magnitude after "
                "preprocessing; neither proves causation."
            )

except (AttributeError, KeyError, ValueError) as error:
    st.warning(
        f"Could not calculate feature importance: {error}"
    )

# -----------------------------
# DOWNLOAD TEST PREDICTIONS
# -----------------------------
st.divider()

st.subheader("5. Export Test Predictions")

st.dataframe(
    predictions.head(100),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Test Predictions",
    data=predictions.to_csv(index=False).encode("utf-8"),
    file_name="test_predictions.csv",
    mime="text/csv",
)

st.caption(
    "House Price Prediction Studio | Model Evaluation"
)
