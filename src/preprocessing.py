
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(dfv):
    """Clean the housing data and create prediction features."""

    dfv = dfv.copy()

    # 1. Remove duplicate records
    dfv = dfv.drop_duplicates()

    # 2. Convert the target to numeric
    dfv["price"] = pd.to_numeric(
        dfv["price"], errors="coerce"
    )

    # 3. Remove rows with invalid target values
    dfv = dfv.dropna(subset=["price"])
    dfv = dfv[dfv["price"] > 0].copy()

    # 4. Create a renovation indicator
    if "yr_renovated" in dfv.columns:
        dfv["was_renovated"] = (
            pd.to_numeric(
                dfv["yr_renovated"], errors="coerce"
            ).fillna(0) > 0
        ).astype(int)

    # 5. Extract useful information from the sale date
    if "date" in dfv.columns:
        sale_date = pd.to_datetime(
            dfv["date"], errors="coerce"
        )

        if sale_date.notna().any():
            dfv["sale_year"] = sale_date.dt.year
            dfv["sale_month"] = sale_date.dt.month

    # 6. Separate target and input features
    y = dfv["price"]

    # Exclude the target and raw identifiers/address fields
    drop_columns = [
        "price",
        "street",
        "country",
        "date",
    ]

    X = dfv.drop(
        columns=drop_columns,
        errors="ignore"
    )

    # 7. Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # 8. Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # 9. Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ]
    )

    # 10. Combine both preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
        ],
        remainder="drop"
    )

    return X, y, preprocessor

# Load the dataset
df = pd.read_csv(
    "C:/Users/Chamanpreet Kaur/OneDrive/Desktop/House Price Prediction/data/data.csv"
)

# Call the function
X, y, preprocessor = prepare_data(df)

# Display the results
print("Dataset loaded successfully!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeature columns:")
print(X.columns.tolist())
print("\nTarget preview:")
print(y.head())

print("\nPreprocessor created successfully!")


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Identify column types
numerical_columns = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# 2. Numerical pipeline
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 3. Categorical pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# 4. Combine pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ],
    remainder="drop",
    sparse_threshold=0.0,
)
