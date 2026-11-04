
from src.data_loader import load_data
from src.preprocessing import prepare_data

dfv = load_data()

X, y, preprocessor = prepare_data(dfv)

print("Original dataset shape:", dfv.shape)
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget summary:")
print(y.describe())

print("\nMissing target values:", y.isna().sum())
print("Invalid target values:", (y <= 0).sum())

print("\nNumerical features:")
print(X.select_dtypes(include="number").columns.tolist())

print("\nCategorical features:")
print(X.select_dtypes(include=["object", "category"]).columns.tolist())

print("\nPreprocessing pipeline:")
print(preprocessor)

print("\nPreprocessing setup validated!")
