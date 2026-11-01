import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

# Add flower names
df["species"] = df["target"].map(
    dict(enumerate(iris.target_names))
)

# Save dataset
df.to_csv("data/iris.csv", index=False)

print("Iris dataset created successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())