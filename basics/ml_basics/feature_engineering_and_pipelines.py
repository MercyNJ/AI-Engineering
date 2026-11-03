
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Encoding categorical variables
df = pd.DataFrame({
    "city": ["Nairobi", "Lagos", "Cairo", "Nairobi", "Lagos"],
    "tier": ["gold", "silver", "gold", "bronze", "silver"],
})

city_encoded = pd.get_dummies(df["city"], prefix="city")
print("One-hot encoded cities:")
print(city_encoded)

tier_order = {"bronze": 0, "silver": 1, "gold": 2}
df["tier_encoded"] = df["tier"].map(tier_order)
print("\nOrdinal encoded tiers:")
print(df["tier_encoded"])


# Scaling numerical features
X_example = np.array([
    [20, 30000],
    [35, 60000],
    [50, 90000],
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_example)
print("\nScaled features:")
print(X_scaled)


# Handling missing values
X_missing = np.array([
    [1, 2, np.nan],
    [4, np.nan, 6],
    [7, 8, 9],
])

imputer = SimpleImputer(strategy="mean")
X_filled = imputer.fit_transform(X_missing)
print("\nImputed values:")
print(X_filled)


# Creating new features
customers = pd.DataFrame({
    "age": [25, 40, 32],
    "income": [30000, 60000, 45000],
    "transaction_amount": [500, 2000, 800],
})

customers["age_income_ratio"] = (
    customers["age"] / (customers["income"] + 1)
)
customers["log_amount"] = np.log1p(
    customers["transaction_amount"]
)
print("\nEngineered features:")
print(customers)


# Build a complete classification pipeline
data = pd.DataFrame({
    "age": [22, 35, 28, 45, 31, 52, 26, 40, 33, 48,
            24, 38, 29, 55, 36, 43, 27, 50, 32, 41],
    "income": [25000, 60000, 35000, 80000, 45000, 90000,
               30000, 70000, 50000, 85000, 28000, 65000,
               40000, 95000, 55000, 75000, 32000, 88000,
               48000, 72000],
    "tenure": [2, 8, 3, 10, 5, 12, 2, 9, 6, 11,
               1, 7, 4, 12, 6, 9, 3, 11, 5, 8],
    "city": ["Nairobi", "Lagos", "Cairo", "Nairobi", "Lagos",
             "Cairo", "Nairobi", "Lagos", "Cairo", "Nairobi",
             "Lagos", "Cairo", "Nairobi", "Lagos", "Cairo",
             "Nairobi", "Lagos", "Cairo", "Nairobi", "Lagos"],
    "plan": ["basic", "premium", "basic", "premium", "standard",
             "premium", "basic", "standard", "standard", "premium",
             "basic", "standard", "basic", "premium", "standard",
             "premium", "basic", "premium", "standard", "premium"],
    "churn": [1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
              1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
})

# Introduce missing values for the demonstration.
data.loc[2, "income"] = np.nan
data.loc[7, "city"] = np.nan

data["age_income_ratio"] = data["age"] / (data["income"] + 1)

X = data.drop(columns="churn")
y = data["churn"]

numeric_features = [
    "age", "income", "tenure", "age_income_ratio"
]
categorical_features = ["city", "plan"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
    )),
])

full_pipeline.fit(X_train, y_train)

predictions = full_pipeline.predict(X_test)
print("\nPipeline accuracy:")
print(accuracy_score(y_test, predictions))