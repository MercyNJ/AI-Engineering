
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Simulated property data
properties = pd.DataFrame({
    "location": [
        "Ruiru", "Ruiru", "Juja", "Kiambu", "Ruiru",
        "Juja", "Kiambu", "Ruiru", "Juja", "Kiambu",
        "Ruiru", "Juja", "Kiambu", "Ruiru", "Juja",
        "Kiambu", "Ruiru", "Juja", "Kiambu", "Ruiru",
    ],
    "property_type": [
        "Apartment", "House", "Apartment", "House", "Apartment",
        "House", "Apartment", "House", "Apartment", "House",
        "Apartment", "House", "Apartment", "Apartment", "House",
        "House", "Apartment", "Apartment", "House", "House",
    ],
    "bedrooms": [
        1, 3, 2, 4, 2, 3, 2, 4, 1, 3,
        3, 2, 1, 2, 4, 3, 1, 2, 4, 3,
    ],
    "bathrooms": [
        1, 2, 1, 3, 2, 2, 1, 3, 1, 2,
        2, 1, 1, 2, 3, 2, 1, 1, 3, 2,
    ],
    "area_sqm": [
        45, 120, 75, 160, 85, 110, 70, 150, 40, 115,
        105, 80, 42, 78, 170, 125, 48, 72, 155, 118,
    ],
    "furnished": [
        "no", "yes", "no", "yes", "yes", "no", "yes",
        "no", "no", "yes", "yes", "no", "no", "yes",
        "no", "yes", "no", "yes", "no", "yes",
    ],
    "year_built": [
        2018, 2015, 2020, 2012, 2019, 2016, 2021, 2014,
        2022, 2017, 2018, 2020, 2023, 2019, 2011, 2016,
        2022, 2021, 2013, 2017,
    ],
    "price_kes": [
        25000, 75000, 35000, 95000, 45000, 60000, 40000,
        85000, 22000, 70000, 65000, 38000, 24000, 42000,
        100000, 78000, 26000, 36000, 90000, 72000,
    ],
})

# Introduce missing values to demonstrate preprocessing.
properties.loc[3, "area_sqm"] = None
properties.loc[8, "year_built"] = None
properties.loc[11, "furnished"] = None

# Create a feature from the property's construction year.
properties["property_age"] = 2026 - properties["year_built"]

# Separate inputs from the target.
X = properties.drop(columns=["price_kes", "year_built"])
y = properties["price_kes"]

numeric_features = [
    "bedrooms", "bathrooms", "area_sqm", "property_age"
]
categorical_features = [
    "location", "property_type", "furnished"
]

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

property_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )),
])

property_pipeline.fit(X_train, y_train)

predictions = property_pipeline.predict(X_test)

rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print(f"RMSE: KES {rmse:,.2f}")
print(f"R²: {r2:.3f}")