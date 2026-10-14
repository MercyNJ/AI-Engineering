import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Generate illustrative property listings.
rng = np.random.default_rng(42)
n_properties = 240

locations = rng.choice(
    ["Ruiru", "Juja", "Kiambu", "Thika"],
    size=n_properties,
)
property_types = rng.choice(
    ["Apartment", "House"],
    size=n_properties,
)
bedrooms = rng.integers(1, 6, size=n_properties)
area_sqm = np.maximum(
    25,
    bedrooms * 25 + rng.normal(10, 15, size=n_properties),
).round(1)

location_adjustment = pd.Series(locations).map({
    "Ruiru": 0,
    "Juja": -3000,
    "Kiambu": 5000,
    "Thika": -2000,
}).to_numpy()

type_adjustment = np.where(
    property_types == "House",
    12000,
    0,
)

rent_kes = (
    5000
    + bedrooms * 7000
    + area_sqm * 180
    + location_adjustment
    + type_adjustment
    + rng.normal(0, 5000, size=n_properties)
).round(0)

properties = pd.DataFrame({
    "location": locations,
    "property_type": property_types,
    "bedrooms": bedrooms,
    "area_sqm": area_sqm,
    "rent_kes": np.maximum(5000, rent_kes),
})

numeric_features = ["bedrooms", "area_sqm"]
categorical_features = ["location", "property_type"]

X = properties[numeric_features + categorical_features]
y = properties["rent_kes"]


# Reserve a test set for the final evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# Preprocess numerical and categorical features.
preprocessor = ColumnTransformer([
    (
        "numeric",
        StandardScaler(),
        numeric_features,
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features,
    ),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42)),
])


# Establish baseline performance using cross-validation.
baseline_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

baseline_rmse = -baseline_scores.mean()

print(f"Baseline CV RMSE: KES {baseline_rmse:,.0f}")


# Grid Search: test every specified combination.
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

grid_rmse = -grid_search.best_score_

print("\nGrid Search")
print("Best parameters:", grid_search.best_params_)
print(f"Best CV RMSE: KES {grid_rmse:,.0f}")
print(
    f"Improvement over baseline: "
    f"KES {baseline_rmse - grid_rmse:,.0f}"
)


# Randomised Search: explore a wider range of settings.
param_dist = {
    "model__n_estimators": randint(50, 301),
    "model__max_depth": [None, 5, 10, 15, 20],
    "model__min_samples_split": randint(2, 16),
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)

random_rmse = -random_search.best_score_

print("\nRandomised Search")
print("Best parameters:", random_search.best_params_)
print(f"Best CV RMSE: KES {random_rmse:,.0f}")


# Inspect the top Grid Search results.
results = pd.DataFrame(grid_search.cv_results_)
results["mean_rmse"] = -results["mean_test_score"]

results = results.sort_values("mean_rmse")

print("\nTop Grid Search results:")
print(
    results[
        [
            "param_model__n_estimators",
            "param_model__max_depth",
            "param_model__min_samples_split",
            "mean_rmse",
            "std_test_score",
        ]
    ].head(5).to_string(index=False)
)


# Select the search with the lower cross-validation RMSE.
best_search = min(
    [grid_search, random_search],
    key=lambda search: -search.best_score_,
)

# Evaluate the selected model on the untouched test set.
final_model = best_search.best_estimator_
test_predictions = final_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = root_mean_squared_error(y_test, test_predictions)

print("\nFinal evaluation")
print(f"Selected search: {best_search.__class__.__name__}")
print(f"Test MAE: KES {test_mae:,.0f}")
print(f"Test RMSE: KES {test_rmse:,.0f}")