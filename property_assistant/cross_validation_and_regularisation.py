
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Illustrative rental listings
properties = pd.DataFrame({
    "location": [
        "Ruiru", "Juja", "Kiambu", "Ruiru", "Juja",
        "Kiambu", "Ruiru", "Juja", "Kiambu", "Ruiru",
        "Juja", "Kiambu", "Ruiru", "Juja", "Kiambu",
        "Ruiru", "Juja", "Kiambu", "Ruiru", "Juja",
        "Kiambu", "Ruiru", "Juja", "Kiambu", "Ruiru",
        "Juja", "Kiambu", "Ruiru", "Juja", "Kiambu",
    ],
    "property_type": [
        "Apartment", "House", "Apartment", "House", "Apartment",
        "House", "Apartment", "House", "Apartment", "House",
        "Apartment", "House", "Apartment", "House", "Apartment",
        "House", "Apartment", "House", "Apartment", "House",
        "Apartment", "House", "Apartment", "House", "Apartment",
        "House", "Apartment", "House", "Apartment", "House",
    ],
    "bedrooms": [
        1, 3, 2, 4, 2, 3, 1, 4, 2, 3,
        2, 4, 1, 3, 2, 4, 1, 3, 2, 4,
        2, 3, 1, 4, 2, 3, 2, 4, 1, 3,
    ],
    "area_sqm": [
        38, 120, 70, 160, 75, 135, 42, 150, 68, 115,
        80, 145, 45, 125, 72, 155, 40, 118, 78, 140,
        74, 110, 43, 158, 69, 122, 82, 148, 41, 128,
    ],
    "rent_kes": [
        12000, 65000, 28000, 90000, 32000, 75000, 15000,
        85000, 30000, 60000, 35000, 80000, 16000, 68000,
        29000, 88000, 14000, 62000, 34000, 78000,
        31000, 58000, 17000, 92000, 27000, 64000,
        36000, 83000, 13000, 70000,
    ],
})

numeric_features = ["bedrooms", "area_sqm"]
categorical_features = ["location", "property_type"]

X = properties[numeric_features + categorical_features]
y = properties["rent_kes"]


# Three-way split: training, validation, and final test.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42
)

print(
    f"Train: {len(X_train)}, "
    f"Validation: {len(X_val)}, Test: {len(X_test)}"
)


# Preprocess numerical and categorical features.
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


# Compare Ridge (L2) and Lasso (L1) using cross-validation.
models = {
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=100.0, max_iter=10000),
}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    scores = cross_val_score(
        pipeline,
        X_trainval,
        y_trainval,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )

    rmse_scores = -scores
    print(
        f"\n{name} CV RMSE: "
        f"KES {rmse_scores.mean():,.0f} "
        f"+/- {rmse_scores.std():,.0f}"
    )

    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)

    print(
        f"{name} validation MAE: "
        f"KES {mean_absolute_error(y_val, val_predictions):,.0f}"
    )


# Select Ridge for the final demonstration and evaluate on the test set.
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0)),
])

final_pipeline.fit(X_trainval, y_trainval)
test_predictions = final_pipeline.predict(X_test)

print(
    "\nFinal test MAE:",
    f"KES {mean_absolute_error(y_test, test_predictions):,.0f}",
)
print(f"Final test R²: {r2_score(y_test, test_predictions):.3f}")


# Learning curve: inspect how training size affects performance.
learning_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0)),
])

train_sizes, train_scores, val_scores = learning_curve(
    learning_pipeline,
    X_trainval,
    y_trainval,
    cv=5,
    scoring="neg_root_mean_squared_error",
    train_sizes=np.linspace(0.3, 1.0, 5),
)

# Convert negative RMSE scores back to positive errors.
train_rmse = -train_scores.mean(axis=1)
val_rmse = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
plt.plot(train_sizes, val_rmse, marker="o", label="Validation RMSE")
plt.xlabel("Number of training examples")
plt.ylabel("RMSE (KES)")
plt.title("Property Rent Prediction Learning Curve")
plt.legend()
plt.show()


# Stratified K-Fold: classify property type.
X_class = properties[numeric_features + ["location"]]
y_class = properties["property_type"]

class_preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numeric_features,
    ),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        ["location"],
    ),
])

classifier = Pipeline([
    ("preprocessor", class_preprocessor),
    ("model", DecisionTreeClassifier(max_depth=3, random_state=42)),
])

stratified_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

class_scores = cross_val_score(
    classifier,
    X_class,
    y_class,
    cv=stratified_cv,
    scoring="accuracy",
)

print("\nProperty type classification fold scores:", class_scores.round(3))
print(f"Mean accuracy: {class_scores.mean():.3f}")