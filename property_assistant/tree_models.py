# DECISION TREES, RANDOM FOREST & XGBOOST — PROPERTY ASSISTANT

import os

import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier


# 1. LOAD THE PROPERTY DATA

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)

print(f"Dataset: {len(properties)} properties")

# The models need numbers, so convert furnished from Yes/No to 1/0.
properties["furnished"] = properties["furnished"].map({"Yes": 1, "No": 0})

# suitable is stored as Yes/No text, so convert it to 1/0 too.
properties["suitable"] = properties["suitable"].map({"Yes": 1, "No": 0})


# 2. PREPARE THE FEATURES AND TARGET

features = [
    "bedrooms",
    "bathrooms",
    "price_kes",
    "furnished"
]

X = properties[features]

# 0 = not suitable
# 1 = suitable
y = properties["suitable"]


# 3. SPLIT THE DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")


# 4. DECISION TREE

# max_depth limits how many questions deep the tree can go.
tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree.fit(X_train, y_train)

# score() returns the model's accuracy.
print(f"\nDecision Tree accuracy: {tree.score(X_test, y_test):.2%}")

# export_text() shows the questions the tree asks about each property.
print("\nDecision Tree Rules:")
print(
    export_text(
        tree,
        feature_names=features
    )
)


# 5. RANDOM FOREST

# n_estimators is the number of decision trees in the forest.
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

print(f"\nRandom Forest accuracy: {rf.score(X_test, y_test):.2%}")


# 6. XGBOOST

# XGBoost builds trees sequentially to improve previous errors.
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print(f"\nXGBoost accuracy: {model.score(X_test, y_test):.2%}")


# 7. FEATURE IMPORTANCE

# Shows which property details the Random Forest relied on most.
importances = pd.Series(
    rf.feature_importances_,
    index=features
)

print("\nFeature Importance:")
print(importances.sort_values(ascending=False))


# 8. PREDICT A NEW PROPERTY

new_property = [[
    3,       # bedrooms
    2,       # bathrooms
    80000,   # price
    1        # furnished
]]

tree_prediction = tree.predict(new_property)[0]
rf_prediction = rf.predict(new_property)[0]
xgb_prediction = model.predict(new_property)[0]

print(
    f"\nDecision Tree: "
    f"{'Suitable' if tree_prediction == 1 else 'Not Suitable'}"
)

print(
    f"Random Forest: "
    f"{'Suitable' if rf_prediction == 1 else 'Not Suitable'}"
)

print(
    f"XGBoost: "
    f"{'Suitable' if xgb_prediction == 1 else 'Not Suitable'}"
)