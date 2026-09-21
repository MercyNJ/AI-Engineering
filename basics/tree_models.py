# DECISION TREES, RANDOM FORESTS & XGBOOST

# These are supervised learning algorithms that can be used
# for classification and, in some cases, regression.

# Decision trees make predictions by asking a series of
# questions about the features.


# 1. DECISION TREES

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

# DecisionTreeClassifier creates a decision tree for classification.
# max_depth limits how many questions deep the tree can go.
# random_state keeps the results reproducible.
tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# fit() trains the tree using the data.
tree.fit(X, y)

# export_text() displays the tree's decision rules as text.
print(export_text(
    tree,
    feature_names=list(iris.feature_names)
))

# score() returns the model's accuracy.
print(f"Train accuracy: {tree.score(X, y):.3f}")


# A single decision tree can easily overfit the training data.
# Random Forest reduces this problem by combining many trees.


# 2. RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42
)

# RandomForestClassifier creates a model made up of many decision trees.
# max_depth=None lets each tree grow without a depth limit.
# n_jobs=-1 uses all available CPU cores.
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Train the Random Forest.
rf.fit(X_train, y_train)

print(f"Test accuracy: {rf.score(X_test, y_test):.3f}")


# Feature importance shows which features contributed most
# to the Random Forest's predictions.
importances = pd.Series(
    rf.feature_importances_,
    index=data.feature_names
)

print(importances.sort_values(ascending=False).head(5))


# 3. XGBOOST

# XGBoost is another tree-based ensemble algorithm.
# It builds trees sequentially, with each new tree helping
# correct errors made by previous trees.

import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# eval_set is data XGBoost checks its performance on while training.
# verbose=False hides the progress output for each tree.
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print(f"XGBoost test accuracy: {model.score(X_test, y_test):.3f}")


# 4. KEY HYPERPARAMETERS

# n_estimators
# Number of trees in the ensemble.
#
# max_depth
# Maximum depth of each tree.
# Lower values can help reduce overfitting.
#
# learning_rate
# Controls how much each new XGBoost tree contributes.
#
# subsample
# Percentage of training rows used by each XGBoost step.
#
# colsample_bytree
# Percentage of features considered when building each XGBoost tree.


# KEY IDEAS

# Decision Tree
# Makes predictions using a series of feature-based decisions.

# Random Forest
# Combines many decision trees and aggregates their predictions.

# XGBoost
# Builds trees sequentially, with each tree helping correct
# errors made by previous trees.

# Ensemble
# A model that combines multiple models to make predictions.