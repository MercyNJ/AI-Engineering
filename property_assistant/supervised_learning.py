# Supervised Learning — Property Assistant

import os

# pandas is a Python library for working with tabular data.

import pandas as pd

# train_test_split() divides data into training and testing sets.

from sklearn.model_selection import train_test_split

# KNeighborsClassifier creates a K-Nearest Neighbours classification model.

from sklearn.neighbors import KNeighborsClassifier

# Tools for evaluating classification performance.

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the property data

# Read the property dataset from the CSV file.

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)

print(f"Dataset: {len(properties)} properties")
print(properties.head())

# Step 2: Prepare the features and target

# Features are the property details the model uses to make a prediction.

# KNN needs numbers, so convert furnished from Yes/No to 1/0.

properties["furnished"] = properties["furnished"].map({"Yes": 1, "No": 0})

features = [
    "bedrooms",
    "bathrooms",
    "price_kes",
    "furnished"
]

# X contains the features.

X = properties[features]

# y contains the target the model is trying to predict.

# suitable = 1 means suitable

# suitable = 0 means not suitable

y = properties["suitable"]

# Step 3: Split into training and testing sets

# Training data is used to teach the model.

# Testing data is used later to evaluate the model.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# Step 4: Create and train the model

# KNN looks at nearby/similar properties to make a prediction.

model = KNeighborsClassifier(n_neighbors=3)

# fit() trains the model using the training data.

model.fit(X_train, y_train)

print("Training complete!")

# Step 5: Make predictions

# predict() uses the trained model to predict suitability for properties in the test set.

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:     ", y_test.to_numpy())

# Step 6: Evaluate performance

# accuracy_score() calculates the percentage of correct predictions.

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")

# Step 7: Confusion matrix

# Shows correct and incorrect predictions for each class.

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Step 8: Classification report

# Shows precision, recall, and F1-score for each class.

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Suitable", "Suitable"]
    )
)