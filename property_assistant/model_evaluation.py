# MODEL EVALUATION — PROPERTY ASSISTANT

import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 1. LOAD THE PROPERTY DATA

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)

print(f"Dataset: {len(properties)} properties")

# KNN needs numbers, so convert furnished from Yes/No to 1/0.
properties["furnished"] = properties["furnished"].map({"Yes": 1, "No": 0})


# 2. PREPARE THE FEATURES AND TARGET

# Features are the property details used by the model.
features = [
    "bedrooms",
    "bathrooms",
    "price_kes",
    "furnished"
]

X = properties[features]

# suitable is the target the model is trying to predict.
y = properties["suitable"]


# 3. SPLIT THE DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")


# 4. CREATE AND TRAIN THE MODEL

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

print("Training complete!")


# 5. MAKE PREDICTIONS

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:     ", y_test.to_numpy())


# 6. EVALUATE THE MODEL

# pos_label="Yes" tells sklearn which class to treat as "positive".
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Yes")
recall = recall_score(y_test, y_pred, pos_label="Yes")
f1 = f1_score(y_test, y_pred, pos_label="Yes")

print("\nEvaluation Metrics")
print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")


# 7. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 8. CLASSIFICATION REPORT

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Suitable", "Suitable"]
    )
)