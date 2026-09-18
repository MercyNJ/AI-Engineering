# CLASSIFICATION — PROPERTY ASSISTANT

import os

import pandas as pd

from sklearn.linear_model import LogisticRegression


# 1. LOAD THE PROPERTY DATA

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)

print(f"Dataset: {len(properties)} properties")

# LogisticRegression needs numbers, so convert furnished from Yes/No to 1/0.
properties["furnished"] = properties["furnished"].map({"Yes": 1, "No": 0})

# suitable is stored as Yes/No text, so convert it to 1/0 too.
properties["suitable"] = properties["suitable"].map({"Yes": 1, "No": 0})


# 2. PREPARE THE DATA

# Features are the property details used to make the prediction.
features = [
    "bedrooms",
    "bathrooms",
    "price_kes",
    "furnished"
]

X = properties[features]

# suitable is the target the model is trying to predict.
# 0 = not suitable
# 1 = suitable
y = properties["suitable"]


# 3. CREATE THE MODEL

# max_iter is raised because price_kes is on a much larger scale
# than the other features, which the default 100 iterations isn't
# enough to fit.
model = LogisticRegression(max_iter=1000)


# 4. TRAIN THE MODEL

model.fit(X, y)

print("Training complete!")


# 5. MAKE A PREDICTION

# Example property:
# 3 bedrooms
# 2 bathrooms
# KES 80,000
# furnished
new_property = [[3, 2, 80000, 1]]

prediction = model.predict(new_property)[0]

print(
    f"Prediction: {'Suitable' if prediction == 1 else 'Not Suitable'}"
)


# 6. GET THE PROBABILITY

probabilities = model.predict_proba(new_property)[0]

print(f"Not suitable probability: {probabilities[0]:.0%}")
print(f"Suitable probability: {probabilities[1]:.0%}")