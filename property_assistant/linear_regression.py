# LINEAR REGRESSION — PROPERTY ASSISTANT

import os

import pandas as pd

from sklearn.linear_model import LinearRegression


# 1. LOAD THE PROPERTY DATA

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)

print(f"Dataset: {len(properties)} properties")

# LinearRegression needs numbers, so convert furnished from Yes/No to 1/0.
properties["furnished"] = properties["furnished"].map({"Yes": 1, "No": 0})


# 2. PREPARE THE DATA

# These property details will be used to predict price.
features = [
    "bedrooms",
    "bathrooms",
    "furnished"
]

X = properties[features]

# price_kes is the value the model is trying to predict.
y = properties["price_kes"]


# 3. CREATE THE MODEL

model = LinearRegression()


# 4. TRAIN THE MODEL

model.fit(X, y)

print("Training complete!")


# 5. UNDERSTAND WHAT THE MODEL LEARNED

# Each coefficient shows how the corresponding feature
# affects the predicted price while the other features are considered.
for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: KES {coefficient:,.0f}")

print(f"Base price: KES {model.intercept_:,.0f}")


# 6. PREDICT A NEW PROPERTY PRICE

# Example property:
# 3 bedrooms
# 2 bathrooms
# furnished
new_property = [[3, 2, 1]]

predicted_price = model.predict(new_property)

print(
    f"Predicted price: KES {predicted_price[0]:,.0f}"
)