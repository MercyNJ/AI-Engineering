
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. property data.
properties = pd.DataFrame({
    "location": [
        "Ruiru", "Juja", "Kiambu", "Thika", "Ruiru",
        "Juja", "Kiambu", "Thika", "Ruiru", "Juja",
        "Kiambu", "Thika", "Ruiru", "Juja", "Kiambu",
        "Thika", "Ruiru", "Juja", "Kiambu", "Thika",
        "Ruiru", "Juja", "Kiambu", "Thika", "Ruiru",
        "Juja", "Kiambu", "Thika", "Ruiru", "Juja",
    ],
    "bedrooms": [
        1, 2, 3, 4, 2, 1, 4, 3, 1, 2,
        3, 4, 2, 1, 3, 4, 1, 2, 4, 3,
        2, 1, 3, 4, 1, 2, 4, 3, 2, 1,
    ],
    "area_sqm": [
        35, 55, 110, 150, 65, 40, 140, 120, 38, 60,
        100, 160, 70, 42, 115, 145, 36, 58, 155, 105,
        68, 39, 108, 148, 37, 62, 142, 118, 72, 41,
    ],
    "rent_kes": [
        12000, 20000, 55000, 85000, 25000, 15000, 80000,
        65000, 14000, 22000, 50000, 90000, 28000, 16000,
        58000, 82000, 13000, 21000, 88000, 60000, 27000,
        15500, 53000, 87000, 12500, 23000, 79000, 62000,
        30000, 17000,
    ],
})

features = ["location", "bedrooms", "area_sqm"]
X = properties[features]
y = properties["rent_kes"]


# 2. Split the data before training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# 3. Convert location names into numerical features.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            ["location"],
        ),
    ],
    remainder="passthrough",
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression()),
])

model.fit(X_train, y_train)


# 4. Evaluate predictions on the test set.
predictions = model.predict(X_test)

print("Property Rent Prediction")
print(f"Test R²: {r2_score(y_test, predictions):.3f}")
print(f"Test MSE: {mean_squared_error(y_test, predictions):,.2f}")
print(
    f"Test RMSE: "
    f"{np.sqrt(mean_squared_error(y_test, predictions)):,.2f} KES"
)


# 5. Predict rent for a new property.
new_property = pd.DataFrame([{
    "location": "Ruiru",
    "bedrooms": 2,
    "area_sqm": 65,
}])

predicted_rent = model.predict(new_property)[0]
print(f"\nPredicted monthly rent: KES {predicted_rent:,.0f}")


# 6. Plot actual versus predicted test rents.
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, s=70)

lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    label="Perfect predictions",
)

plt.xlabel("Actual monthly rent (KES)")
plt.ylabel("Predicted monthly rent (KES)")
plt.title("Actual vs Predicted Property Rents")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()