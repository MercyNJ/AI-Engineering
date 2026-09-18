import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# AI PROPERTY ASSISTANT — DATA VISUALIZATION
# ============================================================

# Load our property dataset (path resolved relative to this file,
# so the script works no matter which directory it's run from)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
df = pd.read_csv(CSV_PATH)

print("=== Property Dataset ===")
print(df.head())


# ============================================================
# 1. BAR CHART — PROPERTY COUNT BY LOCATION
# ============================================================

location_counts = df["property_location"].value_counts()

plt.figure(figsize=(10, 5))

plt.bar(
    location_counts.index,
    location_counts.values
)

plt.xlabel("Location")
plt.ylabel("Number of Properties")
plt.title("Number of Properties by Location")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 2. HISTOGRAM — PROPERTY PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df["price_kes"],
    bins=10,
    edgecolor="white"
)

plt.xlabel("Property Price (KES)")
plt.ylabel("Number of Properties")
plt.title("Distribution of Property Prices")

plt.tight_layout()
plt.show()


# ============================================================
# 3. SCATTER PLOT — PRICE VS BEDROOMS
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["bedrooms"],
    df["price_kes"]
)

plt.xlabel("Number of Bedrooms")
plt.ylabel("Property Price (KES)")
plt.title("Property Price vs Number of Bedrooms")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 4. BAR CHART — AVERAGE PRICE BY PROPERTY TYPE
# ============================================================

average_price = (
    df.groupby("property_type")["price_kes"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

plt.bar(
    average_price.index,
    average_price.values
)

plt.xlabel("Property Type")
plt.ylabel("Average Price (KES)")
plt.title("Average Property Price by Property Type")

plt.tight_layout()
plt.show()


# ============================================================
# 5. TRAINING CURVE — SIMULATED PROPERTY MODEL
# ============================================================

epochs = list(range(1, 11))

train_loss = [
    2.3, 1.8, 1.4, 1.1, 0.9,
    0.75, 0.65, 0.58, 0.54, 0.51
]

val_loss = [
    2.4, 1.9, 1.5, 1.2, 1.05,
    0.95, 0.88, 0.84, 0.82, 0.81
]

plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    train_loss,
    label="Training Loss",
    linewidth=2
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss",
    linewidth=2,
    linestyle="--"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Property Recommendation Model Training")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 6. MULTIPLE CHARTS — PROPERTY DASHBOARD
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

fig.suptitle(
    "AI Property Assistant — Property Dashboard",
    fontsize=15
)


# Top-left: properties by location
axes[0, 0].bar(
    location_counts.index,
    location_counts.values
)

axes[0, 0].set_title("Properties by Location")
axes[0, 0].set_xlabel("Location")
axes[0, 0].set_ylabel("Number of Properties")

axes[0, 0].tick_params(axis="x", rotation=45)


# Top-right: price distribution
axes[0, 1].hist(
    df["price_kes"],
    bins=10,
    edgecolor="white"
)

axes[0, 1].set_title("Property Price Distribution")
axes[0, 1].set_xlabel("Price (KES)")
axes[0, 1].set_ylabel("Number of Properties")


# Bottom-left: price vs bedrooms
axes[1, 0].scatter(
    df["bedrooms"],
    df["price_kes"]
)

axes[1, 0].set_title("Price vs Bedrooms")
axes[1, 0].set_xlabel("Bedrooms")
axes[1, 0].set_ylabel("Price (KES)")


# Bottom-right: average price by property type
axes[1, 1].bar(
    average_price.index,
    average_price.values
)

axes[1, 1].set_title("Average Price by Property Type")
axes[1, 1].set_xlabel("Property Type")
axes[1, 1].set_ylabel("Average Price (KES)")


plt.tight_layout()
plt.show()

print("Property visualization complete!")