import os

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# DATA
# ============================================================

# Load our property data from the CSV file (path resolved relative
# to this file, so the script works no matter which directory it's run from)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)


# ============================================================
# 2×2 PROPERTY ASSISTANT DASHBOARD
# ============================================================

# Create 4 charts:
#
#   [0, 0]  [0, 1]
#   [1, 0]  [1, 1]
#
# fig = the entire dashboard
# axes = the individual charts
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Add one main title to the dashboard.
fig.suptitle("Property Assistant Dashboard", fontsize=15, fontweight="bold")


# ============================================================
# TOP-LEFT: PROPERTIES BY TYPE
# ============================================================

# value_counts() counts how many properties
# belong to each property type.
property_types = properties["property_type"].value_counts()

# bar() creates a bar chart.
axes[0, 0].bar(
    property_types.index,
    property_types.values,
    edgecolor="white"
)

axes[0, 0].set_title("Properties by Type")
axes[0, 0].set_xlabel("Property Type")
axes[0, 0].set_ylabel("Number of Properties")

# Rotate the property type names if they are long.
axes[0, 0].tick_params(axis="x", rotation=45)


# ============================================================
# TOP-RIGHT: PRICE VS BEDROOMS
# ============================================================

# scatter() creates individual points.
#
# X-axis = number of bedrooms
# Y-axis = property price
#
# This helps us see whether properties with
# more bedrooms generally have higher prices.
axes[0, 1].scatter(
    properties["bedrooms"],
    properties["price_kes"],
    s=80,
    edgecolors="white"
)

axes[0, 1].set_xlabel("Bedrooms")
axes[0, 1].set_ylabel("Price (KES)")
axes[0, 1].set_title("Property Price vs Bedrooms")

# Add guide lines to make the chart easier to read.
axes[0, 1].grid(True, alpha=0.3)


# ============================================================
# BOTTOM-LEFT: AVERAGE PRICE BY LOCATION
# ============================================================

# groupby() groups the properties by location.
#
# mean() then calculates the average price
# for each location.
average_price = (
    properties
    .groupby("property_location")["price_kes"]
    .mean()
    .sort_values()
)

# bar() creates a bar chart showing
# the average property price for each location.
axes[1, 0].bar(
    average_price.index,
    average_price.values,
    edgecolor="white"
)

axes[1, 0].set_title("Average Property Price by Location")
axes[1, 0].set_xlabel("Location")
axes[1, 0].set_ylabel("Average Price (KES)")

# Rotate location names so they do not overlap.
axes[1, 0].tick_params(axis="x", rotation=45)


# ============================================================
# BOTTOM-RIGHT: PROPERTY PRICE DISTRIBUTION
# ============================================================

# hist() creates a histogram.
#
# A histogram shows how many properties
# fall within different price ranges.
axes[1, 1].hist(
    properties["price_kes"],
    bins=10,
    edgecolor="white"
)

axes[1, 1].set_xlabel("Price (KES)")
axes[1, 1].set_ylabel("Number of Properties")
axes[1, 1].set_title("Property Price Distribution")


# ============================================================
# DISPLAY THE DASHBOARD
# ============================================================

# Adjust spacing between the charts.
plt.tight_layout()

# Display the completed dashboard.
plt.show()

print("Property Assistant dashboard complete!")