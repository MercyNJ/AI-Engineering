import pandas as pd
import numpy as np


# ── Property listings dataset ──
data = {
    "property": [
        "Kilimani Apartment",
        "Ruiru Townhouse",
        "Karen House",
        "Kasarani Apartment",
        "Westlands Apartment",
        "Syokimau House",
        "Kiambu Apartment",
        "Lavington Apartment",
        "Ruaka Townhouse",
        "Embakasi Apartment"
    ],

    "location": [
        "Kilimani",
        "Ruiru",
        "Karen",
        "Kasarani",
        "Westlands",
        "Syokimau",
        "Kiambu",
        "Lavington",
        "Ruaka",
        "Embakasi"
    ],

    "property_type": [
        "Apartment",
        "Townhouse",
        "House",
        "Apartment",
        "Apartment",
        "House",
        "Apartment",
        "Apartment",
        "Townhouse",
        "Apartment"
    ],

    "bedrooms": [2, 3, 4, 2, 3, 4, 2, 3, 3, 2],

    "price_kes": [
        75000,
        55000,
        150000,
        50000,
        None,
        85000,
        45000,
        120000,
        65000,
        40000
    ],

    "rating": [
        4.6,
        4.2,
        4.8,
        4.0,
        4.7,
        4.3,
        3.9,
        4.9,
        4.4,
        3.8
    ],

    "furnished": [
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "No",
        "No",
        "Yes",
        "Yes",
        None
    ]
}


# ── Convert dictionary into a DataFrame ──
df = pd.DataFrame(data)


# ── Display raw dataset ──
print("=== Raw Property Dataset ===")
print(df.to_string())


# ── Check for missing values ──
print(f"\nMissing values:\n{df.isnull().sum()}")


# ── Clean missing values ──

# Replace missing price with the average price
df["price_kes"] = df["price_kes"].fillna(df["price_kes"].mean())

# Replace missing furnished value
df["furnished"] = df["furnished"].fillna("unknown")


# ── Check dataset after cleaning ──
print("\n=== After Cleaning ===")
print(f"Missing values: {df.isnull().sum().sum()}")


# ── Create price category ──
def price_category(price):
    if price > 100000:
        return "High"

    if price > 60000:
        return "Medium"

    return "Low"


df["price_category"] = df["price_kes"].apply(price_category)


# ── Average price by location ──
print("\n=== Average Price by Location ===")

by_location = (
    df.groupby("location")["price_kes"]
    .mean()
    .sort_values(ascending=False)
)

print(by_location.to_string())


# ── Find the top 3 properties by rating ──
print("\n=== Top 3 Properties ===")

top3 = (
    df.sort_values("rating", ascending=False)
    .head(3)[
        ["property", "location", "rating", "price_category"]
    ]
)

print(top3.to_string(index=False))