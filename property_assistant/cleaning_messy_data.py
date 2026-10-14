import pandas as pd


# Illustrative rental listings with messy values.
raw_properties = pd.DataFrame({
    "listing_id": [
        "P001", "P002", "P003", "P004",
        "P005", "P006", "P002", "P008",
    ],
    "location": [
        "ruiru", " JUJA ", "KIAMBU", "Ruiru",
        "thika", "JUJA", " JUJA ", "Nairobi",
    ],
    "property_type": [
        "apartment", "House", " APARTMENT ", "HOUSE",
        "Apartment", "house", "House", "Apartment",
    ],
    "bedrooms": [
        "2", 3, "1", 4, 2, None, 3, "studio",
    ],
    "area_sqm": [
        65, 120, 40, 150, None, 110, 120, 55,
    ],
    "rent_kes": [
        "25000", 65000, 15000, 85000,
        30000, 55000, 65000, -5000,
    ],
})

print("Raw property listings:")
print(raw_properties)


def clean_property_data(raw_df):
    """Clean rental listings while preserving identifiable records."""
    df = raw_df.copy()

    # Standardise text fields.
    text_columns = ["listing_id", "location", "property_type"]

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    df["location"] = df["location"].str.title()
    df["property_type"] = df["property_type"].str.title()

    # Convert numeric fields; invalid text becomes missing.
    numeric_columns = ["bedrooms", "area_sqm", "rent_kes"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Treat non-positive values as invalid for these fields.
    df.loc[df["bedrooms"] <= 0, "bedrooms"] = float("nan")
    df.loc[df["area_sqm"] <= 0, "area_sqm"] = float("nan")
    df.loc[df["rent_kes"] <= 0, "rent_kes"] = float("nan")

    # A listing needs an ID, location, type, bedroom count,
    # and rent to be usable for this rental-price example.
    required_columns = [
        "listing_id",
        "location",
        "property_type",
        "bedrooms",
        "rent_kes",
    ]

    df = df.dropna(subset=required_columns).copy()

    # Remove duplicate listing IDs, keeping the first valid record.
    df = df.drop_duplicates(subset=["listing_id"], keep="first")

    # Fill missing floor area using the median of retained listings.
    if not df.empty:
        df["area_sqm"] = df["area_sqm"].fillna(
            df["area_sqm"].median()
        )

    return df.reset_index(drop=True)


cleaned_properties = clean_property_data(raw_properties)

print("\nCleaned property listings:")
print(cleaned_properties)

print(f"\nListings before cleaning: {len(raw_properties)}")
print(f"Listings after cleaning: {len(cleaned_properties)}")

print("\nRemaining missing values:")
print(cleaned_properties.isna().sum())