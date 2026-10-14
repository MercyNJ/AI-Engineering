import numpy as np
import pandas as pd


# Create a deliberately messy dataset.
raw = pd.DataFrame({
    "name": [
        "Amara", "Brian", None, "Diana", "Evan",
        "Fatima", "Brian", "Grace",
    ],
    "score": [
        92, "85", 78, None, -5, 110, "85", "not recorded",
    ],
    "city": [
        "Nairobi", "nairobi", " LAGOS ", "Accra",
        "Accra", "NAIROBI", "nairobi", "Kampala",
    ],
    "grade": [
        "A", "B", "B", "B", "A", "A", "B", "C",
    ],
})

print("Raw data:")
print(raw)

print("\nMissing values:")
print(raw.isna().sum())


def clean_student_data(raw_df):
    """Clean student records using explicit validation rules."""
    df = raw_df.copy()

    # Standardise text fields.
    df["name"] = df["name"].astype("string").str.strip()
    df["city"] = df["city"].astype("string").str.strip().str.title()
    df["grade"] = df["grade"].astype("string").str.strip().str.upper()

    # Convert scores to numbers; invalid text becomes missing.
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Replace blank names and cities with missing values.
    df["name"] = df["name"].replace("", pd.NA)
    df["city"] = df["city"].replace("", pd.NA)

    # Remove records with invalid scores.
    df = df[df["score"].between(0, 100)].copy()

    # Fill missing values where a sensible default is available.
    if not df.empty:
        df["score"] = df["score"].fillna(df["score"].median())

    df["name"] = df["name"].fillna("Unknown")
    df["city"] = df["city"].fillna("Unknown")

    # Remove exact duplicate records.
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


cleaned = clean_student_data(raw)

print("\nCleaned data:")
print(cleaned)

print("\nData types after cleaning:")
print(cleaned.dtypes)

print(f"\nRows before cleaning: {len(raw)}")
print(f"Rows after cleaning: {len(cleaned)}")