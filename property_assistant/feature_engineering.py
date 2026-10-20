
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def demonstrate_date_features():
    print("\n1. DATE FEATURES")

    df = pd.DataFrame({
        "listed_at": pd.to_datetime([
            "2026-10-12 09:30",
            "2026-10-17 14:15",
            "2026-06-20 22:00",
            "2026-10-19 08:00",
        ])
    })

    df["listing_hour"] = df["listed_at"].dt.hour
    df["listing_day_of_week"] = df["listed_at"].dt.dayofweek
    df["listing_month"] = df["listed_at"].dt.month
    df["listed_on_weekend"] = (
        df["listing_day_of_week"].isin([5, 6]).astype(int)
    )
    df["listed_in_morning"] = (
        df["listing_hour"] < 12
    ).astype(int)

    print(df.to_string(index=False))


def demonstrate_scaling():
    print("\n2. NORMALISATION")

    data = pd.DataFrame({
        "price_kes": [4_500_000, 8_000_000, 12_500_000, 6_000_000],
        "bedrooms": [1, 3, 4, 2],
        "size_sqft": [550, 1200, 1800, 850],
    })

    standard_scaled = StandardScaler().fit_transform(data)
    minmax_scaled = MinMaxScaler().fit_transform(data)

    print("Original property data:")
    print(data.to_string(index=False))

    print("\nStandardScaler:")
    print(pd.DataFrame(
        standard_scaled, columns=data.columns
    ).round(3).to_string(index=False))

    print("\nMinMaxScaler:")
    print(pd.DataFrame(
        minmax_scaled, columns=data.columns
    ).round(3).to_string(index=False))


def demonstrate_categorical_encoding():
    print("\n3. CATEGORICAL ENCODING")

    df = pd.DataFrame({
        "location": ["Ruiru", "Kilimani", "Ruiru", "Westlands", "Kilimani"],
        "property_type": [
            "apartment", "apartment", "maisonette",
            "apartment", "maisonette",
        ],
        "listing_type": ["rent", "sale", "sale", "rent", "sale"],
    })

    encoded = pd.get_dummies(
        df,
        columns=["location", "property_type", "listing_type"],
        dtype=int,
    )

    print(encoded.to_string(index=False))


def demonstrate_interaction_features():
    print("\n4. INTERACTION FEATURES")

    df = pd.DataFrame({
        "price_kes": [4_500_000, 8_000_000, 12_500_000, 6_000_000],
        "buyer_budget_kes": [5_000_000, 10_000_000, 10_000_000, 6_500_000],
        "size_sqft": [550, 1200, 1800, 850],
        "bedrooms": [1, 3, 4, 2],
    })

    df["budget_ratio"] = (
        df["price_kes"] / df["buyer_budget_kes"].replace(0, np.nan)
    )

    df["remaining_budget_kes"] = (
        df["buyer_budget_kes"] - df["price_kes"]
    )

    df["price_per_sqft_kes"] = (
        df["price_kes"] / df["size_sqft"].replace(0, np.nan)
    )

    df["bedrooms_per_1000_sqft"] = (
        df["bedrooms"] / df["size_sqft"].replace(0, np.nan) * 1000
    )

    print(df.round(2).to_string(index=False))


def build_property_dataset():
    rng = np.random.default_rng(42)
    n = 1000

    locations = rng.choice(
        ["Ruiru", "Kilimani", "Westlands", "Syokimau", "Thika"],
        size=n,
        p=[0.25, 0.20, 0.18, 0.20, 0.17],
    )

    property_types = rng.choice(
        ["apartment", "maisonette", "bungalow"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )

    listing_types = rng.choice(
        ["sale", "rent"],
        size=n,
        p=[0.65, 0.35],
    )

    bedrooms = rng.integers(1, 6, n)

    size_sqft = np.maximum(
        300,
        bedrooms * rng.normal(360, 65, n)
        + rng.normal(100, 180, n),
    )

    location_factor = pd.Series(locations).map({
        "Ruiru": 0.78,
        "Kilimani": 1.35,
        "Westlands": 1.50,
        "Syokimau": 0.88,
        "Thika": 0.70,
    }).to_numpy()

    type_factor = pd.Series(property_types).map({
        "apartment": 1.0,
        "maisonette": 1.22,
        "bungalow": 1.15,
    }).to_numpy()

    sale_price = (
        (1_800_000 + size_sqft * 7_000)
        * location_factor
        * type_factor
    )

    monthly_rent = (
        (8_000 + size_sqft * 18)
        * location_factor
        * type_factor
    )

    price_kes = np.where(
        np.array(listing_types) == "sale",
        sale_price,
        monthly_rent,
    )

    price_kes *= rng.normal(1.0, 0.12, n)
    price_kes = np.maximum(price_kes, 1_000)

    listed_at = (
        pd.to_datetime("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 600, n), unit="D")
        + pd.to_timedelta(rng.integers(0, 24, n), unit="h")
    )

    buyer_budget_kes = price_kes * rng.uniform(0.65, 1.35, n)

    df = pd.DataFrame({
        "listed_at": listed_at,
        "price_kes": price_kes,
        "bedrooms": bedrooms,
        "size_sqft": size_sqft,
        "location": locations,
        "property_type": property_types,
        "listing_type": listing_types,
        "buyer_budget_kes": buyer_budget_kes,
    })

    df["listing_hour"] = df["listed_at"].dt.hour
    df["listing_day_of_week"] = df["listed_at"].dt.dayofweek
    df["listing_month"] = df["listed_at"].dt.month
    df["listed_on_weekend"] = (
        df["listing_day_of_week"].isin([5, 6]).astype(int)
    )
    df["listed_in_morning"] = (
        df["listing_hour"] < 12
    ).astype(int)

    df["budget_ratio"] = (
        df["price_kes"] / df["buyer_budget_kes"].clip(lower=1)
    )

    df["remaining_budget_kes"] = (
        df["buyer_budget_kes"] - df["price_kes"]
    )

    df["price_per_sqft_kes"] = (
        df["price_kes"] / df["size_sqft"].clip(lower=1)
    )

    df["bedrooms_per_1000_sqft"] = (
        df["bedrooms"] / df["size_sqft"].clip(lower=1) * 1000
    )

    suitability = (
        (df["budget_ratio"] <= 0.90)
        & (df["bedrooms"] >= 2)
        & (df["size_sqft"] >= 650)
    ).astype(int)

    noise_indices = rng.choice(n, size=int(n * 0.08), replace=False)
    suitability.iloc[noise_indices] = 1 - suitability.iloc[noise_indices]

    df["suitable_match"] = suitability

    return df


def train_and_explain_model():
    print("\n5. COMPLETE PROPERTY FEATURE-ENGINEERING PIPELINE")

    df = build_property_dataset()

    numeric_features = [
        "price_kes",
        "bedrooms",
        "size_sqft",
        "buyer_budget_kes",
        "listing_hour",
        "listing_day_of_week",
        "listing_month",
        "listed_on_weekend",
        "listed_in_morning",
        "budget_ratio",
        "remaining_budget_kes",
        "price_per_sqft_kes",
        "bedrooms_per_1000_sqft",
    ]

    categorical_features = [
        "location",
        "property_type",
        "listing_type",
    ]

    X = df[numeric_features + categorical_features]
    y = df["suitable_match"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ])

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.1%}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not suitable", "Suitable"],
        zero_division=0,
    ))

    feature_names = (
        model.named_steps["preprocessing"].get_feature_names_out()
    )
    coefficients = model.named_steps["classifier"].coef_[0]

    print("\nLargest absolute model coefficients:")

    for name, coefficient in sorted(
        zip(feature_names, coefficients),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:12]:
        print(f"{name:45} {coefficient:+.3f}")

    print("\nPermutation importance on held-out data:")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
    )

    for name, importance in sorted(
        zip(X_test.columns, result.importances_mean),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name:30} {importance:+.3f}")

    print(
        "\nTeaching note: this is synthetic data with a simplified, noisy "
        "suitability label. It is not a real property recommendation policy."
    )


if __name__ == "__main__":
    demonstrate_date_features()
    demonstrate_scaling()
    demonstrate_categorical_encoding()
    demonstrate_interaction_features()
    train_and_explain_model()
