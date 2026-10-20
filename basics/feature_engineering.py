
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
        "timestamp": pd.to_datetime([
            "2026-01-12 09:30",
            "2026-01-17 14:15",
            "2026-06-20 22:00",
            "2026-10-19 08:00",
        ])
    })

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_morning"] = (df["hour"] < 12).astype(int)

    print(df.to_string(index=False))


def demonstrate_scaling():
    print("\n2. NORMALISATION")

    data = pd.DataFrame({
        "salary_kes_thousands": [1200, 800, 2000, 1500],
        "accuracy": [0.94, 0.87, 0.91, 0.96],
    })

    standard_scaler = StandardScaler()
    standard_scaled = standard_scaler.fit_transform(data)

    minmax_scaler = MinMaxScaler()
    minmax_scaled = minmax_scaler.fit_transform(data)

    print("Original data:")
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
        "city": ["Nairobi", "Lagos", "Accra", "Nairobi", "Lagos"],
        "plan": ["monthly", "one-time", "weekly", "monthly", "one-time"],
    })

    encoded = pd.get_dummies(df, columns=["city", "plan"], dtype=int)
    print(encoded.to_string(index=False))


def demonstrate_interaction_features():
    print("\n4. INTERACTION FEATURES")

    df = pd.DataFrame({
        "hours_studied": [2, 4, 6, 8, 0],
        "sleep_hours": [8, 7, 6, 7, 8],
        "previous_grade": [50, 65, 72, 88, 40],
    })

    df["study_sleep_product"] = (
        df["hours_studied"] * df["sleep_hours"]
    )

    df["score_per_hour"] = (
        df["previous_grade"]
        / df["hours_studied"].replace(0, np.nan)
    )

    print(df.round(2).to_string(index=False))


def build_student_dataset():
    rng = np.random.default_rng(42)
    n = 600

    hours_studied = rng.uniform(0, 10, n)
    sleep_hours = rng.uniform(4, 10, n)
    previous_grade = rng.uniform(35, 95, n)

    study_plan = rng.choice(
        ["daily", "weekends", "irregular"],
        size=n,
        p=[0.50, 0.30, 0.20],
    )

    timestamps = (
        pd.to_datetime("2026-01-01")
        + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
        + pd.to_timedelta(rng.integers(6, 24, n), unit="h")
    )

    plan_bonus = pd.Series(study_plan).map({
        "daily": 8,
        "weekends": 3,
        "irregular": -3,
    }).to_numpy()

    score = (
        25
        + hours_studied * 4.5
        + previous_grade * 0.35
        + (sleep_hours - 7) * 2
        + plan_bonus
        + rng.normal(0, 8, n)
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "hours_studied": hours_studied,
        "sleep_hours": sleep_hours,
        "previous_grade": previous_grade,
        "study_plan": study_plan,
        "score": np.clip(score, 0, 100),
    })

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_morning"] = (df["hour"] < 12).astype(int)

    df["study_sleep_product"] = (
        df["hours_studied"] * df["sleep_hours"]
    )

    df["score_per_hour"] = (
        df["score"] / df["hours_studied"].clip(lower=0.5)
    )

    df["passed"] = (df["score"] >= 60).astype(int)

    return df


def train_and_explain_model():
    print("\n5. COMPLETE FEATURE-ENGINEERING PIPELINE")

    df = build_student_dataset()

    numeric_features = [
        "hours_studied",
        "sleep_hours",
        "previous_grade",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_morning",
        "study_sleep_product",
        "score_per_hour",
    ]

    categorical_features = ["study_plan"]

    X = df[numeric_features + categorical_features]
    y = df["passed"]

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
        target_names=["Not passed", "Passed"],
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
    )[:10]:
        print(f"{name:40} {coefficient:+.3f}")

    print("\nPermutation importance on test data:")

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


if __name__ == "__main__":
    demonstrate_date_features()
    demonstrate_scaling()
    demonstrate_categorical_encoding()
    demonstrate_interaction_features()
    train_and_explain_model()