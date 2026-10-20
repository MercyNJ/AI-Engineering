
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

properties = pd.DataFrame({
    "price_kes": [
        2_500_000, 3_000_000, 3_500_000, 4_000_000, 4_500_000,
        5_000_000, 5_500_000, 6_000_000, 6_500_000, 7_000_000,
        7_500_000, 8_000_000, 8_500_000, 9_000_000, 9_500_000,
        10_000_000, 3_200_000, 4_800_000, 6_800_000, 8_800_000,
        2_800_000, 3_800_000, 5_800_000, 7_800_000, 9_800_000,
        4_200_000, 5_200_000, 6_200_000, 7_200_000, 8_200_000,
        3_600_000, 4_600_000, 5_600_000, 6_600_000, 7_600_000,
        8_600_000, 9_600_000, 3_400_000, 5_400_000, 7_400_000,
    ],
    "bedrooms": [
        1, 2, 2, 3, 2, 3, 3, 3, 4, 3,
        4, 4, 4, 4, 5, 5, 2, 3, 4, 4,
        1, 2, 3, 4, 5, 2, 3, 3, 4, 4,
        2, 3, 3, 4, 4, 4, 5, 2, 3, 4,
    ],
    "size_sqft": [
        450, 650, 800, 950, 900, 1100, 1250, 1350, 1500, 1450,
        1700, 1800, 1900, 2050, 2200, 2400, 700, 1150, 1550, 2000,
        500, 850, 1300, 1750, 2300, 1000, 1200, 1400, 1650, 1850,
        750, 1050, 1280, 1520, 1720, 1950, 2250, 820, 1320, 1680,
    ],
    "buyer_budget_kes": [
        3_000_000, 3_200_000, 4_000_000, 4_200_000, 4_000_000,
        5_500_000, 5_000_000, 6_500_000, 6_000_000, 7_500_000,
        8_000_000, 7_500_000, 9_000_000, 8_500_000, 10_000_000,
        9_000_000, 3_500_000, 5_000_000, 7_000_000, 8_000_000,
        2_500_000, 4_000_000, 6_000_000, 8_500_000, 9_000_000,
        4_500_000, 5_000_000, 6_500_000, 7_000_000, 8_500_000,
        3_800_000, 5_000_000, 5_500_000, 7_000_000, 8_000_000,
        8_000_000, 10_000_000, 3_000_000, 6_000_000, 7_000_000,
    ],
})

X = properties[[
    "price_kes",
    "bedrooms",
    "size_sqft",
    "buyer_budget_kes",
]]

y = (
    properties["price_kes"] <= properties["buyer_budget_kes"]
).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

models = {
    "Shallow tree": DecisionTreeClassifier(
        max_depth=1,
        random_state=42,
    ),
    "Balanced tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    ),
    "Deep tree": DecisionTreeClassifier(
        max_depth=None,
        random_state=42,
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train),
    )
    test_accuracy = accuracy_score(
        y_test,
        model.predict(X_test),
    )
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    print(f"\n{name}")
    print(f"Training accuracy: {train_accuracy:.1%}")
    print(f"Test accuracy:     {test_accuracy:.1%}")
    print(f"CV accuracy:       {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")