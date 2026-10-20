
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Illustrative property and buyer data.
properties = pd.DataFrame({
    "price_kes": [
        2_500_000, 3_000_000, 3_500_000, 4_000_000, 4_500_000,
        5_000_000, 5_500_000, 6_000_000, 6_500_000, 7_000_000,
        7_500_000, 8_000_000, 8_500_000, 9_000_000, 9_500_000,
        10_000_000, 3_200_000, 4_800_000, 6_800_000, 8_800_000,
        2_800_000, 3_800_000, 5_800_000, 7_800_000, 9_800_000,
        4_200_000, 5_200_000, 6_200_000, 7_200_000, 8_200_000,
    ],
    "bedrooms": [
        1, 2, 2, 3, 2, 3, 3, 3, 4, 3,
        4, 4, 4, 4, 5, 5, 2, 3, 4, 4,
        1, 2, 3, 4, 5, 2, 3, 3, 4, 4,
    ],
    "size_sqft": [
        450, 650, 800, 950, 900, 1100, 1250, 1350, 1500, 1450,
        1700, 1800, 1900, 2050, 2200, 2400, 700, 1150, 1550, 2000,
        500, 850, 1300, 1750, 2300, 1000, 1200, 1400, 1650, 1850,
    ],
    "buyer_budget_kes": [
        3_000_000, 3_200_000, 4_000_000, 4_200_000, 4_000_000,
        5_500_000, 5_000_000, 6_500_000, 6_000_000, 7_500_000,
        8_000_000, 7_500_000, 9_000_000, 8_500_000, 10_000_000,
        9_000_000, 3_500_000, 5_000_000, 7_000_000, 8_000_000,
        2_500_000, 4_000_000, 6_000_000, 8_500_000, 9_000_000,
        4_500_000, 5_000_000, 6_500_000, 7_000_000, 8_500_000,
    ],
})

# Target: 1 = within budget, 0 = over budget.
properties["within_budget"] = (
    properties["price_kes"] <= properties["buyer_budget_kes"]
).astype(int)

feature_columns = [
    "price_kes",
    "bedrooms",
    "size_sqft",
    "buyer_budget_kes",
]

X = properties[feature_columns]
y = properties["within_budget"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42),
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.0%}")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Over budget", "Within budget"],
        zero_division=0,
    ))

# Predict for a new property and buyer.
new_property = pd.DataFrame([{
    "price_kes": 5_500_000,
    "bedrooms": 3,
    "size_sqft": 1_250,
    "buyer_budget_kes": 6_000_000,
}])

model = models["Logistic Regression"]
prediction = model.predict(new_property)[0]
probability = model.predict_proba(new_property)[0, 1]

print("\nNew property")
print(f"Price: KSh {new_property.iloc[0]['price_kes']:,.0f}")
print(f"Buyer budget: KSh {new_property.iloc[0]['buyer_budget_kes']:,.0f}")
print(f"Prediction: {'Within budget' if prediction else 'Over budget'}")
print(f"Estimated probability of being within budget: {probability:.1%}")