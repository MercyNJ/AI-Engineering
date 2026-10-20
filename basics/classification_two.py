
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Features: hours studied and previous grade.
X = np.array([
    [1, 40], [2, 45], [2, 55], [3, 52], [3, 60],
    [4, 58], [4, 65], [5, 62], [5, 70], [6, 68],
    [6, 75], [7, 72], [7, 80], [8, 78], [8, 85],
    [9, 82], [9, 88], [10, 90], [10, 95], [1, 35],
])

# Target: 1 = pass, 0 = fail.
y = np.array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
])

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
        target_names=["Fail", "Pass"],
        zero_division=0,
    ))

# Demonstrate a prediction using Logistic Regression.
model = models["Logistic Regression"]
new_student = np.array([[6, 68]])

prediction = model.predict(new_student)[0]
probabilities = model.predict_proba(new_student)[0]

print("\nNew student: 6 hours studied, previous grade 68")
print(f"Prediction: {'PASS' if prediction == 1 else 'FAIL'}")
print(f"Estimated probability of passing: {probabilities[1]:.1%}")