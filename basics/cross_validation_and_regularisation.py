
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Three-way split for classification
X, y = load_breast_cancer(return_X_y=True)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.1765,
    random_state=42,
    stratify=y_trainval,
)

print(
    f"Train: {len(X_train)}, "
    f"Validation: {len(X_val)}, Test: {len(X_test)}"
)

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

classifier.fit(X_train, y_train)
val_predictions = classifier.predict(X_val)

print(
    "Validation accuracy:",
    f"{accuracy_score(y_val, val_predictions):.3f}",
)


# Stratified K-Fold cross-validation
X, y = load_breast_cancer(return_X_y=True)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_val_score(
    model, X, y, cv=skf, scoring="accuracy"
)

print("\nStratified fold scores:", scores.round(3))
print(f"Mean accuracy: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")


# Learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y,
    cv=skf,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1,
)

plt.plot(
    train_sizes, train_scores.mean(axis=1), label="Training"
)
plt.plot(
    train_sizes, val_scores.mean(axis=1), label="Validation"
)
plt.xlabel("Number of training examples")
plt.ylabel("Accuracy")
plt.title("Learning Curves")
plt.legend()
plt.show()


# Ridge (L2) and Lasso (L1) regression
housing = fetch_california_housing()
X_reg, y_reg = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

models = {
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.01, max_iter=10000),
}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )

    rmse_scores = -scores
    print(
        f"{name} CV RMSE: "
        f"{rmse_scores.mean():.3f} "
        f"+/- {rmse_scores.std():.3f}"
    )

    pipeline.fit(X_train, y_train)
    print(f"{name} test R²: {pipeline.score(X_test, y_test):.3f}")