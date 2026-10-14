import pandas as pd
from scipy.stats import randint
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)


# Load the dataset and reserve an untouched test set.
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

baseline_model = RandomForestClassifier(random_state=42)

baseline_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
)

baseline_score = baseline_scores.mean()

print(f"Baseline CV accuracy: {baseline_score:.3f}")


# Grid Search: test every specified combination.
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

print("\nGrid Search")
print("Best parameters:", grid_search.best_params_)
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
print(
    f"Improvement over baseline: "
    f"{grid_search.best_score_ - baseline_score:+.3f}"
)


# Randomised Search: sample combinations from wider ranges.
param_dist = {
    "n_estimators": randint(50, 301),
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": randint(2, 16),
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_train, y_train)

print("\nRandomised Search")
print("Best parameters:", random_search.best_params_)
print(f"Best CV accuracy: {random_search.best_score_:.3f}")


# Inspect the top Grid Search results.
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values(
    "mean_test_score",
    ascending=False,
)

print("\nTop Grid Search results:")
print(
    results[
        [
            "param_n_estimators",
            "param_max_depth",
            "mean_test_score",
            "std_test_score",
        ]
    ].head(5).to_string(index=False)
)


# Select the search with the better CV score.
best_search = max(
    [grid_search, random_search],
    key=lambda search: search.best_score_,
)

final_model = best_search.best_estimator_
test_predictions = final_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)

print("\nFinal evaluation")
print(f"Selected search: {best_search.__class__.__name__}")
print(f"Test accuracy: {test_accuracy:.3f}")