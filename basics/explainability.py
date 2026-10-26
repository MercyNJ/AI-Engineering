
"""
Model Explainability — Understanding What Your Model Learned

Covers:
- Training a classification model
- Global vs. local explanations
- Built-in feature importance
- SHAP global and local explanations
- Detecting misleading shortcuts
- Why explanations do not prove causation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


RANDOM_STATE = 42


def create_dataset():
    """Generate sample applicant data with understandable feature names."""
    X, y = make_classification(
        n_samples=1200,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.6, 0.4],
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )

    feature_names = [
        "income",
        "employment_history",
        "debt_level",
        "credit_history",
        "application_score",
        "location_score",
    ]

    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="approved")

    return X, y


def train_model(X_train, X_test, y_train, y_test):
    """Train and evaluate a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    return model


def explain_builtin_feature_importance(model, feature_names):
    """
    Global explanation:
    Which features did the Random Forest rely on most?
    """
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    print("\n=== GLOBAL FEATURE IMPORTANCE ===")
    print(importance.to_string())

    importance.sort_values().plot(
        kind="barh",
        figsize=(9, 5),
        title="Global Feature Importance — Random Forest",
    )

    plt.xlabel("Importance score")
    plt.tight_layout()
    plt.show()

    return importance


def explain_one_prediction(model, X_test, y_test, row_index=7):
    """
    Local explanation using a simple feature-perturbation approach.

    This compares the original prediction with predictions where one
    feature at a time is replaced by its training-set median.
    It is an illustration, not a faithful attribution method.
    """
    row = X_test.iloc[[row_index]].copy()
    original_prediction = model.predict_proba(row)[0, 1]

    print("\n=== LOCAL EXPLANATION ===")
    print(f"Test row: {row_index}")
    print(f"Actual class: {y_test.iloc[row_index]}")
    print(f"Predicted probability of class 1: {original_prediction:.3f}")
    print(f"Predicted class: {model.predict(row)[0]}")

    changes = {}

    for feature in X_test.columns:
        modified_row = row.copy()
        modified_row[feature] = X_test[feature].median()

        modified_probability = model.predict_proba(modified_row)[0, 1]
        changes[feature] = modified_probability - original_prediction

    changes = pd.Series(changes).sort_values()

    print("\nChange in predicted probability after replacing")
    print("one feature with its test-set median:")
    print(changes.to_string())

    changes.sort_values().plot(
        kind="barh",
        figsize=(9, 5),
        title="Illustrative Local Feature Perturbation",
    )

    plt.axvline(0, linewidth=1)
    plt.xlabel("Probability change (modified minus original)")
    plt.tight_layout()
    plt.show()

    print(
        "\nImportant: this is a simple diagnostic, not SHAP. "
        "Changing one feature can create unrealistic combinations, "
        "and the result does not prove causation."
    )


def explain_with_shap(model, X_train, X_test, row_index=7):
    """Use SHAP for global and local explanations."""
    try:
        import shap
    except ImportError:
        print(
            "\nSHAP is not installed. Install it with:\n"
            "pip install shap"
        )
        return

    print("\n=== SHAP EXPLANATIONS ===")

    # A small background sample helps SHAP establish a reference.
    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=RANDOM_STATE,
    )

    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer(X_test)

    # Global explanation: average absolute contribution per feature.
    print("\nDisplaying the global SHAP summary plot...")
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=X_test.columns.tolist(),
        show=False,
    )
    plt.tight_layout()
    plt.show()

    # Local explanation: contributions for one prediction.
    print(f"\nDisplaying SHAP explanation for test row {row_index}...")
    print("Red generally pushes the output higher; blue pushes it lower.")
    print("The exact output scale depends on the model and SHAP settings.")

    # For binary classifiers, SHAP may return a single explanation
    # object or a separate output for each class, depending on version.
    if len(shap_values.shape) == 3:
        local_explanation = shap_values[row_index, :, 1]
    else:
        local_explanation = shap_values[row_index]

    shap.plots.waterfall(local_explanation, show=False)
    plt.tight_layout()
    plt.show()


def demonstrate_shortcut_learning():
    """
    Demonstrate how a model can achieve high test accuracy by relying
    on a misleading shortcut that will fail when the environment changes.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    sample_count = 1200

    # The true signal is only moderately informative.
    true_signal = rng.normal(size=sample_count)
    labels = (true_signal + rng.normal(scale=1.0, size=sample_count) > 0).astype(int)

    # In this training environment, the shortcut almost reveals the label.
    shortcut = np.where(
        rng.random(sample_count) < 0.95,
        labels,
        1 - labels,
    )

    X = pd.DataFrame({
        "medical_signal": true_signal,
        "hospital_marker": shortcut,
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    normal_accuracy = accuracy_score(y_test, model.predict(X_test))

    # Simulate a new hospital where the marker no longer relates to the label.
    shifted_test = X_test.copy()
    shifted_test["hospital_marker"] = rng.integers(
        0, 2, size=len(shifted_test)
    )

    shifted_accuracy = accuracy_score(
        y_test,
        model.predict(shifted_test),
    )

    print("\n=== SHORTCUT LEARNING DEMONSTRATION ===")
    print(f"Accuracy with original marker: {normal_accuracy:.2%}")
    print(f"Accuracy after marker changes: {shifted_accuracy:.2%}")
    print("\nGlobal feature importance:")
    print(
        pd.Series(
            model.feature_importances_,
            index=X.columns,
        ).sort_values(ascending=False).to_string()
    )

    print(
        "\nLesson: high test accuracy does not guarantee that a model "
        "learned the intended signal. A feature-importance result can "
        "raise suspicion, but further testing is needed to confirm "
        "whether the model relies on a shortcut."
    )


def main():
    X, y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("\nFeature names:", X.columns.tolist())

    model = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    explain_builtin_feature_importance(model, X.columns)

    explain_one_prediction(
        model,
        X_test,
        y_test,
        row_index=7,
    )

    explain_with_shap(
        model,
        X_train,
        X_test,
        row_index=7,
    )

    demonstrate_shortcut_learning()


if __name__ == "__main__":
    main()
