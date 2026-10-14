import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 1. Multinomial Naive Bayes for text classification.
messages = [
    "Win a free phone now",
    "Meeting at three tomorrow",
    "Claim your free prize",
    "Please review the project",
    "Congratulations you won money",
    "The deadline is Friday",
    "Free cash waiting for you",
    "Can we discuss the property",
    "Click here to claim your prize",
    "The viewing is scheduled for Monday",
    "Win money with this offer",
    "I would like to rent an apartment",
]

# 1 = spam, 0 = genuine message.
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

text_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("naive_bayes", MultinomialNB()),
])

text_model.fit(messages, labels)

new_messages = [
    "Claim your free money now",
    "I want to schedule a property viewing",
]

predictions = text_model.predict(new_messages)
probabilities = text_model.predict_proba(new_messages)

print("Text classification:")
for message, prediction, probability in zip(
    new_messages, predictions, probabilities
):
    label = "Spam" if prediction == 1 else "Genuine"
    print(f"\nMessage: {message}")
    print(f"Prediction: {label}")
    print(f"Class probabilities: {probability.round(3)}")


# 2. Gaussian Naive Bayes for numerical features.
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)

nb_predictions = gaussian_nb.predict(X_test)

print("\nGaussian Naive Bayes:")
print(f"Test accuracy: {accuracy_score(y_test, nb_predictions):.3f}")

nb_scores = cross_val_score(
    GaussianNB(),
    X,
    y,
    cv=5,
    scoring="accuracy",
)
print(f"Mean CV accuracy: {nb_scores.mean():.3f}")


# 3. Linear SVM.
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

linear_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1.0)),
])

linear_scores = cross_val_score(
    linear_svm,
    X,
    y,
    cv=5,
    scoring="accuracy",
)

print("\nLinear SVM:")
print(f"Mean CV accuracy: {linear_scores.mean():.3f}")


# 4. RBF SVM for non-linear decision boundaries.
rbf_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])

rbf_scores = cross_val_score(
    rbf_svm,
    X,
    y,
    cv=5,
    scoring="accuracy",
)

print("\nRBF SVM:")
print(f"Mean CV accuracy: {rbf_scores.mean():.3f}")


# 5. Compare SVM hyperparameter settings.
settings = [
    {"C": 0.1, "gamma": "scale"},
    {"C": 1.0, "gamma": "scale"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 1.0, "gamma": 0.01},
]

print("\nRBF SVM settings comparison:")

for params in settings:
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=params["C"],
                gamma=params["gamma"],
            ),
        ),
    ])

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
    )

    print(
        f"C={params['C']}, gamma={params['gamma']}: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )


# 6. Fit and evaluate a final example.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

rbf_svm.fit(X_train, y_train)
final_predictions = rbf_svm.predict(X_test)

print("\nRBF SVM test report:")
print(classification_report(
    y_test,
    final_predictions,
    target_names=cancer.target_names,
))