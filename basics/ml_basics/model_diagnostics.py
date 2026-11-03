
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

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
        max_depth=4,
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