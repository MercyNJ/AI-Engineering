import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 1. Classify property enquiry messages.
messages = [
    "I need a two bedroom apartment in Ruiru",
    "I want to schedule a house viewing",
    "Looking for a rental near the university",
    "Is the apartment available next month",
    "Please send me the property details",
    "I need a house for my family",
    "Claim your free prize now",
    "Win cash today by clicking this link",
    "Congratulations you have won free money",
    "Click here for a guaranteed reward",
    "Claim your bonus and free gift",
    "You have won a prize collect it now",
    "Can I view the apartment on Saturday",
    "What is the monthly rent for this house",
    "I am searching for a studio in Juja",
    "Please share the location of the property",
    "Free cash waiting for you click now",
    "Win a phone and claim your money",
    "I would like to rent a house in Kiambu",
    "Is parking available at the apartment",
]

# 0 = genuine enquiry, 1 = spam.
labels = [
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 0, 0,
]

text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("naive_bayes", MultinomialNB()),
])

X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)

text_pipeline.fit(X_train, y_train)
predictions = text_pipeline.predict(X_test)

print("Property enquiry classification:")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Genuine enquiry", "Spam"],
    zero_division=0,
))

new_messages = [
    "I want to view a two bedroom house in Ruiru",
    "Congratulations claim your free cash prize",
]

new_predictions = text_pipeline.predict(new_messages)

for message, prediction in zip(new_messages, new_predictions):
    label = "Spam" if prediction == 1 else "Genuine enquiry"
    print(f"{label}: {message}")


# 2. property listing data.
properties = pd.DataFrame({
    "location": [
        "Ruiru", "Juja", "Kiambu", "Thika", "Ruiru",
        "Juja", "Kiambu", "Thika", "Ruiru", "Juja",
        "Kiambu", "Thika", "Ruiru", "Juja", "Kiambu",
        "Thika", "Ruiru", "Juja", "Kiambu", "Thika",
        "Ruiru", "Juja", "Kiambu", "Thika", "Ruiru",
        "Juja", "Kiambu", "Thika", "Ruiru", "Juja",
    ],
    "bedrooms": [
        1, 2, 3, 4, 2, 1, 4, 3, 1, 2,
        3, 4, 2, 1, 3, 4, 1, 2, 4, 3,
        2, 1, 3, 4, 1, 2, 4, 3, 2, 1,
    ],
    "area_sqm": [
        35, 55, 110, 150, 65, 40, 140, 120, 38, 60,
        100, 160, 70, 42, 115, 145, 36, 58, 155, 105,
        68, 39, 108, 148, 37, 62, 142, 118, 72, 41,
    ],
    "rent_kes": [
        12000, 20000, 55000, 85000, 25000, 15000, 80000,
        65000, 14000, 22000, 50000, 90000, 28000, 16000,
        58000, 82000, 13000, 21000, 88000, 60000, 27000,
        15500, 53000, 87000, 12500, 23000, 79000, 62000,
        30000, 17000,
    ],
})

# Create a simple target from the rental price.
properties["rent_category"] = properties["rent_kes"].apply(
    lambda rent: "Premium" if rent >= 50000 else "Budget"
)

features = ["bedrooms", "area_sqm", "rent_kes"]
X = properties[features]
y = properties["rent_category"]


# 3. Compare linear and RBF SVM classifiers.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

models = {
    "Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0)),
    ]),
    "RBF SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ]),
}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print(f"Mean CV accuracy: {scores.mean():.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")


# 4. Predict a category for a new illustrative property.
best_example = models["RBF SVM"]

new_property = pd.DataFrame([{
    "bedrooms": 2,
    "area_sqm": 65,
    "rent_kes": 28000,
}])

best_example.fit(X_train, y_train)
prediction = best_example.predict(new_property)[0]

print(f"\nPredicted rental category: {prediction}")