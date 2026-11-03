# CLASSIFICATION — PREDICTING CATEGORIES

# Classification is a supervised learning problem.
# It predicts a category rather than a continuous number.

# Examples:
# Binary classification → spam / not spam
# Multi-class classification → cat / dog / bird


# 1. IMPORTS

# LogisticRegression is a classification algorithm.
from sklearn.linear_model import LogisticRegression

import numpy as np


# 2. PREPARE THE DATA

# Features:
# monthly income (KES thousands)
# months at current job
X = np.array([
    [25, 3],
    [80, 24],
    [15, 1],
    [120, 36],
    [30, 6],
    [95, 30]
])

# Target:
# 0 = declined
# 1 = approved
y = np.array([
    0,
    1,
    0,
    1,
    0,
    1
])


# 3. CREATE THE MODEL

model = LogisticRegression()


# 4. TRAIN THE MODEL

# fit() trains the classifier using the features and known labels.
model.fit(X, y)


# 5. MAKE A PREDICTION

# New applicant:
# income = KES 60,000
# months at current job = 10
applicant = [[60, 10]]

decision = model.predict(applicant)[0]

print(
    f"Decision: {'Approve' if decision == 1 else 'Decline'}"
)


# 6. GET THE PROBABILITY

# predict_proba() returns the probability for each class.
probabilities = model.predict_proba(applicant)[0]

print(f"Decline probability: {probabilities[0]:.0%}")
print(f"Approval probability: {probabilities[1]:.0%}")


# 7. UNDERSTANDING CLASSIFICATION

# A classifier uses the features to determine which category
# an input is most likely to belong to.

# For binary classification:
#
# Class 0 → Decline
# Class 1 → Approve
#
# The model calculates probabilities for both classes.
# The class with the higher probability becomes the prediction.


# 8. DECISION BOUNDARY

# A classifier creates a boundary that separates the classes.
#
# With logistic regression, the decision boundary is based on
# a weighted combination of the input features.


# KEY TERMS

# Classification
# Predicting a category or label.

# Binary classification
# Choosing between two categories.

# Multi-class classification
# Choosing between three or more categories.

# Class
# One of the possible categories the model can predict.

# predict()
# Returns the predicted class.

# predict_proba()
# Returns the probability for each possible class.

# Decision boundary
# The boundary used by a classifier to separate different classes.

# Logistic Regression
# A classification algorithm that uses probabilities to predict classes.