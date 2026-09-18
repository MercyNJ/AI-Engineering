# Supervised Learning

# scikit-learn (sklearn) is a Python library for machine learning.

# It provides tools for preparing data, training models, and evaluating models.

# Import load_iris(), a function that loads the built-in Iris dataset.

from sklearn.datasets import load_iris

# Import train_test_split(), a function that divides data into training and testing sets.

from sklearn.model_selection import train_test_split

# Import KNeighborsClassifier, a class used to create a K-Nearest Neighbours classification model.

from sklearn.neighbors import KNeighborsClassifier

# Import accuracy_score(), a function that measures how many predictions were correct.

from sklearn.metrics import accuracy_score

# pandas is a Python library for working with tabular data.

import pandas as pd

# Step 1: Load the data

# load_iris() loads the Iris dataset.

# The returned object contains the data, labels, and other information.

iris = load_iris()

# X contains the features used by the model.

# Each flower has 4 measurements.

X = iris.data

# y contains the labels the model is trying to predict.

# 0 = setosa, 1 = versicolor, 2 = virginica

y = iris.target

# .shape gives the dimensions of the data.

# X.shape[0] = number of flowers

# X.shape[1] = number of features

print(f"Dataset: {X.shape[0]} flowers, {X.shape[1]} features")

# target_names contains the names of the flower classes.

print(f"Classes: {iris.target_names}")

# Step 2: Split into training and testing sets

# Training data is used to teach the model.

# Testing data is used later to check the model's performance.

#

# We should never train and test on the same data.

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,     # 20% for testing, 80% for training
random_state=42    # keeps the split reproducible
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# Step 3: Choose a model and train it

# KNeighborsClassifier is a class for creating a KNN classification model.

# n_neighbors=3 means the model looks at the 3 nearest examples when deciding the class of a new example.

model = KNeighborsClassifier(n_neighbors=3)

# fit() trains the model using the training data.

# X_train contains the features and y_train contains the correct labels.

model.fit(X_train, y_train)

print("Training complete!")

# Step 4: Make predictions on the test set

# predict() uses the trained model to make predictions.

y_pred = model.predict(X_test)

print("Predictions:", y_pred[:10])
print("Actual:     ", y_test[:10])

# Step 5: Evaluate performance

# accuracy_score() compares the actual labels with the predictions.

accuracy = accuracy_score(y_test, y_pred)

# :.2% displays the accuracy as a percentage with 2 decimal places.

print(f"Accuracy: {accuracy:.2%}")

# K-Nearest Neighbours (KNN)

# KNN looks at the most similar examples it has already seen.

#

# For example, if a new flower is closest to 3 known flowers:

#

# Flower 1 → setosa

# Flower 2 → setosa

# Flower 3 → versicolor

#

# The majority vote is setosa, so KNN predicts setosa.

# Understanding the results

# Import two more functions for evaluating classification models.

from sklearn.metrics import confusion_matrix, classification_report

# confusion_matrix() shows which predictions were correct and which classes were confused with each other.

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Example:

#

# [[10  0  0]

# [ 0  9  1]

# [ 0  0 10]]

#

# Rows represent the actual classes.

# Columns represent the predicted classes.

# The diagonal represents correct predictions.

# classification_report() gives precision, recall, and F1-score.

#

# precision → how often predictions for a class were correct

# recall    → how many actual examples of a class were found

# F1-score  → combines precision and recall

#

# target_names replaces 0, 1, and 2 with the flower names.

print(
classification_report(
y_test,
y_pred,
target_names=iris.target_names
)
)

# Key terms

# Library

# A collection of reusable code.

# Examples: pandas and scikit-learn.

# Module

# A part of a library containing related tools.

# Example: sklearn.datasets.

# Function

# Reusable code that performs a specific task.

# Examples: load_iris(), train_test_split(), accuracy_score().

# Class

# A blueprint used to create objects.

# Example: KNeighborsClassifier.

# Object

# A specific instance created from a class.

# In this example, model is an object created from the KNeighborsClassifier class.

# Feature

# Information given to the model to help it make a prediction.

# Here, the flower measurements are the features.

# Label / Target

# The correct answer the model is trying to predict.

# Here, the flower species is the target.

# Training

# The process where the model learns patterns from training data.

# Prediction

# The answer produced by the trained model.

# Classification

# A machine learning problem where the model predicts a category.

# Accuracy

# The proportion of predictions that were correct.

# Basic machine learning flow

# Data

# ↓

# Split into training and testing data

# ↓

# Create a model

# ↓

# fit() → train the model

# ↓

# predict() → make predictions

# ↓

# Evaluate the predictions
