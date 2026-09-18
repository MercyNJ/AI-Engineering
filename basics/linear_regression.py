# LINEAR REGRESSION — PREDICTING NUMBERS

# Linear regression predicts a continuous number.
# It finds the line that best fits the data.

# In its simplest form:
#
# prediction = (weight × input) + bias
#
# Training means finding the best values for the weight and bias.


# 1. IMPORTS

# LinearRegression is a class for creating a linear regression model.
from sklearn.linear_model import LinearRegression

import numpy as np


# 2. PREPARE THE DATA

# House sizes in square metres.
# X contains the feature used to make predictions.
sizes = np.array([
    [45],
    [60],
    [80],
    [100],
    [120],
    [150]
])

# Sale prices in KES.
# y contains the values the model is trying to predict.
prices = np.array([
    2_800_000,
    3_600_000,
    4_500_000,
    5_600_000,
    6_500_000,
    7_900_000
])


# 3. CREATE THE MODEL

model = LinearRegression()


# 4. TRAIN THE MODEL

# fit() finds the weight and bias that best fit the training data.
model.fit(sizes, prices)


# 5. UNDERSTAND WHAT THE MODEL LEARNED

# coef_ contains the learned weight.
# Here it represents the estimated price change for each additional m².
print(f"Price per m²: KES {model.coef_[0]:,.0f}")

# intercept_ is the learned base value when the input is 0.
print(f"Base price:   KES {model.intercept_:,.0f}")


# 6. MAKE A PREDICTION

# Predict the price of a new 90m² house.
predicted = model.predict([[90]])

print(f"Predicted price for 90m²: KES {predicted[0]:,.0f}")


# 7. HOW LINEAR REGRESSION LEARNS

# 1. Start with initial weight and bias values.
# 2. Make predictions.
# 3. Measure the prediction errors.
# 4. Adjust the weight and bias to reduce the error.
# 5. Repeat until the model finds a good fit.

# Mean Squared Error (MSE) measures error by squaring the differences
# between actual and predicted values.

# Gradient descent is an optimisation method that adjusts model parameters
# to reduce the error.


# 8. WHEN LINEAR REGRESSION WORKS WELL

# Good for:
# - Roughly straight-line relationships
# - Small, clean datasets
# - Problems where predictions need to be easy to explain

# It can struggle with:
# - Curved relationships
# - Very complex data
# - Relationships involving complex feature interactions


# KEY TERMS

# Feature
# An input used by the model to make a prediction.

# Target
# The value the model is trying to predict.

# Weight
# The learned value that determines how much a feature affects the prediction.

# Bias
# The learned base value added to the prediction.

# MSE
# Mean Squared Error. A measure of prediction error that gives more weight
# to larger errors.

# Gradient Descent
# An optimisation method used to reduce model error by adjusting parameters.

# Continuous value
# A numerical value that can take many possible values.
# Examples: price, distance, temperature, delivery time.