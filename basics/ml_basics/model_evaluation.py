# HOW DO YOU KNOW IF YOUR MODEL IS GOOD?

# Choosing the right metric is important when evaluating a model.
# A model can look good with one metric while performing poorly in another.

# 1. OVERFITTING

# Overfitting happens when a model memorises the training data
# instead of learning patterns that generalise to new data.

# Underfitting → model is too simple and misses patterns.
# Just right   → model learns useful patterns and generalises well.
# Overfitting  → model performs very well on training data
#                but poorly on new/test data.


# 2. CLASSIFICATION METRICS

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# Accuracy → of all predictions, how many were correct?
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2%}")

# Precision → when the model predicts positive, how often is it correct?
print(f"Precision: {precision_score(y_true, y_pred):.2%}")

# Recall → of all actual positives, how many did the model catch?
print(f"Recall:    {recall_score(y_true, y_pred):.2%}")

# F1 → combines precision and recall into one metric.
print(f"F1 Score:  {f1_score(y_true, y_pred):.2%}")


# 3. REGRESSION METRICS

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

actual_prices = [500000, 750000, 320000, 950000, 680000]
predicted_prices = [490000, 780000, 310000, 920000, 700000]

# MAE → average absolute difference between actual and predicted values.
mae = mean_absolute_error(actual_prices, predicted_prices)

# RMSE → similar to MAE, but gives more weight to large errors.
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

# R² → measures how well the model explains the variation in the data.
r2 = r2_score(actual_prices, predicted_prices)

print(f"MAE:  KES {mae:,.0f}")
print(f"RMSE: KES {rmse:,.0f}")
print(f"R²:   {r2:.3f}")


# 4. CHOOSING A METRIC

# The most useful metric depends on what the model is trying to do.

# Classification:
# Accuracy  → overall correctness
# Precision → avoid false positives
# Recall    → avoid missing positive cases
# F1        → balance precision and recall

# Regression:
# MAE  → average size of errors
# RMSE → penalises large errors more
# R²   → how well the model explains the data


# 5. REAL-WORLD EXAMPLE

# Imagine a medical model detecting tuberculosis.
#
# If only 5% of patients have TB and the model always predicts "no TB":
#
# Accuracy = 95%
#
# But the model catches zero real TB cases.
#
# This shows why accuracy alone can sometimes be misleading.
# Recall would be important when missing a positive case is very costly.


# KEY TERMS

# Accuracy
# Percentage of all predictions that were correct.

# Precision
# Of the cases predicted as positive, how many were actually positive.

# Recall
# Of all actual positive cases, how many were correctly identified.

# F1 Score
# A balance between precision and recall.

# MAE
# Average absolute difference between actual and predicted values.

# RMSE
# Measures prediction error while giving more weight to larger errors.

# R²
# Measures how well a regression model explains variation in the target.

# Overfitting
# When a model learns the training data too closely and performs poorly
# on new data.

# Underfitting
# When a model is too simple to learn the important patterns in the data.
