
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 1. Prepare the data.
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
scores = np.array([42, 53, 58, 65, 70, 76, 82, 85, 91, 94])

# Scikit-learn expects X to be two-dimensional.
X = hours.reshape(-1, 1)
y = scores

# Keep some data aside for evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 2. Train the model.
model = LinearRegression()
model.fit(X_train, y_train)

print("Linear Regression")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(
    f"Equation: score = "
    f"{model.coef_[0]:.2f} * hours + {model.intercept_:.2f}"
)


# 3. Make predictions.
test_hours = np.array([5, 7, 9]).reshape(-1, 1)
predictions = model.predict(test_hours)

print("\nPredictions:")
for hour, prediction in zip(test_hours.ravel(), predictions):
    print(f"{hour} hours -> predicted score: {prediction:.1f}")


# 4. Evaluate on data not used for training.
test_predictions = model.predict(X_test)

print("\nTest evaluation:")
print(f"R²: {r2_score(y_test, test_predictions):.3f}")
print(f"MSE: {mean_squared_error(y_test, test_predictions):.2f}")
print(
    f"RMSE: {np.sqrt(mean_squared_error(y_test, test_predictions)):.2f}"
)


# 5. Plot the training data, test data, and regression line.
plt.figure(figsize=(9, 5))

plt.scatter(
    X_train.ravel(),
    y_train,
    label="Training data",
    s=70,
)
plt.scatter(
    X_test.ravel(),
    y_test,
    label="Test data",
    marker="x",
    s=90,
)

x_line = np.linspace(0, 11, 100).reshape(-1, 1)
plt.plot(
    x_line,
    model.predict(x_line),
    label="Regression line",
    linewidth=2,
)

plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.title("Study Hours vs Exam Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()