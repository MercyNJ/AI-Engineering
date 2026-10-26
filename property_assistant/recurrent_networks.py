
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# 1. Generate illustrative monthly rent prices

# properties.csv has no dates, so generate 10 years of monthly prices
# on the same price_kes scale as the dataset.
np.random.seed(42)

months = 120
time = np.arange(months)

# Upward trend + yearly ups and downs + random noise
prices = (
    60_000
    + time * 300
    + 3_000 * np.sin(time / 6)
    + np.random.normal(0, 1_500, months)
)

prices = prices.astype("float32")


# 2. Normalize the prices

sequence_length = 12
number_of_sequences = months - sequence_length
split_index = int(number_of_sequences * 0.8)

# Use only the training months for min/max, so the test months stay unseen.
training_prices = prices[:split_index + sequence_length]

minimum_price = training_prices.min()
maximum_price = training_prices.max()

normalized_prices = (
    (prices - minimum_price)
    / (maximum_price - minimum_price)
)


# 3. Create sequences for training

X = []
y = []

# Sliding window: the previous 12 months are the input, the next month is the target.
for i in range(number_of_sequences):
    X.append(normalized_prices[i:i + sequence_length])
    y.append(normalized_prices[i + sequence_length])

# np.newaxis adds a feature dimension: (samples, 12) -> (samples, 12, 1)
X = np.array(X, dtype="float32")[..., np.newaxis]
y = np.array(y, dtype="float32")

# Split by time, not randomly: train on earlier months, test on later months.
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# 4. Build a GRU model

model = keras.Sequential([
    layers.Input(shape=(sequence_length, 1)),
    layers.GRU(32),
    layers.Dense(16, activation="relu"),
    # No activation: the output is a price value, not a probability.
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"]
)

model.summary()


# 5. Train the model

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# shuffle=False keeps the months in order.
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)


# 6. Evaluate predictions on later observations

test_loss, test_mae = model.evaluate(
    X_test, y_test, verbose=0
)

# Convert the error back from the 0-1 scale to KES.
test_mae_kes = test_mae * (maximum_price - minimum_price)

print(f"Test loss: {test_loss:.6f}")
print(f"Normalized test MAE: {test_mae:.4f}")
print(f"Test MAE: KES {test_mae_kes:,.0f}")


# 7. Predict the next month's price

recent_prices = normalized_prices[-sequence_length:]

latest_sequence = recent_prices.reshape(
    1, sequence_length, 1
)

predicted_normalized_price = model.predict(
    latest_sequence, verbose=0
)[0, 0]

predicted_price = (
    predicted_normalized_price
    * (maximum_price - minimum_price)
    + minimum_price
)

print(f"Latest price: KES {prices[-1]:,.0f}")
print(f"Predicted next-month price: KES {predicted_price:,.0f}")


# 8. Save the model

# models/ is in .gitignore, so the saved model is not committed.
MODEL_PATH = Path("models/property_price_gru.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)

print(
    "\nThese results use synthetic data. "
    "They are for learning, not real property price forecasts."
)
