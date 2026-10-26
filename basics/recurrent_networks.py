
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers


# 1. Load and prepare the IMDB sentiment dataset

vocabulary_size = 10000
sequence_length = 200

# Each review is already a list of word numbers.
# num_words keeps only the 10,000 most common words.
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
    num_words=vocabulary_size
)

# pad_sequences() makes every review 200 words long:
# short reviews are padded with 0s, long reviews are cut.
X_train = keras.utils.pad_sequences(
    X_train, maxlen=sequence_length
)
X_test = keras.utils.pad_sequences(
    X_test, maxlen=sequence_length
)

print("Training data:", X_train.shape)
print("Test data:", X_test.shape)


# 2. Build a simple RNN

simple_rnn = keras.Sequential([
    layers.Input(shape=(sequence_length,)),
    # Embedding turns each word number into a vector of 64 values.
    layers.Embedding(vocabulary_size, 64),
    # SimpleRNN reads the words one at a time and keeps a memory of what it has seen.
    layers.SimpleRNN(64),
    layers.Dense(1, activation="sigmoid")
])

simple_rnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

simple_rnn.summary()


# 3. Build an LSTM

lstm_model = keras.Sequential([
    layers.Input(shape=(sequence_length,)),
    layers.Embedding(vocabulary_size, 64),
    # LSTM has gates that decide what to remember and forget, so it handles long sequences better.
    # return_sequences=True passes the output for every word to the next LSTM layer.
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(1, activation="sigmoid")
])

lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

lstm_model.summary()


# 4. Build a GRU

gru_model = keras.Sequential([
    layers.Input(shape=(sequence_length,)),
    layers.Embedding(vocabulary_size, 64),
    # GRU is a simpler, faster version of LSTM with fewer gates.
    layers.GRU(64),
    layers.Dense(1, activation="sigmoid")
])

gru_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

gru_model.summary()


# 5. Build a bidirectional LSTM

bidirectional_model = keras.Sequential([
    layers.Input(shape=(sequence_length,)),
    layers.Embedding(vocabulary_size, 64),
    # Bidirectional reads the review forwards and backwards, then combines both.
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(1, activation="sigmoid")
])

bidirectional_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

bidirectional_model.summary()


# 6. Train one model

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = lstm_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping]
)


# 7. Evaluate the trained model

test_loss, test_accuracy = lstm_model.evaluate(
    X_test, y_test
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# 8. Predict sentiment for sample reviews

predictions = lstm_model.predict(X_test[:5])

for index, prediction in enumerate(predictions):
    probability = float(prediction[0])
    sentiment = "positive" if probability >= 0.5 else "negative"

    print(f"Review {index + 1}: {sentiment}")
    print(f"Positive sentiment probability: {probability:.2%}")


# 9. Save the trained model

# models/ is in .gitignore, so the saved model is not committed.
MODEL_PATH = Path("models/imdb_lstm_model.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
lstm_model.save(MODEL_PATH)
