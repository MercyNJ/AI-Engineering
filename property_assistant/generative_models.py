import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
LATENT_DIM = 3
EPOCHS = 50
SAMPLE_COUNT = 10
OUTPUT_DIR = "outputs/property_generation"

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_synthetic_properties(count=2000):
    locations = np.random.choice(
        ["Ruiru", "Kilimani", "Westlands", "Embakasi"],
        size=count,
        p=[0.35, 0.25, 0.25, 0.15],
    )

    bedrooms = np.random.choice(
        [1, 2, 3, 4],
        size=count,
        p=[0.2, 0.4, 0.3, 0.1],
    )

    bathrooms = np.clip(
        bedrooms + np.random.choice(
            [-1, 0, 0, 1],
            size=count,
        ),
        1,
        5,
    )

    location_price = {
        "Ruiru": 30000,
        "Kilimani": 50000,
        "Westlands": 85000,
        "Embakasi": 30000,
    }

    base_prices = np.array(
        [location_price[location] for location in locations]
    )

    monthly_rent = (
        base_prices
        + bedrooms * 12000
        + bathrooms * 5000
        + np.random.normal(0, 6000, size=count)
    )

    monthly_rent = np.maximum(monthly_rent, 40000)

    furnished = np.random.choice(
        [0, 1],
        size=count,
        p=[0.7, 0.3],
    )

    monthly_rent += furnished * 15000

    return pd.DataFrame(
        {
            "location": locations,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "monthly_rent_kes": monthly_rent,
            "furnished": furnished,
        }
    )


def prepare_data(properties):
    location_mapping = {
        "Ruiru": 0,
        "Kilimani": 1,
        "Westlands": 2,
        "Embakasi": 3,
    }

    encoded = properties.copy()

    encoded["location"] = encoded["location"].map(
        location_mapping
    ) / 3.0

    encoded["bedrooms"] = encoded["bedrooms"] / 4.0
    encoded["bathrooms"] = encoded["bathrooms"] / 5.0

    # Divide by 250,000 so every rent fits between 0 and 1
    encoded["monthly_rent_kes"] = (
        encoded["monthly_rent_kes"] / 250000.0
    )

    feature_columns = [
        "location",
        "bedrooms",
        "bathrooms",
        "monthly_rent_kes",
        "furnished",
    ]

    return (
        encoded[feature_columns].to_numpy(dtype="float32"),
        location_mapping,
    )


class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_variance = inputs

        noise = tf.random.normal(tf.shape(mean))

        return mean + tf.exp(0.5 * log_variance) * noise


def build_vae(input_dim):
    encoder_inputs = keras.Input(shape=(input_dim,))

    x = layers.Dense(32, activation="relu")(encoder_inputs)
    x = layers.Dense(16, activation="relu")(x)

    mean = layers.Dense(LATENT_DIM)(x)
    log_variance = layers.Dense(LATENT_DIM)(x)

    # TensorFlow ops can't be applied directly to Keras inputs, so sampling runs inside a layer.
    latent = Sampling()([mean, log_variance])

    encoder = keras.Model(
        encoder_inputs,
        [mean, log_variance, latent],
        name="property_encoder",
    )

    decoder_inputs = keras.Input(shape=(LATENT_DIM,))

    x = layers.Dense(16, activation="relu")(decoder_inputs)
    x = layers.Dense(32, activation="relu")(x)
    decoder_outputs = layers.Dense(
        input_dim,
        activation="sigmoid",
    )(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs,
        name="property_decoder",
    )

    return encoder, decoder


class PropertyVAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        values = data[0] if isinstance(data, tuple) else data

        with tf.GradientTape() as tape:
            mean, log_variance, latent = self.encoder(
                values,
                training=True,
            )

            reconstructed = self.decoder(
                latent,
                training=True,
            )

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(values - reconstructed),
                    axis=1,
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + log_variance
                    - tf.square(mean)
                    - tf.exp(log_variance),
                    axis=1,
                )
            )

            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(
            total_loss,
            self.trainable_variables,
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def generate_properties(decoder, location_mapping, count):
    latent_vectors = tf.random.normal(
        shape=(count, LATENT_DIM)
    )

    generated = decoder.predict(
        latent_vectors,
        verbose=0,
    )

    generated = np.clip(generated, 0.0, 1.0)

    inverse_locations = {
        value / 3.0: key
        for key, value in location_mapping.items()
    }

    location_values = np.array(
        list(inverse_locations.keys())
    )

    generated_locations = [
        inverse_locations[
            min(
                location_values,
                key=lambda value: abs(value - row[0]),
            )
        ]
        for row in generated
    ]

    bedrooms = np.clip(
        np.rint(generated[:, 1] * 4),
        1,
        4,
    ).astype(int)

    bathrooms = np.clip(
        np.rint(generated[:, 2] * 5),
        1,
        5,
    ).astype(int)

    monthly_rent = np.maximum(
        np.rint(generated[:, 3] * 250000),
        40000,
    ).astype(int)

    furnished = (generated[:, 4] >= 0.5).astype(int)

    return pd.DataFrame(
        {
            "location": generated_locations,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "monthly_rent_kes": monthly_rent,
            "furnished": furnished,
        }
    )


def main():
    properties = create_synthetic_properties()

    print("Example training records:")
    print(properties.head())

    training_data, location_mapping = prepare_data(properties)

    encoder, decoder = build_vae(
        input_dim=training_data.shape[1]
    )

    vae = PropertyVAE(encoder, decoder)

    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

    vae.fit(
        training_data,
        epochs=EPOCHS,
        batch_size=32,
        verbose=0,
    )

    generated_properties = generate_properties(
        decoder,
        location_mapping,
        SAMPLE_COUNT,
    )

    output_path = f"{OUTPUT_DIR}/generated_properties.csv"

    generated_properties.to_csv(output_path, index=False)

    print("\nGenerated synthetic properties:")
    print(generated_properties.to_string(index=False))

    print(f"\nSaved generated records to: {output_path}")

    plt.figure(figsize=(8, 5))

    plt.hist(
        properties["monthly_rent_kes"],
        bins=30,
        alpha=0.6,
        label="Training data",
    )

    plt.hist(
        generated_properties["monthly_rent_kes"],
        bins=10,
        alpha=0.6,
        label="Generated samples",
    )

    plt.xlabel("Monthly rent (KES)")
    plt.ylabel("Number of properties")
    plt.title("Original vs generated synthetic rents")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/rent_comparison.png",
        dpi=150,
    )
    plt.close()


if __name__ == "__main__":
    main()
