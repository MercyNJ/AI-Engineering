import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
LATENT_DIM = 32
BATCH_SIZE = 128
EPOCHS = 5
OUTPUT_DIR = "outputs/generative_models"

tf.keras.utils.set_random_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_mnist():
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(10000, seed=SEED)
    dataset = dataset.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )

    return x_train, dataset


def save_image_grid(images, filename, columns=5):
    images = np.asarray(images)
    rows = int(np.ceil(len(images) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2, rows * 2),
    )
    axes = np.array(axes).reshape(-1)

    for index, axis in enumerate(axes):
        axis.axis("off")

        if index < len(images):
            axis.imshow(images[index].squeeze(), cmap="gray")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)


# --------------------------------------------------
# 1. Variational Autoencoder (VAE)
# --------------------------------------------------

class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_variance = inputs

        noise = tf.random.normal(tf.shape(mean))

        return mean + tf.exp(0.5 * log_variance) * noise


def build_vae():
    encoder_inputs = keras.Input(shape=(28, 28, 1))

    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    mean = layers.Dense(LATENT_DIM, name="latent_mean")(x)
    log_variance = layers.Dense(
        LATENT_DIM,
        name="latent_log_variance",
    )(x)

    latent = Sampling()([mean, log_variance])

    encoder = keras.Model(
        encoder_inputs,
        [mean, log_variance, latent],
        name="vae_encoder",
    )

    decoder_inputs = keras.Input(shape=(LATENT_DIM,))

    x = layers.Dense(128, activation="relu")(decoder_inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(28 * 28, activation="sigmoid")(x)

    decoder_outputs = layers.Reshape((28, 28, 1))(x)

    decoder = keras.Model(
        decoder_inputs,
        decoder_outputs,
        name="vae_decoder",
    )

    return encoder, decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss"
        )
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(
            name="kl_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data

        with tf.GradientTape() as tape:
            mean, log_variance, latent = self.encoder(
                images,
                training=True,
            )

            reconstructed = self.decoder(
                latent,
                training=True,
            )

            pixel_loss = keras.losses.binary_crossentropy(
                images,
                reconstructed,
            )

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(pixel_loss, axis=(1, 2))
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

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def train_vae(dataset):
    encoder, decoder = build_vae()
    vae = VAE(encoder, decoder)

    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

    vae.fit(dataset, epochs=EPOCHS)

    random_latent_vectors = tf.random.normal(
        shape=(25, LATENT_DIM)
    )

    generated_images = decoder.predict(
        random_latent_vectors,
        verbose=0,
    )

    save_image_grid(
        generated_images,
        f"{OUTPUT_DIR}/vae_generated.png",
    )

    return encoder, decoder, vae


# --------------------------------------------------
# 2. Generative Adversarial Network (GAN)
# --------------------------------------------------

def build_generator():
    model = keras.Sequential(
        [
            keras.Input(shape=(LATENT_DIM,)),
            layers.Dense(256),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dense(512),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dense(28 * 28, activation="tanh"),
            layers.Reshape((28, 28, 1)),
        ],
        name="generator",
    )

    return model


def build_discriminator():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(512),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dense(1),
        ],
        name="discriminator",
    )

    return model


def train_gan(dataset):
    generator = build_generator()
    discriminator = build_discriminator()

    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.5,
    )

    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.5,
    )

    loss_function = keras.losses.BinaryCrossentropy(
        from_logits=True
    )

    @tf.function
    def train_step(real_images):
        batch_size = tf.shape(real_images)[0]

        noise = tf.random.normal(
            shape=(batch_size, LATENT_DIM)
        )

        with tf.GradientTape() as generator_tape, \
                tf.GradientTape() as discriminator_tape:

            fake_images = generator(noise, training=True)

            real_scores = discriminator(
                real_images,
                training=True,
            )

            fake_scores = discriminator(
                fake_images,
                training=True,
            )

            discriminator_loss = (
                loss_function(
                    tf.ones_like(real_scores),
                    real_scores,
                )
                + loss_function(
                    tf.zeros_like(fake_scores),
                    fake_scores,
                )
            )

            generator_loss = loss_function(
                tf.ones_like(fake_scores),
                fake_scores,
            )

        generator_gradients = generator_tape.gradient(
            generator_loss,
            generator.trainable_variables,
        )

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss,
            discriminator.trainable_variables,
        )

        generator_optimizer.apply_gradients(
            zip(
                generator_gradients,
                generator.trainable_variables,
            )
        )

        discriminator_optimizer.apply_gradients(
            zip(
                discriminator_gradients,
                discriminator.trainable_variables,
            )
        )

        return generator_loss, discriminator_loss

    for epoch in range(EPOCHS):
        generator_losses = []
        discriminator_losses = []

        for real_images in dataset:
            real_images = real_images * 2.0 - 1.0

            generator_loss, discriminator_loss = train_step(
                real_images
            )

            generator_losses.append(float(generator_loss))
            discriminator_losses.append(float(discriminator_loss))

        print(
            f"GAN Epoch {epoch + 1}/{EPOCHS} | "
            f"Generator loss: {np.mean(generator_losses):.4f} | "
            f"Discriminator loss: "
            f"{np.mean(discriminator_losses):.4f}"
        )

    noise = tf.random.normal(shape=(25, LATENT_DIM))
    generated_images = generator(noise, training=False)

    generated_images = (generated_images + 1.0) / 2.0

    save_image_grid(
        generated_images.numpy(),
        f"{OUTPUT_DIR}/gan_generated.png",
    )

    return generator, discriminator


def main():
    _, dataset = load_mnist()

    print("\nTraining the VAE...")
    train_vae(dataset)

    print("\nTraining the GAN...")
    train_gan(dataset)

    print(f"\nGenerated image grids saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
