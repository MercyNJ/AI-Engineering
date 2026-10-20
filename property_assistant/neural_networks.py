
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def demonstrate_single_neuron():
    print("1. A PROPERTY-MATCHING NEURON")

    features = {
        "size_scaled": 0.7,
        "bedrooms_scaled": 0.5,
        "budget_fit_scaled": 0.8,
    }

    weights = {
        "size_scaled": 0.4,
        "bedrooms_scaled": 0.3,
        "budget_fit_scaled": 0.7,
    }

    bias = -0.2

    weighted_sum = sum(
        features[name] * weights[name]
        for name in features
    ) + bias

    output = max(0, weighted_sum)

    print(f"Weighted sum: {weighted_sum:.3f}")
    print(f"ReLU output:  {output:.3f}")


def build_property_dataset(n=1500, seed=42):
    rng = np.random.default_rng(seed)

    price_millions = rng.uniform(2, 25, n)
    buyer_budget_millions = rng.uniform(3, 22, n)
    bedrooms = rng.integers(1, 6, n)
    size_sqft = np.maximum(
        300,
        bedrooms * rng.normal(350, 60, n)
        + rng.normal(100, 150, n),
    )
    distance_km = rng.uniform(1, 30, n)

    budget_ratio = price_millions / buyer_budget_millions

    suitable = (
        (budget_ratio <= 1.0)
        & (bedrooms >= 2)
        & (size_sqft >= 650)
        & (distance_km <= 20)
    ).astype(int)

    noise_indices = rng.choice(
        n,
        size=int(n * 0.08),
        replace=False,
    )
    suitable[noise_indices] = 1 - suitable[noise_indices]

    return pd.DataFrame({
        "price_millions": price_millions,
        "buyer_budget_millions": buyer_budget_millions,
        "bedrooms": bedrooms,
        "size_sqft": size_sqft,
        "distance_km": distance_km,
        "suitable_match": suitable,
    })


def train_neural_network():
    print("\n2. TRAINING A PROPERTY-MATCHING NETWORK")

    torch.manual_seed(42)

    df = build_property_dataset()

    feature_columns = [
        "price_millions",
        "buyer_budget_millions",
        "bedrooms",
        "size_sqft",
        "distance_km",
    ]

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["suitable_match"].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(
        y_train.reshape(-1, 1),
        dtype=torch.float32,
    )
    y_test = torch.tensor(
        y_test.reshape(-1, 1),
        dtype=torch.float32,
    )

    model = nn.Sequential(
        nn.Linear(len(feature_columns), 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100

    for epoch in range(epochs):
        model.train()

        logits = model(X_train)
        loss = loss_function(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3}/{epochs} "
                f"| Training loss: {loss.item():.4f}"
            )

    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_test))
        predictions = (probabilities >= 0.5).int()

    y_true = y_test.numpy().ravel()
    y_pred = predictions.numpy().ravel()

    print(f"\nTest accuracy: {accuracy_score(y_true, y_pred):.1%}")
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Not suitable", "Suitable"],
        zero_division=0,
    ))

    print("Example property predictions:")

    for index in range(5):
        print(
            f"Property {index + 1}: "
            f"{probabilities[index].item():.1%} estimated match probability"
        )

    print(
        "\nThis is synthetic training data for learning. "
        "These predictions are not real property recommendations."
    )


if __name__ == "__main__":
    demonstrate_single_neuron()
    train_neural_network()
