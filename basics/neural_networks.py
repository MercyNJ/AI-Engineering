
import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def demonstrate_single_neuron():
    print("1. A SINGLE NEURON")

    inputs = [0.5, 0.8, 0.2]
    weights = [0.4, -0.3, 0.7]
    bias = 0.1

    weighted_sum = sum(
        weight * value
        for weight, value in zip(weights, inputs)
    ) + bias

    output = max(0, weighted_sum)

    print(f"Weighted sum: {weighted_sum:.3f}")
    print(f"ReLU output:  {output:.3f}")


def train_neural_network():
    print("\n2. TRAINING A NEURAL NETWORK")

    torch.manual_seed(42)

    X, y = make_moons(
        n_samples=1000,
        noise=0.25,
        random_state=42,
    )

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
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
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
        test_logits = model(X_test)
        probabilities = torch.sigmoid(test_logits)
        predictions = (probabilities >= 0.5).int()

    y_true = y_test.numpy().ravel()
    y_pred = predictions.numpy().ravel()

    print(f"\nTest accuracy: {accuracy_score(y_true, y_pred):.1%}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Example probabilities:")
    for probability in probabilities[:5].numpy().ravel():
        print(f"{probability:.3f}")


if __name__ == "__main__":
    demonstrate_single_neuron()
    train_neural_network()
