import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. YOUR FIRST PLOT
# ============================================================

# Data we want to plot
epochs = [1, 2, 3, 4, 5]
loss = [2.3, 1.8, 1.4, 1.1, 0.9]

# plot() creates a line connecting our data points
plt.plot(epochs, loss)

# xlabel() gives the horizontal axis a name
plt.xlabel("Epoch")

# ylabel() gives the vertical axis a name
plt.ylabel("Training Loss")

# title() gives the chart a title
plt.title("Model Training Curve")

# show() displays the chart
plt.show()


# ============================================================
# 2. FIGURE SIZE
# ============================================================

# figure() creates the overall canvas for the chart.
# figsize=(10, 5) means 10 inches wide and 5 inches tall.
plt.figure(figsize=(10, 5))

plt.plot(epochs, loss)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss")

plt.show()


# ============================================================
# 3. TWO LINES ON THE SAME CHART
# ============================================================

epochs = [1, 2, 3, 4, 5]

train_loss = [2.3, 1.8, 1.4, 1.1, 0.9]
val_loss = [2.4, 1.9, 1.5, 1.2, 1.05]

plt.figure(figsize=(10, 5))

# First line
plt.plot(
    epochs,
    train_loss,
    label="Training Loss"
)

# Second line
plt.plot(
    epochs,
    val_loss,
    label="Validation Loss",
    linestyle="--"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

# legend() displays the labels we gave the lines
plt.legend()

# grid() adds horizontal and vertical guide lines
plt.grid(True)

plt.show()


# ============================================================
# 4. BAR CHART
# ============================================================

models = [
    "GPT-4o",
    "Claude-3",
    "Gemini",
    "Llama-3"
]

accuracy = [
    0.942,
    0.938,
    0.921,
    0.897
]

# bar() creates a bar chart.
# The first argument provides the categories.
# The second argument provides their values.
plt.bar(models, accuracy)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")

plt.show()


# ============================================================
# 5. ADDING VALUES TO THE BARS
# ============================================================

plt.figure(figsize=(10, 5))

bars = plt.bar(models, accuracy)

# zip() pairs each bar with its accuracy value.
for bar, acc in zip(bars, accuracy):

    # text() writes text onto the chart.
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{acc:.1%}",
        ha="center",
        va="bottom"
    )

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")

plt.show()


# ============================================================
# 6. SCATTER PLOT
# ============================================================

latency = [320, 280, 310, 180]
accuracy = [0.942, 0.938, 0.921, 0.897]

# scatter() creates individual points instead of bars or lines.
#
# Here we are asking:
# "Is there a relationship between latency and accuracy?"
plt.scatter(latency, accuracy)

plt.xlabel("Latency (ms)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Latency")

plt.grid(True)

plt.show()


# ============================================================
# 7. HISTOGRAM
# ============================================================

# Generate 100 confidence scores for demonstration.
confidences = np.random.beta(8, 2, 100)

# hist() shows how often values occur within ranges.
plt.hist(confidences, bins=10)

plt.xlabel("Confidence Score")
plt.ylabel("Count")
plt.title("Distribution of Confidence Scores")

plt.show()


# ============================================================
# 8. SUBPLOTS
# ============================================================

# subplots() allows us to create multiple charts
# inside one figure.
#
# 1 row, 2 columns = two charts side by side.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))


# -------------------------
# First chart
# -------------------------

axes[0].plot(epochs, train_loss)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")


# -------------------------
# Second chart
# -------------------------

axes[1].bar(models, accuracy)

axes[1].set_title("Model Accuracy")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("Accuracy")


# tight_layout() adjusts spacing so
# labels and titles don't overlap.
plt.tight_layout()

plt.show()