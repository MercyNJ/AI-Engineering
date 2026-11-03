
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# DATA
# ============================================================

models   = ["GPT-4o", "Claude-3", "Gemini", "Llama-3", "Mistral"]
accuracy = [0.942, 0.938, 0.921, 0.897, 0.871]
latency  = [320, 280, 310, 180, 95]

# range(1, 11) creates numbers from 1 to 10.
# list() converts them into a normal Python list.
epochs     = list(range(1, 11))

train_loss = [2.3, 1.8, 1.4, 1.1, 0.9, 0.75, 0.65, 0.58, 0.54, 0.51]
val_loss   = [2.4, 1.9, 1.5, 1.2, 1.05, 0.95, 0.88, 0.84, 0.82, 0.81]

# NumPy generates 500 random confidence values.
# The values are between 0 and 1.
confidences = np.random.beta(8, 2, 500)


# ============================================================
# 2×2 DASHBOARD
# ============================================================

# subplots() creates multiple charts inside one figure.
#
# 2, 2 means:
#   2 rows
#   2 columns
#
# This gives us 4 charts:
#
#   [0, 0]  [0, 1]
#   [1, 0]  [1, 1]
#
# fig = the entire figure
# axes = the individual chart areas
#
# figsize controls the overall size.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# suptitle() adds one main title to the whole figure.
fig.suptitle("AI Model Dashboard", fontsize=15, fontweight="bold")


# ============================================================
# TOP-LEFT: ACCURACY COMPARISON
# ============================================================

# [0, 0] means the first row and first column.
#
# bar() creates a bar chart.
axes[0, 0].bar(models, accuracy, color="#f97316", edgecolor="white")

axes[0, 0].set_title("Model Accuracy")
axes[0, 0].set_ylabel("Accuracy")

# set_ylim() controls the minimum and maximum
# values shown on the Y-axis.
axes[0, 0].set_ylim(0.85, 0.96)


# Add the accuracy percentage above each bar.
#
# zip() pairs each bar with its accuracy value.
#
# enumerate() also gives us the position number (i).
for i, (bar_val, acc) in enumerate(zip(axes[0, 0].patches, accuracy)):

    # get_x() and get_width() help us find
    # the horizontal centre of the bar.
    #
    # get_height() gives us the height of the bar.
    #
    # text() places the percentage on the chart.
    axes[0, 0].text(
        bar_val.get_x() + bar_val.get_width()/2,
        acc + 0.001,
        f"{acc:.1%}",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )


# ============================================================
# TOP-RIGHT: LATENCY VS ACCURACY
# ============================================================

# [0, 1] means the first row and second column.
#
# scatter() creates individual points.
#
# X-axis = latency
# Y-axis = accuracy
axes[0, 1].scatter(
    latency,
    accuracy,
    s=120,
    color="#6366f1",
    edgecolors="white",
    zorder=5
)


# Add the model name next to each point.
#
# enumerate() gives us:
#   i    = position/index
#   name = model name
for i, name in enumerate(models):

    # annotate() adds text near a specific point.
    #
    # latency[i] and accuracy[i] give us
    # the coordinates of that model.
    axes[0, 1].annotate(
        name,
        (latency[i], accuracy[i]),
        xytext=(6, 3),
        textcoords="offset points",
        fontsize=8
    )


axes[0, 1].set_xlabel("Latency (ms)")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Accuracy vs Speed")

# grid() adds guide lines to the chart.
# alpha controls how transparent the grid is.
axes[0, 1].grid(True, alpha=0.3)


# ============================================================
# BOTTOM-LEFT: TRAINING CURVES
# ============================================================

# [1, 0] means the second row and first column.
#
# plot() creates a line chart.
#
# epochs = X-axis
# train_loss = Y-axis
axes[1, 0].plot(
    epochs,
    train_loss,
    label="Train",
    color="#f97316",
    linewidth=2
)


# Add the validation loss as a second line.
#
# linestyle="--" makes this line dashed.
axes[1, 0].plot(
    epochs,
    val_loss,
    label="Val",
    color="#6366f1",
    linewidth=2,
    linestyle="--"
)


axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].set_title("Training Curve")

# legend() displays the labels:
#   Train
#   Val
axes[1, 0].legend()

axes[1, 0].grid(True, alpha=0.3)


# ============================================================
# BOTTOM-RIGHT: CONFIDENCE DISTRIBUTION
# ============================================================

# [1, 1] means the second row and second column.
#
# hist() creates a histogram.
#
# A histogram shows how often values
# occur within different ranges.
axes[1, 1].hist(
    confidences,
    bins=25,
    color="#22c55e",
    edgecolor="white",
    alpha=0.85
)


# mean() calculates the average confidence score.
#
# axvline() adds a vertical line at that value.
axes[1, 1].axvline(
    confidences.mean(),
    color="#dc2626",
    linestyle="--",
    label=f"Mean: {confidences.mean():.2f}"
)


axes[1, 1].set_xlabel("Confidence")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Confidence Distribution")

# Display the label for the mean line.
axes[1, 1].legend()


# ============================================================
# DISPLAY THE DASHBOARD
# ============================================================

# tight_layout() adjusts the spacing between
# the charts so that things do not overlap.
plt.tight_layout()

# show() displays the completed dashboard.
plt.show()

# Normal Python print statement.
print("Dashboard complete!")
