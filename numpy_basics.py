import numpy as np

# Model names and simulated accuracy scores
model_names = ["GPT-4o", "Claude-3-Sonnet", "Gemini-Pro", "Llama-3-70B",
               "Mistral-7B", "Phi-3-Medium", "Command-R+", "Claude-3-Haiku"]

accuracy = np.array([0.942, 0.938, 0.921, 0.897, 0.871, 0.883, 0.912, 0.905])

print("=== Model Accuracy Report ===")
print(f"Models: {len(accuracy)}")
print(f"Scores: {accuracy}\n")

# Calculate summary statistics
print("--- Summary Statistics ---")
print(f"Mean:   {np.mean(accuracy):.4f}")
print(f"Std:    {np.std(accuracy):.4f}")
print(f"Max:    {np.max(accuracy):.4f}")
print(f"Min:    {np.min(accuracy):.4f}")

# Find the index of the highest accuracy
best_idx = np.argmax(accuracy)
print(f"\nBest model: {model_names[best_idx]} ({accuracy[best_idx]:.1%})")

# Filter models with accuracy above 90%
mask = accuracy > 0.90
top_models = [model_names[i] for i in range(len(model_names)) if mask[i]]
print(f"\nModels above 90%: {top_models}")

# Scale accuracy scores to a 0–1 range
normalised = (accuracy - accuracy.min()) / (accuracy.max() - accuracy.min())

print("\nNormalised scores:")
for name, norm in zip(model_names, normalised):
    print(f"  {name}: {norm:.3f}")