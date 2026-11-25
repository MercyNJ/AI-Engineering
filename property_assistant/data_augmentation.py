
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Generate sample property data

np.random.seed(42)

number_of_properties = 500

locations = ["Kilimani", "Ruiru", "Karen", "Westlands", "Thika"]
property_types = ["Apartment", "House", "Bedsitter", "Maisonette"]

properties = pd.DataFrame({
    "user_location": np.random.choice(locations, number_of_properties),
    "property_location": np.random.choice(locations, number_of_properties),
    "property_type": np.random.choice(property_types, number_of_properties),
    "bedrooms": np.random.randint(1, 6, number_of_properties),
    "bathrooms": np.random.randint(1, 5, number_of_properties),
    "price_kes": np.random.randint(15000, 150001, number_of_properties),
    "furnished": np.random.choice(["Yes", "No"], number_of_properties)
})


# 2. Create illustrative suitability labels

def calculate_suitability(row):

    score = 0

    if row["user_location"] == row["property_location"]:
        score += 1

    if row["price_kes"] <= 100000:
        score += 1

    if row["bedrooms"] >= 2 and row["bathrooms"] >= 1:
        score += 1

    if row["furnished"] == "Yes":
        score += 1

    return 1 if score >= 2 else 0


properties["suitable"] = properties.apply(
    calculate_suitability,
    axis=1
)

print("Original property dataset:")
print(properties.head())

print("\nOriginal dataset shape:", properties.shape)


# 3. Split the data before augmentation

shuffled_properties = properties.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

split_index = int(len(shuffled_properties) * 0.8)

train_data = shuffled_properties.iloc[:split_index].copy()
validation_data = shuffled_properties.iloc[split_index:].copy()

print("\nOriginal training records:", len(train_data))
print("Validation records:", len(validation_data))


# 4. Create augmented property records

def augment_property(row):

    new_row = row.copy()

    # Introduce a small price variation

    price_change = np.random.uniform(0.95, 1.05)

    new_row["price_kes"] = int(
        new_row["price_kes"] * price_change
    )

    # Occasionally change the bedroom count

    if np.random.random() < 0.3:

        new_row["bedrooms"] += np.random.choice([-1, 1])

        new_row["bedrooms"] = np.clip(
            new_row["bedrooms"],
            1,
            5
        )

    # Occasionally change the bathroom count

    if np.random.random() < 0.3:

        new_row["bathrooms"] += np.random.choice([-1, 1])

        new_row["bathrooms"] = np.clip(
            new_row["bathrooms"],
            1,
            5
        )

    # Recalculate the label after modifying the property

    new_row["suitable"] = calculate_suitability(new_row)

    return new_row


# 5. Generate additional training records

augmented_records = []

for _, row in train_data.iterrows():

    augmented_records.append(
        augment_property(row)
    )

augmented_data = pd.DataFrame(augmented_records)

expanded_train_data = pd.concat(
    [train_data, augmented_data],
    ignore_index=True
)


# 6. Compare original and augmented data

print("\nOriginal training records:", len(train_data))
print("Generated records:", len(augmented_data))
print("Expanded training records:", len(expanded_train_data))

print("\nOriginal label distribution:")
print(train_data["suitable"].value_counts())

print("\nExpanded label distribution:")
print(expanded_train_data["suitable"].value_counts())


# 7. Visualise the original and augmented records

plt.figure(figsize=(10, 6))

plt.scatter(
    train_data["price_kes"],
    train_data["bedrooms"],
    label="Original properties",
    alpha=0.6
)

plt.scatter(
    augmented_data["price_kes"],
    augmented_data["bedrooms"],
    label="Augmented properties",
    alpha=0.6
)

plt.xlabel("Property Price (KES)")
plt.ylabel("Number of Bedrooms")
plt.title("Original vs Augmented Property Data")
plt.legend()
plt.grid(True)
plt.show()


# 8. Verify that validation data was not augmented

print("\nValidation dataset shape:", validation_data.shape)

print(
    "Validation data remains unchanged:",
    len(validation_data) == len(properties) - len(train_data)
)


# 9. Inspect some augmented records

print("\nSample augmented property records:")
print(augmented_data.head())
