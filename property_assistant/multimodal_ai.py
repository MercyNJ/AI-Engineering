import os
import base64
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from openai import OpenAI

# Create a sample property image
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

property_path = output_dir / "sample_property.png"

image = Image.new("RGB", (900, 600), "#e8e8e8")
draw = ImageDraw.Draw(image)

# Room walls and floor
draw.rectangle((80, 70, 820, 520), fill="#f5f0e6", outline="#333333", width=5)
draw.rectangle((80, 400, 820, 520), fill="#b99b78")

# Window
draw.rectangle((180, 110, 380, 260), fill="#b9dce8", outline="#444444", width=5)
draw.line((280, 110, 280, 260), fill="#444444", width=4)
draw.line((180, 185, 380, 185), fill="#444444", width=4)

# Sofa
draw.rectangle((450, 310, 720, 410), fill="#668b78", outline="#333333", width=4)
draw.rectangle((470, 280, 700, 340), fill="#7ca38e", outline="#333333", width=4)

# Door
draw.rectangle((90, 270, 170, 400), fill="#9b7653", outline="#333333", width=4)

# Room details
draw.rectangle((490, 130, 690, 150), fill="#8b6b4f")
draw.text((490, 160), "Wall shelf", fill="#333333")
draw.text((100, 540), "Sample property living room", fill="#333333")

image.save(property_path)

# Display the sample property image
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.title("Sample Property Image")
plt.show()

# Convert the image to a base64 data URL
with open(property_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

image_url = f"data:image/png;base64,{encoded_image}"

# Connect to the OpenAI API
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Set your OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

# Ask the multimodal model to describe the property
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a property analysis assistant. "
                "Describe only features that are visibly supported by the image. "
                "Do not invent property details such as location, price, "
                "number of bedrooms, or measurements."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this property's living room. "
                        "Identify visible features, furniture, natural lighting, "
                        "and any details a prospective tenant may want to know."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        },
    ],
)

print("\nProperty Image Analysis:")
print(response.choices[0].message.content)

# Ask a specific visual question
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Is there a visible window in this room? "
                        "Describe its position and appearance."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ],
)

print("\nProperty Visual Question Answering:")
print(response.choices[0].message.content)

# Demonstrate combining image analysis with property requirements
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You assist users in evaluating rental properties. "
                "Distinguish between details visible in an image and details "
                "that require additional information. Never assume that an "
                "unseen feature is present."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I am looking for a furnished rental property with good "
                        "natural lighting. Based only on this image, identify "
                        "which of these requirements can be visually assessed "
                        "and which need further verification."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        },
    ],
)

print("\nProperty Requirement Assessment:")
print(response.choices[0].message.content)

print("\nImportant:")
print(
    "Image analysis can help assess visible property features, but details "
    "such as rent, location, room dimensions, and actual property condition "
    "must be verified separately."
)