import os
import base64
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from openai import OpenAI

# Create a sample receipt image so the example is self-contained
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

receipt_path = output_dir / "sample_receipt.png"

image = Image.new("RGB", (700, 450), "white")
draw = ImageDraw.Draw(image)

draw.text((40, 30), "SAMPLE STORE", fill="black")
draw.text((40, 80), "Receipt No: 1045", fill="black")
draw.text((40, 120), "Date: 14 March 2026", fill="black")
draw.text((40, 180), "Groceries                 KES 1,500", fill="black")
draw.text((40, 220), "Household items           KES 650", fill="black")
draw.text((40, 260), "Transport                 KES 300", fill="black")
draw.text((40, 330), "TOTAL                     KES 2,450", fill="black")

image.save(receipt_path)

# Display the receipt
plt.figure(figsize=(9, 5))
plt.imshow(image)
plt.axis("off")
plt.title("Sample Receipt")
plt.show()

# Convert the image to a base64 data URL
with open(receipt_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

image_url = f"data:image/png;base64,{encoded_image}"

# Connect to the OpenAI API
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Set your OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

# Send text and image together in one request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an assistant that reads documents and extracts information. "
                "Only report information that is visible in the image. "
                "If something is unclear, say so."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Read this receipt. Extract the store name, receipt number, "
                        "issue date, and total amount."
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

print("\nExtracted Receipt Information:")
print(response.choices[0].message.content)

# Visual question answering
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "How many individual purchases are listed on this receipt?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ],
)

print("\nVisual Question Answering:")
print(response.choices[0].message.content)

# Demonstrate that image interpretation should be verified
print("\nImportant:")
print(
    "AI-extracted information should be checked against the original document, "
    "especially when handling financial or legal information."
)