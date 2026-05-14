from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

print("Loading BLIP model... (first time takes 1-2 minutes, downloads the model)")

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("Model loaded successfully!")

# Open your test image
image = Image.open("test.jpg").convert("RGB")
print(f"Image loaded — size: {image.size}")

# Prepare image for the model
inputs = processor(image, return_tensors="pt")
print("Image preprocessed into tensors")

# Generate caption
print("Generating caption...")
output = model.generate(**inputs, max_new_tokens=30)

# Decode the output tokens into readable text
caption = processor.decode(output[0], skip_special_tokens=True)

print("\n" + "="*40)
print(f"CAPTION: {caption}")
print("="*40)