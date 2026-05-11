# This is your Day 1 test — just checking all packages load correctly
print("Testing imports...")

import torch
print(f"PyTorch version: {torch.__version__}")

from transformers import BlipProcessor, BlipForConditionalGeneration
print("BLIP imported successfully")

import gradio as gr
print("Gradio imported successfully")

from PIL import Image
print("Pillow imported successfully")

import matplotlib
print("Matplotlib imported successfully")

print("\nAll good! You are ready for Day 2.")