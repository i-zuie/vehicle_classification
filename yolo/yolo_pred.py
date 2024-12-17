import os
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
from ultralytics import YOLO

# Initialize model
model = YOLO("../runs/classify/train/weights/best.pt")

# Define the path to the test folder
test_dir = Path('../data/test')

# Collect the images and true labels
images = []
true_labels = []

for class_dir in test_dir.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name
        for image_path in class_dir.iterdir():
            if image_path.suffix in ['.jpg', '.png', '.jpeg']:  # Filter by image extension
                images.append(image_path)
                true_labels.append(class_name)

# Make predictions
pred_labels = []
for image_path in images:
    results = model.predict(source=image_path, save=False, show=False)
    predicted_class = results[0].names[results[0].probs.top1]  # Get predicted class
    pred_labels.append(predicted_class)

# Calculate overall accuracy
accuracy = accuracy_score(true_labels, pred_labels)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# Generate a classification report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, zero_division=0))
