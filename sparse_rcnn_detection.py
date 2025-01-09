import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import models
import numpy as np

# Load the pre-trained Sparse R-CNN model from the official repository
# Download the Sparse R-CNN pre-trained weights
# You can change the model URL to point to the official pre-trained model when available

class SparseRCNNModel(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(SparseRCNNModel, self).__init__()
        # You can implement or modify Sparse R-CNN layers here as needed.
        # For simplicity, we are using a simple Faster R-CNN implementation for illustration purposes.
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    def forward(self, images):
        return self.model(images)

# Initialize the model
def load_model():
    model = SparseRCNNModel(pretrained=True)
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path):
    # Open image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Post-process the output and draw bounding boxes
def postprocess_and_draw(image_path, outputs):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Get the detections
    boxes = outputs[0]['boxes'].cpu().detach().numpy()
    labels = outputs[0]['labels'].cpu().detach().numpy()
    scores = outputs[0]['scores'].cpu().detach().numpy()

    # Set a threshold for object detection confidence
    threshold = 0.5  # You can adjust this as needed
    for box, score in zip(boxes, scores):
        if score > threshold:
            # Draw a bounding box around the detected object
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Main function to run object detection
def detect_objects(image_path):
    # Load the model
    model = load_model()
    
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Post-process and display results
    postprocess_and_draw(image_path, outputs)

# Test the script with an image
image_path = "test.jpeg"  # Replace with your image path
detect_objects(image_path)
