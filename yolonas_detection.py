from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLOv8 model (YOLO-NAS variant)
def load_model():
    model = YOLO("yolov8n.pt")  # Use the yolov8n model or other YOLOv8 variants
    return model

# Detect objects using the YOLOv8 model
def detect_objects(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib

    # Perform inference with the YOLO model
    model = load_model()
    results = model(image_path)  # The results will contain the predictions (boxes, labels, and scores)

    # Process results and display bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores (each detection)
        labels = result.names  # Class labels

        for box, confidence, label in zip(boxes, confidences, labels):
            x1, y1, x2, y2 = box
            if confidence > 0.5:  # You can adjust the confidence threshold
                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_rgb, f'{label} {confidence:.2f}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image with bounding boxes
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Test the script with an image
image_path = "test.jpeg"  # Replace with your image path
detect_objects(image_path)
