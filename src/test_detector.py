
import cv2
from src.detector import VehicleDetector

detector = VehicleDetector()

image = cv2.imread("data/test.jpg")

if image is None:
    print("Error: Image not found!")
    exit()

detections = detector.detect(image)

print("\nVehicle Detection Results:")

for vehicle in detections:
    print(
        f"{vehicle['class_name']} | "
        f"Confidence: {vehicle['confidence']:.2f} | "
        f"Box: {vehicle['bbox']}"
    )

print(f"\nTotal Vehicles: {len(detections)}")
