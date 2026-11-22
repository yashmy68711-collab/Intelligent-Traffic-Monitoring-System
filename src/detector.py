
from ultralytics import YOLO


class VehicleDetector:

    def __init__(self, model_path="yolov8n.pt", confidence=0.40):

        print("Loading YOLO model...")

        self.model = YOLO(model_path)
        self.confidence = confidence

        # COCO class IDs
        self.vehicle_classes = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

        print("YOLO model loaded successfully!")

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=list(self.vehicle_classes.keys()),
            verbose=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": self.vehicle_classes[class_id],
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections
