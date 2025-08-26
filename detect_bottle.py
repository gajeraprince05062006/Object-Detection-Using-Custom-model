import torch
import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    # Run object detection
    results = model(frame)
    detections = results.pandas().xyxy[0]

    if detections.empty:
        print("[DEBUG] No objects detected.")
    else:
        print("[DEBUG] Detected:")
        print(detections[['name', 'confidence']])

    # Draw all detections for visual debug
    for i, row in detections.iterrows():
        label = row['name']
        conf = row['confidence']
        if conf > 0.2:  # Lower confidence threshold
            x1, y1 = int(row['xmin']), int(row['ymin'])
            x2, y2 = int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    cv2.imshow("Bottle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
