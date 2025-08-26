import cv2
import os
import time

# ==== CONFIG ====
dataset_type = "train"
output_dir = "custom_dataset"
class_names = ['Parle-G', 'Crunchex-salted']
# ================

image_dir = os.path.join(output_dir, "images", dataset_type)
label_dir = os.path.join(output_dir, "labels", dataset_type)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Available Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# Select class
while True:
    try:
        class_id = int(input("Enter class ID: "))
        if 0 <= class_id < len(class_names):
            break
    except:
        pass
    print("Invalid input.")

print("\nPress SPACE to capture image and crop manually.")
print("Press ESC to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == 32:  # SPACE
        timestamp = int(time.time() * 1000)
        raw_image_name = f"{timestamp}.jpg"
        temp_image_path = os.path.join(image_dir, raw_image_name)

        # Save full image temporarily for drawing ROI
        cv2.imwrite(temp_image_path, frame)
        print("Image captured. Select object only.")

        roi = cv2.selectROI("Draw tight box on Parle-G only", frame)
        cv2.destroyWindow("Draw tight box on Parle-G only")
        x, y, w, h = roi

        if w == 0 or h == 0:
            print("Skipped — no box drawn.")
            os.remove(temp_image_path)
            continue

        # Crop the object only
        cropped = frame[y:y+h, x:x+w]

        # Save cropped image
        cropped_image_name = f"cropped_{timestamp}.jpg"
        image_path = os.path.join(image_dir, cropped_image_name)
        label_path = os.path.join(label_dir, cropped_image_name.replace(".jpg", ".txt"))
        cv2.imwrite(image_path, cropped)

        # YOLO box is now whole image: 0 0.5 0.5 1.0 1.0
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

        print(f"✅ Saved: {image_path}")
        print(f"✅ Label: {label_path} (full box, object only)")

cap.release()
cv2.destroyAllWindows()
