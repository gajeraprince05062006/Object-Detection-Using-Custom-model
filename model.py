import cv2
from ultralytics import YOLO
import pygame
import os
import mysql.connector
from mysql.connector import Error

# ---------- Database Configuration ----------
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "smart"  # 🔁 Update with your actual DB name

# ---------- Sound Setup ----------
pygame.mixer.init()
sound_path = os.path.join(os.path.dirname(__file__), 'static', 'beep.mp3')

def play_sound():
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        print("🔊 Beep sound played")
    except Exception as e:
        print(f"[ERROR] Could not play sound: {e}")

# ---------- Global Shared List ----------
detected_items = []  # 🔁 Shared with Flask app

# ---------- Get Product Info from DB ----------
def get_product_info(name):
    try:
        connection = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        cursor = connection.cursor(dictionary=True, buffered=True)

        clean_name = name.strip().lower()
        query = "SELECT name, price FROM products WHERE LOWER(name) = %s"
        cursor.execute(query, (clean_name,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        return result
    except Error as e:
        print(f"[ERROR] MySQL Error in get_product_info: {e}")
        return None

# ---------- Store Detected Item in DB ----------
def store_detected_item(name, price):
    try:
        connection = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        cursor = connection.cursor()
        query = "INSERT INTO detected_items (name, price) VALUES (%s, %s)"
        cursor.execute(query, (name, float(price)))
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Stored in DB: {name} - ₹{price}")
    except Error as e:
        print(f"[ERROR] MySQL Error in store_detected_item: {e}")

# ---------- Test Database Connection ----------
def test_database():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        if connection.is_connected():
            print("✅ MySQL connection successful.")
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES LIKE 'products'")
            if cursor.fetchone():
                print("✅ Table found: products")
            else:
                print("❌ Missing table: products")

            cursor.execute("SHOW TABLES LIKE 'detected_items'")
            if cursor.fetchone():
                print("✅ Table found: detected_items")
            else:
                print("❌ Missing table: detected_items")

            cursor.close()
            connection.close()
            return True
    except Error as e:
        print(f"[ERROR] MySQL Error: {e}")
        return False

# ---------- Main Detection Function ----------
def run_model():
    global detected_items
    print("🚀 Starting Smart Trolley System")
    print("=" * 60)

    # Clear previous cart items
    detected_items.clear()

    if not test_database():
        print("❌ Cannot start: Database test failed.")
        return

    try:
        model = YOLO("final.pt")  # 🔁 Replace with your actual model path
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("📸 Camera ready. Press 'q' to stop.")
    print("=" * 60)

    detected_labels = set()

    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame")
            break

        results = model(frame, conf=0.5, verbose=False)
        detected_this_frame = 0

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names.get(cls_id, f"id_{cls_id}")
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detected_this_frame += 1

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    clean_name = class_name.strip()

                    if clean_name not in detected_labels:
                        print(f"🎯 New Detection: {clean_name}")
                        detected_labels.add(clean_name)

                        product = get_product_info(clean_name)
                        if product:
                            store_detected_item(product['name'], product['price'])
                            play_sound()
                            detected_items.append(product)  # ✅ Add to cart list
                        else:
                            print(f"❌ Product not found in DB: {clean_name}")

        if detected_this_frame == 0:
            print("🔍 No object detected in this frame.")

        cv2.imshow("Smart Trolley", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting system.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- Manual Run ----------
if __name__ == "__main__":
    run_model()
