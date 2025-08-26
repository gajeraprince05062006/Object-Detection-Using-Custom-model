import cv2
import easyocr
import re
import os

reader = easyocr.Reader(['en'])

def extract_price(text):
    pattern = r'(₹|Rs\.?)\s?\d+'
    match = re.findall(pattern, text)
    return match

def detect_price(image_path):
    image = cv2.imread(image_path)

    # 🔴 Crop region where price is likely printed
    # Adjust these coordinates based on your packet design
    h, w = image.shape[:2]
    roi = image[int(h*0.0):int(h*0.3), int(w*0.6):int(w*0.98)]  # Top-right corner

    result = reader.readtext(roi)
    prices_found = []

    for detection in result:
        text = detection[1]
        prices = extract_price(text)
        if prices:
            prices_found.extend(prices)

    if prices_found:
        print(f"[{image_path}] Prices detected: {prices_found}")
    else:
        print(f"[{image_path}] No price detected.")

image_folder = r'D:\Project-II\custom_dataset\test\images'

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        detect_price(os.path.join(image_folder, filename))
