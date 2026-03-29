import cv2
import numpy as np
import os

# ===== CONFIG =====
INPUT_IMAGE = "wtest2.jpg"
IMG_DIR = "data/input"
REF_DIR = "data/target"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

# ===== โหลดภาพ =====
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise ValueError("โหลดภาพไม่ได้")

# ===== helper functions =====

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = v.astype(np.int16)  # 👈 แก้ตรงนี้
    v = np.clip(v + value, 0, 255)

    final_hsv = cv2.merge((h, s, v.astype(np.uint8)))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def adjust_orange(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # สีส้มอยู่ประมาณ H = 10-25
    mask = (h >= 10) & (h <= 25)

    s = s.astype(np.int16)
    s[mask] = np.clip(s[mask] + value, 0, 255)

    final_hsv = cv2.merge((h, s.astype(np.uint8), v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# ===== generate =====

counter = 1

def save_pair(processed_img, idx):
    global counter

    img_name = f"img_{idx:04d}.jpg"
    ref_name = f"img_{idx:04d}.jpg"

    cv2.imwrite(os.path.join(IMG_DIR, img_name), processed_img)
    cv2.imwrite(os.path.join(REF_DIR, ref_name), img)

# ===== 1. เพิ่มแสง +5 =====
for i in range(1, 101):
    new_img = adjust_brightness(img, i * 5)
    save_pair(new_img, counter)
    counter += 1

# ===== 2. เพิ่มสีส้ม +5 =====
for i in range(1, 101):
    new_img = adjust_orange(img, i * 5)
    save_pair(new_img, counter)
    counter += 1

# ===== 3. ลดแสง -1 =====
for i in range(1, 101):
    new_img = adjust_brightness(img, -i * 1)
    save_pair(new_img, counter)
    counter += 1

# ===== 4. ลดสีส้ม -1 =====
for i in range(1, 101):
    new_img = adjust_orange(img, -i * 1)
    save_pair(new_img, counter)
    counter += 1

print("เสร็จแล้ว! ได้ 400 ภาพ")