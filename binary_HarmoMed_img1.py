# =========================
# IMPORT
# =========================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import time
from datetime import datetime
from skimage.color import deltaE_ciede2000

from binary_path_img import (
    get_next_filename,
    add_new_row_to_csv
)

DEBUG = True

def log(*msg):
    if DEBUG:
        print(*msg)

# =========================
# OBJECT DETECTION
# =========================
def detection(image_path):

    log("\nSTEP 1 : OBJECT DETECTION")

    model_name = "indicator-j3riv"
    version = 4
    api_key = "6GfuPFK2Ue4Fvh9EZiQJ"

    api_url = f"https://detect.roboflow.com/{model_name}/{version}?api_key={api_key}&format=json"

    with open(image_path, "rb") as image_file:
        response = requests.post(api_url, files={"file": image_file})

    image = cv2.imread(image_path)

    if response.status_code != 200:
        return image

    data = response.json()

    if "predictions" not in data or len(data["predictions"]) == 0:
        return image

    best = max(data["predictions"], key=lambda x: x["confidence"])

    x, y, w, h = int(best["x"]), int(best["y"]), int(best["width"]), int(best["height"])

    x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
    x2, y2 = min(image.shape[1], x + w // 2), min(image.shape[0], y + h // 2)

    pad = int(0.02 * max(w, h))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)

    return image[y1:y2, x1:x2]

# =========================
# COLOR METRIC
# =========================
def deltaE_cie2000(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return np.mean(deltaE_ciede2000(ref_lab, img_lab))

# =========================
# COLOR PROCESSING (BEST)
# =========================

def gray_world_white_balance(img):
    img = img.astype(np.float32)
    b, g, r = cv2.split(img)

    avg = (np.mean(b) + np.mean(g) + np.mean(r)) / 3

    b *= avg / (np.mean(b) + 1e-6)
    g *= avg / (np.mean(g) + 1e-6)
    r *= avg / (np.mean(r) + 1e-6)

    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def protect_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 1] < 40  # low saturation = white/gray

    img = img.astype(np.float32)
    img[mask] = img[mask] * 0.95 + 255 * 0.05

    return np.clip(img, 0, 255).astype(np.uint8)

# =========================
# LEVEL 2 : SEGMENT + POLY
# =========================

def segment_red_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 🔴 red (tight range)
    red_mask = cv2.inRange(hsv, (0,120,80), (10,255,255)) + \
               cv2.inRange(hsv, (165,120,80), (180,255,255))

    # ⚪ white (strict = medical-grade)
    white_mask = (hsv[:,:,1] < 30) & (hsv[:,:,2] > 180)

    return red_mask, (white_mask.astype(np.uint8) * 255)

def polynomial_fit(src, dst):

    src = src.reshape(-1,3).astype(np.float32)
    dst = dst.reshape(-1,3).astype(np.float32)

    def poly(x):
        R,G,B = x[:,0], x[:,1], x[:,2]
        return np.stack([
            R,G,B,
            R*R,G*G,B*B,
            R*G,R*B,G*B,
            np.ones_like(R)
        ], axis=1)

    X = poly(src)

    coeffs = []
    for i in range(3):
        # 🔥 ridge regularization (กัน unstable)
        XtX = X.T @ X + np.eye(X.shape[1]) * 1e-3
        XtY = X.T @ dst[:,i]
        w = np.linalg.solve(XtX, XtY)
        coeffs.append(w)

    return coeffs

def enhance_red_contrast(img, mask):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # เพิ่ม contrast เฉพาะ red
    L[mask > 0] = cv2.normalize(L[mask > 0], None, 0, 255, cv2.NORM_MINMAX)

    merged = cv2.merge([L, A, B])
    return cv2.cvtColor(np.clip(merged,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def enforce_neutral_white(img, mask):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # 🔥 FIX: force neutral white
    A[mask > 0] = 128
    B[mask > 0] = 128

    merged = cv2.merge([L, A, B])
    return cv2.cvtColor(
        np.clip(merged,0,255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )

def apply_polynomial(img, coeffs):
    h,w,_ = img.shape
    flat = img.reshape(-1,3).astype(np.float32)

    def poly(x):
        R,G,B = x[:,0], x[:,1], x[:,2]
        return np.stack([
            R,G,B,
            R*R,G*G,B*B,
            R*G,R*B,G*B,
            np.ones_like(R)
        ], axis=1)

    X = poly(flat)

    out = np.zeros_like(flat)

    for i in range(3):
        out[:,i] = X @ coeffs[i]

    return np.clip(out.reshape(h,w,3),0,255).astype(np.uint8)


def level2_color_match(reference, target):

    target = cv2.resize(target, (reference.shape[1], reference.shape[0]))

    # 🔥 STEP 0: white balance
    target = gray_world_white_balance(target)

    # STEP 1: segmentation
    red_mask, white_mask = segment_red_white(target)

    h, w = target.shape[:2]
    m_h = int(h * 0.08)
    m_w = int(w * 0.08)

    safe_target = target[m_h:h-m_h, m_w:w-m_w]
    safe_reference = reference[m_h:h-m_h, m_w:w-m_w]
    safe_red = red_mask[m_h:h-m_h, m_w:w-m_w]

    result = target.copy()

    # =========================
    # 🔴 RED → robust polynomial
    # =========================
    if np.sum(safe_red) > 200:

        src = safe_target[safe_red > 0]
        dst = safe_reference[safe_red > 0]

        # 🔥 remove outliers (สำคัญ)
        diff = np.linalg.norm(src - dst, axis=1)
        keep = diff < np.percentile(diff, 85)

        src = src[keep]
        dst = dst[keep]

        coeffs = polynomial_fit(src, dst)
        corrected = apply_polynomial(target, coeffs)

        red_only = (red_mask > 0) & (white_mask == 0)
        result[red_only] = corrected[red_only]

    # =========================
    # ⚪ WHITE → TRUE NEUTRAL (แก้ฟ้า)
    # =========================
    if np.sum(white_mask) > 200:

        result = enforce_neutral_white(result, white_mask)

        # 🔥 stabilize brightness
        ref_white = reference[white_mask > 0]
        if len(ref_white) > 50:
            mean_L = np.mean(cv2.cvtColor(ref_white.reshape(-1,1,3), cv2.COLOR_BGR2LAB)[:,:,0])

            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[:,:,0][white_mask > 0] = mean_L
            result = cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # =========================
    # 🔥 RED contrast recovery
    # =========================
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    red_area = red_mask > 0

    L[red_area] = cv2.normalize(L[red_area], None, 0, 255, cv2.NORM_MINMAX)
    A[red_area] *= 1.08

    result = cv2.cvtColor(
        np.clip(cv2.merge([L, A, B]), 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )

    # =========================
    # 🔥 EDGE PRESERVE
    # =========================
    result = cv2.bilateralFilter(result, 9, 75, 75)

    # =========================
    # 🔥 FINAL GUARD
    # =========================
    result = protect_white(result)

    return result
def enforce_neutral_white(img, mask):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # white ต้องไม่มีสี → A,B ใกล้ 128
    A[mask > 0] = 128
    B[mask > 0] = 128

    merged = cv2.merge([L, A, B])
    return cv2.cvtColor(
        np.clip(merged,0,255).astype(np.uint8),
        cv2.COLOR_LAB2BGR
    )

def better_color_match(reference, target):

    # 1. White balance
    target = gray_world_white_balance(target)

    # 2. LAB
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    tar_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    ref_l, ref_a, ref_b = cv2.split(ref_lab)
    tar_l, tar_a, tar_b = cv2.split(tar_lab)

    # 3. Match สี (a,b)
    scale_a = ref_a.std() / (tar_a.std() + 1e-6)
    scale_b = ref_b.std() / (tar_b.std() + 1e-6)

    tar_a = (tar_a - tar_a.mean()) * scale_a + ref_a.mean()
    tar_b = (tar_b - tar_b.mean()) * scale_b + ref_b.mean()

    # 4. L แบบ soft
    alpha = 0.3
    tar_l = tar_l * (1 - alpha) + ref_l * alpha

    merged = cv2.merge([tar_l, tar_a, tar_b])
    merged = np.clip(merged, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 5. smooth ลด noise
    result = cv2.GaussianBlur(result, (3,3), 0)

    # 6. protect white
    result = protect_white(result)

    return result

def remove_border(img, margin=0.05):
    h, w = img.shape[:2]
    m_h = int(h * margin)
    m_w = int(w * margin)
    return img[m_h:h-m_h, m_w:w-m_w]

def iterative_color_match(reference, target):

    log("\nSTEP 2 : SINGLE PASS COLOR MATCH")

    result = level2_color_match(reference, target)

    diff = deltaE_cie2000(reference, result)
    log("Final DeltaE:", diff)

    return result, [diff], []

# =========================
# APPLY COLOR FULL IMAGE
# =========================

def apply_color_to_full_image(reference, target, full_img):

    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype("float32")
    tar_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    full_lab = cv2.cvtColor(full_img, cv2.COLOR_BGR2LAB).astype("float32")

    for i in range(3):
        mean_ref = ref_lab[:,:,i].mean()
        std_ref = ref_lab[:,:,i].std()

        mean_tar = tar_lab[:,:,i].mean()
        std_tar = tar_lab[:,:,i].std()

        full_lab[:,:,i] = (full_lab[:,:,i] - mean_tar) * (std_ref/(std_tar+1e-6)) + mean_ref

    return cv2.cvtColor(np.clip(full_lab,0,255).astype("uint8"), cv2.COLOR_LAB2BGR)

# =========================
# MAIN PROCESS
# =========================

def processing_img(refpath, targetpath, filename_path):

    start = time.time()

    reference_img = cv2.imread(refpath)
    target_img = cv2.imread(targetpath)

    cropped = detection(targetpath)

    cropped = cv2.resize(cropped, (reference_img.shape[1], reference_img.shape[0]))

    diff_before = deltaE_cie2000(reference_img, cropped)
    log("Initial DeltaE:", diff_before)

    result_img, history, _ = iterative_color_match(reference_img, cropped)

    diff_after = deltaE_cie2000(reference_img, result_img)
    log("Final DeltaE:", diff_after)

    full_corrected = apply_color_to_full_image(reference_img, cropped, target_img)

    # VISUAL
    fig, axes = plt.subplots(2,3,figsize=(18,10))

    axes[0,0].imshow(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("Reference")

    axes[0,1].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title("Target")

    axes[0,2].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axes[0,2].set_title("Detected")

    axes[1,0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title("Corrected")

    axes[1,1].imshow(cv2.cvtColor(full_corrected, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title("Full")

    axes[1,2].plot(history)
    axes[1,2].set_title("DeltaE")

    for ax in axes.flatten():
        ax.axis("off")

    os.makedirs("static/results", exist_ok=True)

    obj_path = f"static/results/{filename_path}_object.jpg"
    full_path = f"static/results/{filename_path}_full.jpg"
    plot_path = f"static/results/{filename_path}_plot.jpg"

    cv2.imwrite(obj_path, result_img)
    cv2.imwrite(full_path, full_corrected)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    log("Processing time:", time.time() - start)

    return obj_path, full_path, plot_path

# =========================
# PIPELINE
# =========================

def run_image_processing_pipeline(reference_path, uploaded_file, csv_file="image_log.csv"):

    UPLOAD_FOLDER = 'static/uploads/'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    name = str(get_next_filename(csv_file))

    add_new_row_to_csv(
        csv_file,
        name,
        datetime.now().strftime("%H:%M:%S"),
        datetime.now().strftime("%d-%m-%Y")
    )

    target_path = os.path.join(UPLOAD_FOLDER, f"{name}.jpg")
    uploaded_file.save(target_path)

    obj, full, plot = processing_img(reference_path, target_path, name)

    return {
        "object_result": obj,
        "full_result": full,
        "plot": plot,
        "filename": name
    }