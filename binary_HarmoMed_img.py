import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import time
from datetime import datetime
from skimage.exposure import match_histograms
from skimage.color import deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pandas as pd

# กำหนด path และสร้างโฟลเดอร์เก็บข้อมูล
csv_file = "result_HarmoMed/image_log.csv"
UPLOAD_FOLDER = 'result_HarmoMed/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดข้อมูลจาก CSV
def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    required_columns = {'filename', 'time', 'day'}
    if not required_columns.issubset(df.columns):
        raise ValueError("ไฟล์ CSV ต้องมีคอลัมน์ชื่อ 'filename', 'time' และ 'day'")
    return df

# เพิ่มข้อมูลใหม่ลง CSV
def add_new_row_to_csv(csv_file, new_filename, time, day):
    df = load_csv_data(csv_file)
    new_row = pd.DataFrame({'filename': [new_filename], 'time': [time], 'day': [day]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False)

# สร้างชื่อไฟล์ถัดไป
def get_next_filename(csv_file):
    try:
        df = load_csv_data(csv_file)
        max_filename = df['filename'].astype(int).max()
        return max_filename + 1 if pd.notnull(max_filename) else 1
    except FileNotFoundError:
        return 1

# บันทึกไฟล์ภาพที่อัปโหลด
def save_uploaded_images(input_files, csv_file):
    next_filename = get_next_filename(csv_file)
    added_files = []

    for uploaded_file_path in input_files:
        new_filename = str(next_filename)
        new_time = datetime.now().strftime("%H:%M:%S")
        new_day = datetime.now().strftime("%d-%m-%Y")

        add_new_row_to_csv(csv_file, new_filename, new_time, new_day)

        new_path = os.path.join(UPLOAD_FOLDER, f"{new_filename}.jpg")
        uploaded_file_path.save(new_path)

        added_files.append(new_filename)
        next_filename += 1

    print(f"เพิ่ม {len(input_files)} รูป: {', '.join(added_files)}")
    return len(input_files), added_files

DEBUG = True

# แสดง progress bar
def progress_bar(progress, total, length=30):
    percent = progress / total
    filled = int(length * percent)
    bar = "█" * filled + "-" * (length - filled)
    print(f"\r[{bar}] {int(percent*100)}%", end="")
    if progress == total:
        print()

def log(*msg):
    if DEBUG:
        print(*msg)

# ตรวจจับวัตถุจากภาพ
def detection(image_path):
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

    best_obj = max(data["predictions"], key=lambda x: x["confidence"])

    x, y, w, h = int(best_obj["x"]), int(best_obj["y"]), int(best_obj["width"]), int(best_obj["height"])

    x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
    x2, y2 = min(image.shape[1], x + w // 2), min(image.shape[0], y + h // 2)
    pad = int(0.01 * max(w, h))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1], x2 + pad)
    y2 = min(image.shape[0], y2 + pad)

    return image[y1:y2, x1:x2]

# ปรับสีจาก source ไป target
def color_transfer(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    lMeanSrc, aMeanSrc, bMeanSrc = cv2.mean(source)[:3]
    lMeanTar, aMeanTar, bMeanTar = cv2.mean(target)[:3]

    lStdSrc, aStdSrc, bStdSrc = source.std(axis=(0,1))
    lStdTar, aStdTar, bStdTar = target.std(axis=(0,1))

    l, a, b = cv2.split(target)

    l = ((l - lMeanTar) * (lStdSrc / (lStdTar + 1e-6))) + lMeanSrc
    a = ((a - aMeanTar) * (aStdSrc / (aStdTar + 1e-6))) + aMeanSrc
    b = ((b - bMeanTar) * (bStdSrc / (bStdTar + 1e-6))) + bMeanSrc

    transfer = cv2.merge([l,a,b])
    transfer = np.clip(transfer,0,255).astype("uint8")

    return cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)

# เพิ่มความเข้มสีส้ม
def orange_region_boost(img, strength=1.2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    hsv[:,:,1] = np.where(mask>0, hsv[:,:,1]*strength, hsv[:,:,1])
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# วิเคราะห์ความต่างสีและส้ม
def analyze_rgb_and_orange_shift(ref, img):
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ref_mean = ref_rgb.mean(axis=(0,1))
    img_mean = img_rgb.mean(axis=(0,1))

    diff_rgb = img_mean - ref_mean

    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])

    ref_mask = cv2.inRange(ref_hsv, lower_orange, upper_orange)
    img_mask = cv2.inRange(img_hsv, lower_orange, upper_orange)

    ref_orange_ratio = np.sum(ref_mask > 0) / ref_mask.size
    img_orange_ratio = np.sum(img_mask > 0) / img_mask.size

    orange_shift = img_orange_ratio - ref_orange_ratio

    return {
        "rgb_diff": diff_rgb,
        "ref_orange_ratio": ref_orange_ratio,
        "img_orange_ratio": img_orange_ratio,
        "orange_shift": orange_shift
    }

# pipeline ปรับสี
def advanced_color_match(reference, target):
    step1 = color_transfer(reference, target)
    step2 = match_histograms(step1, reference, channel_axis=-1).astype(np.uint8)
    step3 = orange_region_boost(step2, strength=1.15)
    return step3

# คำนวณ DeltaE
def deltaE_cie2000(ref, img):
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return np.mean(deltaE_ciede2000(ref_lab, img_lab))

# คำนวณ MSE
def compute_mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

# คำนวณ PSNR
def compute_psnr(img1, img2):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# คำนวณ SSIM
def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)

# ปรับสีแบบ iterative
def iterative_color_match(reference, target, max_iter=10, tolerance=0.3):
    history = []
    orange_history = []

    current = target.copy()
    prev_diff = 999999

    log("\n[START] Iterative Color Matching")

    for i in range(max_iter):
        log(f"Iteration {i+1}/{max_iter}")

        matched = advanced_color_match(reference, current)
        diff = deltaE_cie2000(reference, matched)

        analysis = analyze_rgb_and_orange_shift(reference, matched)
        orange_shift = abs(analysis["orange_shift"])

        history.append(diff)
        orange_history.append(orange_shift)

        log(f"DeltaE: {diff:.4f} | Orange shift: {orange_shift:.4f}")

        progress_bar(i+1, max_iter)

        improvement = prev_diff - diff

        if abs(improvement) < tolerance and orange_shift < 0.01:
            log("[DONE] Converged")
            return matched, history, orange_history

        prev_diff = diff
        current = matched

    log("[DONE] Max iteration reached")
    return current, history, orange_history

# นำค่าสีไปใช้กับภาพเต็ม
def apply_color_to_full_image(reference,target,full_img):
    ref_lab=cv2.cvtColor(reference,cv2.COLOR_BGR2LAB).astype("float32")
    tar_lab=cv2.cvtColor(target,cv2.COLOR_BGR2LAB).astype("float32")

    lMeanSrc,aMeanSrc,bMeanSrc=cv2.mean(ref_lab)[:3]
    lMeanTar,aMeanTar,bMeanTar=cv2.mean(tar_lab)[:3]

    lStdSrc,aStdSrc,bStdSrc=ref_lab.std(axis=(0,1))
    lStdTar,aStdTar,bStdTar=tar_lab.std(axis=(0,1))

    full_lab=cv2.cvtColor(full_img,cv2.COLOR_BGR2LAB).astype("float32")

    l,a,b=cv2.split(full_lab)

    l=((l-lMeanTar)*(lStdSrc/(lStdTar+1e-6)))+lMeanSrc
    a=((a-aMeanTar)*(aStdSrc/(aStdTar+1e-6)))+aMeanSrc
    b=((b-bMeanTar)*(bStdSrc/(bStdTar+1e-6)))+bMeanSrc

    merged=cv2.merge([l,a,b])
    merged=np.clip(merged,0,255).astype("uint8")

    return cv2.cvtColor(merged,cv2.COLOR_LAB2BGR)

# ประมวลผลภาพทั้งหมด
def processing_img(refpath, targetpath, filename_path):
    start = time.time()

    log("\n===== START PROCESS =====")

    log("Loading images...")
    reference_img = cv2.imread(refpath)
    target_img = cv2.imread(targetpath)

    log("Detecting object...")
    cropped = detection(targetpath)

    log("Resizing object...")
    cropped = cv2.resize(
        cropped,
        (reference_img.shape[1], reference_img.shape[0])
    )

    log("Color matching...")
    result_img, history, orange_history = iterative_color_match(reference_img, cropped)

    log("Applying color to full image...")
    full_corrected = apply_color_to_full_image(
        reference_img,
        cropped,
        target_img
    )

    log("Preparing visualization...")

    reference_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    full_rgb = cv2.cvtColor(full_corrected, cv2.COLOR_BGR2RGB)

    log("Calculating metrics...")

    delta_before = deltaE_cie2000(reference_img, cropped)
    delta_after = deltaE_cie2000(reference_img, result_img)

    ssim_val = compute_ssim(reference_img, result_img)
    psnr_val = compute_psnr(reference_img, result_img)
    mse_val = compute_mse(reference_img, result_img)

    log(f"DeltaE Before: {delta_before:.4f}")
    log(f"DeltaE After: {delta_after:.4f}")
    log(f"SSIM: {ssim_val:.4f}")
    log(f"PSNR: {psnr_val:.2f}")
    log(f"MSE: {mse_val:.2f}")

    metrics_text = (
        f"DeltaE Before: {delta_before:.2f}\n"
        f"DeltaE After: {delta_after:.2f}\n"
        f"SSIM: {ssim_val:.3f}\n"
        f"PSNR: {psnr_val:.2f}\n"
        f"MSE: {mse_val:.2f}"
    )

    fig,axes=plt.subplots(2,3,figsize=(18,10))

    axes[0,0].imshow(reference_rgb)
    axes[0,0].set_title("Reference")

    axes[0,1].imshow(target_rgb)
    axes[0,1].set_title("Target")

    axes[0,2].imshow(cropped_rgb)
    axes[0,2].set_title("Detected Object")

    axes[1,0].imshow(result_rgb)
    axes[1,0].set_title("Object Corrected")

    axes[1,1].imshow(full_rgb)
    axes[1,1].set_title("Full Image Corrected")

    axes[1,2].plot(history, marker='o')
    axes[1,2].set_title("DeltaE per Iteration")
    axes[1,2].set_xlabel("Iteration")
    axes[1,2].set_ylabel("DeltaE")
    axes[1,2].grid(True)

    # axes[1,2].text(
    #     0.05, 0.05,
    #     metrics_text,
    #     transform=axes[1,2].transAxes,
    #     fontsize=10,
    #     bbox=dict(facecolor='white', alpha=0.7)
    # )

    for i, v in enumerate(history):
        axes[1,2].text(i, v, f"{v:.2f}", fontsize=8)

    os.makedirs("result_HarmoMed/results",exist_ok=True)

    object_path=f"result_HarmoMed/results/{filename_path}_object.jpg"
    full_path=f"result_HarmoMed/results/{filename_path}_full.jpg"
    plot_path=f"result_HarmoMed/results/{filename_path}_plot.jpg"

    cv2.imwrite(object_path,result_img)
    cv2.imwrite(full_path,full_corrected)

    plt.tight_layout()
    plt.savefig(plot_path,dpi=300)

    return object_path, full_path, plot_path, {
        "deltaE_before": float(delta_before),
        "deltaE_after": float(delta_after),
        "ssim": float(ssim_val),
        "psnr": float(psnr_val),
        "mse": float(mse_val)
    }

# pipeline หลัก
def run_image_processing_pipeline(reference_path,uploaded_file,csv_file=csv_file):
    UPLOAD_FOLDER='result_HarmoMed/uploads/'
    os.makedirs(UPLOAD_FOLDER,exist_ok=True)

    next_filename=get_next_filename(csv_file)
    new_filename=str(next_filename)

    new_time=datetime.now().strftime("%H:%M:%S")
    new_day=datetime.now().strftime("%d-%m-%Y")

    add_new_row_to_csv(csv_file,new_filename,new_time,new_day)

    target_path=os.path.join(UPLOAD_FOLDER,f"{new_filename}.jpg")

    uploaded_file.save(target_path)

    object_path,full_path,plot_path,metrics=processing_img(
        reference_path,
        target_path,
        filename_path=new_filename
    )

    return{
        "object_result":object_path,
        "full_result":full_path,
        "plot":plot_path,
        "metrics":metrics,
        "filename":new_filename
    }