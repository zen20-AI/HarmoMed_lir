import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
from datetime import datetime
from binary_path_img import save_uploaded_images,get_next_filename,add_new_row_to_csv,load_csv_data

def detection(image_path):
    model_name = "indicator-j3riv"
    version = 4
    api_key = "6GfuPFK2Ue4Fvh9EZiQJ"
    api_url = f"https://detect.roboflow.com/{model_name}/{version}?api_key={api_key}&format=json"
    with open(image_path, "rb") as image_file:
        response = requests.post(api_url, files={"file": image_file})
    if response.status_code == 200:
        data = response.json()
        image = cv2.imread(image_path)
        if "predictions" in data and len(data["predictions"]) > 0:
            # output_dir = "static/cropped"
            # os.makedirs(output_dir, exist_ok=True)
            for i, obj in enumerate(data["predictions"]):
                x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
                label = obj["class"]
                confidence = obj["confidence"]
                x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
                x2, y2 = min(image.shape[1], x + w // 2), min(image.shape[0], y + h // 2)
                cropped_obj = image[y1:y2, x1:x2]
                # cropped_filename = os.path.join(output_dir, f"{label}_{i}.jpg")
                # cv2.imwrite(cropped_filename, cropped_obj)
    return cropped_obj

def adjust_brightness_contrast(reference_img, target_img):
    ref = reference_img.astype(np.float32) / 255.0
    target = target_img.astype(np.float32) / 255.0
    ref_avg_brightness = np.mean(ref)
    target_avg_brightness = np.mean(target)

    brightness_diff = ref_avg_brightness - target_avg_brightness
    contrast_diff = np.std(ref) - np.std(target)
    
    adjusted_target = target.copy()

    adjusted_target += brightness_diff
    adjusted_target = (adjusted_target - 0.00001) * (1 + contrast_diff) + 0.001
    adjusted_target = np.clip(adjusted_target, 0, 1)
    
    corrected_img = (adjusted_target * 255).astype(np.uint8)
    return corrected_img, brightness_diff, contrast_diff

def adjust_tone(reference_img, target_img, step=0.000001):
    reference_img = reference_img.astype(np.float32) / 255.0
    target_img = target_img.astype(np.float32) / 255.0
    ref_avg_color = np.mean(reference_img, axis=(0, 1))
    target_avg_color = np.mean(target_img, axis=(0, 1))

    color_diff = ref_avg_color - target_avg_color

    scale_factor = 0.1 
    color_diff *= scale_factor

    adjusted_target = target_img.copy()
    adjusted_target += color_diff 

    adjusted_target[:, :, 0] *= (1 + color_diff[0] * step)
    adjusted_target[:, :, 1] *= (1 + color_diff[1] * step)
    adjusted_target[:, :, 2] *= (1 + color_diff[2] * step)

    adjusted_target = np.clip(adjusted_target, 0, 1)
    return (adjusted_target * 255).astype(np.uint8), color_diff

def calculate_color_difference(reference_img, corrected_img):
    color_diff = np.abs(reference_img.astype(np.float32) - corrected_img.astype(np.float32))
    return color_diff

def apply_adjustments(target_img, brightness_diff, contrast_diff, color_diff):

    target_img = target_img.astype(np.float32) / 255.0

    target_img += brightness_diff

    target_img = (target_img - 0.5) * (1 + contrast_diff) + 0.5

    target_img[:, :, 0] += color_diff[0]
    target_img[:, :, 1] += color_diff[1]
    target_img[:, :, 2] += color_diff[2]

    target_img = np.clip(target_img, 0, 1)

    return (target_img * 255).astype(np.uint8)

def processing_img(refpath, targetpath, filename_path):
    reference_img = cv2.imread(refpath)
    target_img = cv2.imread(targetpath)

    cropped_img_bgr = detection(targetpath)

    # if reference_img is None or target_img is None or cropped_img_bgr is None:
    #     print("Error: ไม่สามารถโหลดภาพได้")
    #     return

    target_img_ts = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    target_img = cv2.resize(cropped_img_bgr, (reference_img.shape[1], reference_img.shape[0]))

    adjusted_img = cv2.imread(targetpath)

    previous_avg_diff = np.inf
    tolerance = 0.01

    brightness_changes = []
    contrast_changes = []
    color_changes = []

    total_brightness_change = 0
    total_contrast_change = 0
    total_color_change = 0

    while True:
        corrected_img, brightness_diff, contrast_diff = adjust_brightness_contrast(reference_img, target_img)
        warm_corrected_img, color_diff = adjust_tone(reference_img, corrected_img, step=0.000001)

        brightness_changes.append(brightness_diff)
        contrast_changes.append(contrast_diff)
        color_changes.append(color_diff)

        total_brightness_change += brightness_diff
        total_contrast_change += contrast_diff
        total_color_change += np.mean(np.abs(color_diff))

        corrected_avg_color = np.mean(corrected_img, axis=(0, 1))
        current_avg_diff = np.abs(np.mean(reference_img, axis=(0, 1)) - corrected_avg_color).sum()

        if current_avg_diff < tolerance:
            break

        target_img = warm_corrected_img

        if abs(previous_avg_diff - current_avg_diff) < tolerance:
            break
        previous_avg_diff = current_avg_diff

    color_diff_result = calculate_color_difference(reference_img, corrected_img)

    reference_img_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    warm_corrected_img_rgb = cv2.cvtColor(warm_corrected_img, cv2.COLOR_BGR2RGB)
    corrected_img_rgb = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)
    cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
    color_diff_rgb_img = cv2.cvtColor(np.clip(color_diff_result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    true_corrected_img = apply_adjustments(
        adjusted_img,
        total_brightness_change,
        total_contrast_change,
        np.array([total_color_change] * 3)
    )
    true_corrected_img_rgb = cv2.cvtColor(true_corrected_img, cv2.COLOR_BGR2RGB)

    corrected_avg_color = np.mean(corrected_img_rgb, axis=(0, 1))
    true_corrected_avg_color = np.mean(true_corrected_img_rgb, axis=(0, 1))
    color_adjustment = corrected_avg_color - true_corrected_avg_color

    adjusted_true_corrected_img = true_corrected_img_rgb.astype(np.float32) + color_adjustment
    adjusted_true_corrected_img = np.clip(adjusted_true_corrected_img, 0, 255).astype(np.uint8)
    true_corrected_img_rgb = adjusted_true_corrected_img

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(reference_img_rgb)
    axes[0, 0].set_title("Reference Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(target_img_ts)
    axes[0, 1].set_title("Original Target Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(warm_corrected_img_rgb)
    axes[0, 2].set_title("Warm Corrected Image")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(color_diff_rgb_img)
    axes[0, 3].set_title("Color Difference")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(corrected_img_rgb)
    axes[1, 0].set_title("Corrected Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cropped_img_rgb)
    axes[1, 1].set_title("Cropped Image")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(true_corrected_img_rgb)
    axes[1, 2].set_title("True Corrected Image")
    axes[1, 2].axis("off")

    axes[1, 3].plot(brightness_changes, label="Brightness Change")
    axes[1, 3].plot(contrast_changes, label="Contrast Change")
    axes[1, 3].plot([np.mean(diff) for diff in color_changes], label="Color Change")
    axes[1, 3].set_title("Changes Over Iterations")
    axes[1, 3].set_xlabel("Iterations")
    axes[1, 3].set_ylabel("Difference Value")
    axes[1, 3].legend()

    ref_avg_color = np.mean(reference_img, axis=(0, 1))
    corrected_avg_color = np.mean(corrected_img, axis=(0, 1))
    warm_corrected_avg_color = np.mean(warm_corrected_img, axis=(0, 1))

    print(f"Average color of reference image (RGB): {ref_avg_color}")
    print(f"Average color of corrected image (RGB): {corrected_avg_color}")
    print(f"Average color of warm corrected image (RGB): {warm_corrected_avg_color}")

    print(f"Total Brightness Adjustment: {total_brightness_change}")
    print(f"Total Contrast Adjustment: {total_contrast_change}")
    print(f"Total Color Difference: {total_color_change}")

    cv2.imwrite(f"static/results/{filename_path}true_corrected_img_rgb1.jpg", cv2.cvtColor(true_corrected_img_rgb, cv2.COLOR_RGB2BGR))
    plt.tight_layout()
    plt.savefig(f"static/results/{filename_path}result_plot.jpg", dpi=300)

def run_image_processing_pipeline(reference_path, uploaded_file, csv_file="image_log.csv"):
    UPLOAD_FOLDER = 'static/uploads/'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    next_filename = get_next_filename(csv_file)
    new_filename = str(next_filename)
    new_time = datetime.now().strftime("%H:%M:%S")
    new_day = datetime.now().strftime("%d-%m-%Y")
    
    add_new_row_to_csv(csv_file, new_filename, new_time, new_day)
    target_path = os.path.join(UPLOAD_FOLDER, f"{new_filename}.jpg")
    uploaded_file.save(target_path)

    processing_img(reference_path, target_path, filename_path=new_filename)

    result_img_path = f"static/results/{new_filename}true_corrected_img_rgb1.jpg"
    plot_path = f"static/results/{new_filename}result_plot.jpg"
    return {
        "result_image": result_img_path,
        "plot": plot_path,
        "filename": new_filename
    }