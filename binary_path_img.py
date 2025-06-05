import os
from PIL import Image
import pandas as pd
from datetime import datetime

csv_file = "image_log.csv"
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    required_columns = {'filename', 'time', 'day'}
    if not required_columns.issubset(df.columns):
        raise ValueError("ไฟล์ CSV ต้องมีคอลัมน์ชื่อ 'filename', 'time' และ 'day'")
    return df

def add_new_row_to_csv(csv_file, new_filename, time, day):
    df = load_csv_data(csv_file)
    new_row = pd.DataFrame({'filename': [new_filename], 'time': [time], 'day': [day]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False)

def get_next_filename(csv_file):
    try:
        df = load_csv_data(csv_file)
        max_filename = df['filename'].astype(int).max()
        return max_filename + 1 if pd.notnull(max_filename) else 1
    except FileNotFoundError:
        return 1

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


# if __name__ == "__main__":
#     input_files = [
#         "true_corrected_img_rgb1.jpg",
#         "true_corrected_img_rgb1.jpg",
#         "true_corrected_img_rgb1.jpg"
#     ]
#     try:
#         file_count, added_files = save_uploaded_images(input_files, csv_file)
#         print(added_files)