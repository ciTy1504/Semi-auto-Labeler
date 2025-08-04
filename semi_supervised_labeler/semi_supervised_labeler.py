import os
import shutil
import subprocess
from src.utils import UNLABELED_DIR, LABELED_DIR, SEED_DATA_DIR


def run_script(script_path):
    try:
        subprocess.run(['python', script_path], check=True, cwd=os.getcwd())
    except subprocess.CalledProcessError as e:
        print(f"Lỗi {script_path}: {e}")
        raise


def bootstrap_initial_data():
    seed_images_dir = os.path.join(SEED_DATA_DIR, 'images')
    seed_labels_dir = os.path.join(SEED_DATA_DIR, 'labels')
    labeled_images_dir = os.path.join(LABELED_DIR, 'images')
    labeled_labels_dir = os.path.join(LABELED_DIR, 'labels')

    os.makedirs(labeled_images_dir, exist_ok=True)
    os.makedirs(labeled_labels_dir, exist_ok=True)

    if not os.listdir(labeled_images_dir) and os.path.exists(seed_images_dir) and os.listdir(seed_images_dir):
        print(f"{os.path.basename(LABELED_DIR)}' trống. Khởi tạo dữ liệu từ {os.path.basename(SEED_DATA_DIR)}")
        for filename in os.listdir(seed_images_dir):
            shutil.copy2(os.path.join(seed_images_dir, filename), labeled_images_dir)
        for filename in os.listdir(seed_labels_dir):
            shutil.copy2(os.path.join(seed_labels_dir, filename), labeled_labels_dir)


def main():
    seed_images_dir = os.path.join(SEED_DATA_DIR, 'images')

    if not os.path.exists(seed_images_dir) or not os.listdir(seed_images_dir):
        print("#"*50)
        print("'seed_data' đang trống")
        print("Gán nhãn thủ công ~100 ảnh bằng Roboflow")
        print("Đặt ảnh vào 'data/seed_data/images'")
        print("Đặt nhãn vào 'data/seed_data/labels'")
        print("Chạy lại")
        print("#"*50)
        return

    bootstrap_initial_data()

    cycle = 1
    while os.listdir(UNLABELED_DIR):

        run_script('src/train.py')

        run_script('src/predict_batch.py')

        try:
            run_script('src/review_app.py')
        except (subprocess.CalledProcessError, KeyboardInterrupt):
            print("Đóng")

        cycle += 1

if __name__ == "__main__":
    main()