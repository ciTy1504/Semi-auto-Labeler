import os
import shutil
from ultralytics import YOLO
from tqdm import tqdm
from src.utils import UNLABELED_DIR, TEMP_BATCH_DIR

BATCH_SIZE = 100
CONFIDENCE_THRESHOLD = 0.4
TARGET_IMG_SIZE = 640

def predict_on_batch():
    possible_model_paths = [
        'runs/detect/train/weights/best.pt',
    ]
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if not model_path:
        print("Ko thấy file 'best.pt', train.py trước")
        return

    if os.path.exists(TEMP_BATCH_DIR):
        shutil.rmtree(TEMP_BATCH_DIR)
    os.makedirs(os.path.join(TEMP_BATCH_DIR, 'images'))
    os.makedirs(os.path.join(TEMP_BATCH_DIR, 'labels'))

    unlabeled_images = sorted([f for f in os.listdir(UNLABELED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not unlabeled_images:
        return

    batch_images_names = unlabeled_images[:BATCH_SIZE]
    model = YOLO(model_path)


    for image_name in tqdm(batch_images_names, desc="Đang dự đoán"):
        source_image_path = os.path.join(UNLABELED_DIR, image_name)
        base_name, _ = os.path.splitext(image_name)

        temp_image_path = os.path.join(TEMP_BATCH_DIR, 'images', image_name)
        shutil.move(source_image_path, temp_image_path)

        model.predict(
            source=temp_image_path,
            save_txt=True,
            conf=CONFIDENCE_THRESHOLD,
            project=TEMP_BATCH_DIR,
            name='predictions',
            exist_ok=True,
            save=False,
            verbose=False,
            imgsz=TARGET_IMG_SIZE
        )

        predicted_label_path = os.path.join(TEMP_BATCH_DIR, 'predictions', 'labels', f"{base_name}.txt")
        final_label_path = os.path.join(TEMP_BATCH_DIR, 'labels', f"{base_name}.txt")

        if os.path.exists(predicted_label_path):
            shutil.move(predicted_label_path, final_label_path)

    temp_yolo_dir = os.path.join(TEMP_BATCH_DIR, 'predictions')
    if os.path.exists(temp_yolo_dir):
        shutil.rmtree(temp_yolo_dir)

if __name__ == '__main__':
    predict_on_batch()