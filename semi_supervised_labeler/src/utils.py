import cv2
import os
import yaml

ROOT_DATA_DIR = "data"
UNLABELED_DIR = os.path.join(ROOT_DATA_DIR, "unlabeled_images")
LABELED_DIR = os.path.join(ROOT_DATA_DIR, "labeled_data")
TEMP_BATCH_DIR = os.path.join(ROOT_DATA_DIR, "temp_batch_for_review")
MANUAL_LABEL_DIR = os.path.join(ROOT_DATA_DIR, "needs_manual_labeling")
SEED_DATA_DIR = os.path.join(ROOT_DATA_DIR, "seed_data")
DATASET_CONFIG = os.path.join(ROOT_DATA_DIR, "dataset.yaml")


def get_class_names():
    with open(DATASET_CONFIG, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']


def draw_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None: return None
    h, w, _ = image.shape
    class_names = get_class_names()

    if not os.path.exists(label_path):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_names[class_id], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)