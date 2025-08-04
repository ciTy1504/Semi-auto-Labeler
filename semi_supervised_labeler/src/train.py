from ultralytics import YOLO
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset.yaml')
MODEL_PATH = 'yolov8n.pt'
EPOCHS = 25
PROJECT_RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')

def train_model():
    labeled_images_path = os.path.join(PROJECT_ROOT, 'data', 'labeled_data', 'images')

    if not os.path.exists(labeled_images_path) or not os.listdir(labeled_images_path):
        print("Éc éc")
        return None

    best_model_path = os.path.join(PROJECT_RUNS_DIR, 'detect/train/weights/best.pt')
    model_to_train = best_model_path if os.path.exists(best_model_path) else MODEL_PATH

    model = YOLO(model_to_train)

    model.train(
        data=DATA_CONFIG_PATH,
        epochs=EPOCHS,
        imgsz=640,
        project=PROJECT_RUNS_DIR,
        name='detect/train',
        exist_ok=True
    )
    print("Huấn luyện xong")
    return os.path.join(PROJECT_RUNS_DIR, 'detect/train/weights/best.pt')


if __name__ == '__main__':
    train_model()