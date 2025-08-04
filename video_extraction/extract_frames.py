import cv2
import os
import shutil
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SEMI_SUPERVISED_PROJECT_DIR = os.path.join(PROJECT_ROOT, "semi_supervised_labeler")

VIDEO_SOURCE_DIR = os.path.join(SCRIPT_DIR, "videos_to_process")
VIDEO_PROCESSED_DIR = os.path.join(SCRIPT_DIR, "videos_processed")
FRAME_OUTPUT_DIR = os.path.join(SEMI_SUPERVISED_PROJECT_DIR, "data", "unlabeled_images")
DESIRED_FPS = 5

def extract_frames_from_videos():
    os.makedirs(VIDEO_SOURCE_DIR, exist_ok=True)
    os.makedirs(VIDEO_PROCESSED_DIR, exist_ok=True)
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

    video_files = [f for f in os.listdir(VIDEO_SOURCE_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print("Xong hết")
        return

    print(f"Còn {len(video_files)} video")
    for video_file in video_files:
        video_path = os.path.join(VIDEO_SOURCE_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Lỗi ko mở đc video {video_file}")
            continue

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0:
            original_fps = 30

        frame_interval = int(original_fps / DESIRED_FPS)
        if frame_interval < 1:
            frame_interval = 1

        frame_count = 0
        saved_frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc="Được") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    output_filename = f"{video_name}_frame_{frame_count}.jpg"
                    output_path = os.path.join(FRAME_OUTPUT_DIR, output_filename)
                    cv2.imwrite(output_path, frame)
                    saved_frame_count += 1

                frame_count += 1
                pbar.update(1)

        cap.release()
        shutil.move(video_path, os.path.join(VIDEO_PROCESSED_DIR, video_file))

if __name__ == "__main__":
    extract_frames_from_videos()