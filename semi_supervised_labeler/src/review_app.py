import gradio as gr
import os
import shutil
from src.utils import draw_boxes, TEMP_BATCH_DIR, LABELED_DIR, MANUAL_LABEL_DIR

TEMP_IMG_DIR = os.path.join(TEMP_BATCH_DIR, "images")
TEMP_LBL_DIR = os.path.join(TEMP_BATCH_DIR, "labels")
FINAL_IMG_DIR = os.path.join(LABELED_DIR, "images")
FINAL_LBL_DIR = os.path.join(LABELED_DIR, "labels")

os.makedirs(FINAL_IMG_DIR, exist_ok=True)
os.makedirs(FINAL_LBL_DIR, exist_ok=True)
os.makedirs(MANUAL_LABEL_DIR, exist_ok=True)

def get_review_list():
    if not os.path.exists(TEMP_IMG_DIR): return []
    return sorted([f for f in os.listdir(TEMP_IMG_DIR)])

image_files = get_review_list()

def process_decision(decision, current_index):
    if current_index >= len(image_files):
        return None, "Đã xong batch này", current_index

    img_name = image_files[current_index]
    base_name = os.path.splitext(img_name)[0]

    src_img_path = os.path.join(TEMP_IMG_DIR, img_name)
    src_lbl_path = os.path.join(TEMP_LBL_DIR, f"{base_name}.txt")

    if decision == "Approve":
        shutil.move(src_img_path, os.path.join(FINAL_IMG_DIR, img_name))
        if os.path.exists(src_lbl_path):
            shutil.move(src_lbl_path, os.path.join(FINAL_LBL_DIR, f"{base_name}.txt"))
        print(f"✔️ Approved: {img_name}")

    elif decision == "Reject":
        shutil.move(src_img_path, os.path.join(MANUAL_LABEL_DIR, img_name))
        if os.path.exists(src_lbl_path):
            os.remove(src_lbl_path)
        print(f"❌ Rejected: {img_name}")

    current_index += 1
    return load_image(current_index)

def load_image(current_index):
    if not image_files:
        return None, "Không có ảnh để duyệt. Hãy chạy predict_batch.py.", 0

    if current_index >= len(image_files):
        return None, f"Đã duyệt xong {len(image_files)} ảnh, đóng cửa sổ này để tiếp tục.", current_index

    img_name = image_files[current_index]
    img_path = os.path.join(TEMP_IMG_DIR, img_name)
    lbl_path = os.path.join(TEMP_LBL_DIR, f"{os.path.splitext(img_name)[0]}.txt")

    display_image = draw_boxes(img_path, lbl_path)
    progress_text = f"Đang duyệt ảnh {current_index + 1} / {len(image_files)}"

    return display_image, progress_text, current_index

def launch_app():
    if not image_files:
        print("Không có ảnh để duyệt. Hãy chạy predict_batch.py trước.")
        return

    with gr.Blocks(title="Duyệt Nhãn Bán Tự Động") as app:
        gr.Markdown("# Giao diện Duyệt Nhãn Bán Tự Động")
        state_index = gr.State(value=0)

        with gr.Row():
            image_display = gr.Image(label="Ảnh với Nhãn Dự Đoán", type="numpy")
            progress_output = gr.Textbox(label="Tiến trình", interactive=False)

        with gr.Row():
            approve_btn = gr.Button("✅ Approve", variant="primary")
            reject_btn = gr.Button("❌ Reject", variant="stop")

        approve_btn.click(lambda idx: process_decision("Approve", idx), [state_index],
                          [image_display, progress_output, state_index])
        reject_btn.click(lambda idx: process_decision("Reject", idx), [state_index],
                         [image_display, progress_output, state_index])

        app.load(load_image, [state_index], [image_display, progress_output, state_index])

    print("Khởi chạy Gradio. Mở trình duyệt tại địa chỉ được cung cấp")
    app.launch()

if __name__ == '__main__':
    launch_app()