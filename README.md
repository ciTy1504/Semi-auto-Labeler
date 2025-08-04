## 📖 Hướng Dẫn Sử Dụng

### Bước 1: Tách Frame từ Video

Nếu dữ liệu dạng video, dùng script này để chuyển thành các frame ảnh

1.  Copy tất cả các file video:
    `video_extraction/videos_to_process/`

2.  Chạy script sau nếu ko code trên PyCharm:
    ```bash
    cd video_extraction
    python extract_frames.py
    cd ..
    ```

> 📝 **Lưu ý:** Tất cả các ảnh sau khi tách sẽ được tự động lưu vào `semi_supervised_labeler/data/unlabeled_images/`

### Bước 2: Chuẩn bị Dữ liệu mồi

Cần gán nhãn thủ công cho một tập dữ liệu nhỏ (khoảng 100-200 ảnh)

1.  Lấy một vài ảnh từ `unlabeled_images/` xong tải lên **Roboflow**.
2.  Gán nhãn
3.  Khi export dữ liệu từ Roboflow, hãy đảm bảo các cài đặt sau:
    - **Format:** `YOLOv8`
    - **Preprocessing:** Thêm bước `Resize` về `640x640`.
4.  Tải file `.zip` về máy và giải nén.
5.  Copy các file đã gán nhãn vào đúng thư mục:
    - Copy toàn bộ **ảnh** (`.jpg`, `.png`) vào: `semi_supervised_labeler/data/seed_data/images/`
    - Copy toàn bộ **file nhãn** (`.txt`) vào: `semi_supervised_labeler/data/seed_data/labels/`

### Bước 3: Cấu hình Dataset

Trước khi chạy, cần khai báo thông tin về các classes

1.  Mở file `semi_supervised_labeler/data/dataset.yaml`.
2.  Chỉnh sửa các trường `nc` (số lượng lớp) và `names` (danh sách tên các lớp) cho phù hợp

    **Ví dụ:**

    ```yaml
    train: ...
    val: ...

    nc: 3 
    names: ['person', 'car', 'dog']
    ```

### Bước 4: Chạy Vòng Lặp Bán Tự Động

Bây giờ, bạn đã sẵn sàng để khởi động quy trình tự động!

1.  Di chuyển vào thư mục chính và chạy script:
    ```bash
    cd semi_supervised_labeler
    python main_workflow.py
    ```

2.  Chương trình sẽ thực hiện các tác vụ sau:
    - Huấn luyện mô hình YOLOv8 đầu tiên từ dữ liệu mồi
    - Dùng mô hình đó để dự đoán trên các ảnh trong `unlabeled_images`.
    - Khởi chạy một giao diện web Gradio.

3.  Mở trình duyệt và truy cập vào đường link hiển thị trong terminal (thường là `http://127.0.0.1:7860`).

4.  Trong giao diện web, duyệt qua từng ảnh:
    - Nhấn **`Approve`** nếu các bounding box là chính xác
    - Nhấn **`Reject`** nếu sai

5.  Sau khi duyệt xong một loạt một batch:
    - **Đóng tab trình duyệt.**
    - Quay lại cửa sổ terminal và nhấn `Ctrl + C` để dừng server Gradio.

> 🔄 Vòng lặp sẽ tự động tiếp tục: Các ảnh bạn vừa `Approve` sẽ được chuyển vào `labeled_data`, và một mô hình mới, tốt hơn sẽ được huấn luyện lại. Quá trình này lặp đi lặp lại cho đến khi không còn ảnh nào trong thư mục `unlabeled_images`.

---

## ✅ Kết quả

Khi vòng lặp kết thúc (thư mục `01_unlabeled_images` trống), toàn bộ dataset đã được gán nhãn hoàn chỉnh của bạn sẽ nằm trong thư mục `2_semi_supervised_labeler/data/02_labeled_data/`. Bạn có thể sử dụng bộ dữ liệu này để huấn luyện một mô hình cuối cùng với chất lượng cao.