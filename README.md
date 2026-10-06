# Nhận diện thiết bị bảo hộ lao động với YOLOv7

Dự án sử dụng YOLOv7 để nhận diện **mũ bảo hộ (Safety-Helmet)** và **áo phản quang (Reflective-Jacket)** trong ảnh. Repository bao gồm mã nguồn YOLOv7, ứng dụng web Flask để tải ảnh lên và xem kết quả, cùng kịch bản kiểm tra nhanh mô hình.

> **Lưu ý về ứng dụng:** Báo cáo thực nghiệm ban đầu đề cập đến Streamlit Dashboard, nhưng mã nguồn hiện có trong repository triển khai giao diện bằng **Flask**. Các hướng dẫn chạy bên dưới dành cho ứng dụng Flask trong `app.py`; notebook Colab, localtunnel và ứng dụng Streamlit không nằm trong repository này.

## Tính năng

- Tải ảnh JPG, JPEG hoặc PNG để chạy nhận diện.
- Hiển thị ảnh kết quả với bounding box, tên lớp và độ tin cậy.
- Trả về danh sách phát hiện và thống kê số lượng mũ bảo hộ, áo phản quang.
- Tự động dùng CUDA GPU nếu khả dụng; nếu không, chạy trên CPU.
- Nạp mô hình một lần khi ứng dụng khởi động.

## Cấu trúc dự án

```text
YOLOv7-Helmet/
├── model/
│   └── YOLOv7.pt                # Trọng số mô hình đã huấn luyện
├── yolov7/                      # Mã nguồn YOLOv7
├── static/                      # CSS, JavaScript và tài nguyên giao diện
├── templates/                   # Giao diện Flask
├── uploads/                     # Ảnh tải lên (được tạo khi chạy)
├── results/                     # Ảnh kết quả (được tạo khi chạy)
├── app.py                       # Ứng dụng web Flask
├── test_model.py                # Kiểm tra nạp mô hình và inference
├── requirements.txt
└── README.md
```

Ứng dụng nạp trọng số từ `model/YOLOv7.pt`. Đặt file này vào thư mục `model/` trước khi chạy ứng dụng. File trọng số hiện có trong workspace không được đưa vào Git vì thư mục `model/` đang được loại trừ trong `.gitignore`.

## Dữ liệu và cấu hình lớp

Bộ dữ liệu thực nghiệm là **Safety Helmet and Reflective Jacket**, tải bằng `kagglehub` về môi trường Google Colab tại:

```text
/root/.cache/kagglehub/datasets/niravnaik/safety-helmet-and-reflective-jacket/
```

Dữ liệu được chia thành ba tập:

| Tập dữ liệu | Số ảnh |
|---|---:|
| Train | 7.350 |
| Valid | 1.575 |
| Test | 1.575 |
| **Tổng** | **10.500** |

Hai lớp trong `data.yaml`:

| ID | Lớp |
|---:|---|
| 0 | Safety-Helmet |
| 1 | Reflective-Jacket |

Kết quả kiểm định dữ liệu được ghi nhận: không có ảnh thiếu nhãn hoặc nhãn lỗi. Phân phối nhãn:

| Tập | Mũ bảo hộ | Áo phản quang |
|---|---:|---:|
| Train | 14.138 | 11.376 |
| Valid | 2.966 | 2.326 |
| Test | 3.087 | 2.347 |
| **Tổng** | **20.191** | **16.049** |

Trung bình khoảng **3,45 đối tượng/ảnh**. Báo cáo thực nghiệm cũng ghi nhận đã trực quan hóa ngẫu nhiên 9 ảnh cùng bounding box ground truth để kiểm tra tọa độ nhãn YOLO.

## Thiết lập và chạy trên Windows

Yêu cầu Python và pip đã được cài đặt.

1. Mở thư mục dự án trong PowerShell hoặc VS Code.
2. Tạo và kích hoạt môi trường ảo:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Cài các gói phụ thuộc:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   Nếu cần chạy bằng NVIDIA GPU, cài phiên bản PyTorch tương thích với CUDA trên máy theo hướng dẫn cài đặt chính thức của PyTorch.

4. Đặt trọng số đã huấn luyện vào `model\YOLOv7.pt`.
5. Khởi chạy ứng dụng:

   ```powershell
   python app.py
   ```

6. Mở [http://127.0.0.1:5000](http://127.0.0.1:5000).

Ứng dụng giới hạn kích thước file tải lên ở 16 MB và chỉ chấp nhận ảnh JPG, JPEG, PNG. Ảnh tải lên và ảnh kết quả được lưu trong `uploads/` và `results/`.

## Kiểm tra mô hình

Chạy kịch bản inference kiểm tra nhanh:

```powershell
python test_model.py
```

Kịch bản nạp trọng số, in thiết bị và tên lớp, sau đó thử inference trên ảnh mẫu `yolov7\inference\images\zidane.jpg` nếu ảnh đó có sẵn.

## Cấu hình inference

- Kích thước ảnh đầu vào: `640 × 640`.
- Ngưỡng confidence mặc định của ứng dụng: `0.50`.
- Ngưỡng IoU của Non-Maximum Suppression: `0.45`.
- Có thể truyền confidence qua query parameter `conf`, ví dụ `/api/detect?conf=0.4`.
- Endpoint nhận ảnh: `POST /api/detect`, trường multipart có tên `image`.

Phản hồi thành công gồm `image_url`, danh sách `detections` (class, confidence và bounding box) cùng `statistics` (tổng phát hiện, số mũ, số áo và confidence).

## Quy trình và kết quả thực nghiệm

Mô hình YOLOv7 pretrained `yolov7.pt` được dùng làm trọng số khởi tạo để fine-tune trên bộ dữ liệu hai lớp. Cấu hình huấn luyện được báo cáo:

| Tham số | Giá trị |
|---|---:|
| Kích thước ảnh | 640 × 640 |
| Batch size | 16 |
| Epoch | 3 |
| Thiết bị | CUDA GPU (0), Tesla T4 |

Sau 3 epoch, báo cáo ghi nhận kết quả tổng thể trên tập validation là **93%**. Đánh giá độc lập trên tập test bằng `best.pt`:

| Chỉ số | Kết quả |
|---|---:|
| Precision | 90,4% |
| Recall | 84,0% |
| mAP@0.5 | 92,3% |
| mAP@0.5:0.95 | 64,0% |

Trong thử nghiệm trực quan được báo cáo, mô hình phát hiện 2 mũ bảo hộ và 1 áo phản quang trong ảnh mẫu; thời gian xử lý được ghi nhận là 23 ms.

> Các con số trên là kết quả được cung cấp trong báo cáo thực nghiệm. Repository này không bao gồm dữ liệu huấn luyện, notebook huấn luyện, log đánh giá hoặc file `best.pt` để tái tạo độc lập các kết quả đó. Thời gian inference thực tế phụ thuộc phần cứng, kích thước ảnh và cấu hình chạy.

## Tương thích PyTorch

YOLOv7 là mã nguồn cũ; báo cáo thực nghiệm đã phải xử lý khác biệt tương thích khi nạp checkpoint trên PyTorch mới (tham số `weights_only=False`) và lỗi chỉ mục confusion matrix khi đánh giá mô hình hai lớp. Ứng dụng hiện tại thiết lập `weights_only=False` khi nạp checkpoint. Việc vá confusion matrix phục vụ đánh giá huấn luyện không được thực hiện bởi ứng dụng Flask.

Chỉ nạp checkpoint đáng tin cậy: `weights_only=False` có thể thực thi nội dung pickle trong file trọng số.

## Lưu trữ và triển khai

Trong thí nghiệm trên Google Colab, trọng số được sao lưu tại `/content/YOLOv7.pt` và Google Drive ở `/content/drive/MyDrive/YOLOv7_Safety/weights/YOLOv7.pt`. Các đường dẫn này thuộc môi trường Colab, không phải đường dẫn trong máy cục bộ.

Để cho phép truy cập ứng dụng từ bên ngoài, cần tự cấu hình dịch vụ hosting hoặc tunnel. Việc sử dụng localtunnel được đề cập trong báo cáo, nhưng không được tự động cấu hình hay khởi chạy bởi `app.py`.

## Mã nguồn YOLOv7

Thư mục `yolov7/` chứa mã nguồn dựa trên repository [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7). Tham khảo README trong thư mục đó để biết chi tiết về kiến trúc, huấn luyện và các thành phần của YOLOv7 gốc.
