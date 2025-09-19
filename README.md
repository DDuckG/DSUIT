# DSUIT

## Yêu cầu

* Python 3 và các thư viện cần thiết (chưa viết), tạm tự cài đi.
* Video thô đặt trong thư mục `data/raw_videos/` với định dạng `.mp4`.
* Có bash để chạy makefile, khuyến khích WSL.
* Data lấy trên drive

```bash
conda create -n dsuit -y
conda activate dsuit
pip install -r requirements.txt
```

## Cách dùng

### 1. Xử lý một video cụ thể

Ví dụ với video `data/raw_videos/01.mp4`:

```bash
make src=01 process-all
```

Kết quả:

* Video đã qua tiền xử lý: `data/clean_videos/01.mp4`
* Detections: `outputs/01/detection/detections.txt`
* Tracks: `outputs/01/tracking/tracks.txt`
* Segmentation: `outputs/01/segmentation/`
* Depth: `outputs/01/depth/`

### 2. Xử lý tất cả video trong thư mục `data/raw_videos/`

```bash
make all
```

Tất cả video sẽ được xử lý tuần tự, kết quả lưu trong `outputs/<video_name>/`.

### 3. Xoá kết quả của một video

```bash
make src=01 clean_dirs
```

Xoá video đã clean và toàn bộ outputs liên quan đến `01.mp4`.

> Các lệnh nhỏ khác tham khảo trong source code Makefile

---
