# INS3155 — Computer Vision Midterm

Dự án giữa kỳ môn **INS3155 — Computer Vision**, chủ đề *Traditional Image Processing for Filtering, 3D Reconstruction, and Image Stitching*.

Dự án sử dụng các phương pháp xử lý ảnh và thị giác máy tính cổ điển, gồm ba phần:

- **Part A — Image Filtering and Enhancement:** so sánh Mean, Gaussian, Median và Laplacian Sharpening trên ảnh có Gaussian noise và Salt-and-Pepper noise.
- **Part B — 3D Reconstruction:** ước lượng hình học epipolar, so sánh disparity từ BM/SGBM và tái dựng đám mây điểm 3D từ cặp ảnh stereo.
- **Part C — Multi-View Image Stitching:** so sánh các bộ trích xuất đặc trưng và ghép bốn ảnh chồng lấn thành panorama.

## Cấu trúc

```text
src/                         Mã nguồn Python
tests/                       Kiểm thử
data/
  part_a/                    Ảnh sạch và thông tin nguồn
  part_b/                    Cặp Middlebury và metadata/calibration
  part_c/                    Bốn ảnh overlap và thứ tự ghép
results/
  part_a/                    Ảnh lọc, grid và bảng metrics
  part_b/                    Matching, disparity, point cloud và metrics
  part_c/                    Matching, panorama và metrics
report_source/               Nguồn LaTeX và hình báo cáo
pyproject.toml               Metadata và dependency Python
```

## Môi trường

Python **3.13 trở lên** theo `pyproject.toml`.

## Dữ liệu

- **Part A:** ảnh sạch làm tham chiếu để tạo nhiễu và đánh giá chất lượng lọc.
- **Part B:** cặp ảnh stereo từ [Middlebury Optical Flow Datasets](https://vision.middlebury.edu/flow/data/), theo nguồn được chỉ định trong đề bài.
- **Part C:** bốn ảnh có vùng chồng lấn phục vụ ghép panorama.

## Trạng thái

Dự án đang ở bước chuẩn bị cấu trúc thư mục; chưa có mã xử lý A/B/C, dữ liệu thực nghiệm hoặc kết quả đánh giá.
