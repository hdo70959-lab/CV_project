# Đọc yêu cầu dự án

Agent phải đọc `AGENTS.md` tại root khi file có trên máy. File đó chứa phạm vi A/B/C, các metric, ràng buộc dữ liệu, báo cáo và tiêu chí hoàn thành; hướng dẫn này chỉ giải thích cách tìm tài liệu nguồn. `AGENTS.md` chỉ lưu local, bị Git ignore và không có trong bản clone mới.

## Tài liệu local bị Git ignore

Thư mục **`yeu_cau/` được giữ trên máy nhưng không đưa lên GitHub** theo yêu cầu người dùng. Git ignore không có nghĩa là bỏ qua yêu cầu bài tập. Các công cụ tìm file mặc định có thể không hiển thị thư mục này.

Từ root dự án, dùng lệnh sau để liệt kê đầy đủ:

```bash
rtk rg --files --hidden --no-ignore yeu_cau/
```

Đọc theo thứ tự:

1. `yeu_cau/INS3155_Yeu_Cau_Cua_Giang_Vien.md` — yêu cầu được ghi nhận từ giảng viên.
2. `yeu_cau/INS3155_Midterm_Project_Requirements.md` — đặc tả chi tiết, checklist, rủi ro và điểm chưa xác nhận.
3. `yeu_cau/Midterm_Exam_Computer_Vision_HK1.26-27 (1).pdf` — đề gốc khi cần đối chiếu.
4. `main.tex` trong `yeu_cau/Midterm Exam_Report (1).zip` — mẫu nguồn báo cáo.

Không sửa/xóa tài liệu gốc, không dùng `git add -f` để đưa tài liệu bị ignore lên repo khi chưa có yêu cầu mới.

## Khi làm trên bản clone mới

Git clone không khôi phục `yeu_cau/` hoặc `AGENTS.md`. Báo rõ những tài liệu đang thiếu và yêu cầu người dùng khôi phục bản local khi nhiệm vụ phụ thuộc vào chúng. Nếu có `AGENTS.md`, đọc file để nắm yêu cầu đã tổng hợp; tiếp tục các phần độc lập có đủ căn cứ.

Không tự tải đề/mẫu khác để thay thế, không khẳng định đã đọc tài liệu không có trên máy, không biến khuyến nghị trong đặc tả thành quy định bắt buộc. Tài liệu nguồn và cập nhật giảng viên đã xác minh vẫn là căn cứ đối chiếu khi có mâu thuẫn.
