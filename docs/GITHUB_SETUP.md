# GitHub riêng cho dự án

## Trạng thái chuẩn bị

- Repository private: `https://github.com/hdo70959-lab/CV_project`; remote `origin` dùng HTTPS.
- Cấu hình Git chỉ áp dụng cho repo này: `user.useConfigOnly=true`, tên commit `namhai0512` và email do người dùng cung cấp đã được đặt cục bộ; không kế thừa danh tính commit của máy.
- `credential.helper` cục bộ để trống để bỏ chuỗi helper kế thừa; helper riêng cho `https://github.com` dùng GitHub CLI với cấu hình `.git/github-cli` và loại bỏ các biến token có thể ghi đè tài khoản.
- Không đổi cấu hình Git global, tài khoản GitHub CLI đang active hoặc SSH config của máy.

Các thiết lập trên nằm trong `.git/config`, không được chia sẻ qua Git. Bản clone mới cần thiết lập lại danh tính và xác thực riêng.

## Danh tính và xác thực

Tài khoản xác thực đã kiểm tra qua API là **`hdo70959-lab`**. Tên commit là `namhai0512`, theo thông tin người dùng cung cấp; tên commit không quyết định tài khoản đăng nhập GitHub.

Không gửi mật khẩu, token hoặc private key vào chat, source code hay README.

## Khi kết nối

1. Đặt `user.name` và `user.email` bằng `git config --local` theo thông tin đã cung cấp.
2. Thiết lập xác thực riêng cho tài khoản mới. Nếu dùng GitHub CLI, dùng `GH_CONFIG_DIR` riêng (có thể đặt trong `.git/github-cli`) cho mọi lệnh đăng nhập và thao tác repo; không switch tài khoản mặc định của máy. Kiểm tra các biến token trong môi trường không ghi đè tài khoản đã chọn, không in giá trị token.
3. Nếu dùng SSH, chọn key riêng và cấu hình `core.sshCommand` cục bộ có `IdentitiesOnly=yes` cùng key tương ứng; không dùng mặc định key/agent hiện tại. Nếu dùng HTTPS, gắn helper riêng cho repo sau khi xác thực tài khoản mới.
4. Xác minh danh tính xác thực và quyền truy cập repo trước khi đặt remote hoặc push. Danh tính commit không quyết định tài khoản đăng nhập GitHub.
5. Chỉ tạo repo/commit/push khi người dùng giao bước đó. Kiểm tra file sẽ đưa lên, quyền phân phối dữ liệu và quyền hiển thị repo trước khi công bố.

Ngày 18/10/2026, người dùng yêu cầu push bản khởi tạo sau khi đăng nhập. Repo được tạo private dưới tài khoản đã xác thực `hdo70959-lab`; giữ nguyên cấu hình tài khoản mặc định của máy. Thư mục `yeu_cau/` chỉ giữ local theo `.gitignore`.
