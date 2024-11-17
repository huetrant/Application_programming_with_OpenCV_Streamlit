import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Hàm tạo bộ lọc tương quan (Correlation Filter)
def create_gaussian_filter(size, sigma=5):
    # Tạo bộ lọc Gaussian 2D
    filter_x = np.linspace(-size//2, size//2, size)
    filter_y = np.linspace(-size//2, size//2, size)
    X, Y = np.meshgrid(filter_x, filter_y)
    gaussian_filter = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return gaussian_filter

# Hàm áp dụng bộ lọc vào một hình ảnh để tạo bản đồ phản hồi (Response Map)
def apply_correlation_filter(frame, filter_kernel):
    # Áp dụng bộ lọc Gaussian lên hình ảnh để tạo bản đồ phản hồi
    response_map = cv2.filter2D(frame, -1, filter_kernel)
    return response_map

# Lưu bộ lọc và bản đồ phản hồi
def save_filter_and_response(filter_kernel, response_map, output_dir, step):
    # Lưu bộ lọc Gaussian
    filter_filename = os.path.join(output_dir, f"correlation_filter_step_{step}.png")
    cv2.imwrite(filter_filename, np.uint8(filter_kernel * 255))  # Chuẩn hóa và lưu dưới dạng ảnh

    # Lưu bản đồ phản hồi
    response_filename = os.path.join(output_dir, f"response_map_step_{step}.png")
    cv2.imwrite(response_filename, np.uint8(response_map * 255))  # Chuẩn hóa và lưu dưới dạng ảnh

    # Lưu dưới dạng ma trận NumPy
    np.save(os.path.join(output_dir, f"correlation_filter_step_{step}.npy"), filter_kernel)
    np.save(os.path.join(output_dir, f"response_map_step_{step}.npy"), response_map)

# Khởi tạo video và CSRT tracker
DATA_DIR = "./data/object_tracking"
cap = cv2.VideoCapture(f"{DATA_DIR}/fish.mp4")  # Đọc video từ thư mục đã chỉ định
tracker = cv2.TrackerCSRT_create()

# Tạo thư mục lưu kết quả nếu chưa tồn tại
output_dir = os.path.join(DATA_DIR, "output_results")
os.makedirs(output_dir, exist_ok=True)

# Đọc khung hình đầu tiên và chọn vùng cần theo dõi
ret, frame = cap.read()
if not ret:
    print("Không thể đọc video")
    exit()

# Chọn vùng cần theo dõi trong video
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

# Bước 1: Tạo bộ lọc tương quan Gaussian
filter_size = 21  # Kích thước bộ lọc
gaussian_filter = create_gaussian_filter(filter_size, sigma=5)

# Lưu bộ lọc tương quan
save_filter_and_response(gaussian_filter, np.zeros_like(frame), output_dir, step=1)

# Bước 2: Áp dụng bộ lọc vào khung hình
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển hình ảnh sang grayscale
response_map = apply_correlation_filter(gray_frame, gaussian_filter)

# Lưu bản đồ phản hồi
save_filter_and_response(gaussian_filter, response_map, output_dir, step=2)

# Hiển thị bộ lọc Gaussian và bản đồ phản hồi dưới dạng heatmap
plt.imshow(gaussian_filter, cmap='gray')
plt.title("Gaussian Correlation Filter")
plt.colorbar()
plt.show()

plt.imshow(response_map, cmap='jet')
plt.title("Response Map")
plt.colorbar()
plt.show()

# Bước 3: Theo dõi đối tượng qua các khung hình (Tracking)
success, bbox = tracker.update(frame)
if success:
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    cv2.putText(frame, "Tracking Failure!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

# Lưu khung hình theo dõi
cv2.imwrite(f"{output_dir}/tracked_frame.png", frame)

# Lưu video theo dõi (nếu cần)
output_video = cv2.VideoWriter(f"{output_dir}/output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
output_video.write(frame)
output_video.release()

cap.release()
cv2.destroyAllWindows()


