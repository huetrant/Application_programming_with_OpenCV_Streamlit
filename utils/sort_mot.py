import cv2
import numpy as np
import os
from filterpy.kalman import KalmanFilter

from sort import Sort



# Tải video
DATA_DIR = "./data/object_tracking"
cap = cv2.VideoCapture(f"{DATA_DIR}/people.mp4") 
output_dir = os.path.join(DATA_DIR, "output_results")
os.makedirs(output_dir, exist_ok=True)
# Kiểm tra nếu video mở thành công
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Lấy thông tin về độ phân giải video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Khởi tạo VideoWriter để lưu video đầu ra
output_path = os.path.join(output_dir, "output_sort_video.mp4") # Đường dẫn lưu video kết quả
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng video mp4
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Khởi tạo SORT (tracker)
tracker = Sort()

# Dùng Haar Cascade để phát hiện đối tượng (ví dụ: người)
# Tải bộ phát hiện người từ OpenCV
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi ảnh về grayscale (để tăng hiệu suất)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện đối tượng (người) trong khung hình
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Chuyển đổi kết quả thành định dạng phù hợp với SORT
    detections = []
    for (x, y, w, h) in boxes:
        # Tạo hộp bao quanh (bounding box) trong định dạng SORT [x_min, y_min, x_max, y_max]
        detections.append([x, y, x + w, y + h])

    # Nếu có phát hiện, áp dụng SORT để theo dõi
    detections = np.array(detections)
    trackers = tracker.update(detections)

    # Vẽ hộp bao quanh và ID của đối tượng
    for d in trackers:
        x1, y1, x2, y2, track_id = d
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Ghi khung hình vào video đầu ra
    out.write(frame)

    # Hiển thị khung hình
    cv2.imshow('SORT Tracking', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
