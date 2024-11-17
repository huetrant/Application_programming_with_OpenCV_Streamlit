import cv2
import os

# Khởi tạo đường dẫn và video
DATA_DIR = "./data/object_tracking"
cap = cv2.VideoCapture(f"{DATA_DIR}/people.mp4") 
output_dir = os.path.join(DATA_DIR, "output_results")
os.makedirs(output_dir, exist_ok=True)

# Lấy thông tin về video gốc (kích thước, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Giảm độ phân giải video (chỉnh giảm kích thước)
new_width = frame_width // 4  # Chia đôi kích thước chiều rộng
new_height = frame_height // 4  # Chia đôi kích thước chiều cao

# Tạo đối tượng ghi video với kích thước mới
output_path = os.path.join(output_dir, "people_tracking_output.avi")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec MJPG cho tốc độ ghi nhanh hơn
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

# Khởi tạo CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Đọc khung đầu tiên và chọn vùng cần theo dõi
ret, frame = cap.read()
if not ret:
    print("Không thể đọc video.")
    cap.release()
    exit()

# Resize khung đầu tiên
frame = cv2.resize(frame, (new_width, new_height))

bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

# Vòng lặp theo dõi
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize khung hình trước khi theo dõi
    frame = cv2.resize(frame, (new_width, new_height))

    # Cập nhật vị trí của đối tượng
    success, bbox = tracker.update(frame)
    
    if success:
        # Vẽ hộp quanh đối tượng nếu theo dõi thành công
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        # Hiển thị cảnh báo nếu mất dấu đối tượng
        cv2.putText(frame, "Tracking Failure!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Ghi khung vào video đầu ra
    out.write(frame)

    # Hiển thị khung
    cv2.imshow("Tracking", frame)
    
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()  # Dừng ghi video
cv2.destroyAllWindows()

print(f"Video theo dõi đã được lưu tại: {output_path}")
