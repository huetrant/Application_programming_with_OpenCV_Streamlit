import os
import cv2
import numpy as np
import streamlit as st


DATA_DIR  = "./data/semantic_keypoint_detection/synthetic_shapes_datasets"

SHAPE_TYPES  = [
    os.path.join(DATA_DIR, "draw_checkerboard"),
    os.path.join(DATA_DIR, "draw_cube"),
    os.path.join(DATA_DIR, "draw_ellipses"),
    os.path.join(DATA_DIR, "draw_lines"),
    os.path.join(DATA_DIR, "draw_multiple_polygons"),
    os.path.join(DATA_DIR, "draw_polygon"),
    os.path.join(DATA_DIR, "draw_star"),
    os.path.join(DATA_DIR, "draw_stripes"),
]

# Khởi tạo các mô hình phát hiện điểm đặc trưng SIFT và ORB
sift_detector = cv2.SIFT_create()
orb_detector = cv2.ORB_create()


# Hàm vẽ các điểm đặc trưng (keypoints) lên ảnh
def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, color=(0, 255, 0), thickness=2):
    for point in keypoints:
        cv2.circle(image, (int(point[1]), int(point[0])), 1, color, thickness)
    return image

# Hàm tính Precision và Recall của các điểm dự đoán
def calculate_precision_recall(true_points: np.ndarray, predicted_points: np.ndarray):
    true_positive, false_positive, false_negative = 0, 0, 0

    # Duyệt qua từng điểm thực để kiểm tra có điểm dự đoán tương ứng gần đúng hay không
    for point in true_points:
        if predicted_points.shape[0] > 0 and np.any(
            np.linalg.norm(predicted_points - point, axis=1) <= 5
        ):
            true_positive += 1
        else:
            false_negative += 1

    # Duyệt qua từng điểm dự đoán để xác định các điểm không có điểm thực tương ứng gần đúng
    for pred_point in predicted_points:
        if true_points.shape[0] == 0 or not np.any(
            np.linalg.norm(true_points - pred_point, axis=1) <= 5
        ):
            false_positive += 1

    # Tính Precision và Recall
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return precision, recall

# Hàm tính Precision và Recall cho một loại hình dạng
def evaluate_shape_precision_recall(shape_index: int, model):
    precision_recall_scores = []

    # Duyệt qua 500 ảnh của loại hình dạng được chỉ định
    for img_index in range(500):
        # Tải điểm thực và ảnh từ thư mục tương ứng
        true_points = np.load(os.path.join(SHAPE_TYPES[shape_index], "points", f"{img_index}.npy"))
        image = cv2.imread(os.path.join(SHAPE_TYPES[shape_index], "images", f"{img_index}.png"))
        
        # Chuyển ảnh sang thang độ xám để phát hiện điểm đặc trưng
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện điểm đặc trưng trên ảnh xám bằng mô hình (SIFT hoặc ORB)
        keypoints = model.detect(gray_image, None)
        
        # Chuyển đổi điểm đặc trưng thành dạng tọa độ
        predicted_points = np.array([[kp.pt[1], kp.pt[0]] for kp in keypoints])
        
        # Tính Precision và Recall cho ảnh và thêm vào danh sách
        precision_recall_scores.append(calculate_precision_recall(true_points, predicted_points))
    
    return precision_recall_scores
