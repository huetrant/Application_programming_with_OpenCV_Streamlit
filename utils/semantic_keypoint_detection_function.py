import os
import cv2
import numpy as np
import streamlit as st



# Hàm vẽ các điểm đặc trưng (keypoints) lên ảnh
def draw_points(
    image: np.ndarray, points: np.ndarray, color=(0, 255, 0), thickness=2, radius=1, outer_radius=1,lineType=cv2.LINE_AA
):
    for point in points:
        # Vẽ vòng tròn đầu tiên (vòng tròn nhỏ) với bán kính nhỏ hơn
        cv2.circle(image, (int(point[1]), int(point[0])), radius, color, thickness, lineType)
        
        # Vẽ vòng tròn thứ hai (vòng tròn lớn) với bán kính lớn hơn
        cv2.circle(image, (int(point[1]), int(point[0])), outer_radius, color, thickness,lineType)
        
    return image

def calculate_precision_recall_image(datatype, index, detector,distance_threshold=3):
    image = cv2.imread(os.path.join(datatype[index], "images", "7.png"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện keypoints
    keypoints = detector.detect(gray, None)
    
    pred =  np.array([[kp.pt[1], kp.pt[0]] for kp in keypoints])
    # Đọc ground truth
    ground_truth = np.load(os.path.join(datatype[index], "points", "7.npy"))
    # Nếu không có điểm nào trong ground truth hoặc dự đoán, trả về (0, 0)
    if len(ground_truth) == 0 or len(pred) == 0:
        return (0, 0)

    true_positive, false_negative, false_positive = 0, 0, 0

    # Đếm true positives và false positives
    for predicted_point in pred:
        distances = np.linalg.norm(ground_truth - predicted_point, axis=1)
        if np.any(distances <= distance_threshold):
            true_positive += 1
        else:
            false_positive += 1

    # Đếm false negatives
    for ground_truth_point in ground_truth:
        distances = np.linalg.norm(pred - ground_truth_point, axis=1)
        if not np.any(distances <= distance_threshold):
            false_negative += 1

    # Tính toán precision và recall
    if true_positive == 0:
        return (0, 0)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return (precision, recall)

def process_image(datatype, index, detector):
    image = cv2.imread(os.path.join(datatype[index], "images", "7.png"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện keypoints
    keypoints = detector.detect(gray, None)
    
    # Đọc ground truth
    ground_truth = np.load(os.path.join(datatype[index], "points", "7.npy"))
    
    # Vẽ keypoints và ground truth lên ảnh
    image = draw_points(image, [(kp.pt[1], kp.pt[0]) for kp in keypoints], (255, 0, 0), 1,1, 3)
    image = draw_points(image, ground_truth, (0, 255, 0),1)
    
    return image