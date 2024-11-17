import os
import cv2
import numpy as np
import streamlit as st
from utils.SuperPoint import SuperPointFrontend
from typing import Tuple, List

# Đường dẫn dữ liệu
DATA_DIR = "./data/semantic_keypoint_detection/"
DATASET_DIR = os.path.join(DATA_DIR, "synthetic_shapes_datasets")
DATATYPES = [
    "draw_checkerboard", "draw_cube", "draw_ellipses", "draw_lines",
    "draw_multiple_polygons", "draw_polygon", "draw_star", "draw_stripes"
]
DATATYPES_PATHS = [os.path.join(DATASET_DIR, dt) for dt in DATATYPES]


# Khởi tạo mô hình và matcher cho từng phương pháp
models = {
    "ORB": cv2.ORB_create(edgeThreshold=0, fastThreshold=0),
    "SIFT": cv2.SIFT_create(),
    "SuperPoint": SuperPointFrontend(weights_path="./models/superpoint/superpoint_v1.pth", nms_dist=4, conf_thresh=0.015, nn_thresh=0.7)
}
matchers = {
    "SIFT": cv2.BFMatcher(),
    "ORB": cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
    "SuperPoint": cv2.BFMatcher()
}


@st.cache_data
def read_image(type_idx: int, name: str):
    """Đọc ảnh và các điểm đặc trưng từ dataset với caching"""
    image_path = os.path.join(DATATYPES_PATHS[type_idx], "images", f"{name}.png")
    points_path = os.path.join(DATATYPES_PATHS[type_idx], "points", f"{name}.npy")
    
    image = cv2.imread(image_path)
    ground_truth = np.load(points_path)
    keypoints = [(x, y) for x, y in ground_truth]

    
    return image, keypoints

def rotate_image(image, angle):
    """Xoay ảnh theo góc với caching"""
    h, w = image.shape[:2]
    matrix_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, matrix_rotation, (w, h))

def rotate_keypoints(size: Tuple[int, int], keypoints: List[cv2.KeyPoint], angle: int) -> Tuple[List[cv2.KeyPoint], List[int]]:
    """Xoay các điểm đặc trưng và trả về các điểm nằm trong kích thước ảnh sau khi xoay"""
    matrix_rotation = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1)
    kps = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

    # Đảm bảo kps có 2 chiều trước khi kết hợp
    if kps.ndim == 1:
        kps = kps.reshape(-1, 2)  # Đảm bảo kps có dạng N x 2 nếu nó là 1D

    kps = np.hstack([kps, np.ones((len(kps), 1))])  # Thêm cột toàn bộ giá trị 1 vào
    rotated_kps = (matrix_rotation @ kps.T).T  # Áp dụng ma trận xoay

    rotated_keypoints = []
    idx = []

    # Lọc các keypoint nằm trong phạm vi của ảnh và giữ lại chỉ số của chúng
    for i, kp in enumerate(rotated_kps):
        if 0 <= kp[0] < size[0] and 0 <= kp[1] < size[1]:
            rotated_keypoints.append(cv2.KeyPoint(kp[0], kp[1], 1))
            idx.append(i)  # Lưu lại chỉ số của keypoint hợp lệ

    return rotated_keypoints, idx
def convert_to_keypoints(point_tuples: List[Tuple[float, float]]) -> List[cv2.KeyPoint]:
    """Chuyển đổi các điểm dạng tuple thành KeyPoint của OpenCV"""
    return [cv2.KeyPoint(y, x, 1) for x, y in point_tuples]
def match_features(image1, image2, original_kp, rotated_kp, method):
    """Khớp các đặc trưng giữa hai ảnh"""
    model = models[method]
    matcher = matchers[method]
    
    descriptors1 = model.compute(image1, original_kp)[1]
    descriptors2 = model.compute(image2, rotated_kp)[1]
    
    if method in ["SIFT", "SuperPoint"]:
        matches = matcher.knnMatch(descriptors2, descriptors1, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance and m.trainIdx == m.queryIdx]
    else:
        matches = matcher.match(descriptors2, descriptors1)
        good_matches = [m for m in matches if m.queryIdx == m.trainIdx]

    return good_matches

def draw_colored_matches(image1, keypoints1, image2, keypoints2, matches):
    # Tạo hình ảnh ghép chứa cả hai ảnh để vẽ các match
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    matched_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    matched_image[:h1, :w1] = image1
    matched_image[:h2, w1:] = image2

    # Lưu các chỉ số của keypoints đã match
    matched_indices1 = {match.queryIdx for match in matches}
    matched_indices2 = {match.trainIdx for match in matches}

    # Vẽ tất cả các keypoints không match với màu đỏ
    for i, kp in enumerate(keypoints1):
        if i not in matched_indices1:
            pt = tuple(map(int, kp.pt))
            cv2.circle(matched_image, pt, 3, (255,0, 0 ), -1)

    for i, kp in enumerate(keypoints2):
        if i not in matched_indices2:
            pt = (int(kp.pt[0] + w1), int(kp.pt[1]))  # Điều chỉnh tọa độ cho image2
            cv2.circle(matched_image, pt, 3, (255,0, 0 ), -1)

    # Vẽ các keypoints match với đường nối màu xanh lá cây
    for match in matches:
        pt1 = tuple(map(int, keypoints1[match.queryIdx].pt))
        pt2 = (int(keypoints2[match.trainIdx].pt[0] + w1), int(keypoints2[match.trainIdx].pt[1]))

        # Vẽ đường nối và keypoint màu xanh lá cho match
        color = (0, 255, 0)  # Màu xanh lá cây
        cv2.line(matched_image, pt1, pt2, color, 1)
        cv2.circle(matched_image, pt1, 3, color, -1)
        cv2.circle(matched_image, pt2, 3, color, -1)

    return matched_image