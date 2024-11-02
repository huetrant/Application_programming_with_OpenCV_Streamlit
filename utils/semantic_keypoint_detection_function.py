import os
import cv2
import numpy as np
import streamlit as st



# Hàm vẽ các điểm đặc trưng (keypoints) lên ảnh
def draw_points(
    image: np.ndarray, points: np.ndarray, color=(0, 255, 0), thickness=2, radius=1, outer_radius=1
):
    for point in points:
        # Vẽ vòng tròn đầu tiên (vòng tròn nhỏ) với bán kính nhỏ hơn
        cv2.circle(image, (int(point[1]), int(point[0])), radius, color, thickness)
        
        # Vẽ vòng tròn thứ hai (vòng tròn lớn) với bán kính lớn hơn
        cv2.circle(image, (int(point[1]), int(point[0])), outer_radius, color, thickness)
        
    return image
