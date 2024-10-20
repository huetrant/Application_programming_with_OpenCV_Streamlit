import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.neighbors import KNeighborsClassifier


import cv2
import numpy as np
import os


# Hàm để tải và xử lý ảnh
def preprocess_image(image_file):
    # Chuyển đổi đối tượng file sang mảng NumPy
    img = np.array(image_file)

    # Lấy kích thước ảnh gốc
    original_height, original_width = img.shape[:2]

    # Kích thước mong muốn cho việc phát hiện
    desired_size = 128

    # Tính tỷ lệ thu nhỏ
    if original_width > original_height:
        scale = desired_size / original_width
    else:
        scale = desired_size / original_height

    # Resize ảnh giữ tỷ lệ
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_img = cv2.resize(img, (new_width, new_height))

    # Tạo vùng đệm (padding) để đưa về kích thước 128x128
    padded_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)

    # Tính toán vị trí để đặt ảnh đã resize vào giữa vùng đệm
    x_offset = (desired_size - new_width) // 2
    y_offset = (desired_size - new_height) // 2

    # Đặt ảnh đã resize vào vùng đệm
    padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    # Chuyển ảnh đã padding sang ảnh xám
    gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)

    return padded_img, gray, scale
# Hàm tính IoU
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Tính tọa độ của góc trên bên trái và góc dưới bên phải
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Tính diện tích của phần giao nhau
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Tính diện tích của hai bounding box
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Tính diện tích hợp
    union_area = box1_area + box2_area - intersection_area

    # Tính IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())
def knn (train,test,k=5):
    dist=[]
    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=distance(test,ix)
        dist.append([d,iy])
    dk=sorted(dist,key=lambda x: x[0])[:k]
    labels=np.array(dk)[:,-1]
    output=np.unique(labels,return_counts=True)
    index=np.argmax(output[1])
    return output[0][index]

import numpy as np
import os

def load_and_prepare_data(face_data, non_face_data):
    """
    Đọc và chuẩn bị dữ liệu từ các file npy, sau đó tạo ra trainset kết hợp từ face_data và non_face_data.

    Parameters:
    working_dir (str): Đường dẫn đến thư mục làm việc chứa dữ liệu.

    Returns:
    trainset (ndarray): Mảng kết hợp từ dữ liệu và nhãn của face_data và non_face_data.
    """

    # Tạo nhãn (label) cho face và non_face
    face_labels = np.ones((face_data.shape[0],))  # 1 cho face
    non_face_labels = np.zeros((non_face_data.shape[0],))  # 0 cho non-face

    # Kết hợp dữ liệu của face và non_face
    data = np.concatenate((face_data, non_face_data), axis=0)

    # Kết hợp nhãn của face và non_face
    labels = np.concatenate((face_labels, non_face_labels), axis=0).reshape((-1, 1))

    # Tạo trainset bằng cách kết hợp dữ liệu và nhãn
    trainset = np.concatenate((data, labels), axis=1)

    return trainset
