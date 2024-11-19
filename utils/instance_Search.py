import streamlit as st
import pickle
import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.vq import vq
from utils.SuperPoint import SuperPointFrontend

MODEL_DIR = "./models/"
SP_Model_DIR = os.path.join(MODEL_DIR,"superpoint")
Instrance_Model_DIR = os.path.join(MODEL_DIR,"instance_Search")

DATA_DIR = "./data/instance_Search"
Dataset = os.path.join(DATA_DIR, "val2017")

superpoint = SuperPointFrontend(
      weights_path = os.path.join(SP_Model_DIR, "superpoint_v1.pth"),  # Đường dẫn tới tệp trọng số của SuperPoint
      nms_dist=4,
      conf_thresh=0.015,
      nn_thresh=0.7,
      )
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh grayscale
    image = image.astype(np.float32) / 255.0  # Chuẩn hóa ảnh
    return image

# Trích xuất đặc trưng sử dụng SuperPoint

def extract_features_superpoint(image, model = superpoint):
    image = preprocess_image(image)
    _, descriptors, _ = model.run(image)
    
    # Kiểm tra nếu descriptors là None
    if descriptors is None:
        print("Không thể trích xuất descriptors từ ảnh.")
        return None
    return descriptors.T  

@st.cache_resource
# Hàm tải codebook và TF-IDF matrix
def load_codebook_and_tfidf(k):
    codebook_path = os.path.join(Instrance_Model_DIR, f"codebook{k}.pkl")
    tfidf_path = os.path.join(Instrance_Model_DIR, f"tfidf_matrix_{k}.pkl")
    
    with open(codebook_path, "rb") as f:
        codebook = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf_matrix = pickle.load(f)
    
    return codebook, tfidf_matrix

# Hàm tải chỉ số và tên ảnh
@st.cache_resource
def load_image_index():
    index_path = os.path.join(Instrance_Model_DIR, "image_index_to_filename.pkl")
    with open(index_path, "rb") as f:
        image_index = pickle.load(f)
    return image_index


# Tính toán TF-IDF vector cho ảnh truy vấn
def calculate_query_tfidf(descriptors, codebook, idf):
    visual_words, _ = vq(descriptors, codebook)
    frequency_vector = np.zeros(len(codebook))
    for word in visual_words:
        frequency_vector[word] += 1
    query_tfidf = frequency_vector * idf
    return query_tfidf

# Tìm các ảnh tương tự
def find_similar_images(query_tfidf, tfidf_matrix, top_n=5):
    query_tfidf = query_tfidf.reshape(1, -1)
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_scores = similarities[top_indices]
    return top_indices, top_scores

def resize_to_fixed_height(img, target_height=300):
    aspect_ratio = img.shape[1] / img.shape[0]  # Tính tỷ lệ chiều rộng / chiều cao
    target_width = int(target_height * aspect_ratio)  # Tính chiều rộng tương ứng
    resized_img = cv2.resize(img, (target_width, target_height))  # Resize ảnh
    return resized_img, target_width
