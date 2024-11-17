import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from utils.instance_Search import (Dataset, load_codebook_and_tfidf,load_image_index, extract_features_superpoint, 
                                   calculate_query_tfidf,find_similar_images,resize_to_fixed_height)

st.set_page_config(
    page_title="Hue Tran_Instance Search",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(
    "Tìm kiếm ảnh truy vấn"
)

def display_dataset():
    st.header("1. Dataset")
    st.write("""
        - Tập dữ liệu được kết hợp từ **Common Objects in Context (COCO)** gồm **5000** ảnh và 
             **The Scene UNderstanding (SUN)** gồm **800** ảnh.     
        - Biểu đồ phân bổ kích thước các ảnh trong tập dữ liệu.
""")
    image_sizes_df = pd.read_csv("./data/instance_Search/image_sizes.csv")

    widths = image_sizes_df["Width"]
    heights = image_sizes_df["Height"]

    # Tạo biểu đồ chấm
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(widths, heights, alpha=0.7, color='green')

    
    ax.set_xlabel("Chiều rộng (pixel)")
    ax.set_ylabel("Chiều cao (pixel)")
    ax.set_title("Biểu đồ phân bố kích thước ảnh trong Dataset")

    # Hiển thị biểu đồ trong Streamlit
    st.columns([0.5, 2.5, 0.5])[1].pyplot(fig)
    

    st.write("""
        - Một số hình ảnh trong tập dữ liệu
""")
    image_paths = []
    for filename in os.listdir(Dataset):
        if filename.endswith((".jpg", ".png", ".jpeg")):  # Điều chỉnh nếu cần
            image_paths.append(os.path.join(Dataset, filename))

    # Sắp xếp các ảnh theo chiều cao
    image_sizes_df['Filepath'] = image_paths
    # Lấy chiều rộng và chiều cao
    widths = image_sizes_df['Width']
    heights = image_sizes_df['Height']

    # Tính sự chênh lệch giữa chiều rộng và chiều cao
    image_sizes_df['Size Difference'] = abs(widths - heights)

    # Chọn những ảnh có sự chênh lệch chiều rộng và chiều cao nhỏ hơn 10% chiều rộng
    threshold = 0.1  # Ngưỡng độ chênh lệch 10% chiều rộng
    filtered_df = image_sizes_df[image_sizes_df['Size Difference'] < widths * threshold]


    # Lấy danh sách ảnh tương tự
    selected_images = filtered_df.head(8)  # Chọn 8 ảnh đầu tiên (hoặc bất kỳ số lượng nào)

    # Chia thành các nhóm 4 ảnh mỗi hàng
    num_images_per_row = 4
    rows = [selected_images.iloc[i:i + num_images_per_row] for i in range(0, len(selected_images), num_images_per_row)]

    # Hiển thị ảnh trong mỗi hàng
    for row in rows:
        cols = st.columns(len(row))  # Tạo các cột cho mỗi ảnh trong hàng
        for col, (_, row_data) in zip(cols, row.iterrows()):
            img = plt.imread(row_data['Filepath'])  # Đọc ảnh từ đường dẫn
            col.image(img, use_column_width=True)

def display_method():
   st.header("2. Thuật toán")

   st.image("./data/instance_Search/Instance_search.png", caption="PIPELINE tìm kiếm hình ảnh sử dụng Superpoint, BoVW và TF-IDF")
   st.markdown("""
        ### Mô tả các bước:
               
        (1). **Trích xuất đặc trưng Superpoint** từ tập dữ liệu hình ảnh.  
        (2). **Phân cụm K-means** để tạo các cụm đặc trưng.  
        (3). **Xây dựng từ điển trực quan** từ các cụm.  
        (4). **Biểu diễn hình ảnh bằng từ trực quan**: Mỗi hình ảnh được mô tả bởi các từ trực quan.  
        (5). **Tạo biểu đồ histogram**: Đếm tần suất các từ trực quan trong từng hình ảnh.  
        (6). **Chuẩn hóa TF-IDF** để tối ưu biểu đồ histogram.  
        (7). **Trích xuất đặc trưng Superpoint** cho ảnh truy vấn.  
        (8),(9). **Mapping đặc trưng của ảnh truy vấn vào BoVW**: Tạo histogram BoVW cho ảnh truy vấn.  
        (10). **Chuẩn hóa TF-IDF cho histogram truy vấn**.  
        (11), (12). **Tính độ tương đồng cosine** giữa histogram của ảnh truy vấn và các hình ảnh trong tập dữ liệu.  
        (13). **Xây dựng ma trận tương đồng** và **xác định top-K** hình ảnh có độ tương đồng cao nhất.  
        (14). **Truy xuất top-K hình ảnh gốc** từ tập dữ liệu.        
        (15). **Hiển thị hình ảnh truy xuất** với độ tương đồng.
        """)



def display_app():
  st.header("3. Ứng dụng")

  col1, col2, col3 = st.columns(3)

  with col1:
      uploaded_file = st.file_uploader("Vui lòng tải lên ảnh cần tìm kiếm", type=["jpg", "jpeg", "png"])

  with col2:
      k = st.selectbox("Chọn số lượng cluster", [50, 100, 150, 200, 250, 300], index=2)

  with col3:
      num_results = st.slider("Chọn số lượng ảnh trả về", min_value=1, max_value=50, value=5, step = 1)

# Tải codebook và tf-idf matrix
  codebook, tfidf_matrix = load_codebook_and_tfidf(k)
  image_index_to_filename = load_image_index()

  if uploaded_file:
    # Đọc và hiển thị ảnh truy vấn
    query_image = np.frombuffer(uploaded_file.read(), np.uint8)
    query_image = cv2.imdecode(query_image, cv2.IMREAD_COLOR)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB) 
    st.image(query_image, caption="Ảnh truy vấn")

    # Trích xuất đặc trưng từ ảnh truy vấn
    descriptors = extract_features_superpoint(query_image)
  # Nút Tìm kiếm
  if st.button("Tìm kiếm"):
    if descriptors is not None:
        # Tính tf-idf vector cho ảnh truy vấn
        idf = np.log(tfidf_matrix.shape[0] / (np.sum(tfidf_matrix > 0, axis=0)+1))
        query_tfidf = calculate_query_tfidf(descriptors, codebook, idf)

        # Tìm các ảnh tương tự
        top_indices, scores = find_similar_images(query_tfidf, tfidf_matrix, top_n=num_results)

        # Hiển thị kết quả tìm kiếm
        st.write("Kết quả tìm kiếm:")
        if len(top_indices) > 0:
          cols_per_row = 5  # Số ảnh trên mỗi hàng
          rows = (len(top_indices) + cols_per_row - 1) // cols_per_row  # Số hàng cần thiết

          for row in range(rows):
              cols = st.columns(cols_per_row)
              for col_idx in range(cols_per_row):
                  img_idx = row * cols_per_row + col_idx
                  if img_idx < len(top_indices):
                      idx = top_indices[img_idx]
                      score = scores[img_idx]
                      filename = image_index_to_filename.get(idx, "Unknown")
                      img_path = os.path.join(Dataset, filename)
                      img = cv2.imread(img_path)
                      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                      # Hiển thị ảnh trong từng cột
                      with cols[col_idx]:
                          st.image(img, caption=f"Độ tương đồng = {score:.2f}", use_column_width=True)

    else:
        st.write("Không thể trích xuất đặc trưng từ ảnh truy vấn. Vui lòng thử lại với ảnh khác.")

display_dataset()
display_method()
display_app()