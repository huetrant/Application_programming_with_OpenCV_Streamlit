import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import torch

st.set_page_config(
    page_title="Hue Tran_Object Tracking",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(
    "OpenCV Object Tracking Algorithms"
)
DATA_DIR ="./data/object_tracking/"
def display_method():
  st.header("1. Thuật toán CSRT - Channel and Spatial Reliability Tracker")
  st.subheader("1.1. Giới thiệu")
  st.write("""
  [**Thuật toán CSRT** (Discriminative Correlation Filter with Channel and Spatial Reliability)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lukezic_Discriminative_Correlation_Filter_CVPR_2017_paper.pdf)
           được Alan Lukežič, Tomáš Vojı́ř, Luka Čehovin Zajc, Jiřı́ Matas và Matej Kristan giới thiệu là một cải tiến 
  của bộ lọc tương quan phân biệt (DCF), được thiết kế để theo dõi đối tượng hiệu quả trong các tình huống phức tạp 
  như che khuất, biến đổi hình dạng và nhiễu từ nền. Thuật toán này kết hợp hai khái niệm chính: 
  **độ tin cậy không gian** và **độ tin cậy kênh**, nhằm cải thiện độ chính xác và tính ổn định của quá trình theo dõi.
  """)


  # Implementation Steps
  st.subheader("1.2. Các bước thực hiện")
  st.subheader("1.2.1. Xây dựng bản đồ độ tin cậy không gian")
  st.write("""
  Bản đồ độ tin cậy không gian xác định các pixel đáng tin cậy trên đối tượng để học và theo dõi.
  **Quy trình:**
  - **Tính xác suất pixel thuộc foreground:**
    - Dựa trên mô hình màu foreground/background.
    - Áp dụng định lý Bayes
  - **Tăng cường tính nhất quán không gian:** Áp dụng thuật toán Markov Random Field để làm mịn phân vùng.
  - **Xử lý biên:** Thực hiện giãn nở hình thái học (morphological dilation) để giữ lại các pixel quan trọng ở biên đối tượng.
  """)

  # 2. Học bộ lọc tương quan
  st.subheader("1.2.2. Học bộ lọc tương quan có ràng buộc không gian")
  st.write("""
  Bản đồ độ tin cậy không gian được sử dụng trong quá trình học bộ lọc tương quan để chỉ học từ các vùng đáng tin cậy.
  **Quy trình tối ưu hóa:**
  - Sử dụng phương pháp **Augmented Lagrangian** và thuật toán **ADMM** để giải bài toán tối ưu hóa:
    - Cập nhật biến kép dựa trên ràng buộc không gian.
    - Cập nhật bộ lọc tương quan qua phép biến đổi Fourier nghịch.
    - Cập nhật nhân tử Lagrange.
  - Hầu hết các phép tính được thực hiện trong miền tần số (frequency domain) để tăng tốc độ.
  """)

  # 3. Đánh giá độ tin cậy kênh
  st.subheader("1.2.3. Đánh giá độ tin cậy kênh")
  st.write("""
  Mỗi kênh đặc trưng được đánh giá độ tin cậy trong hai giai đoạn:
  - **Trong quá trình học:** Đo lường sức mạnh phân biệt dựa trên phản hồi cực đại.
  - **Trong quá trình định vị:** Đánh giá mức độ rõ ràng của đỉnh phản hồi để tránh nhiễu từ các đối tượng gần đối tượng mục tiêu.
  """)

  # 4. Quy trình theo dõi
  st.subheader("1.2.4. Quy trình theo dõi đối tượng")
  st.write("""
  - **Xác định vị trí:**
    - Tính phản hồi trên từng kênh.
    - Trọng số hóa phản hồi và xác định vị trí đỉnh tương ứng với đối tượng.
  - **Cập nhật:**
    - Cập nhật mô hình màu.
    - Xây dựng lại bản đồ độ tin cậy không gian.
    - Tính bộ lọc mới và cập nhật trọng số độ tin cậy kênh.
  """)
  # Practical Considerations
 

def display_example():
    st.header("2. Ví dụ minh họa")  # Sử dụng `st` để tạo header

    # Đường dẫn tới file video
    video_path1 = os.path.join(DATA_DIR, "tracking_output.mp4")
    video_path2 = os.path.join(DATA_DIR, "people_tracking_output.mp4")
    col1,col2 = st.columns(2)
    with col1:
       # Kiểm tra file tồn tại
      if os.path.exists(video_path1):
          with open(video_path1, "rb") as video_file:
              video_bytes = video_file.read()
          st.video(video_bytes) 
          st.markdown("**Video 1**") # Hiển thị video
      else:
          st.error("File video không tồn tại!")
    with col2:
       # Kiểm tra file tồn tại
      if os.path.exists(video_path2):
          with open(video_path2, "rb") as video_file:
              video_bytes = video_file.read()
          st.video(video_bytes)
          st.markdown("**Video 2**")  # Hiển thị video
      else:
          st.error("File video không tồn tại!")

def display_challenge():
  st.header("3. Thách thức")
  data = {
    "Thách thức": [
        "Background Clutters", 
        "Illumination Variations", 
        "Occlusion", 
        "Fast Motion"
    ],
    "Ưu điểm của CSRT": [
        "- Phân biệt tốt giữa đối tượng và nền nhờ bộ lọc không gian và kênh.",
        "- Linh hoạt trong việc xử lý biến đổi ánh sáng nhờ khả năng học và cập nhật bộ lọc.",
        "- Có thể theo dõi đối tượng khi bị che khuất tạm thời, duy trì độ tin cậy với phần còn lại.",
        "- Bộ lọc được cập nhật nhanh chóng, theo kịp chuyển động vừa phải."
    ],
    "Hạn chế của CSRT": [
        "- Khó khăn với nền phức tạp hoặc chuyển động mạnh, dễ bị nhầm lẫn.",
        "- Không hiệu quả với biến đổi ánh sáng mạnh mẽ, làm giảm độ chính xác của các đặc trưng HOG và SIFT.",
        "- Khó khăn khi đối tượng bị che khuất lâu dài hoặc hoàn toàn, không thể theo dõi chính xác.",
        "- Không theo kịp chuyển động quá nhanh, có thể gây trễ trong các tình huống chuyển động giật cục."
    ]
}

 # Chuyển dữ liệu thành DataFrame
  df = pd.DataFrame(data)

  # Tạo CSS để căn giữa tiêu đề cột và ẩn index
  css = """
  <style>
      thead th {
          text-align: center;
      }
      tbody td {
          text-align: left;
      }
  </style>
  """

  # Áp dụng CSS
  st.markdown(css, unsafe_allow_html=True)

  # Hiển thị bảng mà không có index và căn giữa tiêu đề cột
  st.markdown(df.to_html(index=False, escape=False), unsafe_allow_html=True)


display_method()
display_example()
display_challenge()