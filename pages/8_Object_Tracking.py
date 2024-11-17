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
  st.subheader("1.1. Cơ sở lý thuyết của CSRT")
  st.write("""
  - **Channel and Spatial Reliability**
    - **Channel Reliability:** CSRT tận dụng thông tin từ các kênh màu (RGB hoặc chuyển đổi) để xử lý sự thay đổi về ánh sáng, 
  giúp tăng độ chính xác trong theo dõi đối tượng ngay cả khi điều kiện chiếu sáng thay đổi.        
    - **Spatial Reliability**: Để đối phó với occlusion và non-rigid deformations, CSRT dự đoán chuyển động dựa trên mối liên hệ với các khung hình trước, 
  đảm bảo theo dõi ổn định khi đối tượng bị che khuất hoặc thay đổi hình dạng.
  - **Discriminative Correlation Filter (DCF):**  CSRT sử dụng DCF để định vị và ước lượng hình dạng đối tượng. Bộ lọc được huấn luyện từ các mẫu tích cực (đối tượng) và tiêu cực (nền, nhiễu) 
  và được cập nhật liên tục trong quá trình theo dõi.
  """)


  # Implementation Steps
  st.subheader("1.2. Các bước thực hiện")
  col1, col2 = st.columns(2)
  with col1:
    st.write("""
        - **Sample Generation**
          - **Positive/Negative Samples**: Thu thập mẫu tích cực (đối tượng) và tiêu cực (nền/nhiễu) để huấn luyện.
          - **Feature Extraction**: Sử dụng HOG, SIFT hoặc đặc trưng tương tự để trích xuất thông tin từ các mẫu.
            
        - **Training**
          - **Kernel Matrix**: Xây dựng ma trận nhân (ví dụ Gaussian) từ các đặc trưng trích xuất.
          - **Desired Response**: Tạo phản hồi mong muốn, thường là đỉnh Gaussian tại trung tâm đối tượng.
             
        - **Learning the Filter**
          - **Ridge Regression**:  CSRT áp dụng kỹ thuật ridge regression (hồi quy rìa) để học bộ lọc tương quan, giảm sai số giữa phản hồi thực tế và mong muốn, cải thiện độ chính xác.
        
        - **Tracking**
          - **Feature Extraction**: Tiếp tục trích xuất đặc trưng từ khung hình mới.
          - **Correlation Filtering**: Dùng bộ lọc tương quan đã học được áp dụng lên các khung hình mới và tạo ra bản đồ phản hồi. Đỉnh của bản đồ này cho biết vị trí có xác suất cao nhất của đối tượng.
          - **Locate the Target**: CSRT tìm kiếm đỉnh cao nhất trong bản đồ phản hồi để xác định vị trí của đối tượng trong khung hình hiện tại.
    """)
    with col2:
      img = Image.open(os.path.join(DATA_DIR, "flowchart.png"))
      st.columns([1, 2, 1])[1].image(
          img,
          caption="CSRT Flowchart",
      )
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