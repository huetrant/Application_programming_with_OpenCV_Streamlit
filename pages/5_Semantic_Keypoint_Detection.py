import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.semantic_keypoint_detection_function import draw_points

st.set_page_config(
    page_title="Hue Tran _ Semantic Keypoint Detection với thuật toán SIFT và ORB",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Semantic Keypoint Detection bằng thuật toán SIFT và ORB")


DATA = "./data/semantic_keypoint_detection/"
DATA_DIR = os.path.join(DATA,"synthetic_shapes_datasets")
DATATYPES = [
    os.path.join(DATA_DIR, "draw_checkerboard"),
    os.path.join(DATA_DIR, "draw_cube"),
    os.path.join(DATA_DIR, "draw_ellipses"),
    os.path.join(DATA_DIR, "draw_lines"),
    os.path.join(DATA_DIR, "draw_multiple_polygons"),
    os.path.join(DATA_DIR, "draw_polygon"),
    os.path.join(DATA_DIR, "draw_star"),
    os.path.join(DATA_DIR, "draw_stripes"),
]

@st.fragment()
def display_datasets():
    st.header("1. Synthetic Shapes Datasets")
    st.write(
        """
        - **Synthetic Shapes Datasets** là các tập dữ liệu hình ảnh nhân tạo được tạo ra 
        với mục đích phục vụ các bài toán thị giác máy tính như phân loại ảnh, phát hiện vật thể, phân đoạn ảnh, 
        và các bài toán liên quan khác. Các tập dữ liệu này thường chứa hình ảnh các hình dạng hình học cơ bản như hình tròn, hình vuông, tam giác, 
        và nhiều hình dạng khác. 
        - Dataset có tổng cộng có $4000$ ảnh mẫu.
        - Mỗi loại hình học có $500$ ảnh mẫu, mỗi ảnh mẫu có kích thước $160$ x $120$ pixels.
        - Tập dữ liệu gồm $8$ loại hình học cơ bản như sau: 
    """
    )

    cols1 = st.columns(4)
    cols2 = st.columns(4)

    for i in range(4):
        # Vòng lặp đầu tiên xử lý cols1
        points = np.load(os.path.join(DATATYPES[i], "points", "1.npy"))
        image = cv2.imread(os.path.join(DATATYPES[i], "images", "1.png"))
        # Hiển thị ảnh
        cols1[i].image(draw_points(image, points), use_column_width=True)

        # Hiển thị caption in đậm
        caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
        cols1[i].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)

        # Vòng lặp thứ hai xử lý cols2
        points = np.load(os.path.join(DATATYPES[i + 4], "points", "1.npy"))
        image = cv2.imread(os.path.join(DATATYPES[i + 4], "images", "1.png"))
        # Hiển thị ảnh
        cols2[i].image(draw_points(image, points), use_column_width=True)

        # Hiển thị caption in đậm
        caption = DATATYPES[i + 4].replace('\\', '/').split('/')[-1].replace('draw_', '')
        cols2[i].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)


@st.fragment()
def display_metric():
    st.header("3. Độ đo - Metric")
    st.markdown("""
        - Một keypoint được coi là phát hiện đúng nếu:
        $$d \leq 5\ pixels$$
        
            - Trong đó khoảng cách **Euclidean**:
        $$d = \sqrt{(x_{gt} - x_{pred})^2 + (y_{gt} - y_{pred})^2}$$

            - Với:
                - $(x_{gt}, y_{gt})$: Tọa độ của keypoint ground truth (gt)
                - $(x_{pred}, y_{pred})$: Tọa độ của keypoint phát hiện

        - Hai độ đo **Precision** và **Recall** được sử dụng để đánh giá kết quả phát hiện keypoint của hai thuật toán **SIFT** và **ORB**
        với công thức như sau:
        """)
    st.columns([1, 3, 1])[1].image(
        os.path.join(DATA, "Pre_re.png"),
        width=200,
        use_column_width=True,
        caption="Precision và Recall",
    )


display_datasets()
display_metric()