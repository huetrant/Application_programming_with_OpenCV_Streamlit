import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import torch

from utils.semantic_keypoint_detection_function import draw_points
from utils.SuperPoint import SuperPointFrontend
from utils.matching_keypoint import read_image, rotate_image, rotate_keypoints, match_features,convert_to_keypoints, draw_colored_matches

st.set_page_config(
    page_title="Hue_Tran_Matching Keypoint with SIFT, ORB, SuperPoint",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(
    "Matching Keypoint với SIFT, ORB và SuperPoint trên các góc xoay ảnh"
)

DATA_DIR = "./data/semantic_keypoint_detection/"
DATA_MATCHING = "./data/matching_keypoint/"
DATASET_DIR = os.path.join(DATA_DIR,"synthetic_shapes_datasets")
DATATYPES = [
    os.path.join(DATASET_DIR, "draw_checkerboard"),
    os.path.join(DATASET_DIR, "draw_cube"),
    os.path.join(DATASET_DIR, "draw_ellipses"),
    os.path.join(DATASET_DIR, "draw_lines"),
    os.path.join(DATASET_DIR, "draw_multiple_polygons"),
    os.path.join(DATASET_DIR, "draw_polygon"),
    os.path.join(DATASET_DIR, "draw_star"),
    os.path.join(DATASET_DIR, "draw_stripes"),
]

MODEL_DIR = "./models/superpoint"

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

TYPE_MAPPING = {
    "Checkerboard": 0,
    "Cube": 1,
    "Lines": 3,
    "Multiple Polygons": 4,
    "Polygon": 5,
    "Star": 6,
    "Stripes": 7
    }
@st.fragment()
def display_methods():
  st.header("1. Giới thiệu SuperPoint")
  col1, col2 = st.columns(2)

  with col1:
      st.write("""
        SuperPoint được **Daniel DeTone**, **Tomasz Malisiewicz**, **Andrew Rabinovich** giới thiệu vào năm 2018 
        trong bài báo [SuperPoint: Self-Supervised Interest Point Detection and Description](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf), 
              là một mô hình deep learning dùng để phát hiện và mô tả các điểm đặc trưng trong ảnh.
        Gồm các phần chính sau:
        - **Encoder (Bộ mã hóa)**:
          - Đầu vào là ảnh kích thước **W × H**.
          - Dùng các lớp tích chập để trích xuất đặc trưng từ ảnh, tạo ra một **feature map**.
        - **Feature Point Decoder (Bộ giải mã điểm đặc trưng)**:
          - Feature map từ Encoder được đưa vào và giảm kích thước xuống **W/8 × H/8**.
          - Sử dụng các lớp Softmax để tạo ra một **heatmap** có kích thước **W × H**, với mỗi điểm trên heatmap chỉ ra xác suất của các keypoints.
          - Các điểm có xác suất cao nhất trên heatmap sẽ được chọn làm các keypoints.
        - **Descriptor Decoder (Bộ giải mã mô tả đặc trưng)**:
          - Dùng **feature map** từ Encoder, giảm kích thước xuống **W/16 × H/16**.
          - Với mỗi keypoint, sử dụng **bilinear sampling** để tạo ra một **vector descriptor** cho mỗi điểm.
        """)

  with col2:
      st.image("./data/matching_keypoint/superpoint.png", caption='Kiến trúc mô hình SuperPoint', use_column_width=True)
  
  

  # Khởi tạo SuperPointFrontend
  superpoint = SuperPointFrontend(
      weights_path = os.path.join(MODEL_DIR, "superpoint_v1.pth"),  # Đường dẫn tới tệp trọng số của SuperPoint
      nms_dist=4,
      conf_thresh=0.015,
      nn_thresh=0.7,
      )

  # Hàm minh họa
  st.markdown("##### Minh họa **SuperPoint** trên Synthetic Shapes Datasets")

  # Hiển thị 8 ảnh với 2 dòng, mỗi dòng 4 ảnh
  cols = st.columns(4)
  for i in range(8):
      image = cv2.imread(os.path.join(DATATYPES[i], "images", "6.png"))
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0  # SuperPoint yêu cầu ảnh kiểu float32

      # Sử dụng SuperPoint để phát hiện keypoints và descriptors
      keypoints, descriptors, heatmap = superpoint.run(gray)

      # Vẽ các điểm phát hiện và ground truth
      ground_truth = np.load(os.path.join(DATATYPES[i], "points", "6.npy"))
      image = draw_points(
          image, [(kp[1], kp[0]) for kp in keypoints.T], (255, 0, 0), 1, 1, 3

      )
      image = draw_points(image, ground_truth, (0, 255, 0),1)

      caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
      cols[i % 4].image(
          image,
          use_column_width=True,  
      )
      cols[i % 4].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)
      # Chuyển sang cột mới sau mỗi 4 ảnh
      if (i + 1) % 4 == 0:
          cols = st.columns(4)  # Tạo lại các cột sau mỗi dòng 4 ảnh

  st.write(
      """
      - Các hình tròn màu **:green[xanh lục]** là **keypoints** ground truth.
      - Các hình tròn màu **:red[đỏ]** là **keypoints** được phát hiện với ngưỡng bán kính đánh giá là $3$ pixels bởi thuật toán **SuperPoint**.
      """
  )

def display_experiment():
    st.header("2. Thiết lập thí nghiệm")
    st.write("""
    1. **Dataset**: Sử dụng tập **Synthetic Shapes Dataset** gồm **7** loại hình học,
              mỗi loại hình học gồm **500** ảnh. Tổng **3500** ảnh mẫu với kích thước $160$ x $120$ pixels.
    2. **Keypoint groundtruth** được sử dụng để đánh giá mức độ matching của 3 thuật toán **SIFT**, **ORB** và **SuperPoint**.
    3. Đánh giá mức độ **Matching Keypoint** với **SIFT**, **ORB** và **SuperPoint** 
        trên các **góc xoay ảnh** với các góc xoay từ $0\degree$ đến $350\degree$ với mỗi bước nhảy là $10\degree$
    """)
    # Tạo hai cột
    col1, col2 = st.columns(2)

    # Cột 1: SIFT và SuperPoint
    with col1:
        st.markdown("<div style='text-align: center;'><b>Đối với SIFT và SuperPoint</b></div>", unsafe_allow_html=True)
        st.write("""
            - **Brute-Force Matching**: Sử dụng phương pháp **Brute-Force Matching** để tìm các cặp keypoint matching giữa hai ảnh.
            - **Tính khoảng cách giữa các descriptors**: Các descriptors của **SIFT** và **SuperPoint** là các vector liên tục trong không gian 
                 **Euclidean**, sử dụng **cv2.NORM_L2** để tính khoảng cách Euclidean. 
                 Điều này giúp đo lường sự khác biệt giữa các descriptors một cách chính xác hơn.
            - **Lowe's ratio test**: Sau khi tìm được các cặp matching, ta áp dụng **Lowe's ratio test** với tỷ lệ $ratio = 0.75$ để lọc các matches 
                 không tốt. Quy trình này so sánh khoảng cách giữa match tốt nhất và match thứ hai cho mỗi keypoint. 
                 Nếu tỷ lệ giữa hai khoảng cách này quá lớn, match đó sẽ bị loại bỏ vì có thể không đủ đặc trưng.
        """)

    # Cột 2: ORB
    with col2:
        st.markdown("<div style='text-align: center;'><b> Đối với ORB</b></div>", unsafe_allow_html=True)
        st.write("""
            - **Brute-Force Matching**: Giống như **SIFT** và **SuperPoint**, **ORB** cũng sử dụng **Brute-Force Matching** 
                 để tìm các cặp keypoint matching.
            - **Tính khoảng cách giữa các descriptors**: Descriptors của **ORB** là nhị phân (chứa giá trị 0 và 1), sử dụng **cv2.NORM_HAMMING**
                  để tính khoảng cách **Hamming**. Đây là phương pháp đo sự khác biệt giữa các vectors nhị phân, thích hợp với đặc điểm của descriptors nhị phân.
            - **crossCheck = True**: Điều này đảm bảo rằng chỉ giữ lại các matches đối xứng, nghĩa là nếu descriptor $A$ trong ảnh 1 
                 khớp với descriptor $B$ trong ảnh 2, thì descriptor $B$ cũng phải khớp lại với descriptor $A$ trong ảnh 1. 
                 Điều này giúp loại bỏ các cặp matching không chính xác hoặc nhiễu làm tăng độ chính xác.
        """)
    st.markdown("""
    3. Sử dụng độ đo **Accuracy** để đánh giá matching keypoint sau khi xoay ảnh:
                
        $\\text{Accuracy} = \\frac{\\text{Số keypoint matching đúng}}{\\text{Tổng số keypoint được matching}}$
    """)

# Load dữ liệu
accuracy_sift = np.load(os.path.join(DATA_MATCHING,'accuracy_sift.npy'), allow_pickle=True)
accuracy_orb = np.load(os.path.join(DATA_MATCHING,'accuracy_orb.npy'), allow_pickle=True)
accuracy_superpoint = np.load(os.path.join(DATA_MATCHING,'accuracy_superpoint.npy'), allow_pickle=True)

def display_result():
    st.header("3. Kết quả")
    st.subheader("3.1. Biểu đồ Average Accuracy")
    # Danh sách các lựa chọn cho loại hình và trung bình
    dataset_names = [os.path.basename(path).replace("draw_", "").capitalize() for i, path in enumerate(DATATYPES) if i != 2]
    dataset_names.append("Trung bình")  # Thêm tùy chọn "Trung bình"

    # Tạo radio button nằm ngang
    selected_dataset = st.radio("Chọn loại muốn hiển thị kết quả", options=dataset_names, index=dataset_names.index("Trung bình"),horizontal=True)

    # Xử lý để lấy dữ liệu hiển thị dựa trên lựa chọn
    if selected_dataset == "Trung bình":
        sift_data = np.mean(np.delete(accuracy_sift, 2, axis=1), axis=1)
        orb_data = np.mean(np.delete(accuracy_orb, 2, axis=1), axis=1)
        superpoint_data = np.mean(np.delete(accuracy_superpoint, 2, axis=1), axis=1)
    else:
        selected_index = dataset_names.index(selected_dataset)
        if selected_index >=2:
            selected_index+=1
        sift_data = accuracy_sift[:, selected_index]
        orb_data = accuracy_orb[:, selected_index]
        superpoint_data = accuracy_superpoint[:, selected_index]

    sift_data = sift_data[1:]  # Loại bỏ phần tử đầu tiên (góc 0)
    orb_data = orb_data[1:]    # Loại bỏ phần tử đầu tiên (góc 0)
    superpoint_data = superpoint_data[1:]  
    # Chuẩn bị dữ liệu cho biểu đồ
    st.bar_chart({
        "Góc xoay": np.arange(10, 360, 10),  # các góc từ 0 đến 350 với bước 10
        "SIFT": sift_data,
        "ORB": orb_data,
        "SuperPoint": superpoint_data
    },
        x="Góc xoay",
        y =["SIFT","ORB","SuperPoint"],
        x_label = "Góc xoay (°)", 
        y_label = "Average Accuracy",
        stack=False,
        color=["#2ECC71", "#F39C12","#335CFF"],
        use_container_width=True,
    )

    
    st.subheader("3.2. Kết quả matching keypoint trên ảnh cho $3$ thuật toán với các góc xoay")
    st.write(
    """
    - Các **keypoints** **:green[🟢]** là **keypoints matching** đúng.
    - Các **keypoints** **:red[🔴]** là **keypoints** không **matching**.
    """
)
    cols = st.columns([0.5, 1, 1.5]) 

    # Đặt widget vào các cột
    id_image = cols[0].number_input("Chọn tập ảnh thứ", 0, 499, 0, 1)
    angle = cols[1].slider("Góc xoay", 0, 350, 10, 10)

    # Lựa chọn loại ảnh với các checkbox
    options = ["ALL"] + list(TYPE_MAPPING.keys())

    selected_types = []
    checkbox_cols = cols[2].columns(4)  # Tạo 4 cột trong cột thứ ba

    # Hiển thị checkbox theo 2 hàng, mỗi hàng có 4 checkbox
    for i, option in enumerate(options):
        col = checkbox_cols[i % 4]  # Chia đều checkbox vào các cột
        if col.checkbox(option, value=(option == "ALL")):  # Mặc định "ALL" được chọn
            selected_types.append(option)

    # Xử lý lựa chọn "ALL"
    if "ALL" in selected_types:
        selected_type_indices = list(TYPE_MAPPING.values())
    else:
        selected_type_indices = [TYPE_MAPPING[stype] for stype in selected_types if stype != "ALL"]
    cols = st.columns(3)
    cols[0].markdown("<h3 style='text-align: center;'>ORB</h3>", unsafe_allow_html=True)
    cols[1].markdown("<h3 style='text-align: center;'>SIFT</h3>", unsafe_allow_html=True)
    cols[2].markdown("<h3 style='text-align: center;'>SuperPoint</h3>", unsafe_allow_html=True)
    # Đọc và xử lý ảnh
    for type_idx in selected_type_indices:
        image, ground_truth_tuples = read_image(type_idx, f"{id_image}")
        ground_truth = convert_to_keypoints(ground_truth_tuples)  # Convert to KeyPoint objects here
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rotated_image = rotate_image(image, angle)
        rotated_gray_scale = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

        h, w = rotated_gray_scale.shape
        rotated_kp, idx = rotate_keypoints((w, h), ground_truth, angle)
        original_kp = [ground_truth[i] for i in idx]

        # Match keypoints for each method
        matches = {
            method: match_features(
                gray_scale.astype(np.float32) if method == "SuperPoint" else gray_scale,
                rotated_gray_scale.astype(np.float32) if method == "SuperPoint" else rotated_gray_scale,
                original_kp, rotated_kp, method
            ) for method in models
        }

        # Display the results
        cols = st.columns(3)
        for i, method in enumerate(models.keys()):
            matched_image = draw_colored_matches(image, original_kp, rotated_image, rotated_kp, matches[method])
            if len(original_kp) > 0:
                caption = f"Accuracy: {len(matches[method])}/{len(original_kp)} = {len(matches[method]) / len(original_kp):.2f}"
            else:
                caption = f"Accuracy: Không có keypoints so sánh"
            cols[i].image(
                matched_image,
                caption=caption,
                use_column_width=True
            )

    
def display_discussion():
    st.header("4. Thảo luận")
    st.markdown("""
    - **Góc nhỏ (0° - 30°): độ chính xác của cả 3 thuật toán đều khá cao**
        - **SuperPoint** đạt độ chính xác cao nhất (>0.5) nhờ sử dụng mạng nơ-ron sâu, giúp phát hiện và mô tả keypoint rất chính xác trong điều kiện góc xoay nhỏ, nơi biến dạng ít xảy ra.
        - **ORB** và **SIFT** cũng đạt hiệu suất cao, nhưng thấp hơn SuperPoint do đặc trưng của các thuật toán truyền thống. ORB sử dụng bộ mô tả BRIEF với khả năng kháng xoay cơ bản, còn SIFT dựa trên histogram định hướng giúp mô tả keypoint tốt trong điều kiện ít biến dạng.
    - **Góc trung bình (40° - 180°): độ chính xác của cả 3 thuật toán tương đối thấp**
        - **ORB** vượt trội và duy trì hiệu suất ổn định hơn cả. Điều này nhờ vào khả năng kháng góc xoay của ORB thông qua việc chuẩn hóa hướng chính (dominant orientation) của keypoint. Dù BRIEF không mạnh mẽ như các mô tả học sâu, nhưng tính đơn giản của thuật toán giúp ORB hoạt động nhất quán.
        - **SIFT** giảm hiệu suất khi góc xoay tăng, do histogram định hướng không đủ linh hoạt để mô tả đặc trưng trong các góc xoay lớn hơn 90°.
        - **SuperPoint** giảm đáng kể ở một số góc xoay trung bình. Điều này có thể do dữ liệu huấn luyện không bao quát đủ các biến đổi phức tạp, dẫn đến mạng không tổng quát hóa tốt trong điều kiện này.
    - **Kết luận**:
        - **ORB** là phương pháp phù hợp nhất nếu yêu cầu tính ổn định cao trên toàn bộ phạm vi góc xoay, đặc biệt trong các ứng dụng thời gian thực và tài nguyên hạn chế.
        - **SuperPoint** mạnh mẽ hơn ở góc nhỏ nhưng cần cải thiện khả năng kháng biến đổi ở các góc trung bình.
        - **SIFT** vẫn là lựa chọn tốt trong các bài toán không yêu cầu xử lý góc xoay lớn, nhưng không phù hợp khi góc xoay vượt quá 90°. 
    """)


display_methods()
display_experiment()
display_result()
display_discussion()