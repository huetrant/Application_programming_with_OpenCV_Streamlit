import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd

from utils.semantic_keypoint_detection_function import draw_points, calculate_precision_recall_image,process_image

st.set_page_config(
    page_title="Hue Tran _ Semantic Keypoint Detection với thuật toán SIFT và ORB",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Semantic Keypoint Detection bằng thuật toán SIFT và ORB")


DATA_DIR = "./data/semantic_keypoint_detection/"
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
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

precision_recall_sift: np.ndarray = np.load(os.path.join(DATA_DIR, "precision_recall_sift.npy"))
precision_recall_orb: np.ndarray = np.load(os.path.join(DATA_DIR, "precision_recall_orb.npy"))



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
        points = np.load(os.path.join(DATATYPES[i], "points", "6.npy"))
        image = cv2.imread(os.path.join(DATATYPES[i], "images", "6.png"))
        # Hiển thị ảnh
        cols1[i].image(draw_points(image, points,(0, 255, 0), 1), use_column_width=True)

        # Hiển thị caption in đậm
        caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
        cols1[i].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)

        # Vòng lặp thứ hai xử lý cols2
        points = np.load(os.path.join(DATATYPES[i + 4], "points", "6.npy"))
        image = cv2.imread(os.path.join(DATATYPES[i + 4], "images", "6.png"))
        # Hiển thị ảnh
        cols2[i].image(draw_points(image, points, (0, 255, 0), 1),use_column_width=True)

        # Hiển thị caption in đậm
        caption = DATATYPES[i + 4].replace('\\', '/').split('/')[-1].replace('draw_', '')
        cols2[i].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)

@st.fragment()
def display_methods():
    st.header("2. Phương pháp")
    
    # Phần hiển thị của SIFT
    st.subheader("2.1. Thuật toán SIFT")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            
            """
            - **SIFT** (Scale-Invariant Feature Transform) là một thuật toán được phát triển bởi David Lowe vào năm 2004 trong bài báo  [*Distinctive Image Features from Scale-Invariant Keypoints*](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cc58efc1f17e202a9c196f9df8afd4005d16042a).
            Thuật toán này được thiết kế để phát hiện và mô tả các điểm đặc trưng trong ảnh, giúp nhận diện đối tượng bất kể các thay đổi về tỷ lệ, góc xoay, hoặc ánh sáng. 
            -  **SIFT** có các bước chính như sau: 

                1. **Xây dựng không gian đa tỷ lệ:**
                Tạo các phiên bản làm mờ của hình ảnh ở nhiều kích thước khác nhau để tìm ra các điểm đặc trưng có thể nhận diện ở nhiều tỷ lệ.
                2. **Phát hiện các điểm đặc trưng tiềm năng:**
                Xác định các điểm "nổi bật" (cực trị) trong không gian đa tỷ lệ - đây là các điểm đặc trưng tiềm năng.
                3. **Loại bỏ điểm yếu hoặc không ổn định:**
                Loại bỏ các điểm đặc trưng có độ tương phản thấp hoặc nằm trên cạnh, giữ lại các điểm ổn định và đáng tin cậy.
                4. **Gán hướng cho điểm đặc trưng:**
                Gán hướng cho từng điểm đặc trưng để chúng có thể bất biến với các thay đổi xoay của hình ảnh.
                5. **Tạo mô tả đặc trưng cho từng điểm:**
                Mỗi điểm đặc trưng được biểu diễn bằng một vector, mô tả các đặc điểm của nó để dễ dàng so khớp với các hình ảnh khác.
                6. **So khớp các điểm đặc trưng giữa các hình ảnh:**
                So sánh các vector mô tả đặc trưng của các điểm giữa các hình ảnh để tìm các cặp điểm giống nhau, hỗ trợ cho các bài toán như ghép nối và nhận diện đối tượng.
    """)
    with col2:
        st.image(
               os.path.join(DATA_DIR, "The-flowchart-of-the-SIFT-method.png"),use_column_width=True, caption="SIFT Flowchart"
    )
  
    st.markdown("##### Minh họa **SIFT** trên Synthetic Shapes Datasets:")
    
    # Hiển thị 8 ảnh với 2 dòng, mỗi dòng 4 ảnh
    cols = st.columns(4)
    for i in range(8):
        image_sift = process_image(DATATYPES, i, sift)
        
        caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
        cols[i % 4].image(
            image_sift,
            use_column_width=True,  
        )
        cols[i % 4].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)
        # Chuyển sang cột mới sau mỗi 4 ảnh
        if (i + 1) % 4 == 0:
            cols = st.columns(4)  # Tạo lại các cột sau mỗi dòng 4 ảnh

    st.write(
        """
        - Các vòng tròn **:green[🟢]** là **keypoints** ground truth.
        - Các vòng tròn **:red[⭕•]** là **keypoints** được phát hiện với ngưỡng bán kính đánh giá là $3$ pixels bởi thuật toán **SIFT**.
    """
    )
    
    
    # Phần hiển thị của ORB
    st.subheader("2.2. Thuật toán ORB")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            - **ORB** (Oriented FAST and Rotated BRIEF) là một thuật toán được thiết kế để phát hiện và mô tả các đặc trưng (features) trong hình ảnh.
              ORB là sự kết hợp của hai thuật toán nổi tiếng: **FAST** (Features from Accelerated Segment Test) để phát hiện điểm đặc trưng và 
              **BRIEF** (Binary Robust Independent Elementary Features) để tính toán mô tả đặc trưng. 
              ORB đã cải tiến những thuật toán này để khắc phục một số nhược điểm như độ nhạy với xoay hoặc nhiễu, giúp nó trở nên hiệu quả 
              và bền vững hơn khi nhận diện các đặc trưng trong các điều kiện khác nhau. 
              Thuật toán ORB được giới thiệu lần đầu trong bài báo [*ORB: An efficient alternative to SIFT or SURF*](https://www.researchgate.net/profile/Gary-Bradski-4/publication/221111151_ORB_an_efficient_alternative_to_SIFT_or_SURF/links/00b4951c369020213a000000/ORB-an-efficient-alternative-to-SIFT-or-SURF.pdf) vào năm 2011 bởi Ethan Rublee, Vincent Rabaud, Kurt Konolige, và Gary Bradski.

            - **ORB** có các bước chính như sau :
                1. **Tìm vị trí các điểm đặc trưng với FAST:** Sử dụng thuật toán FAST để phát hiện các điểm đặc trưng trong ảnh. 
                FAST nhanh chóng xác định các điểm nổi bật, nhưng không cung cấp thông tin về mức độ "tốt" của từng điểm.
                2. **Lọc và chọn N điểm đặc trưng tốt nhất với Harris Corner Measure:** ORB lọc và giữ lại các điểm đặc trưng quan trọng nhất bằng Harris Corner Measure, 
                chỉ chọn N điểm có giá trị cao nhất để giảm nhiễu và tập trung vào các điểm đáng tin cậy.
                3. **Tính hướng của các điểm đặc trưng với Moment Patch:** Để tăng độ ổn định trước các phép xoay, 
                ORB tính toán hướng cho từng điểm đặc trưng bằng Moment Patch, giúp nhận diện ổn định dù ảnh bị xoay.
                4. **Trích xuất mô tả đặc trưng bằng BRIEF xoay:** ORB sử dụng một phiên bản xoay của thuật toán BRIEF để tạo mô tả đặc trưng nhị phân cho các điểm đã được xác định hướng, 
                giúp chúng ổn định trước các phép xoay và biến đổi hình học.
                5. **Xuất các điểm và mô tả đặc trưng:** Kết quả là tập hợp các điểm đặc trưng cùng với mô tả đặc trưng tương ứng, 
                có thể dùng để nhận diện, ghép nối, hoặc theo dõi đối tượng trong các ứng dụng khác nhau.
            """
        )
    with col2:
        img = Image.open(os.path.join(DATA_DIR, "Flowchart-of-ORB-algorithm.png"))
        st.columns([0.2, 3, 0.2])[1].image(
            img,
            use_column_width=True,
            caption="ORB Flowchart",
        )
        
    st.markdown("##### Minh họa ORB trên Synthetic Shapes Datasets:")
    
    # Hiển thị 8 ảnh với 2 dòng, mỗi dòng 4 ảnh
    cols = st.columns(4)
    for i in range(8):
        image_orb = process_image(DATATYPES, i, orb)

        caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
        cols[i % 4].image(
            image_orb,
            use_column_width=True,  

        )
        cols[i % 4].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)

        # Chuyển sang cột mới sau mỗi 4 ảnh
        if (i + 1) % 4 == 0:
            cols = st.columns(4)  # Tạo lại các cột sau mỗi dòng 4 ảnh

    st.write(
    """
    - Các vòng tròn **:green[🟢]** là **keypoints** ground truth.
    - Các vòng tròn **:red[⭕•]** là **keypoints** được phát hiện với ngưỡng bán kính đánh giá là $3$ pixels bởi thuật toán **ORB**.
    """
    )

@st.fragment()
def display_metric():
    st.header("3. Đánh giá với độ đo")
    col1, col2 = st.columns(2)
    with col1: 
        st.markdown("""
            - Một keypoint được coi là phát hiện đúng nếu:
            $$d \leq Threshold$$
            
                - Trong đó khoảng cách **Euclidean** giữa keypoint thực tế và keypoint dự đoán:
                    
                    $$d = \sqrt{(x_{gt} - x_{pred})^2 + (y_{gt} - y_{pred})^2}$$

                - Với:
                    - $(x_{gt}, y_{gt})$: Tọa độ của keypoint thực tế.
                    - $(x_{pred}, y_{pred})$: Tọa độ của keypoint dự đoán của thuật toán **SIFT** và **ORB**.
                - **Threshold** được thiết lập trong thí nghiệm là 3 pixel.
            - Hai độ đo **Precision** và **Recall** được sử dụng để đánh giá kết quả phát hiện keypoint của hai thuật toán **SIFT** và **ORB**
            với công thức ở hình bên:
            """)
    with col2:
        img = Image.open(os.path.join(DATA_DIR, "Pre_re.png"))
        img = img.resize((img.width // 2, img.height // 2)) 
        st.columns([1, 2.2, 1])[1].image(
            img,
            use_column_width=True,
            caption="Công thức tính Precision và Recall",
        )

@st.fragment()
def display_results():
    st.header("4. Kết quả")

    # Tách riêng các giá trị Precision và Recall
    precision_sift = precision_recall_sift[:, :, 0]
    recall_sift = precision_recall_sift[:, :, 1]
    precision_orb = precision_recall_orb[:, :, 0]
    recall_orb = precision_recall_orb[:, :, 1]

    # Tính toán giá trị trung bình
    avg_precision_sift = precision_sift.mean(axis=1)
    avg_recall_sift = recall_sift.mean(axis=1)
    avg_precision_orb = precision_orb.mean(axis=1)
    avg_recall_orb = recall_orb.mean(axis=1)

    col1, col2= st.columns(2)
    with col1:
        st.markdown("<p style='text-align: center;font-size: 20px;'>Biểu đồ so sánh độ đo Precision giữa SIFT và ORB trên các loại hình.</p>", unsafe_allow_html=True)

        precision_df = pd.DataFrame(
            {
                "shape_type": [
                    DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
                    for i in range(len(DATATYPES))
                ],
                "SIFT": avg_precision_sift,
                "ORB": avg_precision_orb,
            }
        )
        st.bar_chart(
            precision_df,
            x="shape_type",
            stack=False,
            y_label="",
            x_label="Precision",
            horizontal=True,
            color=["#2ECC71", "#F39C12"],
        )

    with col2:
        st.markdown(
            "<p style='text-align: center; font-size: 20px;'>"
            "Biểu đồ so sánh đô đo Recall giữa SIFT và ORB trên các loại hình."
            "</p>",
            unsafe_allow_html=True
        )
        recall_df = pd.DataFrame(
            {
                "shape_type": [
                    DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
                    for i in range(len(DATATYPES))
                ],
                "SIFT": avg_recall_sift,
                "ORB": avg_recall_orb,
            }
        )
        st.bar_chart(
            recall_df,
            x="shape_type",
            stack=False,
            y_label="",
            x_label="Recall",
            horizontal=True,
            color=["#2ECC71", "#F39C12"],
        )
    st.subheader("Nhận xét:")

@st.fragment()
def display_discussion():

    st.header("5. Thảo luận")
    st.markdown("""
        - **ORB** thể hiện kết quả tốt hơn **SIFT** trong các hình dạng như **checkerboard**, **cube**, **multiple polygons**, **polygon**, và **star**. 
            - **ORB**: Tối ưu cho việc phát hiện các góc cạnh nhờ thuật toán **FAST**, vốn được thiết kế để phát hiện các điểm góc. 
                Các đặc trưng góc rõ ràng trong những hình này giúp **ORB**  phát hiện chính xác nhiều keypoints. 
                Nó nhanh và hiệu quả, đặc biệt với các hình có góc cạnh rõ ràng.
            - **SIFT** sử dụng **gradient** để phát hiện các keypoints và mô tả chúng. 
                Trong những hình này, **SIFT** có thể phát hiện được nhiều keypoints, nhưng nó có xu hướng nhạy cảm với các đặc trưng nhỏ hơn, 
                chẳng hạn như các cạnh hoặc góc phụ, mà đôi khi không phải là các keypoints chính. 
                Điều này khiến cho **SIFT** có thể bỏ sót một số keypoints quan trọng. 
                Tuy nhiên, **SIFT** vẫn có khả năng phát hiện keypoints chính xác nhưng có thể chậm hơn **ORB** do phải xử lý các gradient phức tạp.
        """)
    st.markdown("###### Keypoints do SIFT phát hiện **:red[⭕•]** ,**:green[🟢]**: **keypoints** ground truth.")
    sift_columns = st.columns(5)

    for i in range(8):
        if  i == 2 or i == 3 or i == 7 : 
            continue  

        # Đọc ảnh
        image_sift = process_image(DATATYPES, i, sift)
        
        # Điều chỉnh chỉ số cột để không bị thiếu
        col_index = i if i < 2 else (i - 1 if i < 3 else i - 2)
        pre,re = calculate_precision_recall_image(DATATYPES,i,sift)
        caption_with_metrics = f"Precision: {pre:.2f}, Recall: {re:.2f}"
        sift_columns[col_index].image(image_sift, use_column_width=True,caption=caption_with_metrics)
        
        caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
            
    st.markdown("###### Keypoints do ORB phát hiện: **:red[⭕•]** ,**:green[🟢]**: **keypoints** ground truth." )
    orb_columns  = st.columns(5)

    for i in range(8):
        if  i == 2 or i == 3 or i == 7 : 
            continue  
        image_orb = process_image(DATATYPES, i, orb)


        caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')

        col_index = i if i < 2 else (i - 1 if i < 3 else i - 2)
        pre,re = calculate_precision_recall_image(DATATYPES,i,orb)
        caption_with_metrics = f"Precision: {pre:.2f}, Recall: {re:.2f}"
        orb_columns[col_index].image(image_orb, use_column_width=True,caption=caption_with_metrics)
       

        # Tạo caption cho ảnh
        caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
        orb_columns[col_index].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)


    st.markdown("""
                
    - Ngược lại, **SIFT** cho kết quả tốt hơn **ORB** trong việc nhận diện các hình dạng đơn giản như **stripes** và **lines**. 
        - **ORB** gặp khó khăn trong việc phát hiện các đặc trưng tuyến tính như **stripes** và **lines** 
                vì thuật toán này được tối ưu hóa cho việc phát hiện các điểm góc. Các đặc trưng đường thẳng không tạo ra nhiều điểm góc rõ ràng, 
                vì vậy **ORB** có thể bỏ sót các keypoints quan trọng hoặc phát hiện sai các keypoints trên các hình dạng này. 
                Do đó, hiệu suất phát hiện keypoints của **ORB** trong các hình **stripes** và **lines** kém hơn so với SIFT.
        - **SIFT** sử dụng gradient để phát hiện các keypoints, điều này giúp nó xử lý tốt hơn với các đặc trưng tuyến tính như 
                trong các hình stripes và lines. **SIFT** có khả năng phát hiện chính xác keypoints trên các cạnh và đường thẳng, nhờ vào việc phân tích gradient dọc theo các cạnh. 
                Chính vì thế, **SIFT** có hiệu suất tốt hơn **ORB** trong việc phát hiện keypoints ở 2 hình này.
    """)
    
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("###### Keypoints do SIFT phát hiện **:red[⭕•]** ,**:green[🟢]**: **keypoints** ground truth.")
        sift_columns = st.columns(2)
        for i in [3, 7]:
            image_sift = process_image(DATATYPES, i, sift)
            
            col_index = (i - 3) // 4  # Điều chỉnh chỉ số cột
            pre,re = calculate_precision_recall_image(DATATYPES,i,sift)
            caption_with_metrics = f"Precision: {pre:.2f}, Recall: {re:.2f}"
            sift_columns[col_index].image(image_sift, use_column_width=True, caption=caption_with_metrics)
           
            caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
            
            sift_columns[col_index].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("###### Keypoints do ORB phát hiện **:red[⭕•]** ,**:green[🟢]**: **keypoints** ground truth.")

        orb_columns = st.columns(2)
        for i in [3, 7]:
            image_orb= process_image(DATATYPES, i, orb)
            
            col_index = (i - 3) // 4  # Điều chỉnh chỉ số cột
            pre,re = calculate_precision_recall_image(DATATYPES,i,orb)
            caption_with_metrics = f"Precision: {pre:.2f}, Recall: {re:.2f}"
            orb_columns[col_index].image(image_orb, use_column_width=True,caption=caption_with_metrics)
            caption = DATATYPES[i].replace('\\', '/').split('/')[-1].replace('draw_', '')
            
            orb_columns[col_index].markdown(f"<div style='text-align: center; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)

    st.markdown("""
        - Việc **ORB** cho kết quả cao hơn trong các hình dạng phức tạp cho thấy ưu điểm của nó trong việc tối ưu hóa tốc độ và độ chính xác, 
                điều này là một lợi thế lớn trong các ứng dụng thời gian thực. 
                ORB sử dụng các phương pháp đơn giản hơn để phát hiện các điểm đặc trưng, giúp nó hoạt động hiệu quả hơn trên các hình ảnh có nhiều chi tiết và cấu trúc phức tạp.
        - Trong khi đó, **SIFT** mặc dù có độ chính xác cao trong các tình huống cụ thể, 
                nhưng lại yêu cầu nhiều tài nguyên tính toán hơn, làm cho nó không phù hợp với các ứng dụng cần xử lý nhanh. 
                SIFT cũng thường bị ảnh hưởng bởi các yếu tố như ánh sáng và sự biến dạng trong hình ảnh, 
                điều này có thể làm giảm hiệu suất trong những trường hợp cụ thể.
        - Việc lựa chọn phương pháp nào sẽ phụ thuộc vào mục tiêu cụ thể của bài toán. 
                Nếu bài toán yêu cầu phát hiện các hình dạng phức tạp và có nhiều chi tiết, **ORB** có thể là lựa chọn tối ưu. 
                Tuy nhiên, nếu bài toán tập trung vào các đặc trưng rõ ràng và đơn giản, **SIFT** vẫn có thể là một công cụ đáng tin cậy.

    """)


display_datasets()   
display_methods()
display_metric()
display_results()
display_discussion()