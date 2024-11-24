from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Hue Tran_Thuật toán SORT - Simple Online and Real-Time Tracking",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Thuật toán SORT - Simple Online and Real-Time Tracking")

DATA_DIR = "./data/sort_mot/"

def display_intro():
    st.header("1. Giới thiệu")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("""
        - Theo dõi đa đối tượng (Multiple Object Tracking - MOT) là một trong những thách thức lớn trong lĩnh vực thị giác máy tính, với ứng dụng trong các lĩnh vực như xe tự hành, giám sát video, và robot. 
        Các hệ thống MOT hiện đại thường sử dụng các phương pháp phức tạp để xử lý những vấn đề như che khuất dài hạn, tái nhận diện đối tượng, và tích hợp dữ liệu từ nhiều khung hình.
        - [**Simple Online and Real-Time Tracking**](https://arxiv.org/abs/1602.00763) được Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, và Ben Upcroft đề xuất như một giải pháp đơn giản nhưng hiệu quả, tập trung vào hiệu suất và tính khả dụng trong thời gian thực. 
        Hệ thống dựa trên mô hình **tracking-by-detection** trong đó các đối tượng được phát hiện tại mỗi khung hình và liên kết qua các khung hình tiếp theo.
        - SORT tận dụng sức mạnh của các thuật toán cổ điển như:
            - **Kalman Filter:** Dự đoán trạng thái và vị trí của đối tượng.
            - **Hungarian Algorithm:** Gán các phát hiện mới với các đối tượng đang được theo dõi.
        - Mục tiêu của SORT là:
            - Đạt tốc độ cao (260Hz).
            - Cân bằng giữa độ chính xác và hiệu suất.
            - Tối giản hóa thiết kế để tăng tính ứng dụng thực tế, đặc biệt trong các hệ thống yêu cầu thời gian thực như xe tự hành.
        SORT không xử lý các bài toán phức tạp như tái nhận diện đối tượng hoặc che khuất dài hạn, nhưng nhờ thiết kế đơn giản, nó phù hợp làm một phương pháp nền tảng cho các nghiên cứu nâng cao.
        """)
    with col2:
        st.image(f"{DATA_DIR}Sort.png", caption="Đánh giá hiệu suất của SORT so với các thuật toán theo dõi cơ sở.")

def display_method():
    st.header("2. Phương pháp")
    st.subheader("2.1 Phát hiện (Detection)")
    col3, col4, _ = st.columns([3, 1, 0.5])
    with col3:
        st.write("""
            Hệ thống SORT dựa trên các bộ phát hiện đối tượng hiện đại, ví dụ như là Faster R-CNN, để xác định các đối tượng trong từng khung hình:
            - **Faster R-CNN**:
                1. **Đề xuất vùng (Region Proposal):** Xác định các vùng trong ảnh có khả năng chứa đối tượng.
                2. **Phân loại và tinh chỉnh:** Phân loại các vùng này và dự đoán chính xác hộp bao quanh (bounding boxes) của đối tượng.
            - **Kiến trúc mạng:**
                - ZF-Net: tốc độ cao nhưng độ chính xác thấp hơn.
                - VGG16: cung cấp độ chính xác vượt trội.
            - **Ngưỡng tin cậy:** Chỉ những phát hiện có xác suất trên 50% được đưa vào hệ thống theo dõi. 
                 
            Kết quả từ Faster R-CNN được chuyển sang các bước tiếp theo của SORT để dự đoán và gán đối tượng.
        """)
    with col4:
        st.image(f"{DATA_DIR}Faster_R-CNN.webp", caption="Minh họa hoạt động của Faster R-CNN.")

    st.subheader("2.2 Mô hình dự đoán (Estimation Model)")
    col5, _, col6 = st.columns([2, 0.5, 1.5])
    with col5:
        st.write("""
            Sau khi phát hiện, SORT sử dụng **bộ lọc Kalman** để dự đoán trạng thái tiếp theo của các đối tượng:
            - **Trạng thái của đối tượng** được biểu diễn bởi vector:
            $x = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$, 
                - $(u, v)$ là tọa độ của trung tâm của bounding box của đối tượng.
                - $s$ là tỷ lệ giữa chiều rộng và chiều cao của bounding box của đối tượng.
                - $r$ là tỷ lệ giữa chiều rộng và chiều cao của bounding box của đối tượng.
                - $(\dot{u}, \dot{v})$ là vận tốc của trung tâm của bounding box của đối tượng.
                - $\dot{s}$ là vận tốc của tỷ lệ giữa chiều rộng và chiều cao của bounding box của đối tượng.
            - **Dự đoán và cập nhật:**
                - Nếu phát hiện mới được gán, bộ lọc Kalman sẽ cập nhật trạng thái của đối tượng bằng thông tin từ phát hiện mới.
                - Nếu không, trạng thái sẽ được dự đoán đơn giản mà không cần hiệu chỉnh bằng mô hình vận tốc tuyến tính.
                - Nếu tiếp tục không khớp, ID mục tiêu sẽ bị xóa.
        """)
    with col6:
        st.image(f"{DATA_DIR}Estimation Model.webp", caption="Minh họa Estimation Model")

    st.subheader("2.3 Gán dữ liệu (Data Association)")
    col7, col8 = st.columns(2)
    with col7:
        st.write("""
            Để kết nối các phát hiện mới với các đối tượng đang theo dõi, SORT sử dụng thuật toán Hungarian:
            - **Chi phí gán (Cost Matrix):** Được tính bằng IoU giữa các hộp phát hiện và hộp dự đoán.
            - **Ngưỡng IoU tối thiểu:** Các phát hiện có IoU nhỏ hơn ngưỡng này sẽ không được gán.
            - **Quy trình:**
                1. Tính toán ma trận chi phí IoU giữa các phát hiện và các đối tượng.
                2. Dùng thuật toán Hungarian để tối ưu hóa việc gán, đảm bảo mỗi phát hiện chỉ gán cho một đối tượng.
        """)
        st.subheader("2.4 Quản lý đối tượng (Track Management)")
        st.write("""
            SORT quản lý vòng đời của đối tượng qua các khung hình:
            - **Tạo mới đối tượng:**
                - Khi phát hiện không khớp với bất kỳ đối tượng nào, một ID mới sẽ được tạo.
                - Ban đầu, vận tốc của đối tượng được đặt là 0, với phương sai lớn để phản ánh sự không chắc chắn.
            - **Xóa đối tượng:**
                - Nếu một đối tượng không được phát hiện lại sau $T_{Lost}$ khung hình, nó sẽ bị xóa.
                - $T_{Lost}$ được đặt giá trị nhỏ (thường là 1) để tập trung vào theo dõi thời gian thực, ngắn hạn.
        """)
    with col8:
        st.image(f"{DATA_DIR}Data Association.webp", caption="(a): Kết quả từ bộ lọc Kalman trước đó. Hộp màu xanh lá cây trong (b): Hộp phát hiện hiện tại được so khớp bằng Hungarian thông qua IoU. (C): Kết hợp bằng cách gán ID")

def display_example():
    st.header("3. Ví dụ minh họa")
    video_path = f"{DATA_DIR}output_sort_video.mp4"
    st.video(video_path)

def display_conclude():
    st.header("4. Kết luận")
    st.write("""
        Các điểm mạnh của SORT bao gồm:
        - **Đơn giản và hiệu quả**: Hệ thống này có thiết kế đơn giản, dễ dàng triển khai và phù hợp với các hệ thống yêu cầu thời gian thực.
        - **Tốc độ và hiệu suất cao**: SORT đạt tần số cập nhật lên tới 260Hz, vượt trội so với nhiều phương pháp khác.
        - **Ứng dụng rộng rãi**: Nhờ vào tính đơn giản và hiệu quả, SORT có thể được áp dụng trong nhiều lĩnh vực khác nhau, từ theo dõi người trong video đến điều hướng cho xe tự lái.

        Tuy nhiên, phương pháp này vẫn còn hạn chế khi phải xử lý các tình huống phức tạp như che khuất dài hạn hoặc tái nhận diện đối tượng. Do đó, các nghiên cứu trong tương lai có thể tập trung
""")
display_intro()
display_method()
display_example()
display_conclude()