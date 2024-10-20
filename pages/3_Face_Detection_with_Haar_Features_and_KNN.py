import cv2, os
import numpy as np
from PIL import Image
import streamlit as st

from utils.faceDetection_function import (preprocess_image,knn,load_and_prepare_data)

st.set_page_config(
    page_title="Hue_Tran_Face_Detection",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Phát hiện khuôn mặt với Haar Features và KNN")

__DATA_DIR = "./data/face_detection"

face_images = np.load(os.path.join(__DATA_DIR, "Dataface_data.npy"))
nonface_images=np.load(os.path.join(__DATA_DIR, "Datanon_face_data.npy"))

test_images_path = os.path.join(__DATA_DIR, "tests/images")
test_images_name = os.listdir(test_images_path)

ground_truth_path = os.path.join(__DATA_DIR, "tests/labels")
result_images_path = os.path.join(__DATA_DIR, "tests/results")

ground_truth_images = os.listdir(ground_truth_path)
result_images = os.listdir(result_images_path)

positive_images_name = os.listdir(os.path.join(__DATA_DIR, "train/positives"))
negative_images_name = os.listdir(os.path.join(__DATA_DIR, "train/negatives"))

file_path = os.path.join(__DATA_DIR, "iou_results.txt")
IoU_test_k5_file_path = os.path.join(__DATA_DIR, "IoU_test_k5.txt")

cascade = os.path.join(__DATA_DIR, "cascade.xml")
face_cascade = cv2.CascadeClassifier(cascade)
trainset = load_and_prepare_data(face_images, nonface_images)

# Đọc giá trị IoU từ chuỗi
def load_iou_from_file(iou_file_path):
    iou_dict = {}
    with open(iou_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(':')
            if len(parts) == 2:
                image_name = parts[0].strip()
                iou_value = float(parts[1].split('=')[1].strip())
                iou_dict[image_name] = iou_value
    return iou_dict

# Lấy giá trị IoU
iou_values = load_iou_from_file(IoU_test_k5_file_path)

print(f"Kiểu dữ liệu của iou_values: {type(iou_values)}")
print(f"Nội dung của iou_values: {iou_values}")

# Khởi tạo danh sách cho K và IoU
k_values = []
iou_values = []

# Đọc file và tách dữ liệu
with open(file_path, "r") as f:
    for line in f:
        if "K =" in line and "IoU =" in line:
            # Tách giá trị K và IoU
            parts = line.split(",")
            k = int(parts[0].split("=")[1].strip())
            iou = float(parts[1].split("=")[1].strip())
            k_values.append(k)
            iou_values.append(iou)

# Chuyển đổi thành numpy array nếu cần
k_knn = np.array(k_values)
average_iou_each_k = np.array(iou_values)

# Tìm giá trị IoU lớn nhất và K tương ứng
max_iou_index = np.argmax(average_iou_each_k)
max_iou = average_iou_each_k[max_iou_index]
optimal_k = k_knn[max_iou_index]

# ---------------------------------------------


def display_datasets():
    st.header("1. Dataset")
    st.subheader("1.1. Tập train")
    st.markdown(
        """
        - Tập dữ liệu bao gồm:
            - $400$ ảnh chứa khuôn mặt.  
            - $400$ ảnh không chứa khuôn mặt.
        - Mỗi ảnh trong tập dữ liệu đều có kích thước $24$ x $24$.
        """
    )

    # Hiển thị dòng tiêu đề cho 10 ảnh chứa khuôn mặt
    st.write("$10$ ảnh chứa khuôn mặt")

    # Lấy 10 ảnh đầu tiên của positives và hiển thị trong 1 hàng
    _cols = st.columns(10)  # Tạo 10 cột để hiển thị 10 ảnh trên 1 hàng
    for j in range(10):
        _img_path = os.path.join(__DATA_DIR, "train/positives", positive_images_name[10 + j])
        _image = cv2.imread(_img_path)
        _cols[j].image(_image, use_column_width=True)

    # Hiển thị dòng tiêu đề cho 10 ảnh không chứa khuôn mặt
    st.write("$10$ ảnh không chứa khuôn mặt")

    # Lấy 10 ảnh đầu tiên của negatives và hiển thị trong 1 hàng
    _cols = st.columns(10)  # Tạo 10 cột để hiển thị 10 ảnh trên 1 hàng
    for j in range(10):
        _img_path = os.path.join(__DATA_DIR, "train/negatives", negative_images_name[10+j])
        _image = cv2.imread(_img_path)
        _cols[j].image(_image, use_column_width=True, channels="BGR")


    st.subheader("1.2. Tập Test")
    st.markdown(
        """
        - Tập test bao gồm $10$ ảnh chứa khuôn mặt được thu thập trên mạng.
        - Mỗi ảnh trong tập dữ liệu đều có kích thước $300$ x $300$ và chỉ chứa $1$ khuôn mặt.
        - Các ảnh được xác định ground_truth  bằng **opencv haar cascade**.
        - Các hình ảnh trong tập test như sau:
        """
    )

    for i in range(2):
        _cols = st.columns(5)  # Tạo 5 cột cho mỗi hàng
        for j in range(5):
            index = i * 5 + j  # Tính chỉ số của ảnh
            if index < len(test_images_name):  # Kiểm tra để tránh lỗi nếu không có đủ ảnh
                _img_path = os.path.join(test_images_path, test_images_name[index])
                _image = cv2.imread(_img_path)
                _cols[j].image(_image, use_column_width=True, channels="BGR")

def display_training():
    st.header("2. Quá trình huấn luyện")
    st.subheader("2.1. Quá trình huấn luyện Cascades Classifiers")
    st.markdown(
        """
        - Sử dụng **Cascade-Trainer-GUI** của OpenCV để huấn luyện Cascades Classifiers với các tham số như sau:
            - numPos: $400$ - số lượng ảnh chứa khuôn mặt
            - numNeg: $400$ - số lượng ảnh không chứa khuôn mặt
            - numStages: $20$ - số lượng stages
            - number_of_threads: $5$ - số luồng xử lý song song
            - minHitRate: $0.995$ - tỉ lệ nhận diện đúng tối thiểu cho mỗi stage
            - maxFalseAlarmRate: $0.5$ - tỉ lệ nhận diện sai tối đa cho mỗi stage
            - Sample_width: $24$ - Chiều rộng ảnh train.
            - Sample_Height: $24$ - Chiều cao ảnh train.
        - Kết quả sau khi huấn luyện:
            - Số lượng stages: $5$
            - Số lượng features: $13$
        """
    )

    
    st.subheader("2.2. Quá trình huấn luyện KNN")
    st.markdown(
        """
        - Sử dụng model **KNN** để huấn luyện với các tham số như sau:
            - Tham số **k**: $1$ - $19$ với bước nhảy là $2$.
            - Tham số weights: **distance**.
        - Độ đo đuợc sử dụng để đánh giá mô hình là **IoU**.
        """
    )
    st.columns(3)[1].image(
        os.path.join("./images/IoU.webp"),

        use_column_width=True,
        caption="Công thức IoU",
    )
    st.write(
        "- Biểu đồ **Average IoU** của mô hình với các giá trị **k** của **KNN**:"
    )
    st.line_chart(
        {"K": k_knn, "Average IoU": average_iou_each_k},
        x="K",
        y="Average IoU",
        x_label="Tham số k trong KNN",
        y_label="Average IoU",
    )

    st.markdown(
        f"""
        - Kết quả sau khi huấn luyện:
            - Tham số **k** tốt nhất: ${ optimal_k}$.
            - **Average IoU** tốt nhất: ${max_iou}$.
        """
    )

    @st.fragment
    def display_result_test():
        def load_iou_from_file(iou_file_path):
            iou_dict = {}
            with open(iou_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        image_name = parts[0].strip()
                        iou_value = float(parts[1].split('=')[1].strip())
                        iou_dict[image_name] = iou_value
            return iou_dict

        # Lấy giá trị IoU
        iou_values = load_iou_from_file(IoU_test_k5_file_path)

        

        st.subheader("2.3. Kết quả trên tập dữ liệu kiểm thử")
        st.write("### Ảnh Ground Truth và Predict")
        for i in range(0, 10, 5):  # Lặp qua từng nhóm 5 ảnh
            # Hiển thị 5 ảnh Ground Truth
            cols_gt = st.columns(5)  # 5 cột cho 5 ảnh ground truth
            for j in range(5):
                if i + j < 10:  # Kiểm tra giới hạn danh sách
                    ground_truth_image_path = os.path.join(ground_truth_path, ground_truth_images[i + j])
                    img_ground_truth = cv2.imread(ground_truth_image_path)

                    if img_ground_truth is not None:
                        cols_gt[j].image(img_ground_truth, caption=f"Ground Truth {i + j + 1}", use_column_width=True, channels="BGR")

            # Hiển thị 5 ảnh Kết Quả
            cols_result = st.columns(5)  # 5 cột cho 5 ảnh kết quả
            for j in range(5):
                if i + j < 10:  # Kiểm tra giới hạn danh sách
                    result_image_path = os.path.join(result_images_path, result_images[i + j])
                    img_result = cv2.imread(result_image_path)

                    if img_result is not None:
                        image_name = result_images[i + j]
                        # Lấy giá trị IoU tương ứng
                        iou_value = iou_values.get(image_name, "N/A")  # Lấy IoU, nếu không có thì để là "N/A"
                        
                        # Hiển thị ảnh kết quả với IoU
                        cols_result[j].image(img_result, caption=f"Predict {i + j + 1} - IoU: {iou_value:.4f}", use_column_width=True, channels="BGR")


    display_result_test()


def display_result():
    st.header("3. Ứng dụng phát hiện khuôn mặt")
    uploaded_file = st.file_uploader("Chọn một file ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("Đang phát hiện khuôn mặt..."):
            # Đọc ảnh từ uploaded_file và chuyển thành định dạng OpenCV
            image = Image.open(uploaded_file)
            image = np.array(image)  # Chuyển đổi ảnh thành NumPy array để sử dụng với OpenCV

            # Tiền xử lý ảnh
            padded_img, gray, scale = preprocess_image(image)
            faces = face_cascade.detectMultiScale(gray,1.45,3)

            for (x, y, w, h) in faces:
            # Cắt khuôn mặt từ ảnh
                face_region = padded_img[y:y + h, x:x + w]
                face_resized = cv2.resize(face_region, (24, 24))

                # Kiểm tra khuôn mặt bằng KNN
                label = knn(trainset, face_resized.flatten(), k=5)
                
                # Nếu nhãn là 1 (khuôn mặt), vẽ hình chữ nhật
                if label == 1:
                    # Chuyển tọa độ về ảnh gốc
                    x_orig = int(x / scale)
                    y_orig = int(y / scale)
                    w_orig = int(w / scale)
                    h_orig = int(h / scale)

                    # Vẽ hình chữ nhật trên ảnh gốc
                    cv2.rectangle(image, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)
        # Hiển thị ảnh kết quả
    st.image(image, channels="RGB")





display_datasets()
display_training()
display_result()
