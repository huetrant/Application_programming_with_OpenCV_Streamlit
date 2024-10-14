import cv2, os
import numpy as np
from PIL import Image
import streamlit as st

from services.face_detection.extractor import Extractor, get_iou

st.set_page_config(
    page_title="Phát hiện khuôn mặt với Haar Features và KNN",
    page_icon=Image.open("./public/images/logo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Phát hiện khuôn mặt với Haar Features và KNN")

__SERVICE_DIR = "./services/face_detection"

positive_images_name = os.listdir(os.path.join(__SERVICE_DIR, "train/positives"))
negative_images_name = os.listdir(os.path.join(__SERVICE_DIR, "train/negatives"))

k_knn = np.load(os.path.join(__SERVICE_DIR, "k.npy"))
average_iou_each_k = np.load(os.path.join(__SERVICE_DIR, "average_iou_each_k.npy"))

k_knn_max = k_knn[np.argmax(average_iou_each_k)]
average_iou_max = np.max(average_iou_each_k)

X, y = [], []
extractor = Extractor(os.path.join(__SERVICE_DIR, "cascade.xml"), k_knn_max)
for image_name in os.listdir(os.path.join(__SERVICE_DIR, "train/positives")):
    image = cv2.imread(os.path.join(__SERVICE_DIR, "train/positives", image_name))
    X.append(extractor.extract_feature_image(image))
    y.append(1)

for image_name in os.listdir(os.path.join(__SERVICE_DIR, "train/negatives")):
    image = cv2.imread(os.path.join(__SERVICE_DIR, "train/negatives", image_name))
    X.append(extractor.extract_feature_image(image))
    y.append(0)

X = np.array(X)
y = np.array(y)
extractor.fit(X, y)

# ---------------------------------------------


def display_datasets():
    st.header("1. Tập dữ liệu")
    st.subheader("1.1. Tập huấn luyện")
    st.markdown(
        """
        - Tập dữ liệu bao gồm:
            - $400$ ảnh chứa khuôn mặt, được resize lại từ tập dữ liệu ORL.  
            - $400$ ảnh không chứa khuôn mặt, được resize từ tập dữ liệu thu thập trên mạng.
        - Mỗi ảnh trong tập dữ liệu đều có kích thước $24$ x $24$.
        """
    )

    cols = st.columns(2)
    with cols[0]:
        cols[0].columns([1, 2, 1])[1].write("$10$ ảnh chứa khuôn mặt")
        for i in range(2):
            _cols = st.columns(5)
            for j in range(5):
                _img_path = os.path.join(
                    __SERVICE_DIR, "train/positives", positive_images_name[i * 5 + j]
                )
                _image = cv2.imread(_img_path)
                _cols[j].image(_image, use_column_width=True)

    with cols[1]:
        cols[1].columns([1, 2, 1])[1].write("$10$ ảnh không chứa khuôn mặt")
        for i in range(2):
            _cols = st.columns(5)
            for j in range(5):
                _img_path = os.path.join(
                    __SERVICE_DIR, "train/negatives", negative_images_name[i * 5 + j]
                )
                _image = cv2.imread(_img_path)
                _cols[j].image(_image, use_column_width=True, channels="BGR")

    st.subheader("1.2. Tập kiểm thử")
    st.markdown(
        """
        - Tập dữ liệu kiểm thử bao gồm $10$ ảnh chứa khuôn mặt được thu thập trên mạng.
        - Mỗi ảnh trong tập dữ liệu đều có kích thước $1024$ x $1024$ và chỉ chứa $1$ khuôn mặt.
        - Các ảnh được gán nhãn bằng công cụ **opencv_annotation**.
        - Dưới đây là tất cả hình ảnh trong tập dữ liệu kiểm thử:
        """
    )

    with open(os.path.join(__SERVICE_DIR, "tests/labels/labels.txt")) as f:
        for i in range(2):
            _cols = st.columns(5)
            for j in range(5):
                _data = f.readline().strip().split()
                _img_path = os.path.join(__SERVICE_DIR, "tests/images", _data[0])
                x, y, w, h = map(int, _data[2:])
                _image = cv2.imread(_img_path)
                cv2.rectangle(_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                _cols[j].image(_image, use_column_width=True, channels="BGR")


def display_training():
    st.header("2. Quá trình huấn luyện")
    st.subheader("2.1. Quá trình huấn luyện Cascades Classifiers")
    st.markdown(
        """
        - Sử dụng công cụ **opencv_traincascade** để huấn luyện Cascades Classifiers với các tham số như sau:
            - numPos: $400$ (số lượng ảnh chứa khuôn mặt)
            - numNeg: $400$ (số lượng ảnh không chứa khuôn mặt)
            - numStages: $20$ (số lượng stages)
            - minHitRate: $0.995$ (tỉ lệ nhận diện đúng tối thiểu cho mỗi stage)
            - maxFalseAlarmRate: $0.5$ (tỉ lệ nhận diện sai tối đa cho mỗi stage)
            - w: $24$ (Chiều rộng ảnh chứa khuôn mặt)
            - h: $24$ (Chiều cao ảnh chứa khuôn mặt) 
        - Kết quả sau khi huấn luyện:
            - Số lượng stages: $6$
            - Số lượng features: $16$
        - Một số hình ảnh visualize được tạo ra trong quá trình huấn luyện:
        """
    )

    visualize_images = os.listdir(os.path.join(__SERVICE_DIR, "result"))
    for i in range(2):
        _cols = st.columns(3)
        for j in range(3):
            _img_path = os.path.join(
                __SERVICE_DIR, "result", visualize_images[i * 3 + j]
            )
            _image = cv2.imread(_img_path)
            _cols[j].image(_image, use_column_width=True)

    st.subheader("2.2. Quá trình huấn luyện KNN")
    st.markdown(
        """
        - Sử dụng model **kNN** của thư viện **sklearn** để huấn luyện với các tham số như sau:
            - Tham số **k**: $10$ - $200$ với bước nhảy là $10$.
            - Tham số weights: **distance**.
        - Độ đo đuợc sử dụng để đánh giá mô hình là **IoU**.
        """
    )
    st.columns(3)[1].image(
        os.path.join(__SERVICE_DIR, "iou_equation.webp"),
        use_column_width=True,
        caption="Công thức IoU",
    )
    st.write(
        "- Biểu đồ thể hiện **Average IoU** của mô hình với các giá trị **k** của model **kNN**:"
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
            - Tham số **k** tốt nhất: ${k_knn_max}$.
            - **Average IoU** tốt nhất: ${average_iou_max}$.
        """
    )

    @st.fragment
    def display_result_test():
        st.subheader("2.3. Kết quả trên tập dữ liệu kiểm thử")
        with open(os.path.join(__SERVICE_DIR, "tests/labels/labels.txt")) as f:
            for i in range(2):
                _cols = st.columns(5)
                for j in range(5):
                    _data = f.readline().strip().split()
                    _img_path = os.path.join(__SERVICE_DIR, "tests/images", _data[0])
                    _x, _y, _w, _h = map(int, _data[2:])
                    _image = cv2.imread(_img_path)

                    _height, _width = _image.shape[:2]
                    _ground_truth = np.zeros((_height, _width), dtype=np.uint8)
                    _ground_truth[_y : _y + _h, _x : _x + _w] = 1

                    _mask = np.zeros((_height, _width), dtype=np.uint8)
                    with open(
                        os.path.join(__SERVICE_DIR, "tests/results", _data[0] + ".txt")
                    ) as fi:
                        _faces = fi.readlines()
                        for _face in _faces:
                            x, y, w, h = map(int, _face.strip().split())
                            _mask[y : y + h, x : x + w] = 1
                            _image = cv2.rectangle(
                                _image, (x, y), (x + w, y + h), (0, 255, 255), 2
                            )

                    _iou = get_iou(_ground_truth, _mask)
                    _cols[j].image(
                        _image,
                        use_column_width=True,
                        channels="BGR",
                        caption=f"IoU: {_iou:.5f}",
                    )

    display_result_test()


def display_result():
    st.header("3. Ứng dụng phát hiện khuôn mặt")
    uploaded_file = st.file_uploader("Chọn một file ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("Đang phát hiện khuôn mặt..."):
            image = Image.open(uploaded_file)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            faces = extractor.detect_multiscale(image)
            for x, y, w, h in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            st.image(image, channels="BGR")





display_datasets()
display_training()
display_result()
