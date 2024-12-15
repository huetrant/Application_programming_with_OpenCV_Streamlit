import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


st.set_page_config(
    page_title="Hue Tran _ handwriting_letter_recognition",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Handwriting Letter Recognition")
# Đường dẫn lưu trữ bộ dữ liệu MNIST
DATA_DIR = './data/handwriting_letter_recognition/mnist_data.npz'
history_data = np.load('./data/handwriting_letter_recognition/history.npy', allow_pickle=True).item()
model = load_model('./data/handwriting_letter_recognition/mnist_model.h5')
# Hàm tiền xử lý ảnh
def preprocess_image(img):
    img = img.convert('L')  # Chuyển đổi ảnh thành grayscale (1 kênh)
    img = img.resize((28, 28))  # Resize ảnh về kích thước 28x28
    img = np.array(img)  # Chuyển đổi ảnh thành mảng numpy
    img = img.astype('float32') / 255.0  # Chuẩn hóa ảnh
    img = np.expand_dims(img, axis=-1)  # Thêm chiều kênh (grayscale)
    return np.expand_dims(img, axis=0)  # Thêm chiều batch

# Hàm dự đoán và hiển thị kết quả
def predict_image(img):
    # Tiền xử lý ảnh
    img = preprocess_image(img)
    # Dự đoán
    prediction = model.predict(img)
    # Lấy lớp có xác suất cao nhất
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# Kiểm tra xem bộ dữ liệu đã được lưu chưa
if not os.path.exists(DATA_DIR):
    # Nếu bộ dữ liệu chưa được lưu, tải và lưu lại vào tệp
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Chuẩn hóa dữ liệu về phạm vi [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Lưu bộ dữ liệu vào tệp .npz
    np.savez_compressed(DATA_DIR, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
else:
    # Nếu bộ dữ liệu đã được lưu, tải lại từ tệp
    data = np.load(DATA_DIR)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# Hàm hiển thị hình ảnh cho mỗi chữ số (0-9)
def display_mnist_images():
    st.header("1. Dataset")
    st.write("""
        **MNIST** (Modified National Institute of Standards and Technology) là một bộ dữ liệu nổi tiếng trong lĩnh vực học máy, 
             đặc biệt là trong các bài toán phân loại hình ảnh. Được thu thập từ các nhà trường và các nhân viên bưu điện tại Mỹ. 
             Đây là bộ dữ liệu được sử dụng rộng rãi trong nghiên cứu học sâu (deep learning) và học máy (machine learning) 
             để thử nghiệm các mô hình phân loại hình ảnh.

      ### Các đặc điểm của bộ dữ liệu:
      - **Số lượng hình ảnh**: **70,000** (60,000 cho huấn luyện, 10,000 cho kiểm tra).
      - **Kích thước hình ảnh**: Mỗi hình ảnh có kích thước **28x28** pixels.
      - **Nhãn**: Mỗi hình ảnh là một chữ số từ **0** đến **9**.
      - **Màu sắc**: Ảnh đen trắng.
      - Minh họa một số ảnh trong bộ dữ liệu **MNIST**: 
      """)
    # Tạo một dictionary trống để lưu hình ảnh cho mỗi chữ số
    images_by_digit = {i: [] for i in range(10)}
    
    # Lặp qua dữ liệu huấn luyện và thu thập 5 ví dụ cho mỗi chữ số
    for i in range(len(x_train)):
        digit = y_train[i]
        if len(images_by_digit[digit]) < 5:
            images_by_digit[digit].append(x_train[i])
        if all(len(images) == 5 for images in images_by_digit.values()):
            break  # Dừng lại khi đã có 5 hình ảnh cho mỗi chữ số

    # Tạo một container Streamlit để hiển thị 5 hình ảnh cho mỗi chữ số
    fig, axes = plt.subplots(5, 10, figsize=(12, 6))  # Giảm chiều cao của figure xuống
    for i in range(10):
        for j in range(5):
            axes[j, i].imshow(images_by_digit[i][j], cmap='gray')
            axes[j, i].axis('off')
            if j == 0:
                axes[j, i].set_title(f"{i}")
    
    # Điều chỉnh khoảng cách giữa các hàng và cột
    plt.tight_layout(pad=0.5)  # Giảm khoảng cách giữa các ô

    st.pyplot(fig)

def display_model():
    st.header("2. Mô hình sử dụng")
    st.write(
        """
        - Để nhận dạng chữ số viết tay mô hình **CNN** được sử dụng với kiến trúc như sau:
        """

    )
    image1 = Image.open('./data/handwriting_letter_recognition/model1.png')
    image2 = Image.open('./data/handwriting_letter_recognition/model2.png')

    # Tạo hai cột để hiển thị ảnh song song
    col1, col2 = st.columns(2)

    # Hiển thị ảnh trong mỗi cột
    with col1:
        st.image(image1,  use_column_width=True)
        st.write(
        """
        - **Convolutional Layer**: lớp tích chập dùng để trích xuất các đặc trưng của ảnh.
          - **Padding** = 1, **Kernel** = 3x3, **Stride** = 1, **Activation** = RelU
        - **Attention-like Layer**: giúp mô hình tập trung nhiều hơn vào các vùng quan trọng của dữ liệu đầu vào.
          - Một mặt nạ **Attention** được tính qua lớp `Conv2D` với kích thước kernel là (1, 1) và hàm kích hoạt **`sigmoid`**.
          - Kết quả mặt nạ là một **tensor** có cùng kích thước chiều cao, chiều rộng và số kênh như đầu ra của lớp tích chập trước đó.
          - Hàm kích hoạt **`sigmoid`** giúp giới hạn giá trị của mặt nạ trong khoảng [0, 1], biểu thị "mức độ quan trọng" của từng phần trong **tensor** đặc trưng.
          - `tf.keras.layers.Multiply()` thực hiện phép nhân từng phần tử giữa **tensor** đầu ra của lớp tích chập ban đầu (`x`) và mặt nạ **Attention** (`attention`).
          - Kết quả là các đặc trưng quan trọng được tăng cường (nhân với giá trị gần 1), trong khi các đặc trưng ít quan trọng bị giảm bớt (nhân với giá trị gần 0).
        - **Flatten Layer**: Chuyển đổi dữ liệu dạng tensor đa chiều thành dạng 1D, giúp kết nối với các lớp Dense.
        - **Softmax Layer**: Biến đổi đầu ra của mô hình thành các xác suất, dùng để phân loại ảnh vào các lớp.
        - Hàm mất mát **Categorical Cross-Entropy**, thuật toán **Adam**, **learning rate** = $0.001$, **epochs** = $10$
        """

    )

    with col2:
        st.image(image2, use_column_width=True)

def display_result():
    st.markdown("""
    <style>
        .center-text {
            text-align: center;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)
    st.header("3. Kết quả")
    accuracy = history_data['accuracy']
    val_accuracy = history_data['val_accuracy']
    loss = history_data['loss']
    val_loss = history_data['val_loss']

    # Chuyển các giá trị thành DataFrame để dễ dàng hiển thị với st.line_chart
    df_accuracy = pd.DataFrame({
        'Epoch': range(1, len(accuracy) + 1),
        'Training Accuracy': accuracy,
        'Validation Accuracy': val_accuracy
    })

    df_loss = pd.DataFrame({
        'Epoch': range(1, len(loss) + 1),
        'Training Loss': loss,
        'Validation Loss': val_loss
    })
    col1, col2 = st.columns([1, 2])
    # Hiển thị biểu đồ accuracy
    with col1:
      st.markdown('<div class="center-text">Biểu đồ kết quả huấn luyện</div>', unsafe_allow_html=True)
      st.line_chart(df_accuracy.set_index('Epoch'),color=["#00BFFF", "#F39C12"])
      st.line_chart(df_loss.set_index('Epoch'),color=["#2ECC71", "#FF0000"])
    with col2:
      st.markdown('<div class="center-text"><strong>Ma trận nhầm lẫn</strong></div>', unsafe_allow_html=True)
      image = Image.open('./data/handwriting_letter_recognition/confusion_matrix.png')
      st.image(image,  use_column_width=True)

def display_app():
    st.header("4. Ứng dụng")
    # Tải ảnh từ người dùng
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png", "bmp", "gif"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)  # Mở ảnh từ file tải lên
        
        # Sử dụng Streamlit columns để hiển thị ảnh và kết quả song song
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Uploaded Image")  # Hiển thị ảnh gốc
        
        with col2:
            predicted_class = predict_image(img)  # Dự đoán lớp
            st.write(f"Dự đoán: {predicted_class}") 
# Hiển thị hình ảnh MNIST
display_mnist_images()
display_model()
display_result()
display_app()