import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Nhận Diện Cảm Xúc Từ Khuôn Mặt",
    layout="wide",
    initial_sidebar_state="expanded",
)

Model_DIR = "./models/Thi_hoc_ky"
Data_DIR ="./data/Thi_hoc_ky"
# Tải mô hình và Haar Cascade
model = joblib.load("./models/thi_hoc_ky/emotion_recognition_model.pkl")
cascade = os.path.join(Model_DIR, "haarcascade_frontalface_default.xml")
emotions = ["happy", "sadness", "neutral"]  # Danh sách cảm xúc

# Phần 1: Mô tả Dữ Liệu
st.title("Mô Tả Dữ Liệu Cảm Xúc Từ Khuôn Mặt")
st.markdown("""
### Dữ liệu bao gồm các bức ảnh với các cảm xúc khác nhau: happy, sad, và neutral.
Dưới đây là 3 ví dụ khuôn mặt từ dataset cho mỗi loại cảm xúc.
""")

# Hiển thị ảnh từ dataset trong 3 cột song song
col1, col2, col3 = st.columns(3)

# Lấy đường dẫn tới ảnh cho mỗi loại cảm xúc
data_examples = {emotion: os.listdir(os.path.join(Data_DIR, emotion))[:3] for emotion in emotions}

# Hiển thị ảnh
with col1:
    st.subheader("Happy")
    for img_name in data_examples["happy"]:
        img_path = os.path.join(Data_DIR, "happy", img_name)
        img_data = Image.open(img_path)
        st.image(img_data, caption="Happy", use_column_width=True)

with col2:
    st.subheader("sadness")
    for img_name in data_examples["sadness"]:
        img_path = os.path.join(Data_DIR, "sadness", img_name)
        img_data = Image.open(img_path)
        st.image(img_data, caption="sadness", use_column_width=True)

with col3:
    st.subheader("Neutral")
    for img_name in data_examples["neutral"]:
        img_path = os.path.join(Data_DIR, "neutral", img_name)
        img_data = Image.open(img_path)
        st.image(img_data, caption="Neutral", use_column_width=True)



#  Biểu Diễn Sơ Đồ Minh Họa Luồng Xử Lý
# Tiêu đề trang
st.title("Minh Họa Luồng Xử Lý Nhận Diện Cảm Xúc 🎯")

# Mô tả luồng xử lý
st.markdown("""
### **Luồng Xử Lý**
Ứng dụng nhận diện cảm xúc từ khuôn mặt bao gồm các bước chính như sau:

1. **Nhập Dữ Liệu**
   - Người dùng tải ảnh lên hoặc sử dụng webcam.
2. **Tiền Xử Lý Ảnh**
   - Ảnh đầu vào được chuyển sang dạng grayscale và phát hiện khuôn mặt.
3. **Nhận diện khuôn mặt và Trích Xuất Đặc Trưng**
   - Cắt khuôn mặt, resize ảnh về kích thước chuẩn (48x48), chuyển đổi thành vector.
4. **Dự Đoán Cảm Xúc**
   - Sử dụng mô hình SVM đã huấn luyện để dự đoán nhãn cảm xúc (happy, sad, neutral).
5. **Hiển Thị Kết Quả**
   - Vẽ hình xung quanh khuôn mặt và hiển thị nhãn cảm xúc tương ứng.

""")

# Đường dẫn tới tệp sơ đồ
flow_image_path = os.path.join(Data_DIR, "pipeline.png")

# Kiểm tra nếu tệp sơ đồ tồn tại
if os.path.exists(flow_image_path):
    st.markdown("### **Sơ Đồ Minh Họa Luồng Xử Lý**")
    diagram = Image.open(flow_image_path)  # Mở hình ảnh sơ đồ
    st.image(diagram, caption="Luồng Xử Lý Nhận Diện Cảm Xúc", use_column_width=True)
else:
    st.error("Không tìm thấy tệp sơ đồ luồng xử lý. Vui lòng kiểm tra lại đường dẫn tệp!")

#  Ứng Dụng
st.title("Ứng Dụng Nhận Diện Cảm Xúc")
st.markdown("""
### Tải Ảnh hoặc Sử Dụng Webcam để Nhận Diện Cảm Xúc
""")


# Hàm dự đoán cảm xúc
def predict_emotion(face, model, emotions):
    face_resized = cv2.resize(face, (48, 48))  # Resize ảnh về kích thước chuẩn
    face_flatten = face_resized.flatten().reshape(1, -1)  # Chuyển thành vector 1D
    prediction = model.predict(face_flatten)
    return emotions[prediction[0]]

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Tải ảnh lên để nhận diện cảm xúc", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc và hiển thị ảnh gốc
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Chuyển ảnh sang định dạng NumPy

    # Sao chép ảnh để xử lý
    processed_image = image_np.copy()

    # Kiểm tra nếu ảnh có đủ 3 kênh màu
    if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        # Nếu ảnh đã là grayscale
        gray_image = processed_image

    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cascade)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # Lưu trữ cảm xúc để hiển thị
    emotions_detected = []

    for (x, y, w, h) in faces:
        # Tính tọa độ tâm và bán kính hình tròn
        center_x, center_y = x + w // 2, y + h // 2
        radius = max(w, h) // 2

        # Vẽ hình tròn lên ảnh đã xử lý
        cv2.circle(processed_image, (center_x, center_y), radius, (255, 0, 0), 2)

        # Dự đoán cảm xúc
        face = gray_image[y:y+h, x:x+w]
        emotion = predict_emotion(face, model, emotions)
        emotions_detected.append(emotion)

        # Ghi nhãn cảm xúc lên ảnh
        cv2.putText(processed_image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Hiển thị ảnh gốc và ảnh xử lý cạnh nhau
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Ảnh Gốc", use_column_width=True)
    with col2:
        st.image(processed_image, caption="Ảnh Sau Xử Lý", use_column_width=True)

    # Hiển thị kết quả nhận diện cảm xúc
    if emotions_detected:
        st.write("### Kết Quả Nhận Diện Cảm Xúc:")
        for i, emotion in enumerate(emotions_detected, start=1):
            st.write(f"- Khuôn mặt {i}: **{emotion}**")
    else:
        st.write("Không phát hiện khuôn mặt nào!")

# Thêm phần webcam
st.write("## Nhận Diện Cảm Xúc Từ Webcam")
if st.button("Bật Webcam"):
    st.info("Nhấn 'q' trên cửa sổ webcam để thoát.")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cascade)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Không thể mở webcam. Vui lòng kiểm tra lại!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            if face.size > 0:
                emotion = predict_emotion(face, model, emotions)
                # Vẽ hình tròn và ghi nhãn cảm xúc
                center_x, center_y = x + w // 2, y + h // 2
                radius = max(w, h) // 2
                cv2.circle(frame, (center_x, center_y), radius, (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Webcam - Nhận Diện Cảm Xúc", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
