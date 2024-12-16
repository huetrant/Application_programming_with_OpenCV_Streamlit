import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Hue Tran _ Image_processing",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ứng dụng xử lý ảnh")

def flip_image(image, flip_code):
    """Flip the image.
    flip_code: 0 (vertical), 1 (horizontal), -1 (both).
    """
    return cv2.flip(image, flip_code)

def rotate_image(image, angle):
    """Rotate the image by a specified angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def convert_colorspace(image, conversion_code):
    """Convert the image color space."""
    return cv2.cvtColor(image, conversion_code)

def translate_image(image, x_shift, y_shift):
    """Translate the image and crop it to remove the shifted parts."""
    (h, w) = image.shape[:2]
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv2.warpAffine(image, matrix, (w, h))
    x_start = max(0, x_shift)
    y_start = max(0, y_shift)
    x_end = w - max(0, -x_shift)
    y_end = h - max(0, -y_shift)
    return translated_image[y_start:y_end, x_start:x_end]

def crop_image(image, start_x, start_y, width, height):
    """Crop the image."""
    return image[start_y:start_y+height, start_x:start_x+width]

def main():
    
    st.markdown(
            """
            Công cụ này cho phép bạn tải lên một hình ảnh và áp dụng các kỹ thuật xử lý ảnh khác nhau.
            """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            #### Hướng dẫn sử dụng:
            1. **Tải ảnh lên**: Sử dụng công cụ tải lên để chọn một file ảnh (JPG, JPEG hoặc PNG).
            2. **Chọn chức năng**: Chọn một trong các kỹ thuật xử lý từ danh sách.
            3. **Tùy chỉnh tham số**: Tùy chỉnh các tham số phù hợp với kỹ thuật bạn chọn.
            4. **Áp dụng xử lý**: Nhấn nút để xem kết quả xử lý ảnh.
            Hãy tận hưởng việc khám phá các kỹ thuật xử lý ảnh!
            """
        )

    with col2:
        st.markdown(
            """
            #### Các chức năng có sẵn:
            - **Lật ảnh (Flip)**: Lật ảnh theo chiều ngang, dọc hoặc cả hai.
            - **Xoay ảnh (Rotate)**: Xoay ảnh theo góc bất kỳ.
            - **Chuyển đổi không gian màu (Convert Colorspace)**: Chuyển đổi ảnh sang các không gian màu khác (ví dụ: Grayscale, HSV).
            - **Dịch chuyển (Translate)**: Dịch chuyển ảnh theo trục X và Y.
            - **Cắt ảnh (Crop)**: Chọn và cắt một phần cụ thể của ảnh.
            """
        )

    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        pil_image = Image.open(uploaded_file)

        option = st.selectbox("Chọn chức năng:", 
                             ["Lật ảnh (Flip)", "Xoay ảnh (Rotate)", "Chuyển đổi không gian màu (Convert Colorspace)", "Dịch chuyển (Translate)", "Cắt ảnh (Crop)"])

        transformed_image = None

        if option == "Lật ảnh (Flip)":
            flip_code = st.radio("Kiểu lật:", ("Chiều ngang", "Chiều dọc", "Cả hai"),horizontal=True)
            flip_map = {"Chiều ngang": 1, "Chiều dọc": 0, "Cả hai": -1}
            if st.button("Áp dụng lật ảnh"):
                transformed_image = flip_image(image, flip_map[flip_code])

        elif option == "Xoay ảnh (Rotate)":
            angle = st.slider("Chọn góc xoay (độ):", -180, 180, 0)
            if st.button("Áp dụng xoay ảnh"):
                transformed_image = rotate_image(image, angle)

        elif option == "Chuyển đổi không gian màu (Convert Colorspace)":
            colorspace = st.radio("Không gian màu:", ("Grayscale", "HSV", "LAB", "RGB", "XYZ", "YUV", "HLS", "Luv", "YCrCb"),horizontal=True)
            color_map = {
                "Grayscale": cv2.COLOR_BGR2GRAY,
                "HSV": cv2.COLOR_BGR2HSV,
                "LAB": cv2.COLOR_BGR2LAB,
                "RGB": cv2.COLOR_BGR2RGB,
                "XYZ": cv2.COLOR_BGR2XYZ,
                "YUV": cv2.COLOR_BGR2YUV,
                "HLS": cv2.COLOR_BGR2HLS,
                "Luv": cv2.COLOR_BGR2LUV,
                "YCrCb": cv2.COLOR_BGR2YCrCb
            }
            if st.button("Áp dụng chuyển đổi màu"):
                transformed_image = convert_colorspace(image, color_map[colorspace])

        elif option == "Dịch chuyển (Translate)":
            # Giới hạn giá trị dịch chuyển theo chiều rộng và chiều cao của ảnh
            max_x_shift = image.shape[1]  # Chiều rộng ảnh
            max_y_shift = image.shape[0]  # Chiều cao ảnh

            x_shift = st.slider("Dịch chuyển theo trục X (pixel):", -max_x_shift, max_x_shift, 0)
            y_shift = st.slider("Dịch chuyển theo trục Y (pixel):", -max_y_shift, max_y_shift, 0)

            if st.button("Áp dụng dịch chuyển"):
                transformed_image = translate_image(image, x_shift, y_shift)
                


        elif option == "Cắt ảnh (Crop)":
            # Chỉ hiển thị ảnh gốc và cho phép vẽ vùng cắt
            st.markdown("### Vẽ vùng cần cắt:")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Màu nền cho vùng vẽ
                stroke_width=2,
                stroke_color="rgba(255, 0, 0, 0.5)",  # Màu viền vùng vẽ
                background_image=pil_image,  # Chuyển đổi ảnh gốc sang PIL và sử dụng làm nền
                width=image.shape[1],
                height=image.shape[0],
                drawing_mode="rect",  # Chế độ vẽ hình chữ nhật
                key="canvas",
            )

            # Khi người dùng nhấn Submit, lấy vùng cắt và thực hiện
            if st.button("Submit"):
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        # Lấy thông tin vùng cắt
                        x_start = int(objects[0]["left"])
                        y_start = int(objects[0]["top"])
                        width = int(objects[0]["width"])
                        height = int(objects[0]["height"])

                        # Cắt ảnh theo vùng vẽ
                        transformed_image = crop_image(image, x_start, y_start, width, height)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc")
        if transformed_image is not None:
            with col2:
                st.image(transformed_image, caption="Ảnh kết quả")

if __name__ == "__main__":
    main()

    
