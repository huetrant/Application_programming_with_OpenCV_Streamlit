import cv2
import numpy as np
from PIL import Image
import streamlit as st
from utils.grabCut_function import grabcut
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Hue Tran _ GrabCut",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Áp dụng thuật toán GrabCut trong bài toán tách nền")

st.markdown(""" 
        - **Thêm 1 ảnh cần tách nền**
        - **Chọn vùng cần tách nền bằng cách kéo thả chuột trái trong ảnh**  
        - **Sau đó nhấn vào nút **Submit****
        """) 

uploaded_image = st.file_uploader(
    "Chọn hoặc kéo ảnh vào ô bên dưới", type=["jpg", "jpeg", "png"]
)



if uploaded_image is not None:
    # Chia layout thành 3 cột, cột giữa để chứa button
    cols = st.columns([1, 0.5, 1], gap="large")

    raw_image = Image.open(uploaded_image)

    with cols[0]:
        w, h = raw_image.size
        width = min(w, 475)
        height = width * h // w

        # Resize image to maintain aspect ratio
        raw_image = raw_image.resize((width, height), Image.Resampling.LANCZOS)

        canvas_result = st_canvas(
            background_image=raw_image,
            drawing_mode="rect",
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=1,
            width=width,
            height=height,
        )

    # Button nằm ở cột giữa và được căn giữa bằng CSS
    with cols[1]:
        st.markdown(
            """
            <style>
            .center-button {
                display: flex;
                justify-content: center;
                align-items: center; /* Center vertically */
                height: 100%; /* Full height of the column */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Sử dụng class "center-button" để căn giữa
        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        submitBtn = st.button("Submit")
        st.markdown('</div>', unsafe_allow_html=True)

    if canvas_result.json_data is not None and submitBtn:
        if len(canvas_result.json_data.get("objects")) > 1:
            st.error("Chỉ chọn một vùng cần tách nền")
        elif len(canvas_result.json_data.get("objects")) < 1:
            st.error("Hãy chọn một vùng cần tách nền")
        elif len(canvas_result.json_data.get("objects")) == 1:
            with st.spinner("Đang xử lý..."):
                original_image = np.array(raw_image)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
                scale = original_image.shape[1] / width

                min_x = canvas_result.json_data.get("objects")[0]["left"]
                min_y = canvas_result.json_data.get("objects")[0]["top"]
                obj_width = canvas_result.json_data.get("objects")[0]["width"]
                obj_height = canvas_result.json_data.get("objects")[0]["height"]

                res = grabcut(
                    original_image=original_image,
                    rect=(
                        int(min_x * scale),
                        int(min_y * scale),
                        int(obj_width * scale),
                        int(obj_height * scale),
                    ),
                )

                # Resize processed image to match the size of raw image
                res_resized = cv2.resize(res, (width, height))

                # Hiển thị ảnh sau khi tách nền ở cột bên phải
                with cols[2]:
                    cols[2].image(res_resized, channels="BGR", caption="Ảnh kết quả")
