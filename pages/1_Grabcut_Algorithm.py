from PIL import Image
import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from utils.Grabcut.grabCut_function import extract_canvas_objects,init_session_state,create_canvas,create_canvas,process_image



def render_guide():
    """Hiển thị hướng dẫn sử dụng ứng dụng."""
    st.markdown(
        """
        ### Hướng dẫn Sử dụng

        **1. Tải ảnh**: Nhấn **"Chọn ảnh"** để tải lên file (JPG, PNG).  

        **2. Chọn chế độ vẽ**:  
        - **Rect**: Vẽ vùng bao quanh đối tượng cần tách.  
        - **Foreground**: Tô vùng muốn giữ lại.  
        - **Background**: Tô vùng muốn loại bỏ.  

        **3. Tách nền**: Nhấn **"Tách nền"** để xử lý ảnh. 
        """
    )



def render_form():
    """Tạo form với các radio buttons và chọn độ dày nét vẽ cùng hình minh họa."""

    # Hàng đầu tiên: Radio buttons cho các chế độ vẽ
    drawing_mode = st.radio(
        "Chọn chế độ vẽ",
        options=["rect", "sure_fg", "sure_bg"],
        format_func=lambda x: {
            "rect": "Vùng cần tách (rect)",
            "sure_fg": "Vùng giữ lại (foreground)",
            "sure_bg": "Vùng loại bỏ (background)"
        }[x],
        horizontal=True  # Hiển thị các radio button ngang
    )

    # Thiết lập giá trị mặc định cho độ dày nét vẽ
    if drawing_mode == "rect":
        default_thickness = 2  # Độ dày mặc định cho vùng cần tách là 2px
    else:
        default_thickness = 5  # Độ dày mặc định cho vùng giữ lại và loại bỏ là 10px

    # Hàng thứ hai: Chọn độ dày nét vẽ (number_input) và hình minh họa
    # Sử dụng st.columns để chia cột cho Selectbox và hình minh họa
    cols = st.columns([2, 1.5, 7])  # Cột 1 chiếm 3 phần, cột 2 chiếm 1 phần

    # Chọn độ dày nét vẽ từ 1px đến 10px bằng number_input (có nút tăng giảm)
    selected_thickness = cols[0].number_input(
        "Chọn độ dày nét vẽ",
        min_value=1,
        max_value=10,
        value=default_thickness,  # Giá trị mặc định được thay đổi tùy theo chế độ
        step=1,  # Bước tăng giảm là 1
        key="thickness_number_input"
    )

    # Tạo hình minh họa độ dày nét vẽ bằng các đoạn đường thẳng
    images = []
    for thickness in range(1, 11):  # Tạo hình minh họa từ 1px đến 10px
        fig, ax = plt.subplots(figsize=(1.5, 0.5))  # Hình minh họa nhỏ hơn
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.plot([0.1, 0.9], [0.5, 0.5], color='black', linewidth=thickness)
        
        # Lưu hình ảnh vào danh sách
        plt.close(fig)
        image_path = f"line_{thickness}.png"
        fig.savefig(image_path, dpi=150, bbox_inches='tight', transparent=True)
        images.append(image_path)

    # Hiển thị hình minh họa cho độ dày nét vẽ đã chọn trong cột 2
    selected_image = images[selected_thickness - 1]  # Chọn hình minh họa tương ứng với độ dày
    image = Image.open(selected_image)

    # Hiển thị hình ảnh minh họa độ dày nét vẽ
    cols[1].image(image, use_column_width=True)

    return drawing_mode, selected_thickness


# Khởi tạo ứng dụng
init_session_state(["final_mask", "result_grabcut"])
st.set_page_config(
    page_title="Ứng dụng tách nền GrabCut",
    page_icon=":art:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ứng dụng tách nền GrabCut")

# Hướng dẫn sử dụng
render_guide()

# Tải lên ảnh
uploaded_image = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_image:
    drawing_mode, stroke_width = render_form()

    if drawing_mode:
        cols = st.columns(2)
        raw_image = Image.open(uploaded_image)

        with cols[0]:
            canvas_result = create_canvas(raw_image, drawing_mode, stroke_width)
            rects, true_fgs, true_bgs = extract_canvas_objects(canvas_result)

            if drawing_mode == "rect" and len(rects) != 1:
                st.session_state["result_grabcut"] = None
            else:
                with cols[0]:
                    submit_btn = st.button(
                        "Tách nền",
                    )

                if submit_btn:
                    with st.spinner("Đang xử lý..."):
                        result = process_image(
                            raw_image, canvas_result, rects, true_fgs, true_bgs
                        )
                        cols[1].image(result, channels="BGR", caption="Ảnh kết quả", use_column_width=True)
                elif st.session_state["result_grabcut"] is not None:
                    cols[1].image(
                        st.session_state["result_grabcut"],
                        channels="BGR",
                        caption="Ảnh kết quả",
                         use_column_width=True
                    )
