import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas, CanvasResult

def grabcut(image: np.ndarray, rect: np.ndarray, mask: np.ndarray, mode: int):
    """Thực hiện thuật toán GrabCut để tách nền."""
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    final_mask, _, __ = cv2.grabCut(
        img=image,
        mask=mask,
        rect=rect,
        bgdModel=bg_model,
        fgdModel=fg_model,
        iterCount=5,
        mode=mode,
    )
    grabcut_mask = np.where(
        (final_mask == cv2.GC_PR_BGD) | (final_mask == cv2.GC_BGD), 0, 1
    ).astype("uint8")
    return image * grabcut_mask[:, :, np.newaxis], final_mask


def extract_canvas_objects(canvas_data):
    """Lấy các hình chữ nhật và nét vẽ từ canvas."""
    if not canvas_data or not canvas_data.json_data:
        return [], [], []

    objects = canvas_data.json_data.get("objects", [])
    rects = [obj for obj in objects if obj["type"] == "rect"]
    true_fgs = [obj for obj in objects if obj["type"] == "path" and obj["stroke"] == "rgb(0, 255, 0)"]
    true_bgs = [obj for obj in objects if obj["type"] == "path" and obj["stroke"] == "rgb(255, 0, 0)"]

    return rects, true_fgs, true_bgs


def init_session_state(keys):
    """Khởi tạo các giá trị mặc định trong session_state."""
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None
            
def create_canvas(image, drawing_mode, stroke_width):
    """Hiển thị canvas để người dùng vẽ."""
    width, height = image.size
    display_width = 700  # Tăng giá trị chiều rộng hiển thị
    display_height = display_width * height // width

    mode = "rect" if drawing_mode == "rect" else "freedraw"
    color = {
        "rect": "rgb(0, 0, 255)",  # Màu xanh dương cho vùng chọn (rectangle)
        "sure_fg": "rgb(0, 255, 0)",  # Màu xanh lá cho vùng foreground
        "sure_bg": "rgb(255, 0, 0)"  # Màu đỏ cho vùng background
    }[drawing_mode]

    return st_canvas(
        background_image=image,
        drawing_mode=mode,
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=color,
        width=display_width + 1,
        height=display_height + 1,
        key="full_app",
    )



def process_image(image, canvas_data, rects, true_fgs, true_bgs):
    """Xử lý ảnh với thuật toán GrabCut."""
    np_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    original_height, original_width = np_image.shape[:2]
    canvas_height, canvas_width = canvas_data.image_data.shape[:2]

    scale = original_width / canvas_width
    rect = [int(rects[0][key] * scale) for key in ["left", "top", "width", "height"]]

    mask = st.session_state["final_mask"] if st.session_state["final_mask"] is not None else np.zeros((original_height, original_width), np.uint8)

    def apply_paths(paths, label):
        """Áp dụng các đường vẽ lên mask."""
        for path in paths:
            for sub_path in path["path"]:
                points = (np.array(sub_path[1:]) * scale).astype(int).reshape((-1, 2))
                if len(points) == 1:
                    cv2.circle(mask, points[0], path["strokeWidth"], label, -1)
                else:
                    cv2.polylines(mask, [points], False, label, path["strokeWidth"])

    apply_paths(true_fgs, cv2.GC_FGD)
    apply_paths(true_bgs, cv2.GC_BGD)

    mode = cv2.GC_INIT_WITH_RECT if not (true_fgs or true_bgs) else cv2.GC_INIT_WITH_MASK
    result, final_mask = grabcut(np_image, rect, mask, mode)

    st.session_state["final_mask"] = final_mask
    st.session_state["result_grabcut"] = result
    return result