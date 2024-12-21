import cv2, os
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import PIL.Image as Image

from utils.waterShed_function import (get_iou,get_mask_license_plate,license_plate_watershed_segmentation)

st.set_page_config(
    page_title="Hue Tran_Watershed Segmentation",
    page_icon=Image.open("./public/images/logo_For_Me.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <h1 style='text-align: center;'>Áp dụng Watershed Segmentation trong phân đoạn ký tự biển số xe</h1>
    """,
    unsafe_allow_html=True
)

__DATA_DIR = "./data/watershed"
__IMAGES_DIR = os.path.join(__DATA_DIR,"images")
__LABELS_DIR = os.path.join(__DATA_DIR,"labels")

train_images_name = os.listdir(os.path.join(__IMAGES_DIR, "train"))
test_images_name = os.listdir(os.path.join(__IMAGES_DIR, "test"))

train_images, train_labels = [], []
for image_name in train_images_name:
    train_images.append(cv2.imread(os.path.join(__IMAGES_DIR, "train", image_name)))
    label_name = Path(image_name).stem + ".png"
    train_labels.append(
        cv2.imread(
            os.path.join(__LABELS_DIR, "train", label_name), cv2.IMREAD_GRAYSCALE
        )
    )

test_images, test_labels = [], []
for image_name in test_images_name:
    test_images.append(cv2.imread(os.path.join(__IMAGES_DIR, "test", image_name)))
    label_name = Path(image_name).stem + ".png"
    test_labels.append(
        cv2.imread(os.path.join(__LABELS_DIR, "test", label_name), cv2.IMREAD_GRAYSCALE)
    )
  

df = pd.read_csv(os.path.join(__DATA_DIR, "avg.csv"))
average_ious = df.to_numpy()[:, 1:].T

best_param_iou = {
    "kernel_size": 2
    * np.argmax(max(average_ious[i]) for i in range(average_ious.shape[0]))
    + 3,
    "thres": (np.argmax(average_ious) % average_ious.shape[1]) / average_ious.shape[1],
}

def display_datasets():
    st.header("1. Dataset")
    st.write(
        "- Tập dữ liệu bao gồm $4$ ảnh chia thành $2$ ảnh cho tập train và  $2$ ảnh cho tập test"
    )

    cols = st.columns(2)
    with cols[0]:
        st.subheader("1.1. Train")
        sub_cols = st.columns(2)

        for i in range(2):
            sub_cols[i].image(
                train_images[i],
                caption=f"Ảnh {i+1} của tập train",
                 use_container_width=True,
                channels="BGR",
            )
            sub_cols[i].image(
                train_labels[i],
                caption=f"Ground truth ảnh {i+1} của tập train",
                 use_container_width=True,
            )

    with cols[1]:
        st.subheader("1.2. Test")
        sub_cols = st.columns(2)

        for i in range(2):
            sub_cols[i].image(
                test_images[i],
                caption=f"Ảnh {i+1} của tập test",
                 use_container_width=True,
                channels="BGR",
            )
            sub_cols[i].image(
                test_labels[i],
                caption=f"Ground truth ảnh {i+1} của tập test",
                 use_container_width=True,
            )


def display_tranning_process():
    st.header("2. Tranning")

    def display_pipline():
        st.subheader("2.1. Phân đoạn ký tự bằng Watershed Segmentation")
        st.image(os.path.join(__DATA_DIR, "pipeline.png"),caption="Pipeline quá trình phân đoạn ký tự bằng Watershed")
        st.markdown(
            """
            Mô tả các bước trong quá trình phân đoạn ký tự bằng Watershed Segmentation:
            - (1) Chuyển đổi thành ảnh xám.
            - (2) Chuyển đổi thành ảnh nhị phân áp dụng **Otsu's Binarization**.
            - (3) Xác định **Distance Transform**.
            - (4) Xác định **Sure Foreground** dựa trên ngưỡng
                <br>Giá trị **Distance Transform** của pixel  > **Thres** * **Max(Distance)** sẽ được giữ lại
                - **Thres**  là ngưỡng được chọn có giá trị nằm trong đoạn [0,1].
                - **Max(Distance)** là giá trị pixel lớn nhất của ảnh **Distance Transform**.
            - (5) Xác định **Sure Background**.
            - (6) Xác định **Marker**.
            - (7), (8) Từ **Sure background** và **Sure foreground** để tìm ra vùng **Unknown**.
            - (9) Áp dụng thuật toán **Watershed Segmentation** để phân đoạn ký tự.
            - (10) Xác định các đối tượng ký tự nhờ vị trí, chiều cao, chiều rộng.

            """, unsafe_allow_html=True
        )

    def display_metrics():
        st.subheader("2.2. Độ đo IoU")

        st.markdown(
            """
            - **IoU (Intersection over Union)** là một độ đo phổ biến trong các bài toán phân đoạn ảnh và phát hiện đối tượng, được sử dụng để so sánh sự trùng khớp giữa hai vùng: ground truth (vùng thực tế) và predicted area (vùng dự đoán).
            - **IoU** đo lường mức độ chồng lấn giữa hai vùng, tính bằng cách lấy tỉ lệ giữa phần giao và phần hợp của hai vùng. Công thức tính IoU như sau:
            """
        )
        st.columns([1, 2, 1])[1].image(
            os.path.join(__DATA_DIR, "IoU.webp"),
             use_container_width=True,
            caption="Công thức IoU",
        )

    def display_hyperparameters():
        st.subheader("2.3. Tham số tối ưu")

        st.write("- Biểu đồ giá Average IoU trên tập train khi thay đổi threshold và kernel size.")

        st.line_chart(
            {
                "thres": np.linspace(0, 1, average_ious.shape[1]),
                "kernel_size = 3": average_ious[0],
                "kernel_size = 5": average_ious[1],
                "kernel_size = 7": average_ious[2],
                "kernel_size = 9": average_ious[3],
            },
            x="thres",
            y=[
                "kernel_size = 3",
                "kernel_size = 5",
                "kernel_size = 7",
                "kernel_size = 9",
            ],
            y_label="Average IoU",
        )

        st.markdown(
            """
            - Average IoU tốt nhất: ${:.6f}$
            - Tham số cho kết quả Average IoU tốt nhất là:
                - kernel_size = ${}$
                - thres = ${:.2f}$

            """.format(
                np.max(average_ious),
                best_param_iou["kernel_size"],
                best_param_iou["thres"],
            )
        )

         

    def display_visualize():
        st.subheader("2.4. Minh hoạ quá trình huấn luyện trên tập train theo từng bộ tham số")
# Custom CSS to display radio buttons horizontally
        st.markdown(
            """
            <style>
            .horizontal-radio .stRadio > div {
                display: flex;
                flex-direction: row;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Apply the CSS class to the radio button
        kernel_size = st.radio(
            "Chọn kernel size:",
            (3, 5, 7, 9),
            format_func=lambda x: f"{x} x {x}",
            key="kernel_radio",
            label_visibility="visible",
            horizontal=True
        )

        # Slider for threshold
        thres = st.slider(
            "Chọn thres:", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"
        )
        cols = st.columns(4)
        cols[0].columns([2, 2, 1])[1].write("**Ảnh gốc**")
        cols[1].columns([2, 2, 1])[1].write("**Ground truth**")
        cols[2].columns([2, 2, 1])[1].write("**Predict**")
        cols[3].columns([2, 2, 1])[1].write("**Giá trị IoU**")

        for i in range(2):
            cols = st.columns(4, vertical_alignment="center")

            with cols[0]:
                st.image(
                    train_images[i],
                    caption=f"Ảnh {i+1} của tập train",
                     use_container_width=True,
                    channels="BGR",
                )

            with cols[1]:
                st.image(
                    train_labels[i],
                    caption=f"Ground truth ảnh {i+1} của tập train",
                     use_container_width=True,
                )

            with cols[2]:
                masks = license_plate_watershed_segmentation(
                    train_images[i], kernel_size, thres
                )
                mask = get_mask_license_plate(masks)
                _mask = mask.copy()
                _mask[_mask == 1] = 255
                st.image(_mask, caption=f"Dụ đoán đối tượng ảnh {i+1} của tập train")

            with cols[3]:
                _label = np.copy(train_labels[i])
                _label[_label == 255] = 1
                
                # Căn giữa giá trị IoU
                iou_value = get_iou(_label, mask)
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: center; align-items: center; height: 100%;'>
                        <h3>{iou_value:.4f}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    display_pipline()
    display_metrics()
    display_hyperparameters()
    display_visualize()


def display_result():
    st.header("3.Testing")

    cols = st.columns(4)
    cols[0].columns([2, 2, 1])[1].write("**Ảnh gốc**")
    cols[1].columns([2, 2, 1])[1].write("**Ground truth**")
    cols[2].columns([2, 2, 1])[1].write("**Predict**")
    cols[3].columns([2, 2, 1])[1].write("**Giá trị IoU**")

    for i in range(len(test_images)):
        cols = st.columns(4, vertical_alignment="center")

        with cols[0]:
            st.image(
                test_images[i],
                caption=f"Ảnh {i+1} của tập test",
                 use_container_width=True,
                channels="BGR",
            )

        with cols[1]:
            st.image(
                test_labels[i],
                caption=f"Ground truth ảnh {i+1} của tập test",
                 use_container_width=True,
            )

        with cols[2]:
            masks = license_plate_watershed_segmentation(
                test_images[i],
                int(best_param_iou["kernel_size"]),
                best_param_iou["thres"],
            )
            mask = get_mask_license_plate(masks)
            _mask = mask.copy()
            _mask[_mask == 1] = 255
            st.image(
                _mask,
                caption=f"Dụ đoán đối tượng ảnh {i+1} của tập test",
                 use_container_width=True,
            )
            _label = np.copy(test_labels[i])
            _label[_label == 255] = 1
            iou_iou = get_iou(_label, mask)

        
        with cols[3]:
            _label = np.copy(train_labels[i])
            _label[_label == 255] = 1
            
            # Căn giữa giá trị IoU
            st.markdown(
                f"""
                <div style='display: flex; justify-content: center; align-items: center; height: 100%;'>
                    <h3>{iou_iou:.4f}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

display_datasets()
display_tranning_process()
display_result()
