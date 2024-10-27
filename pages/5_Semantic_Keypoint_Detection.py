import os
import cv2
import numpy as np
import streamlit as st


__DATA_DIR = "./data/semantic_keypoint_detection/synthetic_shapes_datasets"
__DATATYPES = [
    os.path.join(__DATA_DIR, "draw_checkerboard"),
    os.path.join(__DATA_DIR, "draw_cube"),
    os.path.join(__DATA_DIR, "draw_ellipses"),
    os.path.join(__DATA_DIR, "draw_lines"),
    os.path.join(__DATA_DIR, "draw_multiple_polygons"),
    os.path.join(__DATA_DIR, "draw_polygon"),
    os.path.join(__DATA_DIR, "draw_star"),
    os.path.join(__DATA_DIR, "draw_stripes"),
]

