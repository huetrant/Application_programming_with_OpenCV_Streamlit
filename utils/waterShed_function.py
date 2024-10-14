import cv2
import numpy as np


def license_plate_watershed_segmentation(
    img: np.ndarray, kernel_size: int, thres: float
) -> cv2.typing.MatLike:

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray_scale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

  
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background 
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    #sure foreground 
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, thres * dist_transform.max(), 255, 0)

    # unknown
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker 
    _, maskers = cv2.connectedComponents(sure_fg)

    
    maskers = maskers + 1

    maskers[unknown == 255] = 0

    # Apply watershed
    maskers = cv2.watershed(img, maskers)

    return maskers


def get_mask_license_plate(masks: cv2.typing.MatLike | np.ndarray) -> np.ndarray:


    h, w = masks.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    labels = np.unique(masks)
    for label in labels:
        if label == -1:
            continue

        digit = np.where(masks == label)
        min_w = np.min(digit[1])
        max_w = np.max(digit[1])
        min_h = np.min(digit[0])
        max_h = np.max(digit[0])

        ratio_w = (max_w - min_w) / w
        ratio_h = (max_h - min_h) / h

        check_ratio_w = 0 <= ratio_w and ratio_w <= 0.3
        check_ratio_h = 0.3 <= ratio_h and ratio_h <= 0.5

        if check_ratio_w and check_ratio_h:
            mask[masks == label] = 1

    return mask


def get_iou(ground_truth: np.ndarray, pred: np.ndarray) -> float:

    intersection = np.logical_and(ground_truth, pred)
    union = np.logical_or(ground_truth, pred)

    iou = np.sum(intersection) / np.sum(union)
    return iou


