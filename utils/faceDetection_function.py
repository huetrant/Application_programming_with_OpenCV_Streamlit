import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.neighbors import KNeighborsClassifier


class Extractor:
    def __init__(self, cascade_file, n_neighbors=5):
        self.haar_features = []
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")

        cascade = ET.parse(cascade_file)
        features = cascade.getroot().findall(".//features/_/rects")
        for feature in features:
            haar_feature = []
            rects = feature.findall("_")
            for rect in rects:
                x, y, w, h, wt = map(int, map(float, rect.text.strip().split()))
                haar_feature.append((x, y, w, h, wt))
            self.haar_features.append(haar_feature)

    def extract_feature_image(self, img):
        """Extract the haar feature for the current image"""
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ii = cv2.integral(image)
        features = []
        for haar_feature in self.haar_features:
            value = 0
            for rect in haar_feature:
                x, y, w, h, wt = rect
                value += wt * (
                    ii[y + h][x + w] - ii[y][x + w] - ii[y + h][x] + ii[y][x]
                )
            features.append(value)
        return np.asarray(features)

    def fit(self, X, y):
        self.knn.fit(X, y)

    def detect_multiscale(self, img, scale_factor=1.1, step_size=10, prob=0.85):
        """Detect objects in the image at multiple scales"""
        faces = []
        org_h, org_w = img.shape[:2]

        scale = 1.0
        while scale > 0.1:
            resized = cv2.resize(img.copy(), (int(org_w * scale), int(org_h * scale)))
            new_h, new_w = resized.shape[:2]

            arr_boxs, arr_features = [], []
            for y in range(0, new_h - 24, step_size):
                for x in range(0, new_w - 24, step_size):
                    roi = resized[y : y + 24, x : x + 24]
                    features = self.extract_feature_image(roi)
                    arr_features.append(features)
                    arr_boxs.append((x, y, 24, 24))

            if len(arr_features) > 0:
                _prob = self.knn.predict_proba(arr_features)
                _cls = self.knn.predict(arr_features)
                for i in range(len(_cls)):
                    if _cls[i] == 1 and _prob[i][1] >= prob:
                        x, y, w, h = arr_boxs[i]
                        real_x, real_y = int(x / scale), int(y / scale)
                        real_w, real_h = int(w / scale), int(h / scale)
                        faces.append((real_x, real_y, real_w, real_h))

            scale /= scale_factor

        return faces


def get_iou(ground_truth: np.ndarray, mask: np.ndarray):
    """Calculate the Intersection over Union (IoU) between the ground truth and the mask"""
    intersection = np.logical_and(ground_truth, mask)
    union = np.logical_or(ground_truth, mask)
    return np.sum(intersection) / np.sum(union)
