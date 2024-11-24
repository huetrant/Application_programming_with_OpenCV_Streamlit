# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

from itertools import product

import numpy as np
import cv2 as cv


class YuNet:
    def __init__(
        self,
        modelPath,
        inputSize=[320, 320],
        confThreshold=0.6,
        nmsThreshold=0.3,
        topK=5000,
        backendId=0,
        targetId=0,
    ):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize)  # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId,
        )

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId,
        )

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return np.empty(shape=(0, 5)) if faces[1] is None else faces[1]


def detect_faces(detector: YuNet, img: np.ndarray, scale_factor: float = 1.1):
    org_h, org_w = img.shape[:2]

    scale = 1.0
    while scale * min(org_w, org_h) > 50:
        resized = cv.resize(img.copy(), (int(org_w * scale), int(org_h * scale)))
        new_h, new_w = resized.shape[:2]

        detector.setInputSize((new_w, new_h))
        faces = detector.infer(resized)

        if len(faces) == 1:
            return (faces, scale)
        if len(faces) > 1:
            return ([], 1.0)

        scale /= scale_factor

    return ([], 1.0)
