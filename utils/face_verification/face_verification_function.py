import cv2
import re
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from services.face_verification.yunet import YuNet
from services.face_verification.sface import SFace
from services.face_verification.db import Repository, Storage
from streamlit.runtime.uploaded_file_manager import UploadedFile


class AccentRemover:
    ACCENT_TABLES = str.maketrans(
        "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
        "áàảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ",
        "A" * 17 + "D" + "E" * 11 + "I" * 5 + "O" * 17 + "U" * 11 + "Y" * 5,
        chr(774) + chr(770) + chr(795) + chr(769) + chr(768) + chr(777) + chr(771) + chr(803),
    )

    @staticmethod
    def remove(txt: str) -> str:
        return txt.translate(AccentRemover.ACCENT_TABLES)


class StudentService:
    def __init__(self) -> None:
        self.repository = Repository("students")
        self.storage = Storage()
        self.detector = YuNet(
            "./services/face_verification/models/face_detection_yunet_2023mar.onnx",
            confThreshold=0.85,
        )
        self.embedder = SFace(
            "./services/face_verification/models/face_recognition_sface_2021dec.onnx"
        )

    def _detect_face(self, img: np.ndarray, scale_factor: float = 1.1):
        org_h, org_w = img.shape[:2]
        scale = 1.0
        while scale * min(org_w, org_h) > 50:
            resized = cv2.resize(img.copy(), (int(org_w * scale), int(org_h * scale)))
            new_h, new_w = resized.shape[:2]
            self.detector.setInputSize((new_w, new_h))
            faces = self.detector.infer(resized)
            if len(faces) == 1:
                return faces, scale
            elif len(faces) > 1:
                return [], 1.0
            scale /= scale_factor
        return [], 1.0

    def find(self, student_id: str = "", student_name: str = ""):
        students = self.repository.index()
        student_id = AccentRemover.remove(student_id)
        student_name = AccentRemover.remove(student_name)

        filtered_students = {
            key: data
            for key, data in students.items()
            if re.search(student_id, AccentRemover.remove(data["id"]), re.IGNORECASE) and
               re.search(student_name, AccentRemover.remove(data["name"]), re.IGNORECASE)
        }

        return filtered_students

    def find_by_student_id(self, student_id: str):
        student = self.repository.db.collection("students").where("id", "==", student_id).stream()
        return {s.id: s.to_dict() for s in student} or None

    def _process_image(self, image_file: UploadedFile) -> np.ndarray:
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_file.getbuffer())))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def insert(self, student_id: str, name: str, card: UploadedFile, selfie: UploadedFile) -> str:
        if self.find_by_student_id(student_id):
            return "Sinh viên đã tồn tại"

        card_img = self._process_image(card)
        selfie_img = self._process_image(selfie)

        card_face, card_scale = self._detect_face(card_img)
        if not card_face:
            return "Không tìm thấy khuôn mặt trên ảnh thẻ sinh viên"

        selfie_face, selfie_scale = self._detect_face(selfie_img)
        if not selfie_face:
            return "Không tìm thấy khuôn mặt trên ảnh chân dung"

        resized_card = cv2.resize(card_img, (int(card_img.shape[1] * card_scale), int(card_img.shape[0] * card_scale)))
        resized_selfie = cv2.resize(selfie_img, (int(selfie_img.shape[1] * selfie_scale), int(selfie_img.shape[0] * selfie_scale)))

        card_feature = self.embedder.infer(resized_card, card_face[0][:-1])
        selfie_feature = self.embedder.infer(resized_selfie, selfie_face[0][:-1])

        _, match = self.embedder.match_feature(card_feature, selfie_feature)
        if match == 0:
            return "Ảnh thẻ sinh viên và ảnh chân dung không cùng một người"

        self.storage.upload(f"TheSV/{student_id}_card.jpg", card)
        self.storage.upload(f"ChanDung/{student_id}_selfie.jpg", selfie)

        docs = {
            "id": student_id,
            "name": name,
            "card": f"TheSV/{student_id}_card.jpg",
            "selfie": f"ChanDung/{student_id}_selfie.jpg",
            "card_face_feature": card_feature[0].tolist(),
            "selfie_face_feature": selfie_feature[0].tolist(),
        }

        self.repository.insert(docs)
        return "Thêm sinh viên thành công"

    def update(
        self, student_id: str, name: str, card: UploadedFile | None, selfie: UploadedFile | None
    ) -> str:
        student = self.find_by_student_id(student_id)
        if not student:
            return f"Sinh viên {student_id} không tồn tại"

        key = list(student.keys())[0]
        student_data = list(student.values())[0]

        card_feature, selfie_feature = None, None

        if card:
            card_img = self._process_image(card)
            card_face, card_scale = self._detect_face(card_img)
            if not card_face:
                return "Không tìm thấy khuôn mặt trên ảnh thẻ sinh viên"
            resized_card = cv2.resize(card_img, (int(card_img.shape[1] * card_scale), int(card_img.shape[0] * card_scale)))
            card_feature = self.embedder.infer(resized_card, card_face[0][:-1])

        if selfie:
            selfie_img = self._process_image(selfie)
            selfie_face, selfie_scale = self._detect_face(selfie_img)
            if not selfie_face:
                return "Không tìm thấy khuôn mặt trên ảnh chân dung"
            resized_selfie = cv2.resize(selfie_img, (int(selfie_img.shape[1] * selfie_scale), int(selfie_img.shape[0] * selfie_scale)))
            selfie_feature = self.embedder.infer(resized_selfie, selfie_face[0][:-1])

        if card and selfie:
            _, match = self.embedder.match_feature(card_feature, selfie_feature)
            if match == 0:
                return "Ảnh thẻ sinh viên và ảnh chân dung không cùng một người"
            self.storage.upload(f"TheSV/{student_id}_card.jpg", card)
            self.storage.upload(f"ChanDung/{student_id}_selfie.jpg", selfie)
            student_data.update({
                "card": f"TheSV/{student_id}_card.jpg",
                "selfie": f"ChanDung/{student_id}_selfie.jpg",
                "card_face_feature": card_feature[0].tolist(),
                "selfie_face_feature": selfie_feature[0].tolist()
            })
        elif card:
            _, match = self.embedder.match_feature(
                card_feature,
                np.array([student_data["selfie_face_feature"]], dtype=card_feature[0].dtype),
            )
            if match == 0:
                return "Ảnh thẻ sinh viên và ảnh chân dung không cùng một người"
            self.storage.upload(f"TheSV/{student_id}_card.jpg", card)
            student_data.update({
                "card": f"TheSV/{student_id}_card.jpg",
                "card_face_feature": card_feature[0].tolist()
            })
        elif selfie:
            _, match = self.embedder.match_feature(
                np.array([student_data["card_face_feature"]], dtype=selfie_feature[0].dtype),
                selfie_feature,
            )
            if match == 0:
                return "Ảnh thẻ sinh viên và ảnh chân dung không cùng một người"
            self.storage.upload(f"ChanDung/{student_id}_selfie.jpg", selfie)
            student_data.update({
                "selfie": f"ChanDung/{student_id}_selfie.jpg",
                "selfie_face_feature": selfie_feature[0].tolist()
            })

        self.repository.update(key, student_data)
        return f"Chỉnh sửa sinh viên {student_id} thành công"

    def delete(self, student_id: str) -> str:
        student = self.find_by_student_id(student_id)
        if not student:
            return f