import json
import datetime
import streamlit as st
from google.cloud import firestore, storage

# Tối ưu bảng chuyển đổi dấu tiếng Việt
accent_map = { 
    **dict.fromkeys("ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ", "A"), 
    "Đ": "D", 
    **dict.fromkeys("ÈÉẺẼẸÊẾỀỂỄỆ", "E"), 
    **dict.fromkeys("ÍÌỈĨỊ", "I"), 
    **dict.fromkeys("ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ", "O"), 
    **dict.fromkeys("ÚÙỦŨỤƯỨỪỬỮỰ", "U"), 
    **dict.fromkeys("ÝỲỶỸỴ", "Y"), 
    **dict.fromkeys("áàảãạăắằẳẵặâấầẩẫậ", "a"), 
    "đ": "d", 
    **dict.fromkeys("èéẻẽẹêếềểễệ", "e"), 
    **dict.fromkeys("íìỉĩị", "i"), 
    **dict.fromkeys("óòỏõọôốồổỗộơớờởỡợ", "o"), 
    **dict.fromkeys("úùủũụưứừửữự", "u"), 
    **dict.fromkeys("ýỳỷỹỵ", "y")
}

accent_tables = str.maketrans(accent_map)

def remove_vietnamese_accent(txt: str) -> str:
    return txt.translate(accent_tables)


class Database:
    def __init__(self) -> None:
        key_dict = json.loads(st.secrets["textkey"])
        self.db = firestore.Client.from_service_account_info(key_dict)
        self.bucket = storage.Client.from_service_account_info(key_dict).get_bucket("face-recognize-7d75c.appspot.com")
        self.timestamp = datetime.datetime.now()

class Repository(Database):
    def __init__(self, collection: str) -> None:
        super().__init__()
        self.collection = collection

    def index(self):
        """Lấy tất cả tài liệu trong collection."""
        print(self.timestamp, f">> INDEX {self.collection}")
        docs = self.db.collection(self.collection).stream()
        return {doc.id: doc.to_dict() for doc in docs}

    def show(self, id: str):
        """Lấy một tài liệu theo id."""
        print(self.timestamp, f">> SHOW {self.collection} {id}")
        doc = self.db.collection(self.collection).document(id).get()
        return doc.to_dict() if doc.exists else None

    def insert(self, docs: dict):
        """Chèn một tài liệu vào collection."""
        print(self.timestamp, f">> INSERT {self.collection}")
        new_doc_ref = self.db.collection(self.collection).document()
        new_doc_ref.set(docs)
        return new_doc_ref

    def update(self, id: str, docs: dict):
        """Cập nhật tài liệu theo id."""
        print(self.timestamp, f">> UPDATE {self.collection} {id}")
        try:
            self.db.collection(self.collection).document(id).update(docs)
            return True
        except Exception as e:
            print(e)
            return False

    def delete(self, id: str):
        """Xóa tài liệu theo id."""
        print(self.timestamp, f">> DELETE {self.collection} {id}")
        try:
            self.db.collection(self.collection).document(id).delete()
            return True
        except Exception as e:
            print(e)
            return False

class Storage(Database):
    def __init__(self) -> None:
        super().__init__()

    def get_url(self, path: str, expires_in: int = 300):
        """Lấy URL đã ký của tệp trong storage."""
        print(self.timestamp, ">> GET URL FROM STORAGE", path)
        return self.bucket.blob(path).generate_signed_url(datetime.timedelta(seconds=expires_in), method="GET")

    def upload(self, path: str, file):
        """Tải tệp lên storage."""
        print(self.timestamp, ">> UPLOAD TO STORAGE", path)
        self.bucket.blob(path).upload_from_file(file)