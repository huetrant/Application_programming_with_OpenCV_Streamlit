import json
import datetime
import logging
import streamlit as st
from google.cloud import firestore, storage

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Database:
    def __init__(self):
        # Load key từ secrets
        key_dict = json.loads(st.secrets["textkey"])
        self.db = self._init_firestore(key_dict)
        self.bucket = self._init_storage(key_dict)

    @staticmethod
    def _init_firestore(key_dict):
        """Khởi tạo Firestore client"""
        return firestore.Client.from_service_account_info(key_dict)

    @staticmethod
    def _init_storage(key_dict):
        """Khởi tạo Storage client và bucket"""
        return storage.Client.from_service_account_info(key_dict).get_bucket(
            "face-recognize-7d75c.appspot.com"
        )


class Repository(Database):
    def __init__(self, collection: str):
        super().__init__()
        self.collection = collection

    def get_collection_ref(self):
        """Lấy tham chiếu đến collection"""
        return self.db.collection(self.collection)

    def index(self):
        """Lấy toàn bộ document trong collection"""
        logging.info(f"INDEX: Fetching all documents from '{self.collection}'")
        try:
            docs = self.get_collection_ref().stream()
            return {doc.id: doc.to_dict() for doc in docs}
        except Exception as e:
            logging.error(f"Error fetching documents: {e}")
            return {}

    def show(self, doc_id: str):
        """Lấy document theo id"""
        logging.info(f"SHOW: Fetching document '{doc_id}' from '{self.collection}'")
        try:
            doc = self.get_collection_ref().document(doc_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logging.error(f"Error fetching document '{doc_id}': {e}")
            return None

    def insert(self, docs: dict):
        """Thêm document mới"""
        logging.info(f"INSERT: Adding new document to '{self.collection}'")
        try:
            doc_ref = self.get_collection_ref().document()
            doc_ref.set(docs)
            return doc_ref
        except Exception as e:
            logging.error(f"Error inserting document: {e}")
            return None

    def update(self, doc_id: str, docs: dict):
        """Cập nhật document theo id"""
        logging.info(f"UPDATE: Updating document '{doc_id}' in '{self.collection}'")
        try:
            self.get_collection_ref().document(doc_id).update(docs)
            return True
        except Exception as e:
            logging.error(f"Error updating document '{doc_id}': {e}")
            return False

    def delete(self, doc_id: str):
        """Xóa document theo id"""
        logging.info(f"DELETE: Deleting document '{doc_id}' from '{self.collection}'")
        try:
            self.get_collection_ref().document(doc_id).delete()
            return True
        except Exception as e:
            logging.error(f"Error deleting document '{doc_id}': {e}")
            return False


class Storage(Database):
    def __init__(self):
        super().__init__()

    def get_url(self, path: str, expires_in: int = 300):
        """Lấy URL ký tên của file"""
        logging.info(f"GET URL: Generating signed URL for '{path}'")
        try:
            return self.bucket.blob(path).generate_signed_url(
                expiration=datetime.timedelta(seconds=expires_in), method="GET"
            )
        except Exception as e:
            logging.error(f"Error generating signed URL for '{path}': {e}")
            return None

    def upload(self, path: str, file):
        """Upload file lên storage"""
        logging.info(f"UPLOAD: Uploading file to '{path}'")
        try:
            self.bucket.blob(path).upload_from_file(file)
        except Exception as e:
            logging.error(f"Error uploading file to '{path}': {e}")
