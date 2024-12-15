import pandas as pd
import streamlit as st
from components.face_verification import get_table_data
from services.face_verification.service import (
    ClassService,
    StudentClassService,
    StudentService,
)

# Khởi tạo các service
studentService = StudentService()
classService = ClassService()
studentClassService = StudentClassService()


@st.cache_data(ttl=3600)
def get_list_classes():
    """Lấy danh sách các lớp học."""
    classes = classService.repository.index()
    return classes


def hidden_all_forms():
    """Ẩn tất cả các form đang hiển thị."""
    for key in st.session_state["forms_state"].keys():
        st.session_state["forms_state"][key] = False


def show_form(form_id):
    """Hiển thị form theo ID."""
    hidden_all_forms()
    st.session_state["forms_state"][form_id] = True


def display_form_add_student():
    """Hiển thị form thêm sinh viên."""
    with st.form(key="form_add"):
        st.markdown("#### Thêm sinh viên")
        first_cols = st.columns(2)
        student_id = first_cols[0].text_input("Mã sinh viên")
        student_name = first_cols[1].text_input("Tên sinh viên")

        second_cols = st.columns(2)
        card = second_cols[0].file_uploader("Ảnh thẻ sinh viên", type=["jpg", "jpeg", "png"])
        selfie = second_cols[1].file_uploader("Ảnh chân dung", type=["jpg", "jpeg", "png"])

        cols = st.columns(2)
        btnSubmit = cols[0].form_submit_button("Thêm")
        cols[1].form_submit_button("Đóng", on_click=hidden_all_forms)

    if btnSubmit:
        is_valid = True
        if not student_id.strip():
            first_cols[0].error("Mã sinh viên không được để trống.")
            is_valid = False
        if not student_name.strip():
            first_cols[1].error("Tên sinh viên không được để trống.")
            is_valid = False
        if not card:
            second_cols[0].error("Ảnh thẻ sinh viên không được để trống.")
            is_valid = False
        if not selfie:
            second_cols[1].error("Ảnh chân dung không được để trống.")
            is_valid = False

        if is_valid:
            result = studentService.insert(student_id, student_name, card, selfie)
            if result == "Thêm sinh viên thành công":
                get_table_data.clear()
                st.toast(result, icon=":material/check:")
            else:
                st.toast(result, icon=":material/error:")


def display_form_edit_student():
    """Hiển thị form chỉnh sửa thông tin sinh viên."""
    if not st.session_state.get("selected_student"):
        st.error("Không tìm thấy thông tin sinh viên cần chỉnh sửa!")
        return

    student = st.session_state["selected_student"]

    with st.form(key="form_edit"):
        st.markdown("#### Chỉnh sửa thông tin sinh viên")
        first_cols = st.columns(2)
        student_id = first_cols[0].text_input("Mã sinh viên", value=student["id"], disabled=True)
        student_name = first_cols[1].text_input("Tên sinh viên", value=student["name"])

        second_cols = st.columns(2)
        card = second_cols[0].file_uploader("Ảnh thẻ sinh viên", type=["jpg", "jpeg", "png"])
        selfie = second_cols[1].file_uploader("Ảnh chân dung", type=["jpg", "jpeg", "png"])

        cols = st.columns(2)
        btnSubmit = cols[0].form_submit_button("Chỉnh sửa")
        cols[1].form_submit_button("Đóng", on_click=hidden_all_forms)

    if btnSubmit:
        is_valid = True
        if not student_name.strip():
            first_cols[1].error("Tên sinh viên không được để trống.")
            is_valid = False

        if is_valid:
            result = studentService.update(student_id, student_name, card, selfie)
            if result == "Cập nhật sinh viên thành công":
                get_table_data.clear()
                st.session_state["selected_student"] = None
                st.toast(result, icon=":material/check:")
            else:
                st.toast(result, icon=":material/error:")


def confirm_delete_student(student_id):
    """Xác nhận và xử lý xóa sinh viên."""
    if st.confirm(f"Bạn có chắc chắn muốn xóa sinh viên với ID {student_id}?"):
        result = studentService.delete(student_id)
        if "thành công" in result:
            st.toast(result, icon=":material/check:")
            get_table_data.clear()
        else:
            st.toast(result, icon=":material/error:")


def display_student_table():
    """Hiển thị bảng danh sách sinh viên với các nút Sửa và Xóa."""
    table_data, students_raw = get_table_data(
        st.session_state["filter_student"]["student_id"],
        st.session_state["filter_student"]["student_name"],
        st.session_state["filter_student"]["class_id"],
    )

    if len(table_data["id"]) == 0:
        st.write("Không có sinh viên nào.")
        return pd.DataFrame(table_data)

    # Thêm cột hành động (Sửa, Xóa)
    table_data["Hành động"] = [
        f"""
        <button id="edit_{student_id}" class="action-btn edit-btn">Sửa</button>
        <button id="delete_{student_id}" class="action-btn delete-btn">Xóa</button>
        """
        for student_id in table_data["id"]
    ]

    data_editor = st.data_editor(
        pd.DataFrame(table_data),
        column_config={
            "id": st.column_config.TextColumn("Mã sinh viên", disabled=True),
            "name": st.column_config.TextColumn("Tên sinh viên", disabled=True),
            "card": st.column_config.ImageColumn("Thẻ sinh viên"),
            "selfie": st.column_config.ImageColumn("Ảnh chân dung"),
            "Hành động": st.column_config.TextColumn("Hành động"),
        },
        use_container_width=True,
        hide_index=True,
        column_order=["id", "name", "card", "selfie", "Hành động"],
    )

    for student_id in table_data["id"]:
        if st.button(f"Sửa {student_id}", key=f"edit_{student_id}"):
            st.session_state["selected_student"] = students_raw.get(student_id)
            show_form("form_edit_student")
        if st.button(f"Xóa {student_id}", key=f"delete_{student_id}"):
            confirm_delete_student(student_id)


@st.fragment()
def display_student_manager():
    """Quản lý hiển thị giao diện chính của ứng dụng."""
    st.header("Quản lý sinh viên")

    # Hiển thị form hành động
    if st.session_state["forms_state"].get("form_add_student"):
        display_form_add_student()
    elif st.session_state["forms_state"].get("form_edit_student"):
        display_form_edit_student()

    # Hiển thị bảng danh sách sinh viên
    display_student_table()
