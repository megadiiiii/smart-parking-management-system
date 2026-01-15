# admin_gui.py - Admin interface (Tiếng Việt, trạng thái 'Vào'/'Ra')
# UI update: tăng kích thước các cột chữ (dài ra), giảm kích thước vùng xem trước,
#            điều chỉnh thumbnail trong bảng cho phù hợp.
import os
import sys
import csv
import sqlite3
from datetime import datetime

from PyQt6.QtCore import Qt, QSize, QDate, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QFont, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QDateEdit, QInputDialog, QGroupBox, QSizePolicy,
    QSpacerItem, QMenu
)

# Try import project's db helper; fallback if missing
try:
    from db import DB_PATH, init_db
except Exception:
    DB_PATH = os.path.join(os.getcwd(), "events.db")
    def init_db():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT,
                status TEXT,
                time_in TEXT,
                time_out TEXT,
                face_image_path TEXT,
                plate_image_path TEXT
            )
        """)
        conn.commit()
        conn.close()

# Column mapping (Thời gian Vào / Ra)
COLUMNS = [
    ("id", "ID"),
    ("plate_text", "Biển"),
    ("status", "Trạng thái"),
    ("time_in", "Thời gian Vào"),
    ("time_out", "Thời gian Ra"),
    ("face_image_path", "Ảnh mặt"),
    ("plate_image_path", "Ảnh biển"),
]

# Thumbnail size used inside table cells
THUMB_W = 160
THUMB_H = 90

# Preview sizes (reduced per request)
PREVIEW_W = 420
PREVIEW_H = 260

DARK_QSS = f"""
QWidget {{ background-color: #1e1f22; color: #e6eef3; font-family: "Segoe UI"; }}
QGroupBox {{ border: 1px solid #2b2d31; border-radius: 8px; padding: 8px; background-color: #151617; }}
QPushButton {{ background-color: #2d8cff; color: white; border-radius: 6px; padding: 6px 10px; }}
QPushButton[secondary="true"] {{ background-color: transparent; border: 1px solid #3b3e44; color: #cfe8ff; }}
QLineEdit, QDateEdit, QComboBox {{ background-color: #121314; border: 1px solid #2b2d31; border-radius: 6px; padding: 6px; color:#e6eef3; }}
QTableWidget {{ background-color: #151617; gridline-color: #2b2d31; border: 1px solid #2b2d31; }}
QHeaderView::section {{ background-color: #202225; color: #dbeefd; padding: 8px; border: 1px solid #2b2d31; }}
QTableWidget::item:selected {{ background-color: #2d8cff; color: white; }}
/* Make in-table image labels look nicer */
QLabel {{
    color: #e6eef3;
}}
"""

class AdminWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quản trị - Lịch sử Vào/Ra")
        self.resize(1400, 920)
        self.setFont(QFont("Segoe UI", 10))

        self._last_deleted = []
        self._undo_timer = None
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(300)
        self.search_timer.timeout.connect(self.on_search)

        self._login_window_ref = None

        self._init_ui()
        self._add_shortcuts()
        init_db()
        self.load_events()

    def _init_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(12,12,12,12)
        root.setSpacing(10)

        # top actions + filters in a group
        table_group = QGroupBox("Lịch sử ra vào")
        tg_layout = QVBoxLayout()
        table_group.setLayout(tg_layout)

        actions = QHBoxLayout()
        self.export_btn = QPushButton("Xuất CSV"); actions.addWidget(self.export_btn); self.export_btn.clicked.connect(self.export_csv)
        self.delete_btn = QPushButton("Xóa mục chọn"); actions.addWidget(self.delete_btn); self.delete_btn.clicked.connect(self.delete_selected)
        self.edit_btn = QPushButton("Sửa biển"); actions.addWidget(self.edit_btn); self.edit_btn.clicked.connect(self.edit_selected_plate)
        self.open_img_btn = QPushButton("Mở ảnh"); actions.addWidget(self.open_img_btn); self.open_img_btn.clicked.connect(self.open_selected_image)
        self.undo_btn = QPushButton("Hoàn tác xóa"); self.undo_btn.setProperty("secondary", True); self.undo_btn.setEnabled(False); self.undo_btn.clicked.connect(self._undo_delete); actions.addWidget(self.undo_btn)
        actions.addStretch()
        self.refresh_btn = QPushButton("Làm mới"); actions.addWidget(self.refresh_btn); self.refresh_btn.clicked.connect(self.load_events)
        self.clear_btn = QPushButton("Xóa bộ lọc"); actions.addWidget(self.clear_btn); self.clear_btn.clicked.connect(self.on_clear)
        self.logout_btn = QPushButton("Đăng xuất"); actions.addWidget(self.logout_btn); self.logout_btn.clicked.connect(self._logout)
        tg_layout.addLayout(actions)

        # filters row with labels
        filters = QHBoxLayout()
        lbl_find = QLabel("Tìm:"); lbl_find.setFixedWidth(40)
        filters.addWidget(lbl_find)
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText("Nhập biển số xe"); self.search_edit.textChanged.connect(self._on_search_text_changed); filters.addWidget(self.search_edit)
        lbl_status = QLabel("Trạng thái:"); lbl_status.setFixedWidth(84)
        filters.addWidget(lbl_status)
        self.status_combo = QComboBox(); self.status_combo.addItems(["Tất cả","Vào","Ra"]); self.status_combo.currentTextChanged.connect(self._on_filter_changed); filters.addWidget(self.status_combo)
        lbl_from = QLabel("Từ:"); lbl_from.setFixedWidth(36); filters.addWidget(lbl_from)
        self.date_from = QDateEdit(); self.date_from.setCalendarPopup(True); self.date_from.setDate(QDate.currentDate().addMonths(-1)); self.date_from.dateChanged.connect(self._on_filter_changed); filters.addWidget(self.date_from)
        lbl_to = QLabel("Đến:"); lbl_to.setFixedWidth(44); filters.addWidget(lbl_to)
        self.date_to = QDateEdit(); self.date_to.setCalendarPopup(True); self.date_to.setDate(QDate.currentDate()); self.date_to.dateChanged.connect(self._on_filter_changed); filters.addWidget(self.date_to)
        filters.addStretch()
        tg_layout.addLayout(filters)

        # table
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self.on_cell_double)
        self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        # make vertical header row height default bigger to fit thumbnails
        self.table.verticalHeader().setDefaultSectionSize(THUMB_H + 12)
        tg_layout.addWidget(self.table)

        # main layout with preview to the right
        main = QHBoxLayout()
        main.addWidget(table_group, stretch=4)  # give more room to table

        preview_group = QGroupBox("Xem trước ảnh")
        pv_layout = QVBoxLayout()
        preview_group.setLayout(pv_layout)
        lbl_face = QLabel("Ảnh khuôn mặt"); lbl_face.setStyleSheet("padding:4px;")
        pv_layout.addWidget(lbl_face)
        self.preview_face = QLabel("Chưa chọn ảnh mặt")
        self.preview_face.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.preview_face.setStyleSheet("background:#0f1112; border-radius:6px;")
        self.preview_face.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pv_layout.addWidget(self.preview_face)
        lbl_plate = QLabel("Ảnh biển số"); lbl_plate.setStyleSheet("padding:4px;")
        pv_layout.addWidget(lbl_plate)
        self.preview_plate = QLabel("Chưa chọn ảnh biển")
        self.preview_plate.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.preview_plate.setStyleSheet("background:#0f1112; border-radius:6px;")
        self.preview_plate.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pv_layout.addWidget(self.preview_plate)
        btns = QHBoxLayout()
        self.btn_open_plate = QPushButton("Mở biển số"); self.btn_open_plate.clicked.connect(lambda: self._open_preview_image("plate")); btns.addWidget(self.btn_open_plate)
        self.btn_open_face = QPushButton("Mở ảnh mặt"); self.btn_open_face.clicked.connect(lambda: self._open_preview_image("face")); btns.addWidget(self.btn_open_face)
        self.btn_clear_preview = QPushButton("Xóa xem trước"); self.btn_clear_preview.setProperty("secondary", True); self.btn_clear_preview.clicked.connect(self._clear_preview); btns.addWidget(self.btn_clear_preview)
        pv_layout.addLayout(btns)

        main.addWidget(preview_group, stretch=1)  # preview smaller

        root.addLayout(main)

        # status/ toast
        self.status_label = QLabel("")
        root.addWidget(self.status_label)

        self.setLayout(root)
        self.setStyleSheet(DARK_QSS)

    # Shortcuts
    def _add_shortcuts(self):
        act_refresh = QAction("Làm mới", self); act_refresh.setShortcut(QKeySequence("F5")); act_refresh.triggered.connect(self.load_events); self.addAction(act_refresh)
        act_logout = QAction("Đăng xuất", self); act_logout.setShortcut(QKeySequence("Ctrl+L")); act_logout.triggered.connect(self._logout); self.addAction(act_logout)
        act_undo = QAction("Hoàn tác xóa", self); act_undo.setShortcut(QKeySequence("Ctrl+Z")); act_undo.triggered.connect(self._undo_delete); self.addAction(act_undo)
        act_del = QAction("Xóa", self); act_del.setShortcut(QKeySequence(Qt.Key.Key_Delete)); act_del.triggered.connect(self.delete_selected); self.addAction(act_del)

    # DB helpers and UI behaviors (status values 'Vào'/'Ra' used consistently)
    def _build_query(self, search_text="", status_filter="Tất cả", date_from=None, date_to=None):
        sql = "SELECT id, plate_text, status, time_in, time_out, face_image_path, plate_image_path FROM events"
        clauses = []; params = []
        if search_text:
            clauses.append("LOWER(plate_text) LIKE ?"); params.append(f"%{search_text.lower()}%")
        if status_filter in ("Vào","Ra"):
            clauses.append("status = ?"); params.append(status_filter)
        if date_from:
            d0 = date_from.toString("yyyy-MM-dd")
            clauses.append("(time_in >= ? OR time_out >= ? OR (time_in IS NULL AND time_out >= ?))")
            params += [d0+" 00:00:00", d0+" 00:00:00", d0+" 00:00:00"]
        if date_to:
            d1 = date_to.toString("yyyy-MM-dd")
            clauses.append("(time_in <= ? OR time_out <= ? OR (time_in IS NULL AND time_out <= ?))")
            params += [d1+" 23:59:59", d1+" 23:59:59", d1+" 23:59:59"]
        if clauses: sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        return sql, params

    def load_events(self):
        try:
            conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
            cur.execute("SELECT id, plate_text, status, time_in, time_out, face_image_path, plate_image_path FROM events ORDER BY id DESC")
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi DB", f"Không đọc được DB: {e}"); rows = []
        if not rows:
            self.table.setRowCount(0); self._show_status("Chưa có dữ liệu nào trong hệ thống.", timeout=4000); return
        self._populate_table(rows)
        self._show_status(f"Đã tải {len(rows)} sự kiện")

    def _populate_table(self, rows):
        self.table.setRowCount(0)
        for r, row in enumerate(rows):
            self.table.insertRow(r)
            for c, val in enumerate(row):
                col_name = COLUMNS[c][0]
                if col_name in ("face_image_path","plate_image_path"):
                    label = QLabel()
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    label.setStyleSheet("background: #0f1112; border-radius:6px;")
                    # fix size so table cell renders at consistent thumbnail size
                    label.setFixedSize(THUMB_W, THUMB_H)
                    if val and os.path.exists(val):
                        try:
                            pix = QPixmap(val).scaled(THUMB_W, THUMB_H, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                            label.setPixmap(pix)
                        except Exception:
                            label.setText("lỗi")
                    else:
                        label.setText("-")
                    self.table.setCellWidget(r, c, label)
                else:
                    item = QTableWidgetItem(str(val) if val is not None else "")
                    if col_name in ("id","status"):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(r, c, item)
            # ensure row height fits the thumbnail
            self.table.setRowHeight(r, THUMB_H + 12)

        # Resize columns (make textual columns longer)
        self.table.setColumnWidth(0,60)
        self.table.setColumnWidth(1,320)   # Biển (wider)
        self.table.setColumnWidth(2,100)
        self.table.setColumnWidth(3,260)   # Thời gian Vào (wider)
        self.table.setColumnWidth(4,260)   # Thời gian Ra (wider)
        self.table.setColumnWidth(5,THUMB_W + 20)  # face thumb column
        self.table.setColumnWidth(6,THUMB_W + 20)  # plate thumb column

    # Search / filter
    def _on_search_text_changed(self, text): self.search_timer.start()
    def _on_filter_changed(self, _=None): self.search_timer.start()

    def on_search(self):
        sql, params = self._build_query(self.search_edit.text().strip(), self.status_combo.currentText(), self.date_from.date(), self.date_to.date())
        try:
            conn = sqlite3.connect(DB_PATH); cur = conn.cursor(); cur.execute(sql, params); rows = cur.fetchall(); conn.close()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi DB", f"Không thể truy vấn DB: {e}"); rows = []
        if not rows:
            self.table.setRowCount(0); self._show_status("Không tìm thấy dữ liệu.", timeout=3000); return
        self._populate_table(rows); self._show_status(f"Đã lọc: {len(rows)} hàng")

    def on_clear(self):
        self.search_edit.clear(); self.status_combo.setCurrentIndex(0)
        self.date_from.setDate(QDate.currentDate().addMonths(-1)); self.date_to.setDate(QDate.currentDate())
        self.load_events()

    # Export / edit / delete / preview
    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Lưu CSV", os.getcwd(), "CSV files (*.csv)")
        if not path: return
        rows = []
        for r in range(self.table.rowCount()):
            row_data = []
            for c, (col_key, _) in enumerate(COLUMNS):
                if col_key in ("face_image_path","plate_image_path"):
                    try:
                        id_item = self.table.item(r, 0)
                        if id_item:
                            ev = self.get_event_by_id(int(id_item.text())); txt = ev.get(col_key) or ""
                        else: txt = ""
                    except Exception: txt = ""
                    row_data.append(txt)
                else:
                    it = self.table.item(r, c); row_data.append(it.text() if it else "")
            rows.append(row_data)
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f); writer.writerow([h for _, h in COLUMNS]); writer.writerows(rows)
            self._show_status(f"Đã xuất {len(rows)} dòng sang {path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi xuất", f"Không thể xuất CSV: {e}")

    def get_event_by_id(self, ev_id):
        conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
        cur.execute("SELECT id, plate_text, status, time_in, time_out, face_image_path, plate_image_path FROM events WHERE id = ?", (ev_id,))
        row = cur.fetchone(); conn.close()
        if not row: return None
        return dict(zip([c[0] for c in COLUMNS], row))

    def delete_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel: self._show_status("Chọn hàng để xóa.", timeout=2000); return
        ids = []
        for idx in sel:
            it = self.table.item(idx.row(), 0)
            if it:
                try: ids.append(int(it.text()))
                except Exception: pass
        if not ids: self._show_status("Không có ID hợp lệ.", timeout=2000); return
        ok = QMessageBox.question(self, "Xác nhận", f"Xóa {len(ids)} hàng?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ok != QMessageBox.StandardButton.Yes: return
        self._last_deleted = []
        try:
            conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
            for i in ids:
                cur.execute("SELECT plate_text, status, time_in, time_out, face_image_path, plate_image_path FROM events WHERE id = ?", (i,))
                row = cur.fetchone()
                if row:
                    self._last_deleted.append({"plate_text":row[0],"status":row[1],"time_in":row[2],"time_out":row[3],"face_image_path":row[4],"plate_image_path":row[5]})
            cur.executemany("DELETE FROM events WHERE id = ?", [(i,) for i in ids])
            conn.commit(); conn.close()
        except Exception as e:
            QMessageBox.critical(self, "Lỗi DB", f"Không thể xóa: {e}"); return
        self.load_events()
        self.undo_btn.setEnabled(True)
        if self._undo_timer: self._undo_timer.stop()
        self._undo_timer = QTimer(self); self._undo_timer.setSingleShot(True); self._undo_timer.timeout.connect(self._clear_undo_buffer); self._undo_timer.start(5000)
        self._show_status("Đã xóa. Có 5s để hoàn tác (Ctrl+Z).", timeout=4000)

    def _undo_delete(self):
        if not self._last_deleted: self._show_status("Không có thao tác để hoàn tác.", timeout=2000); return
        try:
            conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
            for ev in self._last_deleted:
                cur.execute("INSERT INTO events (plate_text, status, time_in, time_out, face_image_path, plate_image_path) VALUES (?, ?, ?, ?, ?, ?)",
                            (ev["plate_text"], ev["status"], ev["time_in"], ev["time_out"], ev["face_image_path"], ev["plate_image_path"]))
            conn.commit(); conn.close()
            self._last_deleted = []; self.undo_btn.setEnabled(False)
            if self._undo_timer: self._undo_timer.stop()
            self.load_events()
            self._show_status("Đã hoàn tác.", timeout=2500)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi DB", f"Không thể hoàn tác: {e}")

    def _clear_undo_buffer(self):
        self._last_deleted = []; self.undo_btn.setEnabled(False)

    def edit_selected_plate(self):
        sel = self.table.selectionModel().selectedRows();
        if not sel: self._show_status("Chọn một hàng để sửa.", timeout=2000); return
        if len(sel)>1: self._show_status("Chỉ chọn một hàng để sửa.", timeout=2000); return
        row_idx = sel[0].row(); id_item = self.table.item(row_idx,0); plate_item = self.table.item(row_idx,1)
        if not id_item or not plate_item: self._show_status("Không có dữ liệu để sửa.", timeout=2000); return
        ev_id = int(id_item.text()); curr_plate = plate_item.text()
        new_plate, ok = QInputDialog.getText(self, "Sửa biển", "Biển mới:", text=curr_plate)
        if not ok: return
        new_plate = new_plate.strip()
        if not new_plate: self._show_status("Biển trống, huỷ.", timeout=2000); return
        try:
            conn = sqlite3.connect(DB_PATH); cur = conn.cursor(); cur.execute("UPDATE events SET plate_text = ? WHERE id = ?", (new_plate, ev_id)); conn.commit(); conn.close()
            self._show_status("Đã cập nhật.", timeout=2000)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi DB", f"Không thể cập nhật: {e}")
        self.load_events()

    def open_selected_image(self):
        col_choice, ok = QInputDialog.getItem(self, "Chọn ảnh", "Mở ảnh nào?", ["Ảnh mặt","Ảnh biển"], 0, False)
        if not ok: return
        col_key = "face_image_path" if col_choice.startswith("Ảnh mặt") else "plate_image_path"
        sel = self.table.selectionModel().selectedRows()
        if not sel: self._show_status("Chọn hàng để mở ảnh.", timeout=2000); return
        opened = 0
        for idx in sel:
            id_item = self.table.item(idx.row(), 0)
            if not id_item: continue
            ev = self.get_event_by_id(int(id_item.text()))
            if not ev: continue
            path = ev.get(col_key)
            if path and os.path.exists(path):
                try:
                    if sys.platform.startswith("darwin"): os.system(f"open \"{path}\"")
                    elif os.name == "nt": os.startfile(path)
                    else: os.system(f"xdg-open \"{path}\"")
                    opened += 1
                except Exception: pass
        self._show_status(f"Đã cố mở {opened} ảnh.", timeout=2500)

    def on_cell_double(self, row, col):
        col_key = COLUMNS[col][0]
        if col_key not in ("face_image_path","plate_image_path"): return
        id_item = self.table.item(row,0)
        if not id_item: return
        ev = self.get_event_by_id(int(id_item.text()));
        if not ev: return
        path = ev.get(col_key)
        if path and os.path.exists(path):
            try:
                if sys.platform.startswith("darwin"): os.system(f"open \"{path}\"")
                elif os.name == "nt": os.startfile(path)
                else: os.system(f"xdg-open \"{path}\"")
            except Exception as e:
                QMessageBox.information(self, "Mở ảnh", f"Không mở được: {e}")
        else:
            QMessageBox.information(self, "Mở ảnh", "Không có ảnh.")

    # selection -> preview
    def on_table_selection_changed(self, selected, deselected):
        sel_rows = self.table.selectionModel().selectedRows()
        if not sel_rows: self._clear_preview(); return
        row = sel_rows[0].row(); id_item = self.table.item(row,0)
        if not id_item: self._clear_preview(); return
        try: ev_id = int(id_item.text())
        except Exception: self._clear_preview(); return
        self._display_preview_for_id(ev_id)

    def _display_preview_for_id(self, ev_id):
        ev = self.get_event_by_id(ev_id)
        if not ev: self._clear_preview(); return
        plate_path = ev.get("plate_image_path") or ""; face_path = ev.get("face_image_path") or ""
        if plate_path and os.path.exists(plate_path):
            try:
                pix = QPixmap(plate_path).scaled(self.preview_plate.width(), self.preview_plate.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.preview_plate.setPixmap(pix); self.preview_plate.setText("")
            except Exception:
                self.preview_plate.setPixmap(QPixmap()); self.preview_plate.setText("Lỗi ảnh biển")
        else:
            self.preview_plate.setPixmap(QPixmap()); self.preview_plate.setText("Không có ảnh biển")
        if face_path and os.path.exists(face_path):
            try:
                pix = QPixmap(face_path).scaled(self.preview_face.width(), self.preview_face.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.preview_face.setPixmap(pix); self.preview_face.setText("")
            except Exception:
                self.preview_face.setPixmap(QPixmap()); self.preview_face.setText("Lỗi ảnh mặt")
        else:
            self.preview_face.setPixmap(QPixmap()); self.preview_face.setText("Không có ảnh mặt")

    def _open_preview_image(self, kind):
        sel_rows = self.table.selectionModel().selectedRows()
        if not sel_rows: self._show_status("Chọn hàng trước khi mở ảnh.", timeout=2000); return
        row = sel_rows[0].row(); id_item = self.table.item(row,0)
        if not id_item: return
        ev = self.get_event_by_id(int(id_item.text()))
        if not ev: return
        key = "plate_image_path" if kind=="plate" else "face_image_path"; path = ev.get(key)
        if path and os.path.exists(path):
            try:
                if sys.platform.startswith("darwin"): os.system(f"open \"{path}\"")
                elif os.name == "nt": os.startfile(path)
                else: os.system(f"xdg-open \"{path}\"")
            except Exception as e:
                QMessageBox.information(self, "Mở ảnh", f"Không mở được: {e}")
        else:
            QMessageBox.information(self, "Mở ảnh", "Không có file.")

    def _clear_preview(self):
        self.preview_plate.setPixmap(QPixmap()); self.preview_plate.setText("Chưa chọn ảnh biển")
        self.preview_face.setPixmap(QPixmap()); self.preview_face.setText("Chưa chọn ảnh mặt")

    # context menu
    def _on_table_context_menu(self, pos: QPoint):
        idx = self.table.indexAt(pos); menu = QMenu(self)
        open_plate = QAction("Mở ảnh biển", self); open_plate.triggered.connect(lambda: self._context_open(idx.row(), "plate")); menu.addAction(open_plate)
        open_face = QAction("Mở ảnh mặt", self); open_face.triggered.connect(lambda: self._context_open(idx.row(), "face")); menu.addAction(open_face)
        menu.addSeparator()
        edit = QAction("Sửa biển", self); edit.triggered.connect(lambda: self._context_edit(idx.row())); menu.addAction(edit)
        delete = QAction("Xóa", self); delete.triggered.connect(lambda: self._context_delete(idx)); menu.addAction(delete)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _context_open(self, row, kind):
        if row<0: self._show_status("Không có hàng được chọn.", timeout=2000); return
        id_item = self.table.item(row,0)
        if not id_item: return
        ev = self.get_event_by_id(int(id_item.text()))
        if not ev: return
        key = "plate_image_path" if kind=="plate" else "face_image_path"; path = ev.get(key)
        if path and os.path.exists(path):
            try:
                if sys.platform.startswith("darwin"): os.system(f"open \"{path}\"")
                elif os.name=="nt": os.startfile(path)
                else: os.system(f"xdg-open \"{path}\"")
            except Exception as e:
                QMessageBox.information(self, "Mở ảnh", f"Không mở được: {e}")
        else:
            self._show_status("Không có file ảnh.", timeout=2000)

    def _context_edit(self, row):
        if row<0: return
        sel = self.table.selectionModel(); sel.clearSelection(); self.table.selectRow(row); self.edit_selected_plate()

    def _context_delete(self, idx):
        if not idx.isValid(): return
        sel = self.table.selectionModel(); sel.clearSelection(); self.table.selectRow(idx.row()); self.delete_selected()

    # helpers
    def _show_status(self, text, timeout=3000):
        self.status_label.setText(text)
        if timeout>0: QTimer.singleShot(timeout, lambda: self.status_label.setText(""))

    # logout / close
    def _logout(self):
        confirm = QMessageBox.question(self, "Đăng xuất", "Bạn có chắc muốn đăng xuất?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes: return
        if hasattr(self, "_login_window_ref") and self._login_window_ref:
            try: self._login_window_ref.show()
            except Exception: pass
        self.close()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Thoát", "Bạn có chắc muốn đóng cửa sổ quản trị?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore(); return
        if hasattr(self, "_login_window_ref") and self._login_window_ref:
            try: self._login_window_ref.show()
            except Exception: pass
        event.accept()

if __name__ == "__main__":
    init_db()
    app = QApplication(sys.argv)
    win = AdminWindow(); win.show(); sys.exit(app.exec())