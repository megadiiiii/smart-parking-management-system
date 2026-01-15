# login_user_gui.py (FULL - Tiếng Việt)
# Pair Face + Plate, modern UI, cameras stacked in left column
# Tính năng:
# - Phím tắt: '1' = chụp Cam1, '2' = chụp Cam2, '0' = reset panel
# - Nếu sau commit (sau khi cả 2 cam quét và ghép cặp) trong 5s không có hành động,
#   panel sẽ tự động clear (reset)
# - Toast non-blocking, Đăng xuất, phím tắt, auto-capture, reset panels
# - Giữ logic pairing, face detection (InsightFace) và YOLO (nếu có)
#
# Yêu cầu: PyQt6, opencv-python, numpy
# Optional: insightface, torch (để dùng YOLO/OCR)
#
# Lưu ý: trạng thái trong DB dùng "Vào"/"Ra"

import os
import sys
import time
import datetime
import warnings
import multiprocessing as mp

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(os.path.join(RESULT_DIR, "face"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "plate"), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "fullface"), exist_ok=True)

# ensure local project path is visible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Limit native threads (stability)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QLineEdit, QPushButton, QMessageBox, QSpinBox, QInputDialog,
    QSizePolicy, QSpacerItem
)

# Optional torch for YOLO plate
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# Try import helper/utils_rotate like webcam.py (support both package & flat files)
HELPER_OK = True
try:
    import function.utils_rotate as utils_rotate
    import function.helper as helper
except Exception:
    try:
        import utils_rotate
        import helper
    except Exception:
        HELPER_OK = False
        utils_rotate = None
        helper = None

# ------------------- CONFIG -------------------
RESULT_DIR = os.path.join(os.getcwd(), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

FACE_DB_DIR = os.path.join(os.getcwd(), "face_db")
os.makedirs(FACE_DB_DIR, exist_ok=True)

PAIR_TTL = 15.0  # seconds for pairing pending

FACE_DET_SIZE = (320, 320)
FACE_TICK_MS = 350
FACE_TICK_SCALE = 0.5
FACE_SIM_THRESHOLD = 0.5

YOLO_DET_PATH = "model/LP_detector_nano_61.pt"
YOLO_OCR_PATH = "model/LP_ocr_nano_62.pt"
YOLO_OCR_CONF = 0.6

# Simple dark style
DARK_QSS = """
QWidget { background: #121212; color: #e6eef3; font-family: "Segoe UI"; }
QGroupBox { border: 1px solid #2b2d31; border-radius: 6px; padding: 6px; background: #141414; }
QPushButton { background: #2d8cff; color: white; border-radius: 6px; padding: 6px 8px; }
QPushButton[secondary="true"] { background: transparent; border: 1px solid #3b3e44; color: #cfe8ff; }
QLabel { color: #e6eef3; }
"""

# ------------------- Utils -------------------
def save_image_numpy(img_bgr, prefix="capture"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if prefix.startswith("plate"):
        sub = "plate"
    elif prefix.startswith("face_full"):
        sub = "fullface"
    elif prefix.startswith("face"):
        sub = "face"
    else:
        sub = "other"
    save_dir = os.path.join(RESULT_DIR, sub)
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{prefix}_{ts}.jpg"
    path = os.path.join(save_dir, fname)
    cv2.imwrite(path, img_bgr)
    return path

def cv_frame_to_qpixmap(frame_bgr, max_width=None, max_height=None):
    if frame_bgr is None:
        return QPixmap()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if max_width or max_height:
        pix = pix.scaled(max_width or w, max_height or h,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return pix

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def load_face_db_embeddings(face_db_dir: str):
    ids, embs = [], []
    if not os.path.isdir(face_db_dir):
        return ids, embs
    for f in os.listdir(face_db_dir):
        if f.endswith(".npy"):
            try:
                ids.append(f.replace(".npy", ""))
                embs.append(np.load(os.path.join(face_db_dir, f)))
            except Exception:
                pass
    return ids, embs

# ------------------- Face Process (InsightFace isolated) -------------------
def face_process_main(in_q: mp.Queue, out_q: mp.Queue,
                      face_db_dir: str,
                      det_size=(320, 320),
                      det_scale=0.5,
                      sim_threshold=0.5):
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        while True:
            item = in_q.get()
            if item is None:
                break
            out_q.put({"bbox": None, "label": None, "err": f"insightface import fail: {e}"})
        return

    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)
    ids, embs = load_face_db_embeddings(face_db_dir)

    while True:
        frame = in_q.get()
        if frame is None:
            break
        try:
            h0, w0 = frame.shape[:2]
            s = det_scale
            if s <= 0 or s > 1:
                s = 0.5
            small = cv2.resize(frame, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_LINEAR)
            faces = app.get(small)
            if not faces:
                out_q.put({"bbox": None, "label": None})
                continue
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = face.normed_embedding
            x1, y1, x2, y2 = face.bbox.astype(float)
            inv = 1.0 / s
            x1 = int(x1 * inv); y1 = int(y1 * inv); x2 = int(x2 * inv); y2 = int(y2 * inv)
            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0, x2))
            y2 = max(0, min(h0, y2))
            bbox = (x1, y1, x2 - x1, y2 - y1)
            label = "UNKNOWN"
            best_id, best_score = None, 0.0
            for sid, semb in zip(ids, embs):
                sscore = cosine(emb, semb)
                if sscore > best_score:
                    best_score, best_id = sscore, sid
            if best_id is not None and best_score >= sim_threshold:
                label = f"ID {best_id}"
            out_q.put({"bbox": bbox, "label": label})
        except Exception:
            out_q.put({"bbox": None, "label": None})

# ------------------- Camera Thread -------------------
class CameraThread(QThread):
    frame_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    def __init__(self, source):
        super().__init__()
        self.source = source
        self._running = False
        self.cap = None

    def run(self):
        try:
            s = self.source
            try:
                s_conv = int(s)
            except Exception:
                s_conv = s
            self.cap = cv2.VideoCapture(s_conv, cv2.CAP_DSHOW if os.name == "nt" else 0)
            if not self.cap.isOpened():
                self.error_signal.emit(f"Không mở được camera: {self.source}")
                return
            self._running = True
            while self._running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue
                self.frame_signal.emit(frame)
                time.sleep(0.01)
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

    def stop(self):
        self._running = False
        self.wait(500)

# ------------------- Camera Widget -------------------
class CameraWidget(QGroupBox):
    def __init__(self, title, mode):
        super().__init__(title)
        self.mode = mode
        self.thread = None
        self.last_frame = None
        self.last_detection = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        controls = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.addItems(["0", "1", "2", "3", "Custom URL"])
        self.combo.currentTextChanged.connect(self._on_combo_change)
        controls.addWidget(QLabel("Nguồn:"))
        controls.addWidget(self.combo)
        self.source_edit = QLineEdit("0")
        controls.addWidget(self.source_edit)
        self.btn_start = QPushButton("Bắt đầu")
        self.btn_start.clicked.connect(self.start_camera)
        controls.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Dừng")
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_stop.setEnabled(False)
        controls.addWidget(self.btn_stop)
        layout.addLayout(controls)

        self.preview = QLabel("Không có video")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#0f1112; border-radius:6px;")
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.preview, stretch=3)

        self.info = QLabel("")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info.setFixedHeight(28)
        layout.addWidget(self.info)

        self.setLayout(layout)

    def _on_combo_change(self, txt):
        if txt == "Custom URL":
            self.source_edit.setText("")
            self.source_edit.setPlaceholderText("rtsp://... or file")
        else:
            self.source_edit.setText(txt)

    def start_camera(self):
        src = self.source_edit.text().strip()
        if src == "":
            QMessageBox.warning(self, "Source empty", "Please provide camera source.")
            return
        if self.thread is not None:
            self.stop_camera()
        self.thread = CameraThread(src)
        self.thread.frame_signal.connect(self._on_frame)
        self.thread.error_signal.connect(self._on_error)
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.preview.clear()
        self.preview.setText("Đã dừng")
        self.info.setText("")

    def _on_error(self, msg):
        QMessageBox.critical(self, "Lỗi camera", msg)
        self.stop_camera()

    def _on_frame(self, frame):
        self.last_frame = frame
        display = frame.copy()
        if self.mode == "face":
            bbox = self.last_detection.get("face_bbox")
            label = self.last_detection.get("face_label")
            if bbox and label:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, str(label), (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                self.info.setText(str(label))
            else:
                self.info.setText("")
        elif self.mode == "plate":
            bbox = self.last_detection.get("plate_bbox")
            text = self.last_detection.get("plate_text")
            if bbox and text:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, str(text), (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                self.info.setText(str(text))
            else:
                self.info.setText("")
        try:
            pix = cv_frame_to_qpixmap(display, max_width=self.preview.width(), max_height=self.preview.height())
            self.preview.setPixmap(pix)
        except Exception:
            pass

    def get_last_frame(self):
        return self.last_frame

# ------------------- Main Window -------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bảng điều khiển - Người dùng")
        self.resize(1320, 820)
        self.setStyleSheet(DARK_QSS)

        # pending and status
        self.pending_face = None
        self.pending_plate = None
        self.plate_last_status = {}  # key -> "Vào" or "Ra"

        # store current displayed image paths for rescaling
        self.display_entry_plate = None
        self.display_entry_face = None
        self.display_exit_plate = None
        self.display_exit_face = None

        # YOLO/face process placeholders
        self._yolo_loaded = False
        self._yolo_detect = None
        self._yolo_ocr = None
        self.face_in = None
        self.face_out = None
        self.face_p = None

        # status label (non-blocking toast)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #cfe8ff;")
        self.status_timer = QTimer(self)
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(lambda: self.status_label.setText(""))

        # auto clear timer (clears panels 5s after commit)
        self._auto_clear_timer = QTimer(self)
        self._auto_clear_timer.setSingleShot(True)
        self._auto_clear_timer.timeout.connect(self._reset_pair_panel)

        # ensure widget receives key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._init_ui()
        self._add_shortcuts()
        self._start_face_process()
        self._start_face_timer()

    def _init_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 3 equal columns layout
        cols = QHBoxLayout()
        cols.setSpacing(12)

        # Left column: cameras stacked
        left_col = QGroupBox("Camera")
        left_layout = QVBoxLayout()
        self.cam1_widget = CameraWidget("Camera 1 (Mặt)", mode='face')
        self.cam2_widget = CameraWidget("Camera 2 (Biển)", mode='plate')
        left_layout.addWidget(self.cam1_widget, stretch=1)
        left_layout.addWidget(self.cam2_widget, stretch=1)

        # camera controls under cameras (capture buttons, auto, reset, logout)
        btn_row = QHBoxLayout()
        self.btn_capture_cam1 = QPushButton("Chụp mặt (Cam1)")
        self.btn_capture_cam1.setToolTip("Chụp ảnh mặt (Vào)")
        self.btn_capture_cam1.clicked.connect(self._capture_face)
        btn_row.addWidget(self.btn_capture_cam1)

        self.btn_capture_cam2 = QPushButton("Chụp biển (Cam2)")
        self.btn_capture_cam2.setToolTip("Chụp ảnh biển (Vào/Ra tuỳ trạng thái)")
        self.btn_capture_cam2.clicked.connect(self._capture_plate)
        btn_row.addWidget(self.btn_capture_cam2)

        left_layout.addLayout(btn_row)

        # Auto + Reset + Logout row
        ar = QHBoxLayout()
        ar.addWidget(QLabel("Auto-pair (s):"))
        self.auto_spin = QSpinBox()
        self.auto_spin.setRange(0, 3600)
        self.auto_spin.setValue(0)
        ar.addWidget(self.auto_spin)
        self.btn_start_auto = QPushButton("Bắt đầu Auto")
        self.btn_start_auto.clicked.connect(self._toggle_auto)
        ar.addWidget(self.btn_start_auto)
        self.btn_reset = QPushButton("Reset Panels")
        self.btn_reset.setProperty("secondary", True)
        self.btn_reset.clicked.connect(self._reset_pair_panel)
        ar.addWidget(self.btn_reset)
        self.btn_logout = QPushButton("Đăng xuất")
        self.btn_logout.setProperty("secondary", True)
        self.btn_logout.clicked.connect(self._logout)
        ar.addWidget(self.btn_logout)
        left_layout.addLayout(ar)

        left_col.setLayout(left_layout)
        left_col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cols.addWidget(left_col, stretch=1)

        # Middle column: Vào
        middle_col = QGroupBox("Vào")
        middle_layout = QVBoxLayout()
        self.entry_plate = QLabel("Ảnh biển (Vào)")
        self.entry_plate.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.entry_plate.setStyleSheet("background:#0f1112; border-radius:6px;")
        self.entry_plate.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.entry_face = QLabel("Ảnh mặt (Vào)")
        self.entry_face.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.entry_face.setStyleSheet("background:#0f1112; border-radius:6px;")
        self.entry_face.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        middle_layout.addWidget(self.entry_plate, stretch=1)
        middle_layout.addWidget(self.entry_face, stretch=1)
        middle_col.setLayout(middle_layout)
        middle_col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cols.addWidget(middle_col, stretch=1)

        # Right column: Ra
        right_col = QGroupBox("Ra")
        right_layout = QVBoxLayout()
        self.exit_plate = QLabel("Ảnh biển (Ra)")
        self.exit_plate.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exit_plate.setStyleSheet("background:#0f1112; border-radius:6px;")
        self.exit_plate.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.exit_face = QLabel("Ảnh mặt (Ra - mới nhất)")
        self.exit_face.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exit_face.setStyleSheet("background:#0f1112; border-radius:6px;")
        self.exit_face.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.exit_plate, stretch=1)
        right_layout.addWidget(self.exit_face, stretch=1)
        right_col.setLayout(right_layout)
        right_col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cols.addWidget(right_col, stretch=1)

        root.addLayout(cols)

        # Metadata (large)
        meta = QHBoxLayout()
        self.lb_plate = QLabel("Biển số: -")
        self.lb_time = QLabel("Thời gian: -")
        self.lb_status = QLabel("Trạng thái: -")
        self.lb_plate.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.lb_time.setFont(QFont("Segoe UI", 11))
        self.lb_status.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        meta.addWidget(self.lb_plate)
        meta.addSpacing(20)
        meta.addWidget(self.lb_time)
        meta.addSpacing(20)
        meta.addWidget(self.lb_status)
        meta.addStretch()
        root.addLayout(meta)

        root.addWidget(self.status_label)

        root.addWidget(QLabel("Ghi chú: Ghi log chỉ khi CẢ mặt + biển được chụp trong TTL."))

        self.setLayout(root)

        # auto timer
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self._auto_capture_tick)
        self.auto_running = False

    # ---------- Shortcuts ----------
    def _add_shortcuts(self):
        # Ctrl+L logout
        act_logout = QAction("Đăng xuất", self)
        act_logout.setShortcut(QKeySequence("Ctrl+L"))
        act_logout.triggered.connect(self._logout)
        self.addAction(act_logout)

        # Ctrl+R reset panels
        act_reset = QAction("Reset Panels", self)
        act_reset.setShortcut(QKeySequence("Ctrl+R"))
        act_reset.triggered.connect(self._reset_pair_panel)
        self.addAction(act_reset)

        # F5 auto-capture tick
        act_f5 = QAction("Auto capture", self)
        act_f5.setShortcut(QKeySequence("F5"))
        act_f5.triggered.connect(self._auto_capture_tick)
        self.addAction(act_f5)

    # ---------- Key handling (new) ----------
    def keyPressEvent(self, event):
        """
        Phím tắt:
         - '1' -> chụp Cam1 (face)
         - '2' -> chụp Cam2 (plate)
         - '0' -> reset panels
        """
        k = event.key()
        if k == Qt.Key.Key_1:
            self._capture_face()
            self.show_toast("Đã chụp Cam1 (phím 1)", timeout=1500)
            return
        if k == Qt.Key.Key_2:
            self._capture_plate()
            self.show_toast("Đã chụp Cam2 (phím 2)", timeout=1500)
            return
        if k == Qt.Key.Key_0:
            self._reset_pair_panel()
            self.show_toast("Đã reset panels (phím 0)", timeout=1500)
            return
        super().keyPressEvent(event)

    # ---------- Face Process ----------
    def _start_face_process(self):
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass
        self.face_in = mp.Queue(maxsize=1)
        self.face_out = mp.Queue(maxsize=1)
        self.face_p = mp.Process(
            target=face_process_main,
            args=(self.face_in, self.face_out, FACE_DB_DIR, FACE_DET_SIZE, FACE_TICK_SCALE, FACE_SIM_THRESHOLD),
            daemon=True
        )
        self.face_p.start()

    def _start_face_timer(self):
        self.face_timer = QTimer()
        self.face_timer.timeout.connect(self._face_tick)
        self.face_timer.start(FACE_TICK_MS)

    def _face_tick(self):
        frame = self.cam1_widget.get_last_frame()
        if frame is None:
            return
        try:
            while True:
                self.face_in.get_nowait()
        except Exception:
            pass
        try:
            self.face_in.put_nowait(frame.copy())
        except Exception:
            pass
        latest = None
        try:
            while True:
                latest = self.face_out.get_nowait()
        except Exception:
            pass
        if latest:
            bbox = latest.get("bbox")
            label = latest.get("label")
            self.cam1_widget.last_detection["face_bbox"] = bbox
            self.cam1_widget.last_detection["face_label"] = label

    # ---------- DB helpers ----------
    def get_latest_in_out_for_plate(self, plate_text):
        import sqlite3
        from db import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT time_in, face_image_path, plate_image_path
            FROM events
            WHERE plate_text = ? AND status = 'Vào'
            ORDER BY id DESC LIMIT 1
        """, (plate_text,))
        row_in = cur.fetchone()
        cur.execute("""
            SELECT time_out, face_image_path, plate_image_path
            FROM events
            WHERE plate_text = ? AND status = 'Ra'
            ORDER BY id DESC LIMIT 1
        """, (plate_text,))
        row_out = cur.fetchone()
        conn.close()
        if row_in:
            in_time, in_face, in_plate = row_in[0], row_in[1], row_in[2]
        else:
            in_time, in_face, in_plate = (None, None, None)
        if row_out:
            out_time, out_face, out_plate = row_out[0], row_out[1], row_out[2]
        else:
            out_time, out_face, out_plate = (None, None, None)
        return (in_time, in_face, in_plate), (out_time, out_face, out_plate)

    def get_latest_face_for_plate(self, plate_text):
        if not plate_text:
            return None
        import sqlite3
        from db import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT face_image_path
            FROM events
            WHERE plate_text = ? AND face_image_path IS NOT NULL AND face_image_path != ''
            ORDER BY id DESC LIMIT 1
        """, (plate_text,))
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
        return None

    def detect_in_out(self, plate_text):
        """Predict next status for a plate: default Vào if none, else toggle between Vào <-> Ra"""
        import sqlite3
        from db import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT status FROM events WHERE plate_text = ? ORDER BY id DESC LIMIT 1", (plate_text,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return "Vào"
        return "Ra" if row[0] == "Vào" else "Vào"

    # ---------- Insert event ----------
    def insert_event(self, plate_text, status, face_path, plate_path):
        import sqlite3
        from db import DB_PATH
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_in = now if status == "Vào" else None
        time_out = now if status == "Ra" else None
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO events
            (plate_text, status, time_in, time_out, face_image_path, plate_image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (plate_text, status, time_in, time_out, face_path, plate_path))
        conn.commit()
        conn.close()

        # After commit, fetch latest Vào and Ra for this plate and update UI panels
        try:
            (in_time, in_face, in_plate), (out_time, out_face, out_plate) = self.get_latest_in_out_for_plate(plate_text)
            self._update_pair_panel(plate_text, (in_time, in_face, in_plate), (out_time, out_face, out_plate))
        except Exception:
            if status == "Vào":
                self.display_entry_plate = plate_path
                self.display_entry_face = face_path
                self._set_label_image(self.entry_plate, self.display_entry_plate)
                self._set_label_image(self.entry_face, self.display_entry_face)
            else:
                self.display_exit_plate = plate_path
                self.display_exit_face = out_face
                self._set_label_image(self.exit_plate, self.display_exit_plate)
                if in_face:
                    self.display_entry_face = in_face
                    self._set_label_image(self.entry_face, self.display_entry_face)

        # Update metadata and toast
        self.lb_plate.setText(f"Biển số: {plate_text}")
        self.lb_time.setText(f"{'Vào' if status == 'Vào' else 'Ra'}: {now}")
        self.lb_status.setText(f"Trạng thái: {status}")
        self.show_toast(f"Đã ghi: {plate_text} ({status})", timeout=4000)
        print("INSERTED:", plate_text, status)

    # ---------- YOLO Plate ----------
    def _ensure_yolo_models(self):
        if self._yolo_loaded:
            return
        self._yolo_loaded = True
        if not HAS_TORCH:
            self._yolo_detect = None
            self._yolo_ocr = None
            return
        try:
            self._yolo_detect = torch.hub.load('yolov5', 'custom', path=YOLO_DET_PATH, source='local')
            self._yolo_ocr = torch.hub.load('yolov5', 'custom', path=YOLO_OCR_PATH, source='local')
            self._yolo_ocr.conf = YOLO_OCR_CONF
        except Exception as e:
            print("YOLO load failed:", e)
            self._yolo_detect = None
            self._yolo_ocr = None

    def _detect_plate_on_capture(self, frame):
        if frame is None:
            return None, None, ""
        if not HAS_TORCH:
            return None, None, ""
        self._ensure_yolo_models()
        if self._yolo_detect is None or self._yolo_ocr is None:
            return None, None, ""
        if not HELPER_OK:
            print("[WARN] helper/utils_rotate not found -> OCR may not work.")
            return None, None, ""
        h0, w0 = frame.shape[:2]
        try:
            results = self._yolo_detect(frame, size=640)
            dets = results.xyxy[0]
            if dets is None or len(dets) == 0:
                return None, None, ""
            det_best = max(dets.tolist(), key=lambda x: x[4])
            x1, y1, x2, y2, conf, cls = det_best
            x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)
            if x2 <= x1 or y2 <= y1:
                return None, None, ""
            crop = frame[y1:y2, x1:x2]
            plate_img_path = save_image_numpy(crop, prefix="plate")
            plate_text = "unknown"
            for cc in range(2):
                for ct in range(2):
                    plate_text = helper.read_plate(self._yolo_ocr, utils_rotate.deskew(crop, cc, ct))
                    if plate_text != "unknown":
                        break
                if plate_text != "unknown":
                    break
            bbox = (x1, y1, x2 - x1, y2 - y1)
            if plate_text == "unknown":
                return None, bbox, plate_img_path
            return plate_text, bbox, plate_img_path
        except Exception as e:
            print("Plate detect error:", e)
            return None, None, ""

    # ---------- Panel helpers ----------
    def _set_label_image(self, label_widget, img_path):
        if not img_path or not os.path.exists(img_path):
            label_widget.setPixmap(QPixmap())
            label_widget.setText("")
            return
        try:
            pix = QPixmap(img_path)
            w = max(10, label_widget.width())
            h = max(10, label_widget.height())
            pix = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label_widget.setPixmap(pix)
            label_widget.setText("")
        except Exception:
            label_widget.setPixmap(QPixmap())
            label_widget.setText("Lỗi ảnh")

    def _update_pair_panel(self, plate_text, in_tuple, out_tuple):
        in_time, in_face, in_plate = in_tuple
        out_time, out_face, out_plate = out_tuple

        # If no explicit Vào data but a Ra exists, try DB lookup for Vào
        if (not in_time and not in_face and not in_plate) and (plate_text and (out_plate or out_time)):
            try:
                (db_in_time, db_in_face, db_in_plate), _ = self.get_latest_in_out_for_plate(plate_text)
                if db_in_time:
                    in_time = in_time or db_in_time
                    in_face = in_face or db_in_face
                    in_plate = in_plate or db_in_plate
            except Exception:
                pass

        # Update metadata
        self.lb_plate.setText(f"Biển số: {plate_text or '-'}")
        times = []
        if in_time: times.append(f"Vào: {in_time}")
        if out_time: times.append(f"Ra: {out_time}")
        self.lb_time.setText("   ".join(times) if times else "-")
        status = "Vào/Ra" if in_time and out_time else ("Chỉ Vào" if in_time else ("Chỉ Ra" if out_time else "-"))
        self.lb_status.setText(f"Trạng thái: {status}")

        # display paths
        self.display_entry_plate = in_plate
        self.display_entry_face = in_face
        self.display_exit_plate = out_plate
        if in_time and out_time:
            self.display_exit_face = self.get_latest_face_for_plate(plate_text) or out_face
        else:
            self.display_exit_face = out_face

        # set images scaled
        self._set_label_image(self.entry_plate, self.display_entry_plate)
        self._set_label_image(self.entry_face, self.display_entry_face)
        self._set_label_image(self.exit_plate, self.display_exit_plate)
        self._set_label_image(self.exit_face, self.display_exit_face)

    def _show_pending_pair_panel(self):
        """
        - Predict plate next status using detect_in_out and show pending plate in matching panel.
        - If predicted == 'Ra' (out), fetch latest Vào record from DB and show both:
            Entry <- DB Vào (plate + face if exist), Exit <- pending Ra (new plate image).
        - Pending face always shown in ENTRY face (face belongs to Vào).
        """
        plate_text = self.pending_plate.get("text") if self.pending_plate else (self.pending_face.get("label") if self.pending_face else "-")
        in_face = self.pending_face.get("img") if self.pending_face else None
        pending_plate_img = self.pending_plate.get("img") if self.pending_plate else None

        in_time = None; in_plate = None; in_face_path = None
        out_time = None; out_plate = None; out_face_path = None

        in_face_path = in_face

        if self.pending_plate:
            predicted = self.detect_in_out(plate_text)
            now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if predicted == "Vào":
                in_plate = pending_plate_img
                in_time = now_ts
            else:
                out_plate = pending_plate_img
                out_time = now_ts
                try:
                    (db_in_time, db_in_face, db_in_plate), _ = self.get_latest_in_out_for_plate(plate_text)
                    if db_in_time:
                        in_time = db_in_time if in_time is None else in_time
                        if db_in_plate:
                            in_plate = db_in_plate
                        if db_in_face and in_face_path is None:
                            in_face_path = db_in_face
                except Exception:
                    pass

        self._update_pair_panel(plate_text, (in_time, in_face_path, in_plate), (out_time, out_face_path, out_plate))

    # ---------- Capture actions ----------
    def _capture_face(self):
        frame = self.cam1_widget.get_last_frame()
        det = self.cam1_widget.last_detection or {}
        if frame is None:
            QMessageBox.warning(self, "Không có khung", "Cam1 chưa có khung hình.")
            return
        face_bbox = det.get("face_bbox")
        face_label = det.get("face_label") or "UnknownFace"
        if face_bbox:
            x, y, w, h = face_bbox
            x2, y2 = x + w, y + h
            x, y = max(0, int(x)), max(0, int(y))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            face_crop = frame[y:y2, x:x2]
            face_path = save_image_numpy(face_crop, prefix="face")
        else:
            face_path = save_image_numpy(frame, prefix="face_full")
        self.pending_face = {"t": time.time(), "label": face_label, "img": face_path, "bbox": face_bbox}
        self._show_pending_pair_panel()
        if face_bbox:
            QMessageBox.information(self, "Mặt OK", f"Đã nhận diện mặt: {face_label}\nBấm 'Chụp biển (Cam2)' để ghép cặp.")
        else:
            QMessageBox.warning(self, "Mặt không rõ", "Chưa thấy mặt rõ trên Cam1.")
        self._try_commit_pair()

    def _capture_plate(self):
        frame = self.cam2_widget.get_last_frame()
        if frame is None:
            QMessageBox.warning(self, "Không có khung", "Cam2 chưa có khung hình.")
            return
        plate_text, plate_bbox, plate_img_path = self._detect_plate_on_capture(frame)
        if not plate_text:
            plate_text, ok = QInputDialog.getText(self, "Nhập biển số", "Không nhận diện, nhập thủ công:")
            if not ok or plate_text.strip() == "":
                return
            plate_text = plate_text.strip()
            if not plate_img_path:
                plate_img_path = save_image_numpy(frame, prefix="plate_full")
        self.cam2_widget.last_detection["plate_text"] = plate_text
        self.cam2_widget.last_detection["plate_bbox"] = plate_bbox
        self.pending_plate = {"t": time.time(), "text": plate_text, "img": plate_img_path, "bbox": plate_bbox}
        self._show_pending_pair_panel()
        was_face = self.pending_face is not None
        self._try_commit_pair()
        if self.pending_face is None and was_face is False:
            QMessageBox.information(self, "Thiếu mặt", "Đã nhận biển số.\nBấm 'Chụp mặt (Cam1)' để ghép cặp.")
        else:
            if self.pending_face is None and self.pending_plate is None:
                QMessageBox.information(self, "Ghép thành công", "Đã ghép cặp Mặt + Biển số thành công!")

    # ---------- Pairing ----------
    def _cleanup_pending(self):
        now = time.time()
        if self.pending_face and (now - self.pending_face["t"] > PAIR_TTL):
            self.pending_face = None
        if self.pending_plate and (now - self.pending_plate["t"] > PAIR_TTL):
            self.pending_plate = None

    def _try_commit_pair(self):
        self._cleanup_pending()
        if not self.pending_face or not self.pending_plate:
            return
        face = self.pending_face
        plate = self.pending_plate
        key = plate["text"] or "UnknownPlate"
        last = self.plate_last_status.get(key)
        status = "Ra" if last == "Vào" else "Vào"
        self.plate_last_status[key] = status
        # commit
        self.insert_event(plate_text=plate["text"] or "UnknownPlate",
                          status=status,
                          face_path=face["img"],
                          plate_path=plate["img"])
        # reset pending after commit
        self.pending_face = None
        self.pending_plate = None

        # Start auto-clear timer: will reset panels after 5s unless user does something
        self.start_auto_clear_timer(5000)

    # ---------- Auto-clear helpers (new) ----------
    def start_auto_clear_timer(self, ms=5000):
        """Bật timer để tự động reset panels sau ms milliseconds."""
        try:
            if self._auto_clear_timer.isActive():
                self._auto_clear_timer.stop()
            self._auto_clear_timer.start(ms)
            self.show_toast(f"Panels sẽ tự xóa sau {ms//1000} giây...", timeout=ms)
        except Exception:
            pass

    def stop_auto_clear_timer(self):
        try:
            if self._auto_clear_timer.isActive():
                self._auto_clear_timer.stop()
        except Exception:
            pass

    # ---------- Reset ----------
    def _reset_pair_panel(self):
        # stop auto clear when user resets manually
        self.stop_auto_clear_timer()
        self.pending_face = None
        self.pending_plate = None
        self.display_entry_plate = None
        self.display_entry_face = None
        self.display_exit_plate = None
        self.display_exit_face = None
        self.entry_plate.setPixmap(QPixmap()); self.entry_plate.setText("Ảnh biển (Vào)")
        self.entry_face.setPixmap(QPixmap()); self.entry_face.setText("Ảnh mặt (Vào)")
        self.exit_plate.setPixmap(QPixmap()); self.exit_plate.setText("Ảnh biển (Ra)")
        self.exit_face.setPixmap(QPixmap()); self.exit_face.setText("Ảnh mặt (Ra - mới nhất)")
        self.lb_plate.setText("Biển số: -"); self.lb_time.setText("Thời gian: -"); self.lb_status.setText("Trạng thái: -")
        self.show_toast("Đã reset panels", timeout=2000)

    # ---------- Auto ----------
    def _toggle_auto(self):
        secs = self.auto_spin.value()
        if secs <= 0:
            QMessageBox.information(self, "Auto", "Chọn >0 giây để bật Auto.")
            return
        if not self.auto_running:
            self.auto_timer.start(secs * 1000)
            self.auto_running = True
            self.btn_start_auto.setText("Dừng Auto")
            self.show_toast("Auto bật", timeout=2000)
        else:
            self.auto_timer.stop()
            self.auto_running = False
            self.btn_start_auto.setText("Bắt đầu Auto")
            self.show_toast("Auto tắt", timeout=2000)

    def _auto_capture_tick(self):
        if self.cam1_widget.thread:
            self._capture_face()
        if self.cam2_widget.thread:
            self._capture_plate()

    # ---------- Toast ----------
    def show_toast(self, text, timeout=3000):
        self.status_label.setText(text)
        self.status_timer.start(timeout)

    # ---------- Resize handling ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # rescale displayed images if present
        if self.display_entry_plate:
            self._set_label_image(self.entry_plate, self.display_entry_plate)
        if self.display_entry_face:
            self._set_label_image(self.entry_face, self.display_entry_face)
        if self.display_exit_plate:
            self._set_label_image(self.exit_plate, self.display_exit_plate)
        if self.display_exit_face:
            self._set_label_image(self.exit_face, self.display_exit_face)
        # refresh camera previews
        try:
            if self.cam1_widget.last_frame is not None:
                self.cam1_widget._on_frame(self.cam1_widget.last_frame)
            if self.cam2_widget.last_frame is not None:
                self.cam2_widget._on_frame(self.cam2_widget.last_frame)
        except Exception:
            pass

    def _set_label_image(self, label_widget, img_path):
        if not img_path or not os.path.exists(img_path):
            label_widget.setPixmap(QPixmap())
            label_widget.setText("")
            return
        try:
            pix = QPixmap(img_path)
            w = max(10, label_widget.width())
            h = max(10, label_widget.height())
            pix = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label_widget.setPixmap(pix)
            label_widget.setText("")
        except Exception:
            label_widget.setPixmap(QPixmap())
            label_widget.setText("Lỗi ảnh")

    # ---------- Logout ----------
    def _logout(self):
        confirm = QMessageBox.question(self, "Đăng xuất", "Bạn có chắc muốn đăng xuất?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            return
        # stop auto-clear timer when logging out
        self.stop_auto_clear_timer()
        if hasattr(self, "_login_window_ref") and self._login_window_ref:
            try:
                self._login_window_ref.show()
            except Exception:
                pass
        self.close()

    # ---------- Clean shutdown ----------
    def closeEvent(self, e):
        try:
            if self.cam1_widget.thread:
                self.cam1_widget.stop_camera()
            if self.cam2_widget.thread:
                self.cam2_widget.stop_camera()
        except Exception:
            pass
        try:
            if self.face_in is not None:
                self.face_in.put(None)
        except Exception:
            pass
        try:
            if self.face_p is not None and self.face_p.is_alive():
                self.face_p.terminate()
        except Exception:
            pass
        super().closeEvent(e)

# ---------- main ----------
def main():
    from db import init_db
    init_db()
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    # start face timer after show
    win._start_face_timer()
    # ensure widget has focus so keyPressEvent works
    win.setFocus()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()