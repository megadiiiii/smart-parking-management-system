# login_modern.py
# Modern login window (Vietnamese). Tạo style giống Figma modal và làm nổi chữ ở input field (placeholder rõ hơn).
#
# Usage:
#  pip install PyQt6
#  python login_modern.py

import sys
import traceback
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QCheckBox,
    QFrame, QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QFont, QCursor, QColor
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve

# Demo credentials (thay bằng auth thực trong production)
VALID_CREDENTIALS = {
    "admin": "admin",
    "user": "user"
}

class ModernLoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._main_window_ref = None  # giữ tham chiếu đến cửa sổ chính để show lại khi logout
        self.setWindowTitle("Đăng nhập - Modern")
        self.resize(520, 360)

        self._init_ui()
        self._apply_animations()

    def _init_ui(self):
        central = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(16, 16, 16, 16)
        central_layout.setSpacing(8)
        central.setLayout(central_layout)
        self.setCentralWidget(central)

        # Card container (dark themed)
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(22, 18, 22, 18)
        card_layout.setSpacing(12)
        card.setLayout(card_layout)

        # Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(26)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 180))
        card.setGraphicsEffect(shadow)

        # Header
        header = QLabel("Đăng nhập")
        header.setObjectName("header")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setFixedHeight(36)
        card_layout.addWidget(header)

        # Subtitle
        subtitle = QLabel("Nhập thông tin tài khoản để tiếp tục")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft)
        subtitle.setFixedHeight(18)
        card_layout.addWidget(subtitle)

        # Username
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Tên đăng nhập")
        self.username_input.setObjectName("input")
        self.username_input.setFixedHeight(40)
        card_layout.addWidget(self.username_input)

        # Password + toggle
        pw_row = QHBoxLayout()
        pw_row.setSpacing(8)
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Mật khẩu")
        self.password_input.setObjectName("input")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setFixedHeight(40)
        pw_row.addWidget(self.password_input)

        self.toggle_pw_btn = QPushButton("👁")
        self.toggle_pw_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.toggle_pw_btn.setFixedSize(40, 40)
        self.toggle_pw_btn.setObjectName("iconbtn")
        self.toggle_pw_btn.setToolTip("Hiện/Ẩn mật khẩu")
        self.toggle_pw_btn.clicked.connect(self._toggle_show_password)
        pw_row.addWidget(self.toggle_pw_btn)

        card_layout.addLayout(pw_row)

        # remember + forgot
        row = QHBoxLayout()
        self.remember_cb = QCheckBox("Ghi nhớ đăng nhập")
        self.remember_cb.setObjectName("remember")
        row.addWidget(self.remember_cb)
        row.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        forgot = QLabel('<a href="#">Quên mật khẩu?</a>')
        forgot.setOpenExternalLinks(False)
        forgot.linkActivated.connect(self._forgot_pw)
        forgot.setObjectName("link")
        row.addWidget(forgot)
        card_layout.addLayout(row)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch()

        self.login_btn = QPushButton("Đăng nhập")
        self.login_btn.setObjectName("primary")
        self.login_btn.setFixedHeight(44)
        self.login_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.login_btn.clicked.connect(self._handle_login)
        self.login_btn.setDefault(True)
        btn_row.addWidget(self.login_btn)

        self.cancel_btn = QPushButton("Hủy")
        self.cancel_btn.setObjectName("secondary")
        self.cancel_btn.setFixedHeight(44)
        self.cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(self.cancel_btn)

        card_layout.addLayout(btn_row)

        # Footer small
        footer = QLabel("Được xây dựng với ❤️  •  PyQt6")
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(footer)

        # Center the card
        central_layout.addStretch()
        h = QHBoxLayout()
        h.addStretch()
        h.addWidget(card)
        h.addStretch()
        central_layout.addLayout(h)
        central_layout.addStretch()

        # Keyboard handling
        self.username_input.returnPressed.connect(self.password_input.setFocus)
        self.password_input.returnPressed.connect(self._handle_login)

        # Apply styles
        self._apply_styles()

    def _apply_styles(self):
        # Dark modern style, inputs made more prominent (white background + dark text)
        self.setStyleSheet("""
        QMainWindow { background: #0b0c0e; }
        QFrame#card {
            background: #14161a;
            border-radius: 14px;
        }
        QLabel#header {
            font-size: 18px;
            font-weight: 700;
            color: #f3f7fb;
        }
        QLabel#subtitle {
            font-size: 12px;
            color: #c7d3db;
        }
        QLabel#footer {
            font-size: 10px;
            color: #98a7b3;
        }

        /* Inputs: white background with dark text for high contrast like Figma modal */
        QLineEdit#input {
            background: #ffffff;
            border: 1px solid #2f3640;
            border-radius: 8px;
            padding-left: 12px;
            padding-right: 12px;
            font-size: 13px;
            color: #101218; /* actual text color */
        }
        QLineEdit#input:focus {
            border: 1px solid #2d8cff;
            background: #ffffff;
        }
        /* Placeholder more visible */
        QLineEdit#input::placeholder {
            color: #6b7782;
        }

        QPushButton#primary {
            background: #0ea5ff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 14px;
            font-weight: 600;
        }
        QPushButton#primary:hover {
            background: #0893df;
        }
        QPushButton#secondary {
            background: #ffffff;
            color: #222831;
            border: 1px solid #2b2d31;
            border-radius: 8px;
            padding: 8px 14px;
        }
        QPushButton#iconbtn {
            background: transparent;
            border: none;
            font-size: 16px;
            color: #dbeefd;
        }
        QCheckBox#remember { color: #cfe8ff; }
        QLabel#link { color: #8ecbff; font-size: 12px; }
        QLabel#link:hover { text-decoration: underline; }
        QPushButton:focus { outline: none; }
        """)

    def _apply_animations(self):
        # Fade-in
        self.setWindowOpacity(0.0)
        anim = QPropertyAnimation(self, b"windowOpacity")
        anim.setDuration(260)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        anim.start()
        self._fade_anim = anim

    def _toggle_show_password(self):
        if self.password_input.echoMode() == QLineEdit.EchoMode.Password:
            self.password_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.toggle_pw_btn.setText("🙈")
        else:
            self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.toggle_pw_btn.setText("👁")

    def _forgot_pw(self):
        QMessageBox.information(self, "Quên mật khẩu", "Chức năng quên mật khẩu chưa được triển khai.")

    def _handle_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên đăng nhập và mật khẩu.")
            return

        ok = VALID_CREDENTIALS.get(username) == password
        if not ok:
            QMessageBox.critical(self, "Đăng nhập thất bại", "Tên đăng nhập hoặc mật khẩu không đúng.")
            self.password_input.clear()
            self.password_input.setFocus()
            return

        # Quyết định mở GUI nào: admin -> admin_gui, khác -> login_user_gui
        if username.lower() == "admin":
            target_module = "admin_gui"
            target_class = "AdminWindow"
        else:
            target_module = "login_user_gui"
            target_class = "MainWindow"

        try:
            module = __import__(target_module)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Lỗi import", f"Không thể import {target_module}.py:\n{e}")
            module = None

        if module and hasattr(module, target_class):
            try:
                WindowClass = getattr(module, target_class)
                main_win = WindowClass()
                # truyền tham chiếu login để main có thể gọi show() khi logout
                try:
                    main_win._login_window_ref = self
                except Exception:
                    pass
                main_win.setWindowTitle("Admin" if username.lower() == "admin" else "Trang người dùng")
                main_win.show()
                self._main_window_ref = main_win
                # ẩn login
                self.hide()
                return
            except Exception as e:
                traceback.print_exc()
                QMessageBox.warning(self, "Lỗi mở GUI", f"Lỗi khi mở {target_class}:\n{e}")

        # Fallback: cửa sổ đơn giản
        try:
            main = QMainWindow()
            main.setWindowTitle("Ứng dụng chính (dự phòng)")
            main.resize(640, 420)
            lbl = QLabel(f"Bạn đã đăng nhập với: {username}", alignment=Qt.AlignmentFlag.AlignCenter)
            font = QFont()
            font.setPointSize(14)
            lbl.setFont(font)
            main.setCentralWidget(lbl)
            main.show()
            self._main_window_ref = main
            self.hide()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Lỗi", f"Không thể mở cửa sổ chính: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = ModernLoginWindow()
    win.show()
    sys.exit(app.exec())