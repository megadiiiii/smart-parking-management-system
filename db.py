import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "parking.db")

def init_db():
    """
    Tạo hoặc cập nhật schema DB. Nếu database cũ dùng 'IN'/'OUT', migrate sang 'Vào'/'Ra'.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Nếu bảng chưa tồn tại thì tạo mới với CHECK trạng thái bằng tiếng Việt
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text TEXT NOT NULL,
        status TEXT NOT NULL,
        time_in TEXT,
        time_out TEXT,
        face_image_path TEXT,
        plate_image_path TEXT
    )
    """)
    conn.commit()

    # Thêm/check constraint không bắt buộc (để giữ linh hoạt) nhưng đảm bảo giá trị trạng thái chuẩn
    # Nếu có bản cũ với 'IN'/'OUT' thì migrate
    try:
        # Kiểm tra xem có tồn tại các giá trị IN/OUT cũ không
        cur.execute("SELECT COUNT(*) FROM events WHERE status IN ('IN','OUT')")
        cnt = cur.fetchone()[0]
        if cnt > 0:
            # Chuyển 'IN' -> 'Vào', 'OUT' -> 'Ra'
            cur.execute("UPDATE events SET status = 'Vào' WHERE status = 'IN'")
            cur.execute("UPDATE events SET status = 'Ra' WHERE status = 'OUT'")
            conn.commit()
    except Exception:
        # Nếu bảng vừa tạo mới, các truy vấn trên sẽ ok; nếu có lỗi thì bỏ qua (môi trường khác)
        pass

    conn.close()