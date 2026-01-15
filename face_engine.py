# face_engine.py (CPU-optimized)
import os
import time
import numpy as np
import cv2
from insightface.app import FaceAnalysis

DB_DIR = "face_db"
SIM_THRESHOLD = 0.5


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class FaceEngine:
    """
    CPU-only friendly:
    - resize trước khi detect để nhẹ CPU
    - chỉ detect mỗi min_interval seconds
    - cache kết quả trong cache_ttl seconds
    """

    def __init__(
        self,
        model_name="buffalo_s",
        det_scale=0.5,         # 0.5 = giảm kích thước frame 1/2
        min_interval=0.25,     # detect tối đa 4 lần/giây
        cache_ttl=1.5          # dùng lại bbox/label trong 1.5s
    ):
        self.det_scale = float(det_scale)
        self.min_interval = float(min_interval)
        self.cache_ttl = float(cache_ttl)

        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0)

        self.ids = []
        self.embs = []
        self.load_db()

        # cache
        self._last_run_time = 0.0
        self._cache_time = 0.0
        self._cache_result = None  # (bbox, label, color)

    def load_db(self):
        self.ids.clear()
        self.embs.clear()
        if not os.path.isdir(DB_DIR):
            os.makedirs(DB_DIR, exist_ok=True)
            return

        for f in os.listdir(DB_DIR):
            if f.endswith(".npy"):
                self.ids.append(f.replace(".npy", ""))
                self.embs.append(np.load(os.path.join(DB_DIR, f)))

    def recognize(self, frame_bgr):
        """
        Return:
          None
          OR ((x, y, w, h), label, (b,g,r))
        """
        now = time.time()

        # dùng cache nếu còn hạn
        if self._cache_result is not None and (now - self._cache_time) < self.cache_ttl:
            return self._cache_result

        # hạn chế tần suất chạy model
        if (now - self._last_run_time) < self.min_interval:
            return self._cache_result  # có thể None

        self._last_run_time = now

        if frame_bgr is None:
            return None

        h, w = frame_bgr.shape[:2]
        scale = self.det_scale
        if scale <= 0 or scale > 1.0:
            scale = 0.5

        # resize để nhẹ CPU
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        faces = self.app.get(small)
        if not faces:
            self._cache_result = None
            self._cache_time = now
            return None

        # lấy mặt to nhất
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        emb = face.normed_embedding

        # bbox trên ảnh small -> scale lại về ảnh gốc
        x1, y1, x2, y2 = face.bbox.astype(np.float32)
        inv = 1.0 / scale
        x1 = int(x1 * inv)
        y1 = int(y1 * inv)
        x2 = int(x2 * inv)
        y2 = int(y2 * inv)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        best_id, best_score = None, 0.0
        for sid, semb in zip(self.ids, self.embs):
            s = cosine(emb, semb)
            if s > best_score:
                best_score, best_id = s, sid

        if best_score >= SIM_THRESHOLD and best_id is not None:
            result = ((x1, y1, x2 - x1, y2 - y1), f"ID {best_id}", (0, 255, 0))
        else:
            result = ((x1, y1, x2 - x1, y2 - y1), "UNKNOWN", (0, 0, 255))

        self._cache_result = result
        self._cache_time = now
        return result
