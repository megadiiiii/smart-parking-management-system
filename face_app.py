import cv2
import os
import time
import numpy as np
from insightface.app import FaceAnalysis

# ================== CONFIG ==================
DB_DIR = "face_db"
os.makedirs(DB_DIR, exist_ok=True)

DETECT_EVERY_N_FRAMES = 5
SIM_THRESHOLD = 0.5
CAMERA_INDEX = 0
# ============================================


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class FaceSystem:
    def __init__(self):
        print("[INIT] Loading InsightFace model...")
        self.app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0)

        self.ids = []
        self.embs = []
        self.load_db()

        self.last_face = None   # (bbox, emb)
        self.last_result = None # (bbox, label, color)

        print(f"[DB] Loaded {len(self.ids)} identities")

    def load_db(self):
        self.ids.clear()
        self.embs.clear()
        for f in os.listdir(DB_DIR):
            if f.endswith(".npy"):
                self.ids.append(f.replace(".npy", ""))
                self.embs.append(np.load(os.path.join(DB_DIR, f)))

    def detect_and_recognize(self, frame):
        faces = self.app.get(frame)
        if not faces:
            self.last_face = None
            self.last_result = None
            return None

        # lấy mặt to nhất
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        x1, y1, x2, y2 = face.bbox.astype(int)
        emb = face.normed_embedding
        self.last_face = ((x1, y1, x2, y2), emb)

        best_id = None
        best_score = 0.0

        for sid, semb in zip(self.ids, self.embs):
            score = cosine_similarity(emb, semb)
            if score > best_score:
                best_score = score
                best_id = sid

        if best_score >= SIM_THRESHOLD:
            label = f"ID {best_id}"
            color = (0, 255, 0)
        else:
            label = "UNKNOWN"
            color = (0, 0, 255)

        self.last_result = ((x1, y1, x2, y2), label, color)
        return self.last_result

    def register_last_face(self):
        if self.last_face is None:
            return None

        _, emb = self.last_face
        new_id = str(int(time.time()))[-8:]
        np.save(os.path.join(DB_DIR, f"{new_id}.npy"), emb)

        self.ids.append(new_id)
        self.embs.append(emb)

        print(f"[REGISTER] New face ID = {new_id}")
        return new_id


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    face_sys = FaceSystem()
    frame_count = 0

    print("\n=== FACE SYSTEM STARTED ===")
    print("[R] Register UNKNOWN face")
    print("[Q] Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        frame_count += 1

        # detect mỗi N frame
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            face_sys.detect_and_recognize(frame)

        # vẽ cache
        if face_sys.last_result:
            (x1, y1, x2, y2), label, color = face_sys.last_result
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        cv2.imshow("Face Recognition", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):
            if face_sys.last_result and face_sys.last_result[1] == "UNKNOWN":
                new_id = face_sys.register_last_face()
                if new_id:
                    face_sys.last_result = (
                        face_sys.last_result[0],
                        f"ID {new_id}",
                        (0, 255, 0)
                    )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
