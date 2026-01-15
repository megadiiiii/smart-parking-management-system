import warnings
warnings.filterwarnings("ignore")

import function.utils_rotate as utils_rotate
import function.helper as helper


import cv2
import torch
import time
import os

# ================== LOAD MODEL ==================
yolo_LP_detect = torch.hub.load(
    'yolov5',
    'custom',
    path='model/LP_detector_nano_61.pt',
    source='local'
)

yolo_LP_ocr = torch.hub.load(
    'yolov5',
    'custom',
    path='model/LP_ocr_nano_62.pt',
    source='local'
)

yolo_LP_ocr.conf = 0.6

# ================== CAMERA ==================
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

SAVE_INTERVAL = 2
last_saved_time = 0

os.makedirs("plates", exist_ok=True)

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]

    # ===== Detect =====
    results = yolo_LP_detect(frame, size=640)
    detections = results.xyxy[0]

    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()

        # ✅ YOLOv5 xyxy đã theo ảnh gốc -> chỉ cần int + clamp
        x1 = int(x1);
        y1 = int(y1);
        x2 = int(x2);
        y2 = int(y2)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        # ===== OCR =====
        plate_text = "unknown"
        for cc in range(2):
            for ct in range(2):
                plate_text = helper.read_plate(
                    yolo_LP_ocr,
                    utils_rotate.deskew(crop, cc, ct)
                )
                if plate_text != "unknown":
                    break
            if plate_text != "unknown":
                break

        # ===== DRAW =====
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # ===== DRAW =====
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            plate_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # ===== SAVE =====
        if plate_text != "unknown":
            now = time.time()
            if now - last_saved_time >= SAVE_INTERVAL:
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"plates/{plate_text}_{ts}.jpg".replace(" ", "_")
                cv2.imwrite(filename, crop)
                print(f"[OK] {plate_text} -> {filename}")
                last_saved_time = now

    cv2.imshow("License Plate Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================== CLEAN ==================
cap.release()
cv2.destroyAllWindows()
