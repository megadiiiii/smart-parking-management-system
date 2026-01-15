import cv2
import numpy as np
import onnxruntime as ort


def preprocess_ocr(img, size=160):
    """
    Resize + normalize ảnh biển số cho OCR ONNX.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (size, size))

    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = img_norm[np.newaxis, np.newaxis, :, :]  # (1,1,H,W)

    return img_norm


def postprocess_ocr(output, charset="0123456789ABCDEFGHJKLPQRSTUVWXYZ"):
    """
    Chuyển từ output model → chuỗi ký tự.
    """
    preds = np.argmax(output, axis=2)[0]  # (seq_len)
    plate = ""

    last_char = -1
    for p in preds:
        if p != last_char and p < len(charset):
            plate += charset[p]
        last_char = p

    return plate if plate != "" else "unknown"


def read_plate_onnx(sess, img):
    """
    Hàm OCR ONNX hoàn chỉnh.
    """
    blob = preprocess_ocr(img)
    input_name = sess.get_inputs()[0].name

    output = sess.run(None, {input_name: blob})[0]  # (1, seq, num_classes)

    return postprocess_ocr(output)
