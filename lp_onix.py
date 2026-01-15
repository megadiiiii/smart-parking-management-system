import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import time
import onnxruntime as ort
import function.utils_rotate as utils_rotate
import function.helper_onix as helper  # mày có viết helper_onix chưa?

# ========= LOAD ONNX ========= #
det_sess = ort.InferenceSession("model/LP_detector_nano_61.onnx", providers=["CPUExecutionProvider"])
ocr_sess = ort.InferenceSession("model/LP_ocr_nano_62.onnx", providers=["CPUExecutionProvider"])

input_name_det = det_sess.get_inputs()[0].name
input_name_ocr = ocr_sess.get_inputs()[0].name

# ========= NMS ========= #
def non_max_suppression(boxes, confs, iou_thres=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    confs = np.array(confs)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep

# ========= YOLO ONNX DETECTION ========= #
def yolo_onnx_detect(sess, img, input_name, conf_thres=0.5):
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (640, 640))
    blob = img_resized[:, :, ::-1].transpose(2,0,1)
    blob = np.ascontiguousarray(blob/255.0, dtype=np.float32)[None]

    outputs = sess.run(None, {input_name: blob})[0]  # (1, 25200, 6)
    outputs = outputs[0]  # (25200, 6)

    boxes = []
    confs = []
    for det in outputs:
        x1, y1, x2, y2, conf, cls = det
        if conf < conf_thres:
            continue
        # scale bbox về size gốc
        x1 = max(0, min(int(x1/640*w0), w0-1))
        y1 = max(0, min(int(y1/640*h0), h0-1))
        x2 = max(0, min(int(x2/640*w0), w0-1))
        y2 = max(0, min(int(y2/640*h0), h0-1))
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        boxes.append([x1, y1, x2, y2])
        confs.append(conf)

    keep = non_max_suppression(boxes, confs)
    return [boxes[i] for i in keep]

# ========= MAIN LOOP ========= #
vid = cv2.VideoCapture(2)
captured = False

while True:
    ret, frame = vid.read()
    if not ret:
        break

    cv2.imshow("Live Cam", frame)

    plates = yolo_onnx_detect(det_sess, frame, input_name_det, conf_thres=0.3)


    for (x1, y1, x2, y2) in plates:
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size == 0:
            continue

        lp = "unknown"
        for cc in range(2):
            for ct in range(2):
                plate_img = utils_rotate.deskew(crop_img, cc, ct)
                if plate_img is None or plate_img.size == 0:
                    continue
                lp = helper.read_plate_onnx(ocr_sess, plate_img)
                if lp != "unknown":
                    filename = f"{lp}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, crop_img)
                    cv2.imshow("Captured Plate", crop_img)
                    print("BIỂN SỐ:", lp)
                    captured = True
                    break
            if captured:
                break
        if captured:
            break

    if captured:
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
