import os
import time
import psutil
import cv2
import numpy as np
from pathlib import Path
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType
)

BASE_DIR = Path('/home/lucia/solar_panels_defects')
VIDEO_ENTRADA = str(BASE_DIR / 'video_dron.mp4')
VIDEO_SALIDA = str(BASE_DIR / 'output_dron.mp4')
WEIGHTS_PATH = str(BASE_DIR / 'new_models' / 'paneles_yolo11n.hef')
LABELS_PATH = str(BASE_DIR / 'new_models' / 'paneles_yolo11n_labels.txt')
VID_STRIDE = 3
CONF_THRESH = 0.4
IOU_THRESH = 0.45

# Tamaño del modelo
model_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024) if os.path.exists(WEIGHTS_PATH) else 0.0

# Cargar etiquetas
labels = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

cap_info = cv2.VideoCapture(VIDEO_ENTRADA)
if not cap_info.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el vídeo: {VIDEO_ENTRADA}")

fps_original = cap_info.get(cv2.CAP_PROP_FPS) or 30
ancho_orig = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
alto_orig = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_info.release()

fps_salida = fps_original / VID_STRIDE

writer = None
process = psutil.Process(os.getpid())
latencies_ms = []
ram_usages_mb = []

# Matriz para decodificación DFL (16 bins: 0..15)
DFL_WEIGHTS = np.arange(16, dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def decode_yolo_dfl_layer(box_layer, cls_layer, stride, orig_w, orig_h, conf_thresh):
    """
    Decodifica una escala espacial concreta (stride 8, 16 o 32)
    box_layer: (H, W, 64)
    cls_layer: (H, W, 3)
    """
    H, W, _ = cls_layer.shape
    scores_raw = sigmoid(cls_layer)  # Probabilidades de clase (H, W, num_classes)
    
    # Máxima probabilidad por celda
    max_scores = np.max(scores_raw, axis=-1)
    mask = max_scores >= conf_thresh
    
    if not np.any(mask):
        return [], [], []

    valid_indices = np.argwhere(mask)
    boxes = []
    scores = []
    class_ids = []

    # Proyección de DFL: reshape a (4, 16) y cálculo de valor esperado
    dfl_tensor = box_layer[mask].reshape(-1, 4, 16)
    dfl_probs = softmax(dfl_tensor, axis=-1)
    dist = np.sum(dfl_probs * DFL_WEIGHTS, axis=-1) * stride  # (N, 4) -> [left, top, right, bottom]

    for idx, (r, c) in enumerate(valid_indices):
        score = float(max_scores[r, c])
        cid = int(np.argmax(scores_raw[r, c]))

        # Centro de la celda de ancla en escala 640x640
        cx = (c + 0.5) * stride
        cy = (r + 0.5) * stride

        left, top, right, bottom = dist[idx]
        x1 = cx - left
        y1 = cy - top
        x2 = cx + right
        y2 = cy + bottom

        # Escalar a dimensiones originales del vídeo
        x1 = int(np.clip(x1 * (orig_w / 640.0), 0, orig_w))
        y1 = int(np.clip(y1 * (orig_h / 640.0), 0, orig_h))
        x2 = int(np.clip(x2 * (orig_w / 640.0), 0, orig_w))
        y2 = int(np.clip(y2 * (orig_h / 640.0), 0, orig_h))

        bw = x2 - x1
        bh = y2 - y1

        if bw > 2 and bh > 2:
            boxes.append([x1, y1, bw, bh])
            scores.append(score)
            class_ids.append(cid)

    return boxes, scores, class_ids


def decode_hailo_yolo11(raw_outputs, orig_w, orig_h, conf_thresh=0.4, iou_thresh=0.45):
    # Separar cabezales por dimensión
    heads = {}
    for name, tensor in raw_outputs.items():
        arr = tensor[0] if tensor.ndim == 4 else tensor
        H, W, C = arr.shape
        stride = 640 // H
        if stride not in heads:
            heads[stride] = {}
        if C == 64:
            heads[stride]['box'] = arr
        elif C == 3:
            heads[stride]['cls'] = arr

    all_boxes = []
    all_scores = []
    all_classes = []

    for stride, head in heads.items():
        if 'box' in head and 'cls' in head:
            b, s, c = decode_yolo_dfl_layer(head['box'], head['cls'], stride, orig_w, orig_h, conf_thresh)
            all_boxes.extend(b)
            all_scores.extend(s)
            all_classes.extend(c)

    results = []
    if all_boxes:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf_thresh, iou_thresh)
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                bx, by, bw, bh = all_boxes[i]
                results.append((bx, by, bx + bw, by + bh, all_scores[i], all_classes[i]))

    return results


print(f"[*] Cargando modelo HEF: {WEIGHTS_PATH}")
hef = HEF(WEIGHTS_PATH)
params = VDevice.create_params()
cap = cv2.VideoCapture(VIDEO_ENTRADA)
frame_count = 0

with VDevice(params) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]

    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    input_vstream_info = network_group.get_input_vstream_infos()[0]
    input_name = input_vstream_info.name

    with network_group.activate():
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % VID_STRIDE != 0:
                    continue

                input_frame = cv2.resize(frame, (640, 640))
                input_frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                input_tensor = np.expand_dims(input_frame_rgb, axis=0)

                t_start = time.perf_counter()
                raw_outputs = infer_pipeline.infer({input_name: input_tensor})
                t_end = time.perf_counter()

                inference_speed_ms = (t_end - t_start) * 1000.0
                latencies_ms.append(inference_speed_ms)
                ram_usages_mb.append(process.memory_info().rss / (1024 * 1024))

                annotated_frame = frame.copy()
                detections = decode_hailo_yolo11(raw_outputs, ancho_orig, alto_orig, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH)

                for (x1, y1, x2, y2, score, class_id) in detections:
                    class_name = labels[class_id] if class_id < len(labels) else f"defecto_{class_id}"
                    label_text = f"{class_name} {score:.2f}"

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    y_text = max(y1 - 6, th + 4)
                    cv2.rectangle(annotated_frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label_text, (x1 + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

                if writer is None:
                    alto, ancho = annotated_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        VIDEO_SALIDA,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps_salida,
                        (ancho, alto),
                    )

                writer.write(annotated_frame)
                cv2.imshow('Detector de defectos', cv2.resize(annotated_frame, (960, 540)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

if latencies_ms:
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    fps_infer = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    avg_ram_mb = sum(ram_usages_mb) / len(ram_usages_mb)

    print("\n" + "="*40)
    print("        MÉTRICAS DE RENDIMIENTO")
    print("="*40)
    print(f"Tamaño del modelo:  {model_size_mb:.2f} MB")
    print(f"Latencia:           {avg_latency_ms:.2f} ms/frame")
    print(f"FPS:                {fps_infer:.2f}")
    print(f"Uso de RAM:         {avg_ram_mb:.2f} MB")
