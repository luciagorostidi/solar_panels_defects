import os
import time
import psutil
import cv2
from ultralytics import YOLO

VIDEO_ENTRADA = 'video_dron.mp4'
VIDEO_SALIDA = 'output_dron.mp4'
WEIGHTS_PATH = 'weights/best.onnx'
VID_STRIDE = 3  # procesa 1 de cada 3 frames para ir mas rapido

# 1. Tamaño del modelo
model_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024) if os.path.exists(WEIGHTS_PATH) else 0.0

# Cargar VUESTRO cerebro entrenado
model = YOLO(WEIGHTS_PATH)

# FPS original del video, para que el output.mp4 se reproduzca a velocidad real
cap_info = cv2.VideoCapture(VIDEO_ENTRADA)
fps_original = cap_info.get(cv2.CAP_PROP_FPS) or 30
cap_info.release()
fps_salida = fps_original / VID_STRIDE

writer = None
process = psutil.Process(os.getpid())
latencies_ms = []
ram_usages_mb = []

# stream=True + source=ruta de video permite que Ultralytics aplique vid_stride
for results in model.predict(
    source=VIDEO_ENTRADA,
    vid_stride=VID_STRIDE,
    imgsz=640,
    conf=0.5,
    stream=True,
    verbose=False,
):
    # Medición de latencia en la inferencia interna de Ultralytics
    # results.speed contiene los ms de preprocesado, inferencia pura y postprocesado
    inference_speed = results.speed.get('inference', 0.0)
    latencies_ms.append(inference_speed)
    ram_usages_mb.append(process.memory_info().rss / (1024 * 1024))

    annotated_frame = results.plot()

    if writer is None:
        alto, ancho = annotated_frame.shape[:2]
        writer = cv2.VideoWriter(
            VIDEO_SALIDA,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps_salida,
            (ancho, alto),
        )

    writer.write(annotated_frame)

    # Mostrar ventana interactiva a tamaño adecuado (960x540)
    cv2.imshow('Detector de defectos', cv2.resize(annotated_frame, (960, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if writer is not None:
    writer.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------
# Cálculos e impresión de métricas finales
# -------------------------------------------------------------
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
    print("="*40 + "\n")