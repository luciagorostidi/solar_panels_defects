import time
from ultralytics import YOLO

inicio = time.time()

# Cargar la arquitectura ligera nano de YOLOv8
model = YOLO('yolov8n.pt')

# Ejecutar entrenamiento local
results = model.train(
    data='data.yaml',
    epochs=40,                 # 40 pasadas completas
    imgsz=640,
    batch=8,
    name='mi_modelo_frutas'
)

fin = time.time()
minutos = (fin - inicio) / 60
print(f'¡Entrenamiento completado en {minutos:.1f} minutos!')
print('El archivo resultante se ha guardado en:')
print('runs/detect/mi_modelo_frutas/weights/best.pt')
