# Fichero: detectar_video.py
import time
import cv2
from pymavlink import mavutil
from ultralytics import YOLO

VIDEO_ENTRADA = 'video_dron.mp4'
VIDEO_SALIDA = 'output_dron.mp4'
VID_STRIDE = 3  # procesa 1 de cada 3 frames para ir mas rapido

NOMBRE_CLASE_SOILING = 'soiling'  # nombre de la clase a vigilar
SEGUNDOS_ENTRE_ALERTAS = 3  # tiempo minimo entre alertas para no inundar

# Conexion MAVLink por UDP: mandamos la alerta a localhost:14551
mavlink_conn = mavutil.mavlink_connection('udpout:127.0.0.1:14551')

# Guardamos el instante de la ultima alerta enviada (None = ninguna todavia)
ultima_alerta = None


def enviar_alerta_mavlink():
    """Envia un mensaje STATUSTEXT con la alerta de soiling, respetando el cooldown."""
    global ultima_alerta

    ahora = time.time()
    if ultima_alerta is not None and (ahora - ultima_alerta) < SEGUNDOS_ENTRE_ALERTAS:
        return  # todavia dentro del cooldown, no se envia nada

    mavlink_conn.mav.statustext_send(
        mavutil.mavlink.MAV_SEVERITY_WARNING,
        b"Solining detectado",
    )
    ultima_alerta = ahora


# Cargar VUESTRO cerebro entrenado
model = YOLO('weights/best.pt')

# FPS original del video, para que el output.mp4 se reproduzca a velocidad real
# (al saltar frames con vid_stride, hay que reducir el fps del video de salida)
cap_info = cv2.VideoCapture(VIDEO_ENTRADA)
fps_original = cap_info.get(cv2.CAP_PROP_FPS) or 30
cap_info.release()
fps_salida = fps_original / VID_STRIDE

writer = None

# stream=True + source=ruta de video permite que Ultralytics aplique vid_stride
for results in model.predict(
    source=VIDEO_ENTRADA,
    vid_stride=VID_STRIDE,
    imgsz=640,
    conf=0.5,
    stream=True,
    verbose=False,
):
    annotated_frame = results.plot()

    # Comprobar si en este frame se ha detectado soiling
    nombres_detectados = [results.names[int(caja.cls)] for caja in results.boxes]
    if NOMBRE_CLASE_SOILING in nombres_detectados:
        enviar_alerta_mavlink()

    if writer is None:
        alto, ancho = annotated_frame.shape[:2]
        writer = cv2.VideoWriter(
            VIDEO_SALIDA,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps_salida,
            (ancho, alto),
        )

    writer.write(annotated_frame)

    # Mostrar ventana interactiva
    cv2.imshow('Detector de defectos', cv2.resize(annotated_frame, (960, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if writer is not None:
    writer.release()
cv2.destroyAllWindows()