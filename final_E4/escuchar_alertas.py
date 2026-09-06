# Fichero: escuchar_alertas.py
# Script sencillo para comprobar que las alertas de detectar_video_alerta.py
# se estan enviando correctamente. Ejecutalo en otra consola, en paralelo
# a detectar_video_alerta.py, y ve mostrando los mensajes que recibe.
from pymavlink import mavutil

conn = mavutil.mavlink_connection('udpin:127.0.0.1:14551')
print("Escuchando alertas en 127.0.0.1:14551... (Ctrl+C para salir)")

while True:
    msg = conn.recv_match(type='STATUSTEXT', blocking=True)
    print("ALERTA RECIBIDA:", msg.text)
