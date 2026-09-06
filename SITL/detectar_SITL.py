#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 prueba_stream.py — Detección YOLO + grabación local + streaming WebRTC
 Controlado por Altitud de Vuelo (> 2m) vía MAVLink hacia Mission Planner (Inbound)
 Emisión de Alertas: Únicamente Datagrama UDP JSON (para Netcat)
==============================================================================
"""

import argparse
import asyncio
import json
import os
import queue
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction

import aiohttp
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import VideoFrame
from ultralytics import YOLO

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None


# ==============================================================================
# CONFIGURACIÓN GENERAL POR DEFECTO
# ==============================================================================
WHIP_URL = "https://customer-qf7p7diwgx16wzce.cloudflarestream.com/f4e0303fe5f16853cb0c3f43735c325akd5e51cc142efbac96a214f460a4daacf/webRTC/publish"
VIEW_URL = "https://customer-qf7p7diwgx16wzce.cloudflarestream.com/d5e51cc142efbac96a214f460a4daacf/iframe"

VIDEO_IN_DEFAULT = "video_dron.mp4"
VIDEO_OUT_DEFAULT = "output_dron.mp4"

WEIGHTS_PATH = "exported_models/solar_panels_yolo11n_int8.onnx"
CONF_THRESHOLD = 0.5
VID_STRIDE = 2

STREAM_WIDTH = 640
STREAM_HEIGHT = 360
STREAM_FPS = 15

# Red y MAVLink
MP_IP_DEFAULT = "192.168.50.232"       # IP de tu PC con Mission Planner
MP_PORT_DEFAULT = 5762                # Puerto configurado en Mission Planner
ALTURA_ACTIVACION_DEFAULT = 2.0        # Metros relativos para activar captura/inferencia
UDP_ALERT_PORT_DEFAULT = 9000          # Puerto donde netcat escucha en tu PC
ANTI_REPETICION_DEFAULT = 5.0          # Segundos de margen entre alertas repetidas


# ==============================================================================
# AntiRepeticion — Filtro de frecuencia para no saturar con avisos
# ==============================================================================
class AntiRepeticion:
    def __init__(self, ventana_segundos):
        self.ventana = ventana_segundos
        self._ultimo_envio = {}

    def permitido(self, clase):
        ahora = time.monotonic()
        anterior = self._ultimo_envio.get(clase)
        if anterior is not None and (ahora - anterior) < self.ventana:
            return False
        self._ultimo_envio[clase] = ahora
        return True


# ==============================================================================
# EnlaceMavlink — Hilo de gestión MAVLink + Telemetría GPS + Emisión UDP
# ==============================================================================
class EnlaceMavlink(threading.Thread):
    def __init__(self, *, target_ip, target_port, sysid, compid,
                 altura_umbral, udp_alert_port, mavlink_proto="udp"):
        super().__init__(daemon=True, name="EnlaceMavlink")
        self._target_ip = target_ip
        self._target_port = target_port
        self._sysid = sysid
        self._compid = compid
        self._altura_umbral = altura_umbral
        self._udp_alert_port = udp_alert_port
        self._mavlink_proto = mavlink_proto

        self._telemetria = {
            "lat": 0.0,
            "lon": 0.0,
            "alt_rel_m": 0.0,
            "alt_msl_m": 0.0,
            "heading": 0.0
        }
        self._video_on = False
        self._lock = threading.Lock()

        self._cola_alertas = queue.Queue()
        self._parar = threading.Event()
        self._listo = threading.Event()
        self.master = None

        # Socket UDP dedicado para emitir los JSON a Netcat
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @property
    def video_on(self):
        with self._lock:
            return self._video_on

    def get_telemetria_actual(self):
        with self._lock:
            return dict(self._telemetria)

    def encolar_alerta(self, clase_defecto):
        with self._lock:
            telem = dict(self._telemetria)
        self._cola_alertas.put({"clase": clase_defecto, "telem": telem})

    def esperar_listo(self, timeout=20):
        return self._listo.wait(timeout)

    def detener(self):
        self._parar.set()
        try:
            self._udp_sock.close()
        except Exception:
            pass

    def _abrir_conexion(self):
        # udpout: conecta hacia el mirror UDP "Inbound" de Mission Planner.
        # tcp: conecta directamente como cliente a un puerto MAVLink TCP de
        # la SITL de ArduPilot (5762/5763 son los puertos secundarios que
        # deja libres además del 5760, que suele usar Mission Planner).
        prefijo = "tcp" if self._mavlink_proto == "tcp" else "udpout"
        endpoint = f"{prefijo}:{self._target_ip}:{self._target_port}"
        print(f"[MAV] Iniciando conexión {self._mavlink_proto.upper()} hacia {endpoint}...")
        self.master = mavutil.mavlink_connection(
            endpoint,
            source_system=self._sysid,
            source_component=self._compid
        )

        # Enviar primer Heartbeat para registrar la Raspberry en Mission Planner
        self._enviar_heartbeat()

        # Solicitar stream de posición (GLOBAL_POSITION_INT / ALTITUDE)
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system or 1,
                self.master.target_component or 1,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION, 4, 1
            )
        except Exception as e:
            print(f"[MAV] Nota al solicitar telemetría: {e}")

        self._listo.set()

    def _enviar_heartbeat(self):
        """Emite un Heartbeat MAVLink como Onboard Controller (Componente auxiliar)."""
        try:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, mavutil.mavlink.MAV_STATE_ACTIVE
            )
        except Exception as e:
            print(f"[MAV] Error enviando Heartbeat: {e}")

    def _enviar_alertas(self, item_alerta):
        clase = item_alerta["clase"]
        telem = item_alerta["telem"]
        timestamp_iso = datetime.now(timezone.utc).isoformat()

        # Envío de datagrama UDP JSON para Netcat (única vía de alerta; ver README)
        payload = {
            "timestamp": timestamp_iso,
            "evento": "DEFECTO_DETECTADO",
            "clase": clase,
            "posicion": {
                "latitud": telem["lat"],
                "longitud": telem["lon"],
                "altitud_relativa_m": round(telem["alt_rel_m"], 2),
                "altitud_msl_m": round(telem["alt_msl_m"], 2),
                "heading_deg": round(telem["heading"], 1)
            }
        }
        try:
            mensaje_json = (json.dumps(payload) + "\n").encode("utf-8")
            self._udp_sock.sendto(mensaje_json, (self._target_ip, self._udp_alert_port))
            print(f"[UDP-JSON] -> Enviado a {self._target_ip}:{self._udp_alert_port} -> {payload}")
        except Exception as e:
            print(f"[UDP-JSON] Error enviando paquete UDP: {e}")

    def run(self):
        try:
            self._abrir_conexion()
        except Exception as e:
            print(f"[MAV] ❌ Error abriendo socket UDP: {e}")
            self._listo.set()
            return

        margen_histeresis = 0.2
        umbral_subida = self._altura_umbral
        umbral_bajada = max(0.5, self._altura_umbral - margen_histeresis)
        ultimo_heartbeat = 0

        while not self._parar.is_set():
            ahora = time.time()

            # Enviar Heartbeat cada 1.0 segundo para mantener abierto el túnel UDP
            if ahora - ultimo_heartbeat >= 1.0:
                self._enviar_heartbeat()
                ultimo_heartbeat = ahora

            # 1. Vaciar cola de alertas
            try:
                while True:
                    self._enviar_alertas(self._cola_alertas.get_nowait())
            except queue.Empty:
                pass

            # 2. Procesar telemetría de posición entrante
            while True:
                try:
                    msg = self.master.recv_match(
                        type=["GLOBAL_POSITION_INT", "ALTITUDE"], blocking=False)
                except (ConnectionResetError, OSError) as e:
                    # En Windows, un socket UDP "conectado" puede lanzar
                    # WinError 10054 (ConnectionResetError) si llega un ICMP
                    # "puerto inalcanzable" de un envío anterior; no es fatal
                    # para MAVLink sobre UDP, así que no se mata el hilo.
                    print(f"[MAV] Aviso: error de red leyendo telemetría ({e}); reintentando...")
                    break
                if msg is None:
                    break

                tipo = msg.get_type()
                with self._lock:
                    if tipo == "GLOBAL_POSITION_INT":
                        self._telemetria["lat"] = msg.lat / 1e7
                        self._telemetria["lon"] = msg.lon / 1e7
                        self._telemetria["alt_rel_m"] = msg.relative_alt / 1000.0
                        self._telemetria["alt_msl_m"] = msg.alt / 1000.0
                        self._telemetria["heading"] = msg.hdg / 100.0
                    elif tipo == "ALTITUDE":
                        self._telemetria["alt_rel_m"] = msg.altitude_relative
                        self._telemetria["alt_msl_m"] = msg.altitude_amsl

                    alt_actual = self._telemetria["alt_rel_m"]
                    if not self._video_on and alt_actual >= umbral_subida:
                        self._video_on = True
                        print(f"[MAV-ALT] Altura: {alt_actual:.2f}m >= {umbral_subida}m -> VÍDEO ON")
                    elif self._video_on and alt_actual <= umbral_bajada:
                        self._video_on = False
                        print(f"[MAV-ALT] Altura: {alt_actual:.2f}m <= {umbral_bajada}m -> VÍDEO OFF (PAUSA)")

            time.sleep(0.05)

        print("[MAV] Hilo MAVLink detenido.")


# ==============================================================================
# Utilidades de Video y Canvas
# ==============================================================================
def nombre_con_fecha(ruta_base):
    raiz, ext = os.path.splitext(ruta_base)
    marca = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{raiz}_{marca}{ext or '.mp4'}"


def _lienzo_pausa(ancho, alto, altura_actual=0.0):
    img = np.zeros((alto, ancho, 3), dtype=np.uint8)
    cv2.putText(img, "PAUSA (EN SUELO)", (int(ancho * 0.18), int(alto * 0.48)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 2, cv2.LINE_AA)
    texto_sub = f"Esperando altura > 2.0m (Actual: {altura_actual:.1f}m)"
    cv2.putText(img, texto_sub, (int(ancho * 0.12), int(alto * 0.60)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return img


# ==============================================================================
# PISTA WEBRTC (YOLO + STREAMING + GRABACIÓN)
# ==============================================================================
class YOLOStreamTrack(VideoStreamTrack):
    def __init__(self, source, video_out_base, enlace, anti_rep, fps=STREAM_FPS):
        super().__init__()
        self.time_base = Fraction(1, fps)
        self._timestamp = 0
        self.is_running = True

        self._source = source
        self._video_out_base = video_out_base
        self.enlace = enlace
        self.anti_rep = anti_rep

        self._es_camara = isinstance(source, int) or (
            isinstance(source, str) and source.startswith("/dev/video"))

        self.cap = None
        self.writer = None
        self.activo = False

        print(f"\n[1/2] Cargando modelo YOLO: {WEIGHTS_PATH}")
        self.model = YOLO(WEIGHTS_PATH, task="detect")
        print("[2/2] Modelo cargado. Esperando que el dron supere 2 metros para grabar e inferir...\n")

    def _arrancar_captura(self):
        print("[VÍDEO] Altura > 2m alcanzada -> Iniciando cámara, inferencia y grabación.")
        if self._es_camara:
            self.cap = cv2.VideoCapture(self._source, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            self.cap = cv2.VideoCapture(self._source)

        if not self.cap.isOpened():
            print(f"❌ No se pudo abrir la fuente de vídeo: {self._source}")
            self.cap = None
            self.is_running = False
            return

        ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps_in = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        salida = nombre_con_fecha(self._video_out_base)
        self.writer = cv2.VideoWriter(
            salida, cv2.VideoWriter_fourcc(*"mp4v"),
            max(1, int(fps_in / VID_STRIDE)), (ancho, alto))
        print(f"[VÍDEO] Grabando archivo local: {salida}")
        self.activo = True

    def _parar_captura(self):
        if not self.activo and self.cap is None and self.writer is None:
            return
        print("[VÍDEO] Dron en suelo/baja cota -> Cerrando archivo de vídeo y pausando inferencia.")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.activo = False

    def _procesar_detecciones(self, resultado):
        if self.enlace is None:
            return
        boxes = getattr(resultado, "boxes", None)
        if boxes is None or getattr(boxes, "cls", None) is None:
            return

        nombres = resultado.names
        clases_en_frame = {
            nombres.get(int(c), str(int(c))) for c in boxes.cls.tolist()
        }
        for clase in clases_en_frame:
            if self.anti_rep.permitido(clase):
                self.enlace.encolar_alerta(clase)

    async def recv(self):
        self._timestamp += 1
        debe_estar_on = self.enlace.video_on if self.enlace is not None else True

        if debe_estar_on and not self.activo:
            self._arrancar_captura()
        elif not debe_estar_on and self.activo:
            self._parar_captura()

        if not self.activo:
            alt = self.enlace.get_telemetria_actual()["alt_rel_m"] if self.enlace else 0.0
            frame = _lienzo_pausa(STREAM_WIDTH, STREAM_HEIGHT, alt)
            video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.pts = self._timestamp
            video_frame.time_base = self.time_base
            await asyncio.sleep(1 / STREAM_FPS)
            return video_frame

        ret, frame = False, None
        for _ in range(VID_STRIDE):
            ret, frame = self.cap.read()
            if not ret:
                break

        if not ret or frame is None:
            if not self._es_camara:
                self.is_running = False
            annotated_frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
        else:
            results = self.model.predict(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
            annotated_frame = results[0].plot()
            self.writer.write(annotated_frame)
            self._procesar_detecciones(results[0])
            annotated_frame = cv2.resize(annotated_frame, (STREAM_WIDTH, STREAM_HEIGHT))

        video_frame = VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        video_frame.pts = self._timestamp
        video_frame.time_base = self.time_base
        return video_frame

    def release(self):
        self._parar_captura()


# ==============================================================================
# Espera a que aiortc reúna todos los candidatos ICE antes de enviar el SDP:
# WHIP (Cloudflare Stream) no usa trickle ICE, así que el offer tiene que
# llevarlos ya todos o la conexión puede fallar de forma intermitente.
# ==============================================================================
async def _esperar_ice_completo(pc):
    if pc.iceGatheringState == "complete":
        return
    fut = asyncio.get_event_loop().create_future()

    @pc.on("icegatheringstatechange")
    def _on_change():
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(True)

    await fut


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
async def main():
    parser = argparse.ArgumentParser(
        description="Streaming WebRTC + Detección YOLO gobernado por altura MAVLink.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video_in", type=str, default=None,
                        help="Fuente de vídeo: archivo ('video.mp4') o cámara ('0', '/dev/video0').")
    parser.add_argument("--video_out", type=str, default=VIDEO_OUT_DEFAULT,
                        help=f"Nombre base de la grabación [Por defecto: '{VIDEO_OUT_DEFAULT}']")
    parser.add_argument("--mp-ip", type=str, default=MP_IP_DEFAULT,
                        help=f"IP del puesto de control / Mission Planner [Por defecto: {MP_IP_DEFAULT}]")
    parser.add_argument("--mp-port", type=int, default=MP_PORT_DEFAULT,
                        help=f"Puerto MAVLink de destino [Por defecto: {MP_PORT_DEFAULT}]")
    parser.add_argument("--mavlink-proto", choices=["udp", "tcp"], default="udp",
                        help="Transporte MAVLink: 'udp' (mirror Inbound de Mission Planner, "
                             "por defecto) o 'tcp' (cliente directo a un puerto MAVLink TCP "
                             "de la SITL de ArduPilot, p. ej. 5762/5763).")
    parser.add_argument("--udp-alert-port", type=int, default=UDP_ALERT_PORT_DEFAULT,
                        help=f"Puerto UDP de alertas JSON para Netcat [Por defecto: {UDP_ALERT_PORT_DEFAULT}]")
    parser.add_argument("--altura-min", type=float, default=ALTURA_ACTIVACION_DEFAULT,
                        help=f"Altitud mínima en metros para activar vídeo [Por defecto: {ALTURA_ACTIVACION_DEFAULT} m]")
    parser.add_argument("--anti-repeticion", type=float, default=ANTI_REPETICION_DEFAULT,
                        help=f"Ventana de anti-repetición en segundos [Por defecto: {ANTI_REPETICION_DEFAULT} s]")
    parser.add_argument("--sin-mavlink", action="store_true",
                        help="Ejecutar sin dron (vídeo e inferencia siempre activos).")

    args = parser.parse_args()

    source = 0 if args.video_in is None and not os.path.exists(VIDEO_IN_DEFAULT) else (
        int(args.video_in) if args.video_in and args.video_in.isdigit() else (args.video_in or VIDEO_IN_DEFAULT)
    )

    print("=" * 75)
    print("      DETECCIÓN YOLO Y TELEMETRÍA — CONTROL POR ALTITUD (>2m)")
    print("=" * 75)
    print(f" • Origen de vídeo        : {source}")
    print(f" • MAVLink Destino        : {args.mavlink_proto}:{args.mp_ip}:{args.mp_port}")
    print(f" • Alertas UDP JSON       : {args.mp_ip}:{args.udp_alert_port}")
    print(f" • Cota de activación     : >= {args.altura_min} metros relativos")
    print(f" • Ver WebRTC en directo  : {VIEW_URL}")
    print("=" * 75)

    enlace = None
    if not args.sin_mavlink:
        if mavutil is None:
            print("❌ pymavlink no está instalado.")
            return
        enlace = EnlaceMavlink(
            target_ip=args.mp_ip,
            target_port=args.mp_port,
            sysid=1,
            compid=199,
            altura_umbral=args.altura_min,
            udp_alert_port=args.udp_alert_port,
            mavlink_proto=args.mavlink_proto,
        )
        enlace.start()
        enlace.esperar_listo(timeout=10)

    anti_rep = AntiRepeticion(args.anti_repeticion)

    pc = RTCPeerConnection()
    track = YOLOStreamTrack(source, args.video_out, enlace, anti_rep)
    pc.addTrack(track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await _esperar_ice_completo(pc)

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post(
            WHIP_URL, data=pc.localDescription.sdp,
            headers={"Content-Type": "application/sdp"}
        ) as resp:
            if resp.status not in (200, 201):
                print(f"❌ Error WebRTC HTTP {resp.status}: {await resp.text()}")
                await pc.close()
                track.release()
                if enlace:
                    enlace.detener()
                return
            answer_sdp = await resp.text()
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer_sdp, type="answer"))
            print("✔ Conexión WebRTC establecida con éxito.\n")

    try:
        while track.is_running:
            await asyncio.sleep(0.5)
    finally:
        track.release()
        await pc.close()
        if enlace is not None:
            enlace.detener()
        print("✔ Transmisión finalizada limpiamente.")


if __name__ == "__main__":
    asyncio.run(main())
