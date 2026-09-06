# SITL

## Contexto

Este subdirectorio recoge la prueba **SITL (Software In The Loop)** del proyecto, realizada con **Mission Planner** y el simulador SITL de ArduPilot, lista para ejecutarse de forma autocontenida (incluye el modelo y el vídeo de ejemplo).

La prueba funciona así:

1. `detectar_SITL.py` se conecta por MAVLink al autopiloto simulado (SITL) y recibe su telemetría, incluida la **altitud relativa**.
2. **Mientras el dron está por debajo de 2 metros de altitud, el vídeo/inferencia no se procesa**; en cuanto **supera los 2 metros, se activa** la captura, la inferencia YOLO sobre el vídeo y el streaming WebRTC.
3. Cuando el modelo **detecta un defecto** durante el vuelo, se emite una alerta: un `STATUSTEXT` MAVLink (visible en el panel de mensajes de Mission Planner) y un datagrama UDP en JSON con la clase detectada y la telemetría, dirigido al puerto donde escucha `server_alertas.py`.

### Por qué ONNX cuantizado (INT8) y no el HEF

El modelo que se usa aquí es el **ONNX cuantizado a INT8** (`solar_panels_yolo11n_int8.onnx`), no el `.hef` compilado para la NPU Hailo-8. La razón es que el `.hef` solo se puede ejecutar con el acelerador Hailo (el HAT) conectado a una Raspberry Pi, y esa combinación —Raspberry + HAT— ya se ha probado con éxito por separado. Para esta prueba SITL interesa poder trabajar en el simulador desde un **ordenador local cualquiera**, sin depender de ese hardware específico, así que se usa el ONNX cuantizado, que corre en CPU con Ultralytics sin necesitar ni Raspberry ni HAT.

## Scripts Python

### `detectar_SITL.py`
Detección YOLO + grabación local + streaming WebRTC, controlado por la altitud de vuelo (> 2 m) leída vía MAVLink desde Mission Planner. Carga el modelo `exported_models/solar_panels_yolo11n_int8.onnx` y procesa `video_dron.mp4` como entrada (`VIDEO_IN_DEFAULT`), generando `output_dron.mp4`. Al detectar un defecto, encola y emite la alerta por dos canales: `STATUSTEXT` MAVLink y datagrama UDP JSON (puerto 9000 por defecto) hacia `server_alertas.py`.

### `server_alertas.py`
El **servidor de alertas**: un receptor UDP muy simple que escucha en el puerto 9000 (`0.0.0.0:9000`) y va imprimiendo por consola, en formato JSON legible, cada alerta que le llega desde `detectar_SITL.py`. Se ejecuta en paralelo (en otra consola, o en el PC que hace de estación de tierra de alertas) mientras corre `detectar_SITL.py`.

## Modelo y datos de ejemplo

- **`exported_models/`** — el modelo que usa `detectar_SITL.py` (`solar_panels_yolo11n_int8.onnx`) junto con sus ficheros hermanos generados por el mismo pipeline de exportación: `solar_panels_yolo11n.hef` (compilado para NPU Hailo-8, no usado por esta prueba — ver más arriba), `solar_panels_yolo11n.onnx` (FP32), `best.pt` (pesos PyTorch de origen), `calib.npy` (imágenes de calibración para la cuantización INT8, ~120 MB), `pipeline_run.log` (log de la exportación) y `solar_panels_yolo11n_ncnn/` (exportación alternativa a formato NCNN).
- **`video_dron.mp4`** — vídeo de ejemplo usado como entrada por `detectar_SITL.py`.

> Nota de tamaño: `exported_models/calib.npy` (~120 MB) y `video_dron.mp4` (~114 MB) superan el límite de 100 MB por fichero de GitHub; están pensados para versionarse con Git LFS (ver `.gitattributes` en la raíz del repositorio, que ya trackea `*.mp4` y `*.npy`).
