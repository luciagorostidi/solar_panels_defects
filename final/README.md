# final

## Contexto

Este subdirectorio recoge el **modelo YOLO11n funcionando en formato HEF** (compilado para el acelerador Hailo-8) junto con el script que lo ejecuta. Es el modelo más completo de los que se han generado: incluye ya las etiquetas de clase, los metadatos y el paquete listo para desplegar en la Raspberry Pi con acelerador Hailo.

Contiene:

- **`detectar_video_metricas_hef.py`** — script de inferencia y medición de métricas (latencia, FPS, uso de RAM) sobre vídeo, usando el modelo HEF a través de `hailo_platform` (carga el HEF, configura el `VDevice`, decodifica manualmente las salidas raw de YOLO11 con DFL y dibuja las detecciones sobre el vídeo). Está pensado para ejecutarse en la Raspberry Pi con el Hailo-8 conectado; la ruta base (`BASE_DIR`) está fijada a `/home/lucia/solar_panels_defects`, la ubicación original en esa máquina — hay que ajustarla si se ejecuta desde otra ruta.
- **`new_models/`** — el modelo HEF y todos los ficheros generados junto a él por el mismo pipeline de exportación (mismo `--tag`, `paneles_yolo11n`), en el mismo sitio relativo que espera el script anterior (`new_models/paneles_yolo11n.hef`, `new_models/paneles_yolo11n_labels.txt`):
  - `paneles_yolo11n.hef` — el modelo compilado para Hailo-8 (el que usa el script).
  - `paneles_yolo11n_labels.txt` / `paneles_yolo11n_metadata.json` — nombres de clase y metadatos del modelo.
  - `paneles_yolo11n_paquete.zip` — zip con los tres ficheros anteriores, listo para copiar a la Raspberry Pi de un solo golpe.
  - `paneles_yolo11n.onnx` / `paneles_yolo11n_int8.onnx` / `paneles_yolo11n_best.pt` / `paneles_yolo11n_calib.npy` — ficheros intermedios del proceso de exportación/cuantización (no imprescindibles para desplegar en la Pi).
  - `paneles_yolo11n_ncnn/` — exportación alternativa a formato NCNN (sin relación con Hailo).
  - `paneles_yolo11n_pipeline_run.log` / `hailort.log` — logs de la generación del modelo y de una ejecución de HailoRT.
  - `README.md` — README original de esa carpeta (autogenerado por el pipeline `generate_all_formats.py`), con el detalle de cada fichero.

> Nota: `paneles_yolo11n_calib.npy` pesa ~120 MB (imágenes de calibración para la cuantización INT8), por encima del límite de 100 MB por fichero de GitHub — si este directorio se sube al repositorio, ese fichero necesitará Git LFS (o excluirse si no es necesario conservarlo).
