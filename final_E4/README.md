# final_E4

## Contexto

Este subdirectorio corresponde a la **Etapa 4 (E4)** del TFM, la fase final de entrenamiento e inferencia del modelo destinado a la **detección de defectos en paneles solares** (por ejemplo, suciedad/*soiling*) a partir de imágenes aéreas capturadas por dron. A diferencia de `Test_Inicial_Frutas` (pruebas exploratorias con un dataset de frutas para validar el flujo de trabajo con YOLO), aquí se trabaja ya con el modelo final entrenado sobre el dataset real de paneles solares del proyecto.

El vídeo `video_dron.mp4` es un **vuelo real de dron sobre una vivienda con paneles solares instalados en el tejado**, usado como entrada para probar el modelo en condiciones realistas.

La carpeta `weights/` contiene los pesos del modelo ya entrenado, exportados en distintos formatos para comparar rendimiento y portabilidad:
- `best.pt`: pesos nativos de PyTorch/Ultralytics (máxima precisión, requiere `ultralytics`/PyTorch para ejecutarse).
- `best.onnx`: exportación a ONNX en precisión completa (FP32), pensada para inferencia más portable/optimizada fuera del entorno de entrenamiento.
- `best_int8.onnx`: exportación a ONNX cuantizada a INT8, con el objetivo de reducir tamaño de modelo y latencia (a costa de una posible pequeña pérdida de precisión), pensada para escenarios con recursos limitados (p. ej. hardware embarcado en el dron).

Los scripts de esta carpeta cubren tres necesidades: (1) generar un vídeo anotado con las detecciones, (2) generar ese mismo vídeo pero además emitiendo alertas en tiempo real por protocolo MAVLink cuando se detecta un defecto, y (3) medir el rendimiento (tamaño de modelo, latencia, FPS y uso de RAM) de las distintas variantes de pesos exportadas, para poder comparar el compromiso precisión/eficiencia entre `best.pt`, `best.onnx` y `best_int8.onnx`.

## Scripts Python

### `detectar_video.py`
Script base de inferencia sobre vídeo. Carga los pesos `weights/best.pt` y ejecuta la predicción sobre `video_dron.mp4` frame a frame, procesando 1 de cada 3 frames (`vid_stride=3`) para acelerar el procesado. Por cada frame anotado:
- Escribe el resultado en `output_dron.mp4`, ajustando el FPS de salida para compensar el salto de frames y mantener la velocidad de reproducción real.
- Muestra el vídeo anotado en una ventana interactiva (960x540), cerrable con la tecla `q`.

Es el equivalente, ya con el modelo final de paneles solares, al script homónimo usado en las pruebas iniciales con frutas.

### `detectar_video_alerta.py`
Extiende `detectar_video.py` añadiendo un sistema de **alertas en tiempo real vía MAVLink** (protocolo estándar de comunicación con drones/autopilotos). Además de generar el vídeo anotado, por cada frame comprueba si entre las clases detectadas aparece la clase `soiling` (suciedad en el panel). Si es así, envía un mensaje `STATUSTEXT` de severidad *warning* a través de una conexión MAVLink UDP (`udpout:127.0.0.1:14551`), respetando un cooldown de 3 segundos entre alertas para no saturar el canal. Simula cómo el sistema de detección podría integrarse con el software de control de vuelo del dron para avisar en vivo de defectos detectados durante el vuelo.

### `escuchar_alertas.py`
Script auxiliar de prueba/depuración para el anterior. Abre una conexión MAVLink en modo escucha (`udpin:127.0.0.1:14551`) y va imprimiendo por consola cada mensaje `STATUSTEXT` que recibe. Está pensado para ejecutarse en paralelo (en otra consola) mientras corre `detectar_video_alerta.py`, para verificar que las alertas de "soiling detectado" se están emitiendo y recibiendo correctamente.

### `detectar_video_metricas.py`
Variante de `detectar_video.py` orientada a **evaluar el rendimiento del modelo exportado a ONNX** (`weights/best.onnx`) en lugar de los pesos nativos `.pt`. Además de generar el vídeo anotado, durante la inferencia recopila:
- **Latencia** por frame (tiempo de inferencia interno reportado por Ultralytics, `results.speed`).
- **Uso de RAM** del proceso (vía `psutil`).

Al finalizar, calcula y muestra por consola un resumen de métricas: tamaño del fichero de pesos en MB, latencia media (ms/frame), FPS de inferencia equivalente y uso medio de RAM (MB). Sirve para cuantificar el coste computacional del modelo en formato ONNX estándar.

### `detectar_video_metricas_int8.py`
Variante de `detectar_video_metricas.py` que apunta a los pesos cuantizados `weights/best_int8.onnx` en lugar de `weights/best.onnx`. Se usa para obtener las mismas métricas (tamaño, latencia, FPS, RAM) pero con el modelo cuantizado a INT8, y así poder **comparar directamente el impacto de la cuantización** frente a la versión ONNX en precisión completa (y frente a los pesos originales `.pt`).
