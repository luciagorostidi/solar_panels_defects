# Test Inicial Frutas

## Contexto

Este subdirectorio recoge las **primeras pruebas exploratorias con YOLO** realizadas antes de abordar el problema real del TFM (detección de defectos en paneles solares). El objetivo de esta fase era puramente de aprendizaje y validación de herramientas: comprobar si YOLOv8 era capaz de entrenarse localmente con un dataset sencillo, y si el modelo resultante era capaz de detectar e identificar correctamente objetos (en este caso, distintos tipos de fruta) en una toma aérea de vídeo grabada con un dron.

El dataset de frutas (`train/`, `valid/`, `test/`, `data.yaml`) procede de [Roboflow](https://universe.roboflow.com/luca-gorostidi-garca-s-workspace/fruits-came9-gbvfi/dataset/2) (ver `README.roboflow.txt` y `README.dataset.txt`) y contiene 12 clases de fruta (manzana, plátano, uva, kiwi, lichi, mango, naranja, melocotón, pera, piña, granada y fresa).

Los vídeos `video_dron.mp4` y `video_iphone.mp4` son las tomas de prueba usadas como entrada para la inferencia (una grabada con dron desde el aire y otra con móvil), y `output.mp4` / `output_dron.mp4` son los vídeos resultantes ya anotados con las detecciones del modelo. `yolov8n.pt` son los pesos base (preentrenados en COCO) de la arquitectura nano de YOLOv8, punto de partida del entrenamiento. La carpeta `runs/detect/` contiene los artefactos generados automáticamente por Ultralytics en cada entrenamiento (métricas, curvas, matriz de confusión, batches de ejemplo y los pesos resultantes `best.pt`/`last.pt` en `weights/`).

Esta fase sirvió como banco de pruebas para validar el flujo de trabajo (entrenar → inferir sobre vídeo → visualizar resultados) que después se reutilizó, ya adaptado, en la fase final del proyecto (`final_E4`).

## Scripts Python

### `explorar_dataset.py`
Script de utilidad para inspeccionar rápidamente el dataset antes de entrenar. Lee `data.yaml`, localiza las carpetas de imágenes de train y validación (siendo tolerante con rutas relativas de distinto formato) y muestra por consola:
- El número de imágenes encontradas en cada partición (train/valid).
- El listado de clases configuradas (`names`) con su índice numérico.

Es un chequeo previo para confirmar que las rutas del `data.yaml` son correctas y que el dataset está completo antes de lanzar un entrenamiento.

### `entrenar_local.py`
Script de entrenamiento. Carga los pesos preentrenados `yolov8n.pt` y lanza `model.train()` con el dataset definido en `data.yaml`, usando 40 épocas, tamaño de imagen 640 y batch de 8. El resultado (pesos, métricas y gráficas) se guarda en `runs/detect/mi_modelo_frutas*/`. Al finalizar, imprime el tiempo total de entrenamiento en minutos y la ruta donde han quedado guardados los pesos finales (`best.pt`).

### `detectar_video.py`
Script de inferencia sobre vídeo. Carga el modelo ya entrenado (`runs/detect/mi_modelo_frutas_v2/weights/best.pt`) y ejecuta la predicción sobre `video_dron.mp4` frame a frame (procesando 1 de cada 3 frames mediante `vid_stride=3` para agilizar el proceso). Por cada frame:
- Dibuja las cajas de detección sobre la imagen (`results.plot()`).
- Escribe el frame anotado en el vídeo de salida `output_dron.mp4`, ajustando el FPS de salida para que se reproduzca a velocidad real pese al salto de frames.
- Muestra el resultado en una ventana interactiva en tiempo real (redimensionada a 960x540), que puede cerrarse pulsando la tecla `q`.

Este script fue la prueba de concepto que demostró la viabilidad de usar YOLO sobre vídeo aéreo, y sirvió de base para los scripts de detección más elaborados de la fase `final_E4`.
