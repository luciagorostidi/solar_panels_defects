# solar_panels_defects

Este repositorio contiene el código utilizado por **Lucía Gorostidi García** durante su Trabajo Fin de Máster (TFM), centrado en la detección de defectos en paneles solares mediante visión por computador (YOLO) e imágenes/vídeo capturados por dron. La memoria completa del TFM en PDF puede descargarse directamente desde GitHub: [`TFM_Lucia_Gorostidi.pdf`](https://github.com/luciagorostidi/solar_panels_defects/blob/main/TFM_Lucia_Gorostidi.pdf).

El contenido está organizado en subdirectorios, cada uno correspondiente a una fase o componente distinto del trabajo. Dentro de cada uno hay un `README.md` propio con el detalle completo de su contexto, de qué hace cada script Python y de qué modelo depende.

- **`Test_Inicial_Frutas/`** — Pruebas exploratorias iniciales con YOLOv8 (entrenamiento e inferencia) sobre un dataset de frutas, usadas para validar el flujo de trabajo antes de abordar el caso real de paneles solares.
- **`final_E4/`** — Etapa 4 (fase final) del TFM: modelo YOLO entrenado y evaluado ya sobre el dataset real de defectos en paneles solares, con scripts de inferencia, alertas y métricas de rendimiento.
- **`final/`** — Modelo YOLO11n final exportado a formato HEF (compilado para el acelerador Hailo-8), junto con el script que lo ejecuta y mide su rendimiento sobre vídeo.
- **`SITL/`** — Prueba SITL (Software In The Loop) con Mission Planner: activa la detección al superar 2 m de altitud y emite alertas por UDP al detectar un defecto.

Ficheros de configuración de Git (`.gitignore`, `.gitattributes`, incluido el tracking de vídeos y ficheros `.npy` pesados vía Git LFS) están unificados en esta carpeta raíz y aplican a todos los subdirectorios.

## Notebook de entrenamiento (Google Colab)

**[`TFM_YOLO.ipynb`](TFM_YOLO.ipynb)** es un cuaderno de **Google Colab** con todo el proceso de entrenamiento de los modelos YOLO11n del TFM: instalación de dependencias, descarga del dataset desde Roboflow y los sucesivos experimentos de entrenamiento (E1 baseline, E2 cambio de versión del dataset, E3 revisión de clases, E4 dataset aumentado — este último es el que da lugar al modelo de `final_E4/`), además de la exportación del modelo a ONNX y su cuantización a INT8.

Para ejecutarlo directamente, sin instalar nada en local:
1. Pulsa el botón **"Open in Colab"** de la primera celda del notebook (o abre directamente [este enlace](https://colab.research.google.com/github/luciagorostidi/solar_panels_defects/blob/main/TFM_YOLO.ipynb)).
2. Ya en Colab, activa una sesión con GPU (`Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU`).
3. Guarda tu API key de Roboflow como secreto de Colab con el nombre `ROBOFLOW_API_KEY` (icono de la llave 🔑 en el panel izquierdo) — el notebook la necesita para descargar el dataset.
4. Ejecuta las celdas en orden (`Entorno de ejecución → Ejecutar todas`).
