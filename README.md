# solar_panels_defects

Este repositorio contiene el código utilizado por **Lucía Gorostidi García** durante su Trabajo Fin de Máster (TFM), centrado en la detección de defectos en paneles solares mediante visión por computador (YOLO) e imágenes/vídeo capturados por dron.

El contenido está organizado en 3 subdirectorios, cada uno correspondiente a una fase o componente distinto del trabajo. Dentro de cada uno hay un `README.md` propio con el detalle completo de su contexto y de los scripts que contiene.

- **`Test_Inicial_Frutas/`** — Pruebas exploratorias iniciales con YOLOv8 (entrenamiento e inferencia) sobre un dataset de frutas, usadas para validar el flujo de trabajo antes de abordar el caso real de paneles solares.
- **`final_E4/`** — Etapa 4 (fase final) del TFM: modelo YOLO entrenado y evaluado ya sobre el dataset real de defectos en paneles solares, con scripts de inferencia, alertas y métricas de rendimiento.
- **`SITL/`** — Pruebas SITL (Software In The Loop) con Mission Planner, para validar el software en un entorno de vuelo simulado.
