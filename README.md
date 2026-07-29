# Detector de objetos con YOLOv8 y Streamlit

Aplicación web de visión artificial que reconoce múltiples objetos con
YOLOv8 Nano y el conjunto de datos COCO. Incluye:

- detección en vivo desde la cámara del navegador;
- análisis de fotografías capturadas o cargadas;
- tabla de objetos, confianza y coordenadas;
- descarga de resultados en CSV.

## Ejecución local

Requiere Python 3.10 o superior.

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Despliegue en Streamlit Community Cloud

1. Publica este directorio en un repositorio de GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io/).
3. Crea una app y selecciona el repositorio, la rama `main` y
   `streamlit_app.py` como archivo de entrada.
4. En la configuración avanzada, usa Python 3.12.

El archivo `yolov8n.pt` debe permanecer en la raíz del repositorio.

> La transmisión en vivo utiliza capturas periódicas mediante HTTPS, por lo
> que no necesita configurar servidores STUN/TURN. El modo de captura
> fotográfica permanece disponible como alternativa.
