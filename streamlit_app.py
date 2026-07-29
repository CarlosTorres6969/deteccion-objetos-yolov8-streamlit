"""Aplicación web para detección de objetos con YOLOv8."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
YOLO_CONFIG_DIR = BASE_DIR / ".yolo_config"
YOLO_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

try:
    import av
    from streamlit_webrtc import webrtc_streamer

    WEBRTC_DISPONIBLE = True
except ImportError:
    WEBRTC_DISPONIBLE = False


MODEL_PATH = BASE_DIR / "yolov8n.pt"
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

st.set_page_config(
    page_title="Visión IA · YOLOv8",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #17202a;
            --muted: #667085;
            --accent: #ff5d3b;
            --accent-soft: #fff1ed;
            --panel: rgba(255, 255, 255, 0.88);
        }
        .stApp {
            background:
                radial-gradient(circle at 6% 8%, rgba(255, 93, 59, .11), transparent 25rem),
                radial-gradient(circle at 96% 22%, rgba(40, 123, 255, .10), transparent 28rem),
                #f7f8fb;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        .hero {
            border: 1px solid rgba(23, 32, 42, .08);
            border-radius: 26px;
            padding: 2.2rem 2.35rem;
            margin-bottom: 1.4rem;
            background: linear-gradient(120deg, rgba(255,255,255,.96), rgba(255,247,244,.90));
            box-shadow: 0 18px 45px rgba(16, 24, 40, .07);
        }
        .eyebrow {
            color: var(--accent);
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
            margin-bottom: .65rem;
        }
        .hero h1 {
            color: var(--ink);
            font-size: clamp(2.15rem, 5vw, 4rem);
            letter-spacing: -.055em;
            line-height: .98;
            margin: 0 0 .9rem 0;
        }
        .hero p {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.65;
            max-width: 720px;
            margin: 0;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            background: #eefbf3;
            border: 1px solid #c8efd6;
            border-radius: 999px;
            color: #18753c;
            font-size: .8rem;
            font-weight: 700;
            padding: .35rem .7rem;
            margin-top: 1.1rem;
        }
        .status-dot {
            width: .48rem;
            height: .48rem;
            border-radius: 50%;
            background: #22a35a;
            box-shadow: 0 0 0 .22rem rgba(34,163,90,.12);
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid rgba(23, 32, 42, .08);
            border-radius: 18px;
            padding: .8rem 1rem;
        }
        [data-testid="stSidebar"] {
            background: rgba(252, 252, 253, .96);
            border-right: 1px solid rgba(23, 32, 42, .08);
        }
        .tiny-note {
            color: var(--muted);
            font-size: .83rem;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando YOLOv8 Nano…")
def cargar_modelo() -> YOLO:
    """Carga una sola instancia del modelo durante la sesión del servidor."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró yolov8n.pt en la raíz del proyecto."
        )
    return YOLO(str(MODEL_PATH))


def construir_tabla(resultado: object, nombres: dict[int, str]) -> pd.DataFrame:
    """Convierte las cajas producidas por YOLO en datos tabulares."""
    columnas = [
        "N.º",
        "Objeto",
        "Confianza",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    filas: list[dict[str, object]] = []
    cajas = getattr(resultado, "boxes", None)

    if cajas is None:
        return pd.DataFrame(columns=columnas)

    for numero, (clase, confianza, coordenadas) in enumerate(
        zip(cajas.cls.tolist(), cajas.conf.tolist(), cajas.xyxy.tolist()),
        start=1,
    ):
        x1, y1, x2, y2 = (round(valor) for valor in coordenadas)
        filas.append(
            {
                "N.º": numero,
                "Objeto": nombres[int(clase)],
                "Confianza": round(float(confianza) * 100, 1),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return pd.DataFrame(filas, columns=columnas)


def detectar_imagen(
    imagen: Image.Image,
    modelo: YOLO,
    confianza: float,
    tamano: int,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Ejecuta la inferencia y devuelve imagen anotada, tabla y confianza media."""
    arreglo_rgb = np.asarray(imagen.convert("RGB"))
    resultado = modelo.predict(
        source=arreglo_rgb,
        conf=confianza,
        imgsz=tamano,
        max_det=100,
        device="cpu",
        verbose=False,
    )[0]
    anotada_bgr = resultado.plot(line_width=2, font_size=12)
    anotada_rgb = anotada_bgr[:, :, ::-1].copy()
    tabla = construir_tabla(resultado, modelo.names)
    promedio = (
        float(resultado.boxes.conf.mean().item() * 100)
        if resultado.boxes is not None and len(resultado.boxes) > 0
        else 0.0
    )
    return anotada_rgb, tabla, promedio


def procesar_fotograma_yolo(
    frame: "av.VideoFrame",
    modelo: YOLO,
    confianza: float,
    tamano: int,
) -> "av.VideoFrame":
    """Anota un fotograma WebRTC y conserva el flujo ante fallos aislados."""
    imagen_bgr = frame.to_ndarray(format="bgr24")
    try:
        resultado = modelo.predict(
            source=imagen_bgr,
            conf=confianza,
            imgsz=tamano,
            max_det=50,
            device="cpu",
            verbose=False,
        )[0]
        imagen_bgr = resultado.plot(line_width=2, font_size=11)
    except Exception:
        pass
    return av.VideoFrame.from_ndarray(imagen_bgr, format="bgr24")


def mostrar_resultados(
    imagen: Image.Image,
    modelo: YOLO,
    confianza: float,
    tamano: int,
) -> None:
    """Renderiza la inferencia de una imagen y sus métricas."""
    with st.spinner("Analizando la escena…"):
        anotada, tabla, promedio = detectar_imagen(
            imagen, modelo, confianza, tamano
        )

    metrica_1, metrica_2, metrica_3 = st.columns(3)
    metrica_1.metric("Detecciones", len(tabla))
    metrica_2.metric(
        "Clases distintas",
        int(tabla["Objeto"].nunique()) if not tabla.empty else 0,
    )
    metrica_3.metric("Confianza media", f"{promedio:.1f} %")

    izquierda, derecha = st.columns([1.35, 1], gap="large")
    with izquierda:
        st.image(
            anotada,
            caption="Resultado procesado por YOLOv8 Nano",
            use_container_width=True,
        )
    with derecha:
        st.subheader("Objetos encontrados")
        if tabla.empty:
            st.info(
                "No hubo detecciones. Prueba con mejor iluminación o reduce "
                "la confianza mínima."
            )
        else:
            tabla_visible = tabla.copy()
            tabla_visible["Confianza"] = tabla_visible["Confianza"].map(
                lambda valor: f"{valor:.1f} %"
            )
            st.dataframe(
                tabla_visible,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Objeto": st.column_config.TextColumn("Objeto"),
                },
            )
            st.download_button(
                "Descargar resultados CSV",
                data=tabla.to_csv(index=False).encode("utf-8-sig"),
                file_name="detecciones_yolov8.csv",
                mime="text/csv",
                use_container_width=True,
            )


def renderizar_modo_imagen(
    modelo: YOLO,
    confianza: float,
    tamano: int,
) -> None:
    """Muestra los controles para captura fotográfica y carga de archivos."""
    st.subheader("Analiza una escena")
    st.caption(
        "Toma una fotografía con tu cámara o carga una imagen JPG/PNG."
    )

    pestana_camara, pestana_archivo = st.tabs(
        ["Usar cámara", "Subir imagen"]
    )
    entrada = None

    with pestana_camara:
        captura = st.camera_input("Captura una fotografía")
        if captura is not None:
            entrada = captura

    with pestana_archivo:
        archivo = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )
        if archivo is not None:
            entrada = archivo

    if entrada is None:
        st.info("La imagen aparecerá aquí cuando la captures o la selecciones.")
        return

    try:
        imagen = Image.open(entrada).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("El archivo no es una imagen válida o está dañado.")
        return

    mostrar_resultados(imagen, modelo, confianza, tamano)


def renderizar_modo_vivo(
    modelo: YOLO,
    confianza: float,
    tamano: int,
) -> None:
    """Muestra detección continua usando la cámara del navegador."""
    st.subheader("Detección en vivo")
    st.caption(
        "Pulsa START, permite el acceso a la cámara y observa las detecciones "
        "directamente sobre el video."
    )

    if not WEBRTC_DISPONIBLE:
        st.warning(
            "El componente de video en vivo no está instalado en este entorno. "
            "Puedes usar el modo «Foto o archivo»."
        )
        return

    def procesar_fotograma(frame: "av.VideoFrame") -> "av.VideoFrame":
        return procesar_fotograma_yolo(
            frame, modelo, confianza, tamano
        )

    webrtc_streamer(
        key="detector-yolov8",
        video_frame_callback=procesar_fotograma,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 960},
                "height": {"ideal": 540},
                "facingMode": "user",
            },
            "audio": False,
        },
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
    )

    st.markdown(
        """
        <p class="tiny-note">
        La primera conexión puede tardar unos segundos. Si la red bloquea
        WebRTC, usa <b>Foto o archivo</b>, que accede a la cámara sin transmitir
        video continuo.
        </p>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Visión artificial · COCO 80</div>
        <h1>Objetos a la vista.</h1>
        <p>
            Detección de múltiples objetos con YOLOv8 Nano. Usa la cámara en
            vivo o analiza una fotografía; cada resultado incluye su clase,
            nivel de confianza y ubicación.
        </p>
        <div class="status-pill">
            <span class="status-dot"></span>
            Modelo listo para inferencia
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Panel de detección")
    modo = st.radio(
        "Modo de entrada",
        ["Video en vivo", "Foto o archivo"],
        index=0,
    )
    confianza = st.slider(
        "Confianza mínima",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help="Solo se muestran detecciones por encima de este umbral.",
    )
    tamano = st.select_slider(
        "Resolución de inferencia",
        options=[320, 416, 480, 640],
        value=480,
        help="Una resolución menor responde más rápido; una mayor puede detectar detalles pequeños.",
    )
    st.divider()
    st.caption("MODELO")
    st.write("**YOLOv8 Nano**")
    st.write("80 clases del conjunto COCO")
    dispositivo = "GPU" if torch.cuda.is_available() else "CPU"
    st.write(f"Inferencia: **{dispositivo}**")

try:
    modelo_yolo = cargar_modelo()
except Exception as error:
    st.error(f"No fue posible cargar el modelo: {error}")
    st.stop()

if modo == "Video en vivo":
    renderizar_modo_vivo(modelo_yolo, confianza, tamano)
else:
    renderizar_modo_imagen(modelo_yolo, confianza, tamano)

st.divider()
st.caption(
    "YOLOv8 Nano · Modelo preentrenado con COCO · "
    "Las predicciones pueden contener errores."
)
