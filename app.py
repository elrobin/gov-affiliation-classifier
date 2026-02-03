"""
Interfaz Streamlit para clasificación de afiliaciones institucionales.

Permite elegir backend (Gemini / Local), configurar API Key o URL,
subir CSV, ver progreso y descargar el resultado.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd
import streamlit as st

from lm_client import get_classifier, set_default_backend
from main import run_classification_pipeline

# Configurar logging para que no sature la consola en Streamlit
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("gov-affiliation-classifier.app")

st.set_page_config(
    page_title="Gov Affiliation Classifier",
    page_icon="🏛️",
    layout="centered",
)

st.title("🏛️ Gov Affiliation Classifier")
st.markdown(
    "Clasifica afiliaciones institucionales (tipo de organización, nivel de gobierno, misión investigación). "
    "Sube un CSV con columnas **afid**, **affiliation**, **country_code**."
)

# ---------------------------------------------------------------------------
# Barra lateral: Backend y configuración
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")
    backend_type = st.radio(
        "Backend LLM",
        options=["gemini", "local"],
        format_func=lambda x: "Google Gemini (API)" if x == "gemini" else "Local (LM Studio / OpenAI-compatible)",
        index=1,
    )

    if backend_type == "gemini":
        api_key = st.text_input(
            "API Key (Gemini)",
            type="password",
            placeholder="Introduce tu GEMINI_API_KEY",
            help="O define la variable de entorno GEMINI_API_KEY.",
        )
        model_name = st.text_input(
            "Modelo Gemini",
            value="gemini-2.0-flash-lite",
            placeholder="gemini-2.0-flash-lite",
            help="Modelo Lite estable (v1beta). Mantén sleep para evitar 429.",
        )
        config = {
            "api_key": api_key or None,
            "model_name": model_name.strip() or None,
        }
    else:
        base_url = st.text_input(
            "URL base (Local)",
            value="http://localhost:1234/v1",
            placeholder="http://localhost:1234/v1",
            help="URL base del servidor LM Studio o compatible OpenAI.",
        )
        model_name = st.text_input(
            "Nombre del modelo (opcional)",
            value="",
            placeholder="local-model",
            help="O deja vacío para usar el valor por defecto.",
        )
        config = {
            "base_url": base_url.strip() or None,
            "model_name": model_name.strip() or None,
        }

    st.divider()
    ror_path = st.text_input(
        "Ruta al ROR JSON (opcional)",
        value="",
        placeholder="v1.74-2025-11-24-ror-data/v1.74-2025-11-24-ror-data.json",
        help="Archivo JSON del Research Organization Registry para enriquecer clasificación.",
    )
    ror_path = ror_path.strip() or None

# ---------------------------------------------------------------------------
# Subida de CSV y procesamiento
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Sube tu CSV",
    type=["csv"],
    help="Columnas requeridas: afid, affiliation, country_code.",
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        st.stop()

    required = {"afid", "affiliation", "country_code"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Faltan columnas requeridas: {missing}. Asegúrate de que el CSV tenga: afid, affiliation, country_code.")
        st.stop()

    st.success(f"CSV cargado: **{len(df)}** filas.")

    if st.button("Clasificar", type="primary"):
        # Validar configuración según backend
        if backend_type == "gemini" and not (config.get("api_key") or __import__("os").environ.get("GEMINI_API_KEY")):
            st.error("Para el backend Gemini necesitas introducir una API Key o definir GEMINI_API_KEY.")
            st.stop()

        try:
            backend = get_classifier(backend_type, config)
            set_default_backend(backend)
        except Exception as e:
            st.error(f"Error al crear el backend: {e}")
            st.stop()

        progress_bar = st.progress(0.0, text="Iniciando clasificación...")
        status_placeholder = st.empty()

        try:
            status_placeholder.info("Procesando lotes (reglas + LLM)...")
            total = len(df)
            # Actualizar progreso de forma aproximada durante el pipeline
            # (el pipeline no devuelve progreso por fila; podemos hacer un callback en el futuro)
            progress_bar.progress(0.2, text="Clasificando afiliaciones...")

            out_df = run_classification_pipeline(
                df.copy(),
                ror_path=ror_path,
                batch_size=8,
                enable_individual_fallback=False,
            )

            progress_bar.progress(1.0, text="Listo.")
            status_placeholder.success("Clasificación completada.")

            # Vista previa
            st.subheader("Vista previa del resultado")
            st.dataframe(out_df.head(20), use_container_width=True)

            # Descarga
            buffer = io.BytesIO()
            out_df.to_csv(buffer, index=False, encoding="utf-8")
            buffer.seek(0)
            st.download_button(
                label="Descargar CSV resultante",
                data=buffer,
                file_name="classified_affiliations.csv",
                mime="text/csv",
            )
        except Exception as e:
            logger.exception("Error durante la clasificación")
            progress_bar.progress(1.0, text="Error.")
            status_placeholder.error(f"Error durante la clasificación: {e}")
            st.exception(e)

else:
    st.info("Sube un archivo CSV para comenzar.")
