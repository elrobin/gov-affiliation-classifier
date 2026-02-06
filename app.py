"""
Streamlit UI for classifying institutional affiliations.

Lets you choose backend (Gemini / Local), set API key or URL,
upload CSV, see progress, and download the result.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd
import streamlit as st

from lm_client import get_classifier, set_default_backend
from main import OUTPUT_COLUMN_ORDER, run_classification_pipeline

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
    "Classify institutional affiliations (organization type, government level, research mission). "
    "Upload a CSV with columns **afid**, **affiliation**, **country_code**."
)

# ---------------------------------------------------------------------------
# Sidebar: Backend and configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    backend_type = st.radio(
        "LLM Backend",
        options=["gemini", "local"],
        format_func=lambda x: "Google Gemini (API)" if x == "gemini" else "Local (LM Studio / OpenAI-compatible)",
        index=1,
    )

    if backend_type == "gemini":
        api_key = st.text_input(
            "API Key (Gemini)",
            type="password",
            placeholder="Enter your GEMINI_API_KEY",
            help="Or set the GEMINI_API_KEY environment variable.",
        )
        model_name = st.text_input(
            "Gemini model",
            value="gemini-2.0-flash-lite",
            placeholder="gemini-2.0-flash-lite",
            help="Stable Lite model (v1beta). Keeps delay to avoid 429.",
        )
        config = {
            "api_key": api_key or None,
            "model_name": model_name.strip() or None,
        }
    else:
        base_url = st.text_input(
            "Base URL (Local)",
            value="http://localhost:1234/v1",
            placeholder="http://localhost:1234/v1",
            help="Base URL of LM Studio or OpenAI-compatible server.",
        )
        model_name = st.text_input(
            "Model name (optional)",
            value="",
            placeholder="local-model",
            help="Or leave blank to use the default.",
        )
        config = {
            "base_url": base_url.strip() or None,
            "model_name": model_name.strip() or None,
        }

    st.divider()
    ror_path = st.text_input(
        "Path to ROR JSON (optional)",
        value="",
        placeholder="v1.74-2025-11-24-ror-data/v1.74-2025-11-24-ror-data.json",
        help="Research Organization Registry JSON file to enrich classification.",
    )
    ror_path = ror_path.strip() or None

# ---------------------------------------------------------------------------
# CSV upload and processing
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV",
    type=["csv"],
    help="Required columns: afid, affiliation, country_code.",
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    required = {"afid", "affiliation", "country_code"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {missing}. Ensure the CSV has: afid, affiliation, country_code.")
        st.stop()

    st.success(f"CSV loaded: **{len(df)}** rows.")

    if st.button("Classify", type="primary"):
        if backend_type == "gemini" and not (config.get("api_key") or __import__("os").environ.get("GEMINI_API_KEY")):
            st.error("For the Gemini backend you need to enter an API key or set GEMINI_API_KEY.")
            st.stop()

        try:
            backend = get_classifier(backend_type, config)
            set_default_backend(backend)
        except Exception as e:
            st.error(f"Error creating backend: {e}")
            st.stop()

        progress_bar = st.progress(0.0, text="Starting classification...")
        status_placeholder = st.empty()

        try:
            status_placeholder.info("Processing batches (rules + LLM)...")
            progress_bar.progress(0.2, text="Classifying affiliations...")

            out_df = run_classification_pipeline(
                df.copy(),
                ror_path=ror_path,
                batch_size=8,
                enable_individual_fallback=False,
            )

            progress_bar.progress(1.0, text="Done.")
            status_placeholder.success("Classification complete.")

            if "sector" not in out_df.columns:
                st.warning("Column 'sector' is not present in the result. The backend may not have returned it.")
            preview_cols = [c for c in OUTPUT_COLUMN_ORDER if c in out_df.columns]
            out_df = out_df[preview_cols]

            st.subheader("Result preview")
            st.dataframe(out_df.head(20), use_container_width=True)

            buffer = io.BytesIO()
            out_df.to_csv(buffer, index=False, encoding="utf-8")
            buffer.seek(0)
            st.download_button(
                label="Download result CSV",
                data=buffer,
                file_name="classified_affiliations.csv",
                mime="text/csv",
            )
        except Exception as e:
            logger.exception("Error during classification")
            progress_bar.progress(1.0, text="Error.")
            status_placeholder.error(f"Error during classification: {e}")
            st.exception(e)

else:
    st.info("Upload a CSV file to get started.")
