# Gov Affiliation Classifier

A Python utility for classifying institutional affiliations using a hybrid approach combining:
- **ROR (Research Organization Registry)** knowledge base for fast matching
- **Rule-based classification** for clear cases (universities, funders, teaching hospitals)
- **LLM classification** for complex or ambiguous cases, via **Local** (LM Studio) or **Cloud** (Google Gemini) backends

The classifier determines:
- **Sector** (`sector`): government, academic, corporate, non_profit, international_organization, or unknown
- **Organization type** (`org_type`): university, research_institute, hospital_clinic, government_agency, military, museum_park_library, company, association_foundation, or unknown
- **Government level** (`gov_level`): federal, state, local, non_applicable, or unknown (primarily for USA; non_applicable for non-government sectors)
- **Local government type** (`gov_local_type`): city, county, other_local, unknown, or non_applicable
- **Research mission category** (`mission_research_category`): NonResearch, AcademicResearch, AppliedResearch, ExperimentalDevelopment, or unknown
- **Research mission flag** (`mission_research`): 0 or 1 (derived from category)

**Note:** This tool is designed for research analysis and does not reproduce official administrative classifications. The LLM returns JSON without a `rationale` field for speed.

## Backends

The classifier supports two backends; choose one at runtime (CLI or Streamlit):

| Backend | Description | Use case |
|--------|-------------|----------|
| **Local (LM Studio)** | OpenAI-compatible API (e.g. LM Studio). Uses `openai` client with `base_url` and optional `model_name`. | Private, offline, or custom models. |
| **Nube (Gemini)** | Google Gemini API via direct HTTP (`requests`). URL: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`. | Google AI Studio; models such as `gemini-2.0-flash-lite`. |

- **Rule of thumb:** Try rule-based classification first; if it returns no result, call the selected LLM backend.
- No API keys are hardcoded: use config, CLI args, or environment variables (`GEMINI_API_KEY`, `LM_STUDIO_BASE_URL`, etc.).

## Features

### 🚀 Fast-Track Classification
The system uses a two-stage approach for efficiency:

1. **Rule-based classification** (fast): For clear cases identified via ROR matching:
   - Universities with education type in ROR
   - Research funding organizations (Enablers)
   - Teaching hospitals with academic links

2. **LLM classification** (when needed): For ambiguous cases, complex organizations, or when ROR data is unavailable. Uses either Local (LM Studio) or Cloud (Gemini) backend.

3. **Robust JSON extraction:** The prompt asks the model to return only a valid JSON object. The client uses a robust extractor that:
   - Strips Markdown fences (e.g. `` ```json ... ``` ``)
   - Finds the first `{ ... }` or `[ ... ]` by matching braces/brackets so that surrounding text or extra text does not break parsing
   - Falls back to regex if brace matching fails, and raises a clear error if no JSON is found

### 📊 Output Fields

The classifier produces a CSV with the following fields:

**Classification fields:**
- `sector`: One of `government`, `academic`, `corporate`, `non_profit`, `international_organization`, `unknown`
- `org_type`: One of `university`, `research_institute`, `hospital_clinic`, `government_agency`, `military`, `museum_park_library`, `company`, `association_foundation`, `unknown`
- `gov_level`: One of `federal`, `state`, `local`, `non_applicable`, `unknown`
- `gov_local_type`: One of `city`, `county`, `other_local`, `unknown`, `non_applicable`
- `mission_research_category`: One of `NonResearch`, `AcademicResearch`, `AppliedResearch`, `ExperimentalDevelopment`, `unknown`
- `mission_research`: Binary (0 or 1), derived from `mission_research_category`
  - `1` if category is `AppliedResearch`, `AcademicResearch`, or `ExperimentalDevelopment`
  - `0` if category is `NonResearch` or `unknown`

**ROR enrichment fields:**
- `ror_id`, `ror_name`, `ror_types`, `ror_country_code`, `ror_state`, `ror_city`, `ror_match_score`, `suggested_org_type_from_ror`

### 📐 Categorization Rules (LLM)

- **Department of...**: Use parent or context to distinguish federal government (e.g. "Dept of Energy, USA") from academic departments (e.g. "Dept of Physics, Harvard"). Government → sector `government`, org_type `government_agency`. Academic → sector `academic`, org_type `university`.
- **Parks / Museums**: "National Park" → `gov_level`: federal. "City Museum" → `gov_level`: local. org_type `museum_park_library`.
- **University extensions / systems**: Always sector `academic`, org_type `university`.
- **Hospitals**: org_type is always `hospital_clinic`; sector can be academic, government, or corporate.
- **ROR match**: Affiliation string takes priority; if the string indicates a different legal entity than the ROR match (e.g. "Inc" vs "University"), ignore the ROR type.

## Current Scope (v1.0)

- **Geographic:** `gov_level` is primarily validated for U.S. affiliations; for other countries, values may be `unknown` or `non_applicable`.
- **Academic use:** This tool is for **research analysis** and does not aim to reproduce official administrative classifications.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Main dependencies (see `requirements.txt` for versions):
   - **pandas** – CSV processing
   - **requests** – HTTP client (used for Gemini API and general requests)
   - **openai** – OpenAI-compatible client (used for Local / LM Studio backend)
   - **python-dotenv** – Environment variable management
   - **rapidfuzz** – Fuzzy matching for ROR
   - **streamlit** – Web UI (optional, for `app.py`)
   - **pydantic** – Schema validation (taxonomy)

3. Ensure you have:
   - For **Local:** An LM Studio (or OpenAI-compatible) server and, if needed, a ROR dump file.
   - For **Gemini:** A Google AI Studio API key (set `GEMINI_API_KEY` or pass via config/UI). No API keys are hardcoded in the codebase.

**Nota (modo local):** Para el modo local, es necesario cargar el modelo en LM Studio y activar el servidor en el puerto 1234.

## Usage

### CLI

**Local backend (default):**
```bash
python main.py --input input.csv --output output.csv
```

**Gemini backend:**
```bash
python main.py --input input.csv --output output.csv --backend gemini --gemini-api-key YOUR_KEY
```
(Or set `GEMINI_API_KEY` in the environment and omit `--gemini-api-key`.)

**Custom ROR path:**
```bash
python main.py --input input.csv --output output.csv --ror-path path/to/ror-data.json
```

### Streamlit UI

```bash
streamlit run app.py
```

In the sidebar: choose **Local** or **Gemini**, set API key (Gemini) or base URL (Local), optional model name and ROR path. Upload a CSV, run classification, and download the result.

### Input CSV Format

Required columns: `afid`, `affiliation`, `country_code` (e.g. `usa`, `gbr`, `fra`).

## Configuration

Environment variables (optional, e.g. in `.env`):

- **Local:** `LM_STUDIO_BASE_URL` (default: `http://localhost:1234/v1`), `LM_STUDIO_MODEL_NAME`, `LM_STUDIO_TIMEOUT`
- **Gemini:** `GEMINI_API_KEY` (required for Gemini backend if not passed via CLI/UI)

No API keys or secrets are hardcoded; always use config, CLI, or environment.

## Troubleshooting

### Error 429 (Google / Gemini – Cuotas)

When using the **Gemini** backend, Google may return **429 Too Many Requests** (rate limit / quota exceeded).

- **Reduce request rate:** The Gemini client uses a **4-second delay** (`time.sleep(4)`) between each individual request to stay under free-tier limits.
- **Use a lighter model:** The default model is `gemini-2.0-flash-lite`, which typically has a more generous free quota. You can change it in the Streamlit sidebar or via config/CLI.
- **Check quota:** In [Google AI Studio](https://aistudio.google.com/) check your project’s quotas and limits. If you hit “limit: 0”, wait or switch to a model with available quota.
- **Retries:** The client retries on transient failures; combined with the 4 s delay, this helps with occasional 429s.

### Local (LM Studio)

- Ensure LM Studio is running and the server URL matches `LM_STUDIO_BASE_URL`.
- If the model name is not the default, set `LM_STUDIO_MODEL_NAME` or pass it via CLI/UI.

### JSON parsing errors

- The robust JSON extractor in `lm_client.py` (`_extract_json_object`) handles fences and surrounding text. If you still see parsing errors, check the model output (e.g. via logs); the prompt instructs the model to return only a valid JSON object.

## Architecture

- **`ror_knowledge.py`:** ROR index and matching.
- **`lm_client.py`:** Rule-based classifier, `LLMBackend` (abstract), `GeminiBackend` (HTTP with `requests`), `LocalBackend` (OpenAI client), robust JSON extraction, normalization.
- **`main.py`:** CLI orchestration; selects backend and runs the classification pipeline.
- **`app.py`:** Streamlit UI; backend selection and configuration in the sidebar.

Classification flow: ROR match → rule-based classification → if no result, call selected LLM backend → normalize and validate response (including robust JSON extraction) → write CSV.

## Dependencies

- `pandas>=2.0` – CSV processing
- `requests>=2.31` – HTTP (Gemini API and general)
- `openai>=1.0.0` – OpenAI-compatible API (Local / LM Studio)
- `python-dotenv>=1.0` – Env loading
- `rapidfuzz>=3.0` – ROR fuzzy matching
- `streamlit>=1.28` – Web UI
- `pydantic>=2.0` – Schemas

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]
