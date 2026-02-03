"""
Clientes para clasificación de afiliaciones mediante LLMs.

Arquitectura híbrida:
- LLMBackend: clase abstracta.
- GeminiBackend: Google Gemini (API) con salida JSON estructurada (response_schema).
- LocalBackend: LM Studio / API compatible con OpenAI, con reintentos para JSON.

Regla de oro: primero se intenta try_rule_based_classification; si devuelve None,
se usa el backend LLM seleccionado.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

import logging
from dotenv import load_dotenv

try:
    from ror_knowledge import RorMatch
except ImportError:
    RorMatch = None  # type: ignore


load_dotenv()

LOGGER = logging.getLogger("gov-affiliation-classifier.lm_client")

# ---------------------------------------------------------------------------
# Prompts (compartidos por todos los backends)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You classify institutional affiliations into a structured taxonomy.

Given:
- an affiliation string,
- a country code (ISO-2), and
- optionally a ROR match,

return ONLY a valid JSON with:

- "org_type": one of ["supranational_organization","government","university","research_institute","company","ngo","hospital","other"]
- "gov_level": ["federal","state","local","unknown","non_applicable"]
- "gov_local_type": ["city","county","other_local","unknown","non_applicable"]
- "mission_research_category": ["NonResearch","Enabler","AppliedResearch","AcademicResearch"]
- "mission_research": 0 or 1
- "rationale": ""

# 1. ORG TYPE RULES

- government = federal, state, or local public entities (ministries, departments, agencies, regulatory bodies, military units).
- university = higher education institutions and systems CONSISTING OF UNIVERSITIES:
  - Single universities, colleges, campuses.
  - Academic departments ("Department of X", "School of Y", "Faculty of Z").
  - Alliances / consortia / systems / networks COMPOSED OF universities or higher education institutions
    (e.g. "University of California System", "Association of American Universities").
- research_institute = organizations primarily dedicated to research (public or private) that are NOT clearly universities.
- company = private for-profit enterprises (Inc, LLC, Corp, Ltd, Technologies, Biosciences, etc.).
- ngo = non-profit organizations, associations, foundations, alliances or networks that are NOT single universities or clear university-only systems.
- hospital = healthcare institutions (hospitals, medical centers, clinics), unless clearly private-company only.
  - Veterans Affairs Medical Centers = government / federal / NonResearch unless explicitly research-focused.
- supranational_organization = UN, EU, WHO, OECD, World Bank, etc.
- other = conceptual or non-institutional entities, or ambiguous cases that do not fit the above types.

Academic departments:
- If the affiliation is a department, school, faculty, college or laboratory clearly associated with a university,
  classify org_type="university" unless there is explicit government context (e.g. "Department of Health, State of X").

"Department of X" pattern - CRITICAL RULE:
- If the affiliation is "Department of X" (or similar patterns like "Department of X, Y" where Y is not a government entity):
  - If X corresponds to an ACADEMIC DISCIPLINE (e.g., Philosophy, Computer Science, Mathematics, Physics, Chemistry, Biology, History, Literature, Educational Sciences, Psychology, Sociology, Economics, etc.), 
    → classify as org_type="university" and mission_research_category="AcademicResearch", 
    EVEN IF the word "University" does not appear in the affiliation string.
  - If X corresponds to an ADMINISTRATIVE or MINISTERIAL FUNCTION (e.g., Energy, Education, Health, Transportation, Defense, Commerce, Labor, Agriculture, Interior, Justice, Treasury, etc.), 
    → classify as org_type="government" and mission_research_category="NonResearch".
  - This rule applies even when the affiliation is just "Department of X" without additional context.

# 2. SPECIAL PATTERNS

Military entities (Air Force, Army, Navy, DoD):
→ org_type="government", gov_level="federal".

University consortia, alliances or systems:
→ If they are alliances / consortia / systems composed of universities or higher education institutions,
   classify as org_type="university", NOT as ngo.

Labs and Centers:
- If clearly linked to a university (e.g. "Center for X, University of Y", "Institute for Z at Harvard University")
  → org_type="university".
- If part of government agencies → org_type="government".
- If clearly corporate (Inc, LLC, Technologies, Biosciences, Labs, Pharmaceuticals, etc.) → org_type="company".
- If independent and primarily research, not clearly a university → org_type="research_institute".

Museums / Heritage / Preservation:
- Usually ngo, unless there is explicit government ownership (City/County/State/National Museum),
  in which case org_type="government".

# 3. GOV LEVEL

- gov_level="federal" for national agencies (NIH, CDC, EPA, USGS, NOAA, VA hospitals, DoD units, national ministries).
- gov_level="state" for state-level agencies, state departments, state universities when they are clearly state public bodies.
- gov_level="local" for city/county/town departments:
  - If city-level → gov_local_type="city".
  - If county-level → gov_local_type="county".
  - If other local public body → gov_local_type="other_local".
- If org_type="government" and the level is unclear → gov_level="unknown", gov_local_type="unknown".

**IMPORTANT CONSTRAINT (E):**
If org_type != "government", you MUST set:
  - gov_level = "non_applicable"
  - gov_local_type = "non_applicable"

Never assign "federal", "state", "local" or "unknown" when org_type is not "government".

# 4. MISSION RESEARCH CATEGORY

Choose exactly one mission_research_category:

- "NonResearch":
  - Administrative units, municipal agencies, service providers.
  - Hospitals without explicit teaching or research role.
  - Purely operational government departments.
- "Enabler":
  - Funding agencies, philanthropic foundations, funders, research councils,
    or organizations whose main role is to enable / finance research rather than perform it.
- "AppliedResearch":
  - Government labs, mission-oriented R&D units, analytics divisions.
  - Corporate R&D centers, industrial labs, applied technology organizations.
- "AcademicResearch":
  - Universities, academic departments, graduate schools.
  - Public or private research institutes with a primarily academic research mission.
  - Academic medical centers and teaching hospitals with strong research activity.

Default mapping (without considering ROR):
- mission_research = 1 if mission_research_category in ["AppliedResearch","AcademicResearch"].
- mission_research = 0 if mission_research_category in ["NonResearch","Enabler"].

# 5. USE OF ROR (IF PROVIDED)

If a valid ROR record is provided (ror_id is not empty), you MUST use it as a strong prior:

- ROR types:
  - "Education": strong cue for org_type="university".
  - "Government": strong cue for org_type="government".
  - "Healthcare": strong cue for org_type="hospital" (or university if clearly a teaching hospital).
  - "Nonprofit": strong cue for org_type="ngo".
  - "Facility", "Research Institute": strong cue for org_type="research_institute".
  - "Company": strong cue for org_type="company".
  - "Funder": strong cue for mission_research_category="Enabler".

ROR and research mission (NEW RULE):
- If THERE IS a valid ROR match:
  - You MUST assume the organization is part of the research ecosystem.
  - mission_research MUST be 1.
  - mission_research_category MUST NOT be "NonResearch".
  - You MUST choose among ["AcademicResearch","AppliedResearch","Enabler"] based on the role:
    - Universities, research institutes, academic hospitals → "AcademicResearch".
    - Government or corporate research labs → "AppliedResearch".
    - Funders, research councils, philanthropic funders → "Enabler".

If there is NO ROR match:
- Use only the affiliation text and country context to infer org_type, gov_level, and mission_research_category.
- In this case, you MAY use "NonResearch" if appropriate.

ROR can be overridden by very strong evidence in the affiliation string,
but in most cases you should align with ROR type and domain.

# 6. OUTPUT FORMAT

Return ONLY a JSON object with all required keys:
- "org_type"
- "gov_level"
- "gov_local_type"
- "mission_research_category"
- "mission_research"
- "rationale"

"rationale" MUST be a string (can be short). No extra text, no explanations outside the JSON.
"""

SYSTEM_PROMPT_BATCH = """\
You classify multiple institutional affiliations into a structured taxonomy.

Given an array of items, each with:
- an id (unique identifier),
- an affiliation string,
- a country code (ISO-2), and
- optionally a ROR match,

return ONLY a valid JSON array with the same number of items, in the EXACT SAME ORDER as the input.

Each item in the output array must be a JSON object with:

- "id": the same id from the input
- "org_type": one of ["supranational_organization","government","university","research_institute","company","ngo","hospital","other"]
- "gov_level": ["federal","state","local","unknown","non_applicable"]
- "gov_local_type": ["city","county","other_local","unknown","non_applicable"]
- "mission_research_category": ["NonResearch","Enabler","AppliedResearch","AcademicResearch"]
- "mission_research": 0 or 1
- "rationale": ""

Follow the same rules as the single-item classification prompt for each item.

CRITICAL: You MUST return a JSON array, not a single object. The array must have exactly the same length and order as the input.
"""


# ---------------------------------------------------------------------------
# JSON schema para Gemini (response_schema)
# ---------------------------------------------------------------------------

GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "org_type": {"type": "string", "enum": ["supranational_organization", "government", "university", "research_institute", "company", "ngo", "hospital", "other"]},
        "gov_level": {"type": "string", "enum": ["federal", "state", "local", "unknown", "non_applicable"]},
        "gov_local_type": {"type": "string", "enum": ["city", "county", "other_local", "unknown", "non_applicable"]},
        "mission_research_category": {"type": "string", "enum": ["NonResearch", "Enabler", "AppliedResearch", "AcademicResearch"]},
        "mission_research": {"type": "integer", "enum": [0, 1]},
        "rationale": {"type": "string"},
    },
    "required": ["org_type", "gov_level", "gov_local_type", "mission_research_category", "mission_research", "rationale"],
}


# ---------------------------------------------------------------------------
# Excepciones y utilidades
# ---------------------------------------------------------------------------

class LMStudioError(RuntimeError):
    """Raised when the LLM request fails or returns invalid data."""


def _is_technical_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a technical error (network, timeout, etc.)
    that should trigger retry/fallback, vs semantic uncertainty.
    """
    try:
        import requests
        if isinstance(exception, (requests.exceptions.ConnectionError,
                                  requests.exceptions.Timeout,
                                  requests.exceptions.RequestException)):
            return True
    except ImportError:
        pass

    if isinstance(exception, LMStudioError):
        error_str = str(exception).lower()
        technical_indicators = ["timeout", "connection", "network", "status_code", "invalid response format"]
        if any(indicator in error_str for indicator in technical_indicators):
            return True

    return False


def _normalize_to_unknown(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a result dict: replace missing/invalid values with 'unknown' or 'non_applicable'."""
    valid_org_types = {
        "supranational_organization", "government", "university", "research_institute",
        "company", "ngo", "hospital", "other", "unknown",
    }
    valid_gov_levels = {"federal", "state", "local", "unknown", "non_applicable"}
    valid_gov_local_types = {"city", "county", "other_local", "unknown", "non_applicable"}
    valid_mission_categories = {"NonResearch", "Enabler", "AppliedResearch", "AcademicResearch", "unknown"}

    org_type = result_dict.get("org_type")
    if not org_type or org_type not in valid_org_types:
        org_type = "unknown"
        result_dict["org_type"] = org_type

    gov_level = result_dict.get("gov_level")
    if not gov_level or gov_level not in valid_gov_levels:
        gov_level = "unknown" if org_type == "government" else "non_applicable"
        result_dict["gov_level"] = gov_level

    gov_local_type = result_dict.get("gov_local_type")
    if not gov_local_type or gov_local_type not in valid_gov_local_types:
        gov_local_type = "unknown" if (org_type == "government" and gov_level == "local") else "non_applicable"
        result_dict["gov_local_type"] = gov_local_type

    mission_category = result_dict.get("mission_research_category")
    if not mission_category or mission_category not in valid_mission_categories:
        mission_category = "unknown"
        result_dict["mission_research_category"] = mission_category

    if mission_category in {"AppliedResearch", "AcademicResearch"}:
        mission_research = 1
    else:
        mission_research = 0
    result_dict["mission_research"] = mission_research

    if "rationale" not in result_dict:
        result_dict["rationale"] = ""

    return result_dict


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object or array from a text response, stripping ``` fences."""
    if not text:
        raise ValueError("Empty response from LLM")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    match_arr = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    match_obj = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match_arr:
        return match_arr.group(0).strip()
    if match_obj:
        return match_obj.group(0).strip()
    return cleaned


def _parsed_to_result_dict(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parsed LLM response to normalized result dict (same shape for all backends)."""
    parsed = _normalize_to_unknown(parsed.copy())
    org_type = parsed.get("org_type")
    if org_type != "government":
        parsed["gov_level"] = "non_applicable"
        parsed["gov_local_type"] = "non_applicable"
    elif parsed.get("gov_level") != "local":
        parsed["gov_local_type"] = "non_applicable"

    for key in ("confidence_org_type", "confidence_gov_level", "confidence_mission_research"):
        if key not in parsed:
            parsed[key] = None
    return parsed


# ---------------------------------------------------------------------------
# Regla basada en reglas (conservada)
# ---------------------------------------------------------------------------

def try_rule_based_classification(
    affiliation: str, country_code: str | None, ror_info: dict | None
) -> dict | None:
    """
    Intenta clasificar una afiliación con reglas conservadoras usando ROR.

    Si las reglas no aplican o el caso es ambiguo, devuelve None (usar LLM).
    """
    if ror_info is None:
        return None

    ror_types = ror_info.get("ror_types", [])
    ror_name = ror_info.get("ror_name", "")
    affiliation_lower = affiliation.lower()
    ror_name_lower = ror_name.lower() if ror_name else ""
    combined_text = f"{affiliation_lower} {ror_name_lower}".strip()

    # Rule A: Clear universities
    if "education" in [t.lower() for t in ror_types]:
        university_patterns = [
            "university", "college", "institute of technology", "polytechnic",
            "school of", "universidad", "université", "universität", "universita",
        ]
        exclusion_patterns = [
            "consortium", "association", "alliance", "coalition", "council of", "network of",
        ]
        if any(p in combined_text for p in university_patterns) and not any(p in combined_text for p in exclusion_patterns):
            return {
                "org_type": "university",
                "gov_level": "non_applicable",
                "gov_local_type": "non_applicable",
                "mission_research_category": "AcademicResearch",
                "mission_research": 1,
                "confidence_org_type": 0.95,
                "confidence_gov_level": 0.95,
                "confidence_mission_research": 0.95,
                "rationale": "",
            }

    # Rule B: Clear funders (Enabler)
    if "funder" in [t.lower() for t in ror_types]:
        funder_patterns = [
            "foundation", "research council", "funding", "science foundation",
            "national science", "research fund", "grant", "funder",
        ]
        if any(p in combined_text for p in funder_patterns):
            gov_patterns = ["national science", "national research", "ministry", "department", "agency", "government"]
            is_government = any(p in combined_text for p in gov_patterns)
            federal_patterns = ["national", "federal", "u.s.", "us "]
            is_federal = any(p in combined_text for p in federal_patterns)
            gov_level = "federal" if (is_government and is_federal) else ("unknown" if is_government else "non_applicable")
            return {
                "org_type": "government" if is_government else "ngo",
                "gov_level": gov_level,
                "gov_local_type": "non_applicable" if gov_level != "local" else "unknown",
                "mission_research_category": "Enabler",
                "mission_research": 0,
                "confidence_org_type": 0.9,
                "confidence_gov_level": 0.8,
                "confidence_mission_research": 0.95,
                "rationale": "",
            }

    # Rule C: Clear teaching hospitals
    if "healthcare" in [t.lower() for t in ror_types]:
        teaching_hospital_patterns = [
            "university hospital", "academic medical center", "teaching hospital",
            "medical center university", "university medical", "academic hospital",
        ]
        if any(p in combined_text for p in teaching_hospital_patterns):
            return {
                "org_type": "hospital",
                "gov_level": "non_applicable",
                "gov_local_type": "non_applicable",
                "mission_research_category": "AcademicResearch",
                "mission_research": 1,
                "confidence_org_type": 0.9,
                "confidence_gov_level": 0.95,
                "confidence_mission_research": 0.9,
                "rationale": "",
            }

    return None


# ---------------------------------------------------------------------------
# Backend abstracto e implementaciones
# ---------------------------------------------------------------------------

BackendConfig = Dict[str, Any]


class LLMBackend(ABC):
    """Backend abstracto para clasificación mediante LLM."""

    @abstractmethod
    def classify_affiliation(
        self,
        affiliation: str,
        country_code: str,
        ror_match: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Clasifica una afiliación. Devuelve dict con org_type, gov_level, etc."""
        ...

    @abstractmethod
    def classify_affiliations_batch(
        self,
        items: List[Dict[str, Any]],
        max_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        """Clasifica varias afiliaciones en una sola llamada. Mismo orden que items."""
        ...


class GeminiBackend(LLMBackend):
    """Backend usando Google Gemini (API) con salida JSON estructurada (response_schema)."""

    def __init__(self, config: BackendConfig):
        self.api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini backend requires api_key in config or GEMINI_API_KEY in environment")
        self.model_name = config.get("model_name", "gemini-1.5-flash")
        self._client = None

    def _get_model(self):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        try:
            config = genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=256,
                response_mime_type="application/json",
                response_schema=GEMINI_RESPONSE_SCHEMA,
            )
        except (TypeError, AttributeError):
            # Algunas versiones no soportan response_schema; usar solo JSON
            config = genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=256,
                response_mime_type="application/json",
            )
        return genai.GenerativeModel(self.model_name, generation_config=config)

    def classify_affiliation(
        self,
        affiliation: str,
        country_code: str,
        ror_match: Optional[Any] = None,
    ) -> Dict[str, Any]:
        user_data: Dict[str, Any] = {
            "affiliation": affiliation.strip(),
            "country_code": country_code.strip(),
        }
        if ror_match is not None and RorMatch is not None:
            user_data["ror_match"] = {
                "ror_id": ror_match.ror_id,
                "ror_name": ror_match.ror_name,
                "ror_types": ror_match.ror_types,
                "ror_country_code": ror_match.ror_country_code,
                "ror_state": ror_match.ror_state,
                "ror_city": ror_match.ror_city,
                "ror_domains": ror_match.ror_domains,
                "match_score": ror_match.match_score,
            }
            if getattr(ror_match, "suggested_org_type_from_ror", None):
                user_data["ror_match"]["suggested_org_type_from_ror"] = ror_match.suggested_org_type_from_ror

        model = self._get_model()
        prompt = f"{SYSTEM_PROMPT}\n\nInput:\n{json.dumps(user_data)}"
        response = model.generate_content(prompt)
        text = response.text if hasattr(response, "text") else (response.candidates[0].content.parts[0].text if response.candidates else "")
        if not text:
            raise LMStudioError("Gemini returned empty response")
        try:
            json_str = _extract_json_object(text)
            parsed = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            raise LMStudioError(f"Gemini returned invalid JSON: {e}") from e
        return _parsed_to_result_dict(parsed)

    def classify_affiliations_batch(
        self,
        items: List[Dict[str, Any]],
        max_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []

        batch_data = []
        for item in items:
            entry: Dict[str, Any] = {
                "id": item["id"],
                "affiliation": str(item["affiliation"]).strip(),
                "country_code": str(item.get("country_code", "")).strip(),
            }
            if "ror_match" in item and item["ror_match"] is not None:
                ror = item["ror_match"]
                entry["ror_match"] = {
                    "ror_id": ror.ror_id,
                    "ror_name": ror.ror_name,
                    "ror_types": ror.ror_types,
                    "ror_country_code": ror.ror_country_code,
                    "ror_state": ror.ror_state,
                    "ror_city": ror.ror_city,
                    "ror_domains": ror.ror_domains,
                    "match_score": ror.match_score,
                }
                if getattr(ror, "suggested_org_type_from_ror", None):
                    entry["ror_match"]["suggested_org_type_from_ror"] = ror.suggested_org_type_from_ror
            batch_data.append(entry)

        model = self._get_model()
        # Gemini with response_schema is single-object; for batch we ask for array in prompt and parse
        prompt = f"{SYSTEM_PROMPT_BATCH}\n\nInput:\n{json.dumps(batch_data)}"
        # For batch, we don't use response_schema (array); we use plain JSON and extract
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model_batch = genai.GenerativeModel(
            self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        response = model_batch.generate_content(prompt)
        text = response.text if hasattr(response, "text") else (response.candidates[0].content.parts[0].text if response.candidates else "")
        if not text:
            raise LMStudioError("Gemini batch returned empty response")
        json_str = _extract_json_object(text)
        parsed_array = json.loads(json_str)

        if not isinstance(parsed_array, list):
            raise LMStudioError(f"Expected JSON array, got {type(parsed_array)}")

        if len(parsed_array) < len(items):
            parsed_array.extend([{}] * (len(items) - len(parsed_array)))
        elif len(parsed_array) > len(items):
            parsed_array = parsed_array[:len(items)]

        results: List[Dict[str, Any]] = []
        for idx, result_item in enumerate(parsed_array):
            try:
                normalized = _parsed_to_result_dict(result_item.copy() if result_item else {})
                normalized["id"] = items[idx]["id"]
                results.append(normalized)
            except Exception as exc:
                LOGGER.warning("Failed to normalize batch item %d: %s", idx, exc)
                results.append({
                    "id": items[idx]["id"],
                    "org_type": "unknown",
                    "gov_level": "unknown",
                    "gov_local_type": "unknown",
                    "mission_research_category": "unknown",
                    "mission_research": 0,
                    "confidence_org_type": None,
                    "confidence_gov_level": None,
                    "confidence_mission_research": None,
                    "rationale": str(exc),
                })
        return results


class LocalBackend(LLMBackend):
    """Backend OpenAI-compatible (LM Studio, etc.) con reintentos para JSON."""

    def __init__(self, config: BackendConfig):
        self.base_url = config.get("base_url") or os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        self.model_name = config.get("model_name") or os.environ.get("LM_STUDIO_MODEL_NAME", "local-model")
        self.timeout = float(config.get("timeout") or os.environ.get("LM_STUDIO_TIMEOUT", "60"))
        self.max_retries = int(config.get("max_retries", 2))

    def _get_client(self):
        from openai import OpenAI
        return OpenAI(base_url=self.base_url, api_key="lm-studio")

    def classify_affiliation(
        self,
        affiliation: str,
        country_code: str,
        ror_match: Optional[Any] = None,
    ) -> Dict[str, Any]:
        user_data: Dict[str, Any] = {
            "affiliation": affiliation.strip(),
            "country_code": country_code.strip(),
        }
        if ror_match is not None and RorMatch is not None:
            user_data["ror_match"] = {
                "ror_id": ror_match.ror_id,
                "ror_name": ror_match.ror_name,
                "ror_types": ror_match.ror_types,
                "ror_country_code": ror_match.ror_country_code,
                "ror_state": ror_match.ror_state,
                "ror_city": ror_match.ror_city,
                "ror_domains": ror_match.ror_domains,
                "match_score": ror_match.match_score,
            }
            if getattr(ror_match, "suggested_org_type_from_ror", None):
                user_data["ror_match"]["suggested_org_type_from_ror"] = ror_match.suggested_org_type_from_ror

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_data)},
        ]
        client = self._get_client()
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256,
                    timeout=min(600, self.timeout * 2),
                )
                content = resp.choices[0].message.content
                json_str = _extract_json_object(content)
                parsed = json.loads(json_str)
                return _parsed_to_result_dict(parsed)
            except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise LMStudioError(f"Invalid JSON response after {self.max_retries + 1} attempts: {e}") from e
        raise LMStudioError("Unreachable")

    def classify_affiliations_batch(
        self,
        items: List[Dict[str, Any]],
        max_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []

        batch_data = []
        for item in items:
            entry: Dict[str, Any] = {
                "id": item["id"],
                "affiliation": str(item["affiliation"]).strip(),
                "country_code": str(item.get("country_code", "")).strip(),
            }
            if "ror_match" in item and item["ror_match"] is not None:
                ror = item["ror_match"]
                entry["ror_match"] = {
                    "ror_id": ror.ror_id,
                    "ror_name": ror.ror_name,
                    "ror_types": ror.ror_types,
                    "ror_country_code": ror.ror_country_code,
                    "ror_state": ror.ror_state,
                    "ror_city": ror.ror_city,
                    "ror_domains": ror.ror_domains,
                    "match_score": ror.match_score,
                }
                if getattr(ror, "suggested_org_type_from_ror", None):
                    entry["ror_match"]["suggested_org_type_from_ror"] = ror.suggested_org_type_from_ror
            batch_data.append(entry)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BATCH},
            {"role": "user", "content": json.dumps(batch_data)},
        ]
        client = self._get_client()

        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=5000,
                    timeout=600,
                )
                content = resp.choices[0].message.content
                json_str = _extract_json_object(content)
                parsed_array = json.loads(json_str)
                break
            except (json.JSONDecodeError, ValueError, KeyError, IndexError):
                if len(items) > 1 and attempt < max_retries:
                    mid = len(items) // 2
                    first = self.classify_affiliations_batch(items[:mid], max_retries - 1)
                    second = self.classify_affiliations_batch(items[mid:], max_retries - 1)
                    return first + second
                raise LMStudioError(f"Invalid batch response after {attempt + 1} attempts") from None

        if not isinstance(parsed_array, list):
            raise LMStudioError(f"Expected JSON array, got {type(parsed_array)}")

        if len(parsed_array) < len(items):
            parsed_array.extend([{}] * (len(items) - len(parsed_array)))
        elif len(parsed_array) > len(items):
            parsed_array = parsed_array[:len(items)]

        results: List[Dict[str, Any]] = []
        for idx, result_item in enumerate(parsed_array):
            try:
                normalized = _parsed_to_result_dict(result_item.copy() if result_item else {})
                normalized["id"] = items[idx]["id"]
                results.append(normalized)
            except Exception as exc:
                LOGGER.warning("Failed to normalize batch item %d: %s", idx, exc)
                results.append({
                    "id": items[idx]["id"],
                    "org_type": "unknown",
                    "gov_level": "unknown",
                    "gov_local_type": "unknown",
                    "mission_research_category": "unknown",
                    "mission_research": 0,
                    "confidence_org_type": None,
                    "confidence_gov_level": None,
                    "confidence_mission_research": None,
                    "rationale": str(exc),
                })
        return results


def get_classifier(
    backend_type: Literal["gemini", "local"],
    config: Optional[BackendConfig] = None,
) -> LLMBackend:
    """
    Factory: devuelve la instancia del backend solicitado.

    - backend_type: "gemini" | "local"
    - config: dict con claves según backend:
      - gemini: api_key, model_name (opcional)
      - local: base_url, model_name, timeout, max_retries (opcionales)
    """
    config = config or {}
    if backend_type == "gemini":
        return GeminiBackend(config)
    if backend_type == "local":
        return LocalBackend(config)
    raise ValueError(f"Unknown backend_type: {backend_type!r}. Use 'gemini' or 'local'.")


# ---------------------------------------------------------------------------
# API de conveniencia (usa backend por defecto para compatibilidad con main.py)
# ---------------------------------------------------------------------------

_default_backend: Optional[LLMBackend] = None


def set_default_backend(backend: LLMBackend) -> None:
    """Establece el backend por defecto (para main.py y scripts)."""
    global _default_backend
    _default_backend = backend


def get_default_backend() -> Optional[LLMBackend]:
    return _default_backend


def classify_affiliation(
    affiliation: str,
    country_code: str,
    ror_match: Optional[Any] = None,
    backend: Optional[LLMBackend] = None,
) -> Dict[str, Any]:
    """
    Clasifica una afiliación usando el backend indicado o el por defecto.
    """
    b = backend or _default_backend
    if b is None:
        b = LocalBackend({})  # fallback: LM Studio por defecto
    return b.classify_affiliation(affiliation, country_code, ror_match)


def classify_affiliations_batch(
    items: List[Dict[str, Any]],
    max_retries: int = 2,
    backend: Optional[LLMBackend] = None,
) -> List[Dict[str, Any]]:
    """
    Clasifica varias afiliaciones en batch usando el backend indicado o el por defecto.
    """
    b = backend or _default_backend
    if b is None:
        b = LocalBackend({})
    return b.classify_affiliations_batch(items, max_retries)
