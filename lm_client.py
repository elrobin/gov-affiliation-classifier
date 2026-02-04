"""
Clientes para clasificación de afiliaciones mediante LLMs.

- LLMBackend: clase abstracta.
- GeminiBackend: Gemini vía HTTP directo (requests), formato Google generateContent.
- LocalBackend: LM Studio vía librería openai (chat completions); 100% compatible, sin requests.

classify_affiliation / classify_affiliations_batch usan el backend seleccionado (get_classifier + set_default_backend).
Regla de oro: primero try_rule_based_classification; si devuelve None, se usa el LLM.
"""
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

import logging
import requests
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
You are an expert classifier of institutional affiliations. Your task is to analyze an input affiliation string and classify it into a strict taxonomy.

Input provided:
1. Affiliation String: The raw name of the entity.
2. Country Code: ISO-2 code (context for government levels).
3. ROR Match (Optional): A potential match from the Research Organization Registry.
   - WARNING: The ROR match might be imprecise (e.g., matching "Northwestern Inc" to "Northwestern University").
   - USE OF ROR (IF PROVIDED): If a valid ROR record is provided (ror_id is not empty), you may use it as a strong prior:
     - ROR types:
       - "Education": strong cue for org_type="university".
       - "Government": strong cue for org_type="government_agency".
       - "Healthcare": strong cue for org_type="hospital_clinic" (or university if clearly a teaching hospital).
       - "Nonprofit": strong cue for org_type="association_foundation".
       - "Facility", "Research Institute": strong cue for org_type="research_institute".
       - "Company": strong cue for org_type="company".
       - "Funder": strong cue for mission_research_category="Enabler".
     - If THERE IS a valid ROR match: You MUST assume the organization is part of the research ecosystem. mission_research_category MUST NOT be "NonResearch". You MUST choose among ["AcademicResearch","AppliedResearch","Enabler"] based on the role: 1) Universities, research institutes, academic hospitals → "AcademicResearch"; 2) Government or corporate research labs → "AppliedResearch"; 3) Funders, research councils, philanthropic funders → "Enabler".
   - CRITICAL RULE: Prioritize the Affiliation String. If the string explicitly indicates a different legal entity than the ROR match (e.g., "Inc" vs "University"), ignore the ROR match type.

Output Schema (JSON only):
Return a single JSON object with these exact keys:

1. "sector": Choose ONE from:
   - "government": Public sector, ministries, armed forces, city councils, public agencies.
   - "academic": Universities, colleges, polytechnics, schools (public or private).
   - "corporate": For-profit companies, Inc, LLC, Ltd.
   - "non_profit": NGOs, foundations, associations, charities (NOT hospitals/universities).
   - "international_organization": Supranational (UN, WHO, NATO, EU).
   - "unknown": Cannot determine.

2. "org_type": Choose ONE from:
   - "university": Degree-granting institutions. INCLUDES: Teaching depts ("Dept of History"), "Cooperative Extensions", "University Systems".
   - "research_institute": Labs, research centers, observatories, think tanks (NOT teaching depts).
   - "hospital_clinic": Hospitals, medical centers (including VA/Military hospitals).
   - "government_agency": Ministries, executive departments, bureaus. (Ex: "Dept of Energy").
   - "military": Combat units, bases (if NOT a hospital or research lab).
   - "museum_park_library": Museums, archives, national/state parks, libraries.
   - "company": General for-profit entities.
   - "association_foundation": Legal entity for NGOs/foundations.
   - "unknown": Other or unclear.

3. "gov_level": Choose ONE from:
   - "federal": National/Central (e.g., "Department of Defense", "National Institutes of Health").
   - "state": State level.
   - "local": City, municipality, county.
   - "non_applicable": For private companies, NGOs, or universities (unless military academies).
   - "unknown": Ambiguous.

4. "mission_research_category": Choose ONE from:
   - "NonResearch": Administrative units, municipal agencies, service providers; Hospitals without explicit teaching or research role; Purely operational government departments.
   - "AcademicResearch": Universities, academic departments, graduate schools; Public or private research institutes with a primarily academic research mission; Academic medical centers and teaching hospitals with strong research activity.
   - "AppliedResearch": Government labs, mission-oriented R&D units, analytics divisions; Corporate R&D centers, industrial labs, applied technology organizations.
   - "ExperimentalDevelopment": Corporate R&D, product development.
   - "Enabler": Funding agencies, philanthropic foundations, funders, research councils or organizations whose main role is to enable / finance research rather than perform it.
   - "unknown": Cannot determine.

5. "gov_local_type": Use "non_applicable" unless sector is government and gov_level is local (then "city", "county", or "other_local").

CRITICAL INSTRUCTIONS FOR AMBIGUITY:
- "Department of...": Check the parent or knowledge base to distinguish between federal government and academic departments. "Dept of Physics, Harvard" -> Sector: Academic. "Dept of Energy, USA" -> Sector: Government.
- "Parks/Museums": "National Park" -> gov_level: federal. "City Museum" -> gov_level: local.
- "University Extensions": ALWAYS Sector: Academic / Org_Type: University.
- "Hospitals": Sector can be Academic, Government, or Corporate. Org_Type is ALWAYS "hospital_clinic". It includes Medical centres and clinics.

**IMPORTANT CONSTRAINT (E):**
If sector is not "government", you MUST set:
  - gov_level = "non_applicable"
  - gov_local_type = "non_applicable"
Never assign "federal", "state", "local" or "unknown" when sector is not "government".

EXAMPLES (Follow these patterns):

Ex 1: Gov Dept vs Academic Dept (Contrast)
Input: "Department of Energy", Country: "US"
JSON: {"sector": "government", "org_type": "government_agency", "gov_level": "federal", "mission_research_category": "NonResearch"}
Input: "Department of Philosophy", Country: "US"
JSON: {"sector": "academic", "org_type": "university", "gov_level": "non_applicable", "mission_research_category": "AcademicResearch"}

Ex 2: Gov Defense Agency (Critical)
Input: "United States Department of Defense, Military Health System", Country: "US"
JSON: {"sector": "government", "org_type": "government_agency", "gov_level": "federal", "mission_research_category": "NonResearch"}

Ex 3: Research Centers (Gov vs Corp Contrast)
Input: "Albany Research Center", Country: "US"
JSON: {"sector": "government", "org_type": "research_institute", "gov_level": "federal", "mission_research_category": "AppliedResearch"}
(Note: Generic 'Research Centers' without 'Inc' are often Gov/Applied).
Input: "Pfizer Global Research and Development", Country: "US"
JSON: {"sector": "corporate", "org_type": "company", "gov_level": "non_applicable", "mission_research_category": "ExperimentalDevelopment"}
(Note: Corporate names imply 'company', even with 'Research' in title).

Ex 4: Hospitals (Military vs Academic)
Input: "Corporal Michael J. Crescenz VAMC", Country: "US"
JSON: {"sector": "government", "org_type": "hospital_clinic", "gov_level": "federal", "mission_research_category": "NonResearch"}
Input: "University of Chicago Medical Center", Country: "US"
JSON: {"sector": "academic", "org_type": "hospital_clinic", "gov_level": "non_applicable", "mission_research_category": "AcademicResearch"}

Ex 5: University Extension
Input: "University of Maine Cooperative Extension", Country: "US"
JSON: {"sector": "academic", "org_type": "university", "gov_level": "non_applicable", "mission_research_category": "AcademicResearch"}

NO "rationale" field is needed. JSON ONLY.
"""

SYSTEM_PROMPT_BATCH = """\
You classify multiple institutional affiliations into the same strict taxonomy as the single-item prompt.

Input: a JSON array of items, each with id, affiliation string, country code (ISO-2), and optional ROR match.

Output: a JSON array with the SAME number of items, in the EXACT SAME ORDER. Each item must be a JSON object with:
- "id": the same id from the input
- "sector": one of ["government","academic","corporate","non_profit","international_organization","unknown"]
- "org_type": one of ["university","research_institute","hospital_clinic","government_agency","military","museum_park_library","company","association_foundation","unknown"]
- "gov_level": one of ["federal","state","local","non_applicable","unknown"]
- "mission_research_category": one of ["NonResearch","AcademicResearch","AppliedResearch","ExperimentalDevelopment","unknown"]

Apply the same CRITICAL rules: Department of... (parent/government vs academic), Parks/Museums (federal vs local), University Extensions (always academic/university), Hospitals (org_type always hospital_clinic). Prioritize affiliation string over ROR when they conflict. NO "rationale" field. JSON array ONLY.
"""


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
    valid_sectors = {"government", "academic", "corporate", "non_profit", "international_organization", "unknown"}
    valid_org_types = {
        "university", "research_institute", "hospital_clinic", "government_agency", "military",
        "museum_park_library", "company", "association_foundation", "unknown",
    }
    valid_gov_levels = {"federal", "state", "local", "unknown", "non_applicable"}
    valid_gov_local_types = {"city", "county", "other_local", "unknown", "non_applicable"}
    valid_mission_categories = {"NonResearch", "Enabler", "AppliedResearch", "AcademicResearch", "ExperimentalDevelopment", "unknown"}

    sector = result_dict.get("sector")
    if not sector or sector not in valid_sectors:
        result_dict["sector"] = "unknown"
    else:
        result_dict["sector"] = sector

    org_type = result_dict.get("org_type")
    if not org_type or org_type not in valid_org_types:
        org_type = "unknown"
        result_dict["org_type"] = org_type

    gov_level = result_dict.get("gov_level")
    if not gov_level or gov_level not in valid_gov_levels:
        gov_level = "unknown" if sector == "government" else "non_applicable"
        result_dict["gov_level"] = gov_level

    gov_local_type = result_dict.get("gov_local_type")
    if not gov_local_type or gov_local_type not in valid_gov_local_types:
        gov_local_type = "unknown" if (sector == "government" and gov_level == "local") else "non_applicable"
        result_dict["gov_local_type"] = gov_local_type

    mission_category = result_dict.get("mission_research_category")
    if not mission_category or mission_category not in valid_mission_categories:
        mission_category = "unknown"
        result_dict["mission_research_category"] = mission_category

    if mission_category in {"AppliedResearch", "AcademicResearch", "ExperimentalDevelopment"}:
        mission_research = 1
    else:
        mission_research = 0
    result_dict["mission_research"] = mission_research

    return result_dict


def _extract_json_object(text: str) -> str:
    """
    Extrae el primer objeto {} o array [] JSON del texto, aunque haya texto alrededor o cercas ```.
    Prioriza el carácter que abre primero: si el JSON empieza por [, extrae array; si por {, extrae objeto.
    Así "[{...}, {...}]" se devuelve completo como array, no solo el primer objeto.
    """
    if not text or not text.strip():
        raise ValueError("Empty response from LLM")

    cleaned = text.strip()
    # Quitar cercas ```json ... ``` o ``` ... ```
    if "```" in cleaned:
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        cleaned = re.sub(r"```\s*", "\n", cleaned).strip()

    # Encontrar el primer { o [ para decidir si extraemos objeto o array
    first_brace = cleaned.find("{")
    first_bracket = cleaned.find("[")

    def extract_object(start: int) -> str:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
        match = re.search(r"\{.*\}", cleaned[start:], flags=re.DOTALL)
        return match.group(0).strip() if match else ""

    def extract_array(start: int) -> str:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "[":
                depth += 1
            elif cleaned[i] == "]":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
        match = re.search(r"\[.*\]", cleaned[start:], flags=re.DOTALL)
        return match.group(0).strip() if match else ""

    # Si el JSON empieza por [, extraer array (para batch); si por {, extraer objeto
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        out = extract_array(first_bracket)
        if out:
            return out
    if first_brace != -1:
        out = extract_object(first_brace)
        if out:
            return out
    if first_bracket != -1:
        out = extract_array(first_bracket)
        if out:
            return out

    raise ValueError("No JSON object or array found in response")


def _normalize_llm_keys(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map common LLM key variants to canonical keys so we don't get 'unknown' when
    the model returns e.g. mission_research (string) instead of mission_research_category,
    or different casing (OrgType, govLevel, etc.).
    """
    if not parsed:
        return parsed
    canonical = dict(parsed)
    key_lower = {k.lower().replace("-", "_"): k for k in parsed}

    def get_canonical(*aliases: str) -> Optional[Any]:
        for a in aliases:
            al = a.lower().replace("-", "_")
            if al in key_lower:
                return parsed.get(key_lower[al])
        return None

    # sector (new)
    v = get_canonical("sector", "Sector")
    if v is not None:
        canonical["sector"] = v

    # org_type
    v = get_canonical("org_type", "OrgType", "orgType", "organization_type")
    if v is not None:
        canonical["org_type"] = v

    # gov_level
    v = get_canonical("gov_level", "GovLevel", "govLevel", "government_level")
    if v is not None:
        canonical["gov_level"] = v

    # gov_local_type
    v = get_canonical("gov_local_type", "GovLocalType", "govLocalType", "local_type")
    if v is not None:
        canonical["gov_local_type"] = v

    # mission_research_category: string (includes ExperimentalDevelopment)
    v = get_canonical("mission_research_category", "MissionResearchCategory", "mission_research_category")
    if v is None:
        v = parsed.get("mission_research")
        if isinstance(v, str) and v in {"NonResearch", "Enabler", "AppliedResearch", "AcademicResearch", "ExperimentalDevelopment"}:
            canonical["mission_research_category"] = v
    else:
        canonical["mission_research_category"] = v

    # mission_research: 0 or 1 (do not overwrite with string category)
    v = get_canonical("mission_research", "MissionResearch", "mission_research_flag")
    if v is not None:
        if isinstance(v, (int, float)) and v in (0, 1):
            canonical["mission_research"] = int(v)
        elif isinstance(v, bool):
            canonical["mission_research"] = 1 if v else 0
        # else: leave as-is; _normalize_to_unknown may infer from category

    # rationale: no longer required; omit so we don't add it to output
    return canonical


def _parsed_to_result_dict(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parsed LLM response to normalized result dict (same shape for all backends)."""
    parsed = _normalize_llm_keys(parsed.copy())
    parsed = _normalize_to_unknown(parsed)
    sector = parsed.get("sector", "unknown")
    if sector != "government":
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
                "sector": "academic",
                "org_type": "university",
                "gov_level": "non_applicable",
                "gov_local_type": "non_applicable",
                "mission_research_category": "AcademicResearch",
                "mission_research": 1,
                "confidence_org_type": 0.95,
                "confidence_gov_level": 0.95,
                "confidence_mission_research": 0.95,
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
                "sector": "government" if is_government else "non_profit",
                "org_type": "government_agency" if is_government else "association_foundation",
                "gov_level": gov_level,
                "gov_local_type": "non_applicable" if gov_level != "local" else "unknown",
                "mission_research_category": "Enabler",
                "mission_research": 0,
                "confidence_org_type": 0.9,
                "confidence_gov_level": 0.8,
                "confidence_mission_research": 0.95,
            }

    # Rule C: Clear teaching hospitals
    if "healthcare" in [t.lower() for t in ror_types]:
        teaching_hospital_patterns = [
            "university hospital", "academic medical center", "teaching hospital",
            "medical center university", "university medical", "academic hospital",
        ]
        if any(p in combined_text for p in teaching_hospital_patterns):
            return {
                "sector": "academic",
                "org_type": "hospital_clinic",
                "gov_level": "non_applicable",
                "gov_local_type": "non_applicable",
                "mission_research_category": "AcademicResearch",
                "mission_research": 1,
                "confidence_org_type": 0.9,
                "confidence_gov_level": 0.95,
                "confidence_mission_research": 0.9,
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
    """
    Backend Gemini vía HTTP directo (requests). Formato Google generateContent;
    no usa la librería openai. URL y payload bajo control total.
    """

    def __init__(self, config: BackendConfig):
        self.api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini backend requires api_key in config or GEMINI_API_KEY in environment")
        # Modelo Lite estable en la API (gemini-2.0-flash-lite)
        self.model_name = (config.get("model_name") or "gemini-2.0-flash-lite").strip().replace("models/", "")
        self.max_retries = int(config.get("max_retries", 2))

    def classify_affiliation(
        self,
        affiliation: str,
        country_code: str,
        ror_match: Optional[Any] = None,
    ) -> Dict[str, Any]:
        affiliation = str(affiliation).strip() if affiliation is not None else ""
        country_code = str(country_code).strip() if country_code is not None else ""
        user_data: Dict[str, Any] = {
            "affiliation": affiliation,
            "country_code": country_code,
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

        prompt = f"{SYSTEM_PROMPT}\n\nInput:\n{json.dumps(user_data)}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        # URL: .../v1beta/models/gemini-2.0-flash-lite:generateContent?key=...
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        for attempt in range(self.max_retries + 1):
            try:
                time.sleep(4)  # Retardo entre peticiones para evitar 429 (límite de cuota)
                response = requests.post(url, json=payload, headers=headers, timeout=120)
                if not response.ok:
                    LOGGER.error("Gemini API error response: %s", response.text)
                    response.raise_for_status()
                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                json_str = _extract_json_object(text)
                parsed = json.loads(json_str)
                return _parsed_to_result_dict(parsed)
            except (json.JSONDecodeError, ValueError, KeyError, IndexError, requests.exceptions.RequestException) as e:
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                if hasattr(e, "response") and e.response is not None:
                    LOGGER.error("Gemini API error body: %s", e.response.text)
                raise LMStudioError(f"Gemini request or parse failed after {self.max_retries + 1} attempts: {e}") from e
        raise LMStudioError("Unreachable")

    def classify_affiliations_batch(
        self,
        items: List[Dict[str, Any]],
        max_retries: int = 2,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []
        results: List[Dict[str, Any]] = []
        for item in items:
            affiliation = str(item.get("affiliation", "") or "").strip()
            country_code = str(item.get("country_code", "") or "").strip()
            ror_match = item.get("ror_match")
            try:
                result = self.classify_affiliation(
                    affiliation=affiliation,
                    country_code=country_code,
                    ror_match=ror_match,
                )
                result["id"] = item["id"]
                results.append(result)
            except Exception as e:
                LOGGER.warning("Gemini individual classification failed for id=%s: %s", item.get("id"), e)
                results.append({
                    "id": item["id"],
                    "sector": "unknown",
                    "org_type": "unknown",
                    "gov_level": "unknown",
                    "gov_local_type": "unknown",
                    "mission_research_category": "unknown",
                    "mission_research": 0,
                    "confidence_org_type": None,
                    "confidence_gov_level": None,
                    "confidence_mission_research": None,
                })
        return results


class LocalBackend(LLMBackend):
    """
    Backend para LM Studio (o cualquier API compatible con OpenAI).
    Usa siempre la librería openai: LM Studio es 100% compatible con el cliente OpenAI
    (chat completions con messages). No usa requests ni formato Google.
    """

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
        affiliation = str(affiliation).strip() if affiliation is not None else ""
        country_code = str(country_code).strip() if country_code is not None else ""
        user_data: Dict[str, Any] = {
            "affiliation": affiliation,
            "country_code": country_code,
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
                if content is None:
                    content = ""
                content = str(content).strip()
                LOGGER.info("CONTENIDO CRUDO DEL LLM: %s", content)
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
                raw = resp.choices[0].message.content
                if raw is None:
                    raw = ""
                LOGGER.info("CONTENIDO CRUDO DEL LLM (batch): %s", raw)
                # Si ya es objeto Python (dict/list), no procesar como texto
                if isinstance(raw, (dict, list)):
                    parsed_array = [raw] if isinstance(raw, dict) else raw
                else:
                    content = str(raw).strip()
                    json_str = _extract_json_object(content)
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        parsed_array = [parsed]
                    elif isinstance(parsed, list):
                        parsed_array = parsed
                    else:
                        parsed_array = [parsed] if parsed is not None else []
                break
            except (json.JSONDecodeError, ValueError, KeyError, IndexError):
                if len(items) > 1 and attempt < max_retries:
                    mid = len(items) // 2
                    first = self.classify_affiliations_batch(items[:mid], max_retries - 1)
                    second = self.classify_affiliations_batch(items[mid:], max_retries - 1)
                    return first + second
                raise LMStudioError(f"Invalid batch response after {attempt + 1} attempts") from None

        if not isinstance(parsed_array, list):
            parsed_array = [parsed_array] if parsed_array is not None else []

        if len(parsed_array) < len(items):
            parsed_array.extend([{}] * (len(items) - len(parsed_array)))
        elif len(parsed_array) > len(items):
            parsed_array = parsed_array[:len(items)]

        results: List[Dict[str, Any]] = []
        for idx in range(len(items)):
            result_item = parsed_array[idx] if idx < len(parsed_array) else {}
            if not isinstance(result_item, dict):
                LOGGER.warning(
                    "Batch item %d is not a dict (type=%s); using empty dict to preserve order.",
                    idx,
                    type(result_item).__name__,
                )
                result_item = {}
            try:
                normalized = _parsed_to_result_dict(result_item.copy())
                normalized["id"] = items[idx]["id"]
                results.append(normalized)
            except Exception as exc:
                LOGGER.warning("Failed to normalize batch item %d: %s", idx, exc)
                results.append({
                    "id": items[idx]["id"],
                    "sector": "unknown",
                    "org_type": "unknown",
                    "gov_level": "unknown",
                    "gov_local_type": "unknown",
                    "mission_research_category": "unknown",
                    "mission_research": 0,
                    "confidence_org_type": None,
                    "confidence_gov_level": None,
                    "confidence_mission_research": None,
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
    affiliation y country_code se convierten a str() antes de enviar al backend.
    """
    b = backend or _default_backend
    if b is None:
        b = LocalBackend({})  # fallback: LM Studio por defecto
    return b.classify_affiliation(
        str(affiliation) if affiliation is not None else "",
        str(country_code) if country_code is not None else "",
        ror_match,
    )


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
