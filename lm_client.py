"""
Client utilities to interact with a local LM Studio instance that exposes
an OpenAI-compatible chat completions API.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import re
import requests
from dotenv import load_dotenv

try:
    from ror_knowledge import RorMatch
except ImportError:
    RorMatch = None  # type: ignore

# Load environment variables from an optional .env file.
load_dotenv()


LM_STUDIO_BASE_URL = os.environ.get(
    "LM_STUDIO_BASE_URL", "http://localhost:1234/v1/chat/completions"
)
LM_STUDIO_MODEL_NAME = os.environ.get("LM_STUDIO_MODEL_NAME", "local-model")
LM_STUDIO_TIMEOUT = float(os.environ.get("LM_STUDIO_TIMEOUT", "60"))

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
- government = federal, state, or local public entities (ministries, departments, agencies, military).  
- university = higher education institutions (not consortia).  
  Academic departments (“Department of X”, “School of Y”) belong to a university unless clearly a government unit.  
- research_institute = organizations primarily dedicated to research (public or private).  
- company = private for-profit enterprises.  
- ngo = non-profits, associations, alliances, networks (not single universities).  
- hospital = healthcare institutions unless clearly private-company.  
  Veterans Affairs Medical Centers = government / federal / NonResearch unless explicitly research-focused.  
- supranational_organization = UN, EU, WHO, etc.  
- other = conceptual or non-institutional entities.  
 
# 2. SPECIAL PATTERNS  
Military entities (Air Force, Army, Navy, DoD):  
→ org_type="government", gov_level="federal".  
 
University consortia or alliances:  
→ not "university"; usually "ngo".  
 
Academic departments:  
→ classify as university unless explicit government context.  
 
Labs and Centers:  
- If linked to a university → AcademicResearch.  
- If part of government → AppliedResearch.  
- If corporate cues (Inc, LLC, Technologies, Biosciences, etc.) → company.  
 
Museums / Heritage / Preservation:  
- Usually ngo unless explicit government ownership (City/County/State).  
 
# 3. GOV LEVEL  
- federal = national agencies (NIH, CDC, EPA, USGS, NOAA, VA hospitals, DoD units).  
- state = state agencies.  
- local = city/county/town departments; set gov_local_type accordingly.  
- If org_type != government → gov_level="non_applicable".  
 
# 4. MISSION RESEARCH CATEGORY  
Choose one:  
- NonResearch (0): administrative units, municipal agencies, service providers, hospitals without research.  
- Enabler (0): funding agencies, foundations.  
- AppliedResearch (1): government labs, mission-oriented R&D, analytics divisions, corporate R&D.  
- AcademicResearch (1): universities, research institutes, academic departments, academic medical centers.  
 
# 5. USE OF ROR (if provided)  
- education → strong cue for university / AcademicResearch.  
- government → if research terms → AppliedResearch; if administrative → NonResearch.  
- healthcare → teaching hospitals → AcademicResearch.  
- funder → Enabler.  
 
# 6. DERIVED FIELD  
mission_research = 1 for AppliedResearch or AcademicResearch; else 0.  
 
# 7. FORMAT  
Return only a JSON object with the required keys.  
"rationale" = "".\
"""


class LMStudioError(RuntimeError):
    """Raised when the LM Studio request fails or returns invalid data."""


def _extract_json_object(text: str) -> str:
    """
    Extract the first JSON object from a text response, stripping ``` fences.

    Raises ValueError if no JSON-like object can be found.
    """
    if not text:
        raise ValueError("Empty response from LM Studio")

    cleaned = text.strip()

    # Remove ```json ... ``` style fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    # Look for the first {...} block
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    # Fallback: assume entire content is JSON
    return cleaned


def try_rule_based_classification(
    affiliation: str, country_code: str | None, ror_info: dict | None
) -> dict | None:
    """
    Attempt to classify an affiliation using conservative rule-based logic with ROR data.
    
    This function implements fast-track classification for clear cases (universities,
    funders, teaching hospitals) to avoid unnecessary LLM calls.
    
    Parameters
    ----------
    affiliation:
        The raw affiliation string.
    country_code:
        Optional ISO country code.
    ror_info:
        Optional dict with ROR match information. Expected keys:
        - ror_types: list of ROR type strings
        - ror_name: ROR organization name
    
    Returns
    -------
    dict or None
        Classification dict with same keys as LLM response if classification is clear,
        None if rules don't apply or case is ambiguous (should use LLM).
    """
    if ror_info is None:
        return None
    
    ror_types = ror_info.get("ror_types", [])
    ror_name = ror_info.get("ror_name", "")
    affiliation_lower = affiliation.lower()
    ror_name_lower = ror_name.lower() if ror_name else ""
    
    # Combine affiliation and ROR name for pattern matching
    combined_text = f"{affiliation_lower} {ror_name_lower}".strip()
    
    # Rule A: Clear universities
    if "education" in [t.lower() for t in ror_types]:
        # Patterns that clearly indicate a university
        university_patterns = [
            "university",
            "college",
            "institute of technology",
            "polytechnic",
            "school of",
            "universidad",
            "université",
            "universität",
            "universita",
        ]
        
        # Check if any pattern matches
        is_university = any(pattern in combined_text for pattern in university_patterns)
        
        # Exclude consortia and associations
        exclusion_patterns = [
            "consortium",
            "association",
            "alliance",
            "coalition",
            "council of",
            "network of",
        ]
        is_excluded = any(pattern in combined_text for pattern in exclusion_patterns)
        
        if is_university and not is_excluded:
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
        # Patterns that clearly indicate a funding organization
        funder_patterns = [
            "foundation",
            "research council",
            "funding",
            "science foundation",
            "national science",
            "research fund",
            "grant",
            "funder",
        ]
        
        is_funder = any(pattern in combined_text for pattern in funder_patterns)
        
        if is_funder:
            # Determine if it's government or NGO based on name patterns
            gov_patterns = [
                "national science",
                "national research",
                "ministry",
                "department",
                "agency",
                "government",
            ]
            is_government = any(pattern in combined_text for pattern in gov_patterns)
            
            # Determine gov_level if it's government
            if is_government:
                # Check for federal indicators (US-specific)
                federal_patterns = ["national", "federal", "u.s.", "us "]
                is_federal = any(pattern in combined_text for pattern in federal_patterns)
                gov_level = "federal" if is_federal else "unknown"
            else:
                gov_level = "non_applicable"
            
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
        # Patterns that clearly indicate a teaching/academic hospital
        teaching_hospital_patterns = [
            "university hospital",
            "academic medical center",
            "teaching hospital",
            "medical center university",
            "university medical",
            "academic hospital",
        ]
        
        is_teaching_hospital = any(pattern in combined_text for pattern in teaching_hospital_patterns)
        
        if is_teaching_hospital:
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
    
    # If no rule applies, return None to use LLM
    return None


def _post_chat_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a chat completion request and return the raw response JSON."""
    response = requests.post(
        LM_STUDIO_BASE_URL,
        json=payload,
        timeout=LM_STUDIO_TIMEOUT,
    )
    if response.status_code != 200:
        raise LMStudioError(
            f"LM Studio returned {response.status_code}: {response.text}"
        )
    return response.json()


def classify_affiliation(
    affiliation: str, country_code: str, ror_match: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Classify whether an affiliation is governmental using LM Studio.

    Parameters
    ----------
    affiliation:
        The raw affiliation string.
    country_code:
        ISO country code (if unknown, send an empty string).
    ror_match:
        Optional RorMatch object with ROR information.

    Returns
    -------
    dict
        Dictionary with keys: org_type, gov_level, gov_local_type, mission_research_category,
        mission_research, rationale.
    """
    # Build user message with affiliation and optional ROR info
    user_data: Dict[str, Any] = {
        "affiliation": affiliation.strip(),
        "country_code": country_code.strip(),
    }
    
    if ror_match is not None:
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
        if ror_match.suggested_org_type_from_ror:
            user_data["ror_match"]["suggested_org_type_from_ror"] = ror_match.suggested_org_type_from_ror
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_data),
        },
    ]
    payload = {
        "model": LM_STUDIO_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 150,  # Increased to accommodate mission_research_category
        # "response_format": {"type": "json_object"},
    }

    completion = _post_chat_completion(payload)
    try:
        content = completion["choices"][0]["message"]["content"]
        json_str = _extract_json_object(content)
        parsed = json.loads(json_str)
    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
        raise LMStudioError(f"Invalid response format from LM Studio: {content!r}") from exc

    # Normalization step: fix inconsistent combinations before validation
    valid_org_types = {
        "supranational_organization",
        "government",
        "university",
        "research_institute",
        "company",
        "ngo",
        "hospital",
        "other",
    }
    valid_gov_levels = {"federal", "state", "local", "unknown", "non_applicable"}
    valid_gov_local_types = {"city", "county", "other_local", "unknown", "non_applicable"}

    # Fix org_type if invalid
    if parsed.get("org_type") not in valid_org_types:
        parsed["org_type"] = "other"

    # Fix gov_level if invalid
    if parsed.get("gov_level") not in valid_gov_levels:
        parsed["gov_level"] = "non_applicable"

    # Fix gov_local_type if invalid
    if parsed.get("gov_local_type") not in valid_gov_local_types:
        parsed["gov_local_type"] = "non_applicable"

    # If org_type is NOT "government", force both gov fields to "non_applicable"
    if parsed.get("org_type") != "government":
        parsed["gov_level"] = "non_applicable"
        parsed["gov_local_type"] = "non_applicable"

    # If org_type is "government" but gov_level is NOT "local", force gov_local_type to "non_applicable"
    if parsed.get("org_type") == "government" and parsed.get("gov_level") != "local":
        parsed["gov_local_type"] = "non_applicable"

    # Validate mission_research_category
    valid_mission_categories = {"NonResearch", "Enabler", "AppliedResearch", "AcademicResearch"}
    mission_category = parsed.get("mission_research_category")
    if mission_category not in valid_mission_categories:
        # Try to fix common variations
        mission_category_lower = str(mission_category).lower() if mission_category else ""
        if "non" in mission_category_lower or "none" in mission_category_lower:
            parsed["mission_research_category"] = "NonResearch"
        elif "enabler" in mission_category_lower or "enable" in mission_category_lower:
            parsed["mission_research_category"] = "Enabler"
        elif "applied" in mission_category_lower:
            parsed["mission_research_category"] = "AppliedResearch"
        elif "academic" in mission_category_lower:
            parsed["mission_research_category"] = "AcademicResearch"
        else:
            parsed["mission_research_category"] = "NonResearch"  # Default fallback
        mission_category = parsed["mission_research_category"]
    
    # Derive mission_research from mission_research_category if not present or inconsistent
    if "mission_research" not in parsed or parsed.get("mission_research") is None:
        parsed["mission_research"] = 1 if mission_category in {"AppliedResearch", "AcademicResearch"} else 0
    else:
        # Ensure consistency: override mission_research based on category
        expected_mission_research = 1 if mission_category in {"AppliedResearch", "AcademicResearch"} else 0
        parsed["mission_research"] = expected_mission_research

    required_keys = {
        "org_type",
        "gov_level",
        "gov_local_type",
        "mission_research_category",
        "mission_research",
        "rationale",
    }
    missing = required_keys - parsed.keys()
    if missing:
        raise LMStudioError(
            f"Missing keys {missing} in LM Studio response: {json.dumps(parsed)}"
        )
    
    # Add confidence fields as empty/None for compatibility (not used but kept for CSV output)
    if "confidence_org_type" not in parsed:
        parsed["confidence_org_type"] = None
    if "confidence_gov_level" not in parsed:
        parsed["confidence_gov_level"] = None
    if "confidence_mission_research" not in parsed:
        parsed["confidence_mission_research"] = None
    
    # Validate org_type values
    valid_org_types = {
        "supranational_organization",
        "government",
        "university",
        "research_institute",
        "company",
        "ngo",
        "hospital",
        "other",
    }
    org_type = parsed.get("org_type")
    if org_type not in valid_org_types:
        raise LMStudioError(
            f"Invalid org_type '{org_type}'. Must be one of {valid_org_types}."
        )
    
    # Enforce that gov_level and gov_local_type must be "non_applicable" when org_type != "government"
    if org_type != "government":
        if parsed.get("gov_level") != "non_applicable":
            raise LMStudioError(
                f"gov_level must be 'non_applicable' when org_type is '{org_type}', "
                f"but got '{parsed.get('gov_level')}'."
            )
        if parsed.get("gov_local_type") != "non_applicable":
            raise LMStudioError(
                f"gov_local_type must be 'non_applicable' when org_type is '{org_type}', "
                f"but got '{parsed.get('gov_local_type')}'."
            )
    
    return parsed

