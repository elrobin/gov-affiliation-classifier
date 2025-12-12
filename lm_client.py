"""
Client utilities to interact with a local LM Studio instance that exposes
an OpenAI-compatible chat completions API.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import logging
import re
import requests
from dotenv import load_dotenv

try:
    from ror_knowledge import RorMatch
except ImportError:
    RorMatch = None  # type: ignore

# Load environment variables from an optional .env file.
load_dotenv()

LOGGER = logging.getLogger("gov-affiliation-classifier.lm_client")


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

- government = federal, state, or local public entities (ministries, departments, agencies, regulatory bodies, military units).
- university = higher education institutions and systems CONSISTING OF UNIVERSITIES:
  - Single universities, colleges, campuses.
  - Academic departments (“Department of X”, “School of Y”, “Faculty of Z”).
  - Alliances / consortia / systems / networks COMPOSED OF universities or higher education institutions
    (e.g. “University of California System”, “Association of American Universities”).
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
- If clearly linked to a university (e.g. “Center for X, University of Y”, “Institute for Z at Harvard University”)
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

# 6. DERIVED FIELD

Normally:
- mission_research = 1 for "AppliedResearch" or "AcademicResearch".
- mission_research = 0 for "NonResearch" or "Enabler".

EXCEPTION:
- If a valid ROR record exists, mission_research MUST be 1,
  and mission_research_category MUST be one of ["AppliedResearch","AcademicResearch","Enabler"].

# 7. OUTPUT FORMAT

Return ONLY a JSON object with all required keys:
- "org_type"
- "gov_level"
- "gov_local_type"
- "mission_research_category"
- "mission_research"
- "rationale"

"rationale" MUST be a string (can be short).

No extra text, no explanations outside the JSON.

# 8. EXAMPLES (COMPLEX CASES)

Example 1: Government department (administrative function)
Affiliation: "Department of Energy"
Country: "US"
→ government, NonResearch

{
  "org_type": "government",
  "gov_level": "federal",
  "gov_local_type": "non_applicable",
  "mission_research_category": "NonResearch",
  "mission_research": 0,
  "rationale": "Federal government department with administrative and policy functions, not primarily a research organization."
}

Example 2: Government department (administrative function)
Affiliation: "Department of Education"
Country: "US"
→ government, NonResearch

{
  "org_type": "government",
  "gov_level": "federal",
  "gov_local_type": "non_applicable",
  "mission_research_category": "NonResearch",
  "mission_research": 0,
  "rationale": "Federal government department with administrative and policy functions, not primarily a research organization."
}

Example 3: Academic department (academic discipline)
Affiliation: "Department of Philosophy"
Country: "US"
→ university, AcademicResearch

{
  "org_type": "university",
  "gov_level": "non_applicable",
  "gov_local_type": "non_applicable",
  "mission_research_category": "AcademicResearch",
  "mission_research": 1,
  "rationale": "Academic department of an academic discipline, performing teaching and research activities."
}

Example 4: Academic department (academic discipline)
Affiliation: "Department of Educational Sciences"
Country: "US"
→ university, AcademicResearch

{
  "org_type": "university",
  "gov_level": "non_applicable",
  "gov_local_type": "non_applicable",
  "mission_research_category": "AcademicResearch",
  "mission_research": 1,
  "rationale": "Academic department of an academic discipline, performing teaching and research activities."
}

Example 5: Academic department (academic discipline)
Affiliation: "Department of Computer Science"
Country: "US"
→ university, AcademicResearch

{
  "org_type": "university",
  "gov_level": "non_applicable",
  "gov_local_type": "non_applicable",
  "mission_research_category": "AcademicResearch",
  "mission_research": 1,
  "rationale": "Academic department of an academic discipline, performing teaching and research activities."
}

Example 6: City public health department
Affiliation: "Division of Family Health, Rhode Island Department of Health"
Country: "US"
→ government, local, NonResearch

{
  "org_type": "government",
  "gov_level": "state",
  "gov_local_type": "other_local",
  "mission_research_category": "NonResearch",
  "mission_research": 0,
  "rationale": "State department of health providing public health services, not primarily a research organization."
}

Example 7: University department
Affiliation: "Department of Computer Science, Stanford University"
Country: "US"
→ university, AcademicResearch

{
  "org_type": "university",
  "gov_level": "non_applicable",
  "gov_local_type": "non_applicable",
  "mission_research_category": "AcademicResearch",
  "mission_research": 1,
  "rationale": "Academic department within a university, with teaching and research."
}

Example 8: Alliance of universities
Affiliation: "Association of American Universities"
Country: "US"
→ university, Enabler or AcademicResearch depending on role

{
  "org_type": "university",
  "gov_level": "non_applicable",
  "gov_local_type": "non_applicable",
  "mission_research_category": "Enabler",
  "mission_research": 1,
  "rationale": "Alliance of research universities that coordinates and enables academic research."
}

Example 9: Corporate R&D center with NO ROR
Affiliation: "3 Dimensional Pharmaceuticals, Inc."
Country: "US"

{
  "org_type": "company",
  "gov_level": "non_applicable",
  "gov_local_type": "non_applicable",
  "mission_research_category": "AppliedResearch",
  "mission_research": 1,
  "rationale": "Private pharmaceutical company with R&D activities."
}

Example 10: Government laboratory WITH ROR
Affiliation: "National Security Technologies, LLC"
Country: "US"
ROR types: "Government, Facility"

{
  "org_type": "government",
  "gov_level": "federal",
  "gov_local_type": "non_applicable",
  "mission_research_category": "AppliedResearch",
  "mission_research": 1,
  "rationale": "Federal government-related facility performing mission-oriented research and development."
}

Example 11: Funder WITH ROR
Affiliation: "Georgia Clinical & Translational Science Alliance"
Country: "US"
ROR types: "Facility, Funder"

{
  "org_type": "university",
  "gov_level": "non_applicable",
  "gov_local_type": "non_applicable",
  "mission_research_category": "Enabler",
  "mission_research": 1,
  "rationale": "Alliance of universities acting as a research enabler and funder in the clinical and translational research ecosystem."
}
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

    # Look for the first {...} block or [...] array
    match_arr = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    match_obj = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    
    if match_arr:
        return match_arr.group(0).strip()
    elif match_obj:
        return match_obj.group(0).strip()

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
        timeout=300,
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


def classify_affiliations_batch(
    items: List[Dict[str, Any]], max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Classify multiple affiliations in a single LLM call (batch processing).

    Parameters
    ----------
    items:
        List of dicts, each with keys: "id", "affiliation", "country_code", and optionally "ror_match".
    max_retries:
        Maximum number of retry attempts with smaller batches if parsing fails.

    Returns
    -------
    List[Dict[str, Any]]
        List of classification results, one per input item, in the same order.
    """
    if not items:
        return []

    # Build batch request
    batch_data = []
    for item in items:
        entry: Dict[str, Any] = {
            "id": item["id"],
            "affiliation": str(item["affiliation"]).strip(),
            "country_code": str(item.get("country_code", "")).strip(),
        }
        if "ror_match" in item and item["ror_match"] is not None:
            ror_match = item["ror_match"]
            entry["ror_match"] = {
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
                entry["ror_match"]["suggested_org_type_from_ror"] = ror_match.suggested_org_type_from_ror
        batch_data.append(entry)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BATCH},
        {
            "role": "user",
            "content": json.dumps(batch_data),
        },
    ]
    payload = {
        "model": LM_STUDIO_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 5000,  # Increased for batch processing
    }

    try:
        completion = _post_chat_completion(payload)
        content = completion["choices"][0]["message"]["content"]
        json_str = _extract_json_object(content)
        parsed_array = json.loads(json_str)
    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
        # If parsing fails and we can retry with smaller batch, do so
        if len(items) > 1 and max_retries > 0:
            LOGGER.warning(
                "Batch parsing failed for %d items, splitting in half and retrying. Error: %s",
                len(items),
                exc,
            )
            mid = len(items) // 2
            first_half = classify_affiliations_batch(items[:mid], max_retries - 1)
            second_half = classify_affiliations_batch(items[mid:], max_retries - 1)
            return first_half + second_half
        else:
            raise LMStudioError(f"Invalid batch response format from LM Studio: {content!r}") from exc

    # Validate that we got an array
    if not isinstance(parsed_array, list):
        raise LMStudioError(f"Expected JSON array, got {type(parsed_array)}: {json.dumps(parsed_array)[:200]}")

    # Validate length matches
    if len(parsed_array) != len(items):
        raise LMStudioError(
            f"Response array length {len(parsed_array)} does not match input length {len(items)}"
        )

    # Normalize and validate each result
    results = []
    id_to_index = {item["id"]: idx for idx, item in enumerate(items)}
    
    for result_item in parsed_array:
        # Normalize each result using the same logic as single-item classification
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
        valid_mission_categories = {"NonResearch", "Enabler", "AppliedResearch", "AcademicResearch"}

        # Fix org_type if invalid
        if result_item.get("org_type") not in valid_org_types:
            result_item["org_type"] = "other"

        # Fix gov_level if invalid
        if result_item.get("gov_level") not in valid_gov_levels:
            result_item["gov_level"] = "non_applicable"

        # Fix gov_local_type if invalid
        if result_item.get("gov_local_type") not in valid_gov_local_types:
            result_item["gov_local_type"] = "non_applicable"

        # If org_type is NOT "government", force both gov fields to "non_applicable"
        if result_item.get("org_type") != "government":
            result_item["gov_level"] = "non_applicable"
            result_item["gov_local_type"] = "non_applicable"

        # If org_type is "government" but gov_level is NOT "local", force gov_local_type to "non_applicable"
        if result_item.get("org_type") == "government" and result_item.get("gov_level") != "local":
            result_item["gov_local_type"] = "non_applicable"

        # Validate mission_research_category
        mission_category = result_item.get("mission_research_category")
        if mission_category not in valid_mission_categories:
            mission_category_lower = str(mission_category).lower() if mission_category else ""
            if "non" in mission_category_lower or "none" in mission_category_lower:
                result_item["mission_research_category"] = "NonResearch"
            elif "enabler" in mission_category_lower or "enable" in mission_category_lower:
                result_item["mission_research_category"] = "Enabler"
            elif "applied" in mission_category_lower:
                result_item["mission_research_category"] = "AppliedResearch"
            elif "academic" in mission_category_lower:
                result_item["mission_research_category"] = "AcademicResearch"
            else:
                result_item["mission_research_category"] = "NonResearch"
            mission_category = result_item["mission_research_category"]

        # Derive mission_research
        if "mission_research" not in result_item or result_item.get("mission_research") is None:
            result_item["mission_research"] = 1 if mission_category in {"AppliedResearch", "AcademicResearch"} else 0
        else:
            expected_mission_research = 1 if mission_category in {"AppliedResearch", "AcademicResearch"} else 0
            result_item["mission_research"] = expected_mission_research

        # Add confidence fields for compatibility
        if "confidence_org_type" not in result_item:
            result_item["confidence_org_type"] = None
        if "confidence_gov_level" not in result_item:
            result_item["confidence_gov_level"] = None
        if "confidence_mission_research" not in result_item:
            result_item["confidence_mission_research"] = None

        # Ensure rationale exists
        if "rationale" not in result_item:
            result_item["rationale"] = ""

        results.append(result_item)

    # Sort results by original input order using id mapping
    results_sorted = [None] * len(items)
    for result in results:
        result_id = result.get("id")
        if result_id in id_to_index:
            results_sorted[id_to_index[result_id]] = result
        else:
            LOGGER.warning("Result with id '%s' not found in input items", result_id)

    # Fill any missing results with error placeholders
    for idx, result in enumerate(results_sorted):
        if result is None:
            results_sorted[idx] = {
                "id": items[idx]["id"],
                "org_type": None,
                "gov_level": None,
                "gov_local_type": None,
                "mission_research_category": None,
                "mission_research": None,
                "confidence_org_type": None,
                "confidence_gov_level": None,
                "confidence_mission_research": None,
                "rationale": "Error: Result missing from batch response",
            }

    return results_sorted

