"""
ROR (Research Organization Registry) knowledge base integration.

This module loads the ROR dump and provides efficient matching functions
to find organizations by name and country code.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from rapidfuzz import fuzz, process
except ImportError:
    # Fallback if rapidfuzz is not available
    fuzz = None
    process = None

LOGGER = logging.getLogger("gov-affiliation-classifier.ror")


_MATCH_CLEAN_PATTERNS = [
    re.compile(r"\bcorp\b", re.IGNORECASE),
    re.compile(r"\bcorporation\b", re.IGNORECASE),
    re.compile(r"\bcompany\b", re.IGNORECASE),
    re.compile(r"\binc\b", re.IGNORECASE),
    re.compile(r"\bdepartment\b", re.IGNORECASE),
    re.compile(r"\bdept\b", re.IGNORECASE),
    re.compile(r"\bprogram\b", re.IGNORECASE),
    re.compile(r"\boffice\b", re.IGNORECASE),
    re.compile(r"\bdivision\b", re.IGNORECASE),
    re.compile(r"\bcenter\b", re.IGNORECASE),
    re.compile(r"\bcollege\s+of\b", re.IGNORECASE),
    re.compile(r"\bcollege\b", re.IGNORECASE),
    re.compile(r"\bschool\s+of\b", re.IGNORECASE),
    re.compile(r"\blaboratory\b", re.IGNORECASE),
    re.compile(r"\blab\b", re.IGNORECASE),
    re.compile(r"\binstitute\b", re.IGNORECASE),
    re.compile(r"\bhospital\b", re.IGNORECASE),
]

_PREFERRED_COUNTRY_BOOSTS = {
    "US": 0.05,
    "CA": 0.04,
    "GB": 0.04,
}
_OTHER_COUNTRY_PENALTY = -0.04


def _clean_match_text(text: str) -> str:
    """Remove generic noise words to improve fuzzy matching."""
    if not text:
        return ""
    cleaned = text
    for pattern in _MATCH_CLEAN_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().lower()


@dataclass
class RorMatch:
    """Result of matching an affiliation against ROR."""

    ror_id: str
    ror_name: str
    ror_types: List[str]
    ror_country_code: Optional[str]
    ror_state: Optional[str]
    ror_city: Optional[str]
    ror_domains: List[str]
    match_score: float
    suggested_org_type_from_ror: Optional[str] = None


class RorIndex:
    """In-memory index of ROR organizations for fast lookup."""

    def __init__(self, organizations: List[Dict[str, Any]]):
        """
        Initialize the ROR index.

        Parameters
        ----------
        organizations:
            List of ROR organization dictionaries.
        """
        self.organizations = organizations
        self._by_normalized_name: Dict[str, List[Dict[str, Any]]] = {}
        self._by_country: Dict[str, List[Dict[str, Any]]] = {}
        self._build_index()

    def _normalize_name(self, name: str) -> str:
        """
        Normalize an organization name for matching.

        Parameters
        ----------
        name:
            Raw organization name.

        Returns
        -------
        str
            Normalized name (lowercase, common prefixes removed).
        """
        # Convert to lowercase
        normalized = name.lower().strip()

        # Remove common prefixes that don't help with matching
        prefixes_to_remove = [
            r"^dept\.?\s+of\s+",
            r"^department\s+of\s+",
            r"^ministry\s+of\s+",
            r"^office\s+of\s+",
            r"^bureau\s+of\s+",
            r"^agency\s+of\s+",
        ]
        for prefix in prefixes_to_remove:
            normalized = re.sub(prefix, "", normalized, flags=re.IGNORECASE)

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return _clean_match_text(normalized)

    def _build_index(self) -> None:
        """Build lookup indices from the organizations list."""
        LOGGER.info("Building ROR index from %d organizations...", len(self.organizations))

        for org in self.organizations:
            # Extract key fields
            ror_id = org.get("id", "")
            name = org.get("name", "")
            country = org.get("country", {})
            country_code = country.get("country_code", "").lower() if country else ""

            if not name:
                continue

            # Index by normalized name
            normalized = self._normalize_name(name)
            if normalized not in self._by_normalized_name:
                self._by_normalized_name[normalized] = []
            self._by_normalized_name[normalized].append(org)

            # Index by country
            if country_code:
                if country_code not in self._by_country:
                    self._by_country[country_code] = []
                self._by_country[country_code].append(org)

            # Also index aliases and acronyms
            aliases = org.get("aliases", [])
            for alias in aliases:
                normalized_alias = self._normalize_name(alias)
                if normalized_alias not in self._by_normalized_name:
                    self._by_normalized_name[normalized_alias] = []
                self._by_normalized_name[normalized_alias].append(org)

            acronyms = org.get("acronyms", [])
            for acronym in acronyms:
                normalized_acronym = self._normalize_name(acronym)
                if normalized_acronym not in self._by_normalized_name:
                    self._by_normalized_name[normalized_acronym] = []
                self._by_normalized_name[normalized_acronym].append(org)

        LOGGER.info(
            "ROR index built: %d normalized names, %d countries",
            len(self._by_normalized_name),
            len(self._by_country),
        )

    def _extract_org_fields(self, org: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant fields from a ROR organization entry.

        Parameters
        ----------
        org:
            ROR organization dictionary.

        Returns
        -------
        dict
            Extracted fields.
        """
        ror_id = org.get("id", "")
        name = org.get("name", "")
        types = org.get("types", [])

        country = org.get("country", {})
        country_code = country.get("country_code", "").lower() if country else ""

        # Extract state/region and city from addresses
        state = None
        city = None
        addresses = org.get("addresses", [])
        if addresses:
            first_addr = addresses[0]
            state = first_addr.get("state", None) or first_addr.get("region", None)
            city = first_addr.get("city", None)

        # Extract domains
        domains = []
        links = org.get("links", [])
        if links:
            domains = [link for link in links if link]

        return {
            "ror_id": ror_id,
            "ror_name": name,
            "ror_types": types,
            "ror_country_code": country_code,
            "ror_state": state,
            "ror_city": city,
            "ror_domains": domains,
        }

    def _suggest_org_type_from_ror_types(self, ror_types: List[str]) -> Optional[str]:
        """
        Suggest org_type based on ROR types.

        Parameters
        ----------
        ror_types:
            List of ROR type strings.

        Returns
        -------
        str or None
            Suggested org_type, or None if unclear.
        """
        if not ror_types:
            return None

        # Map ROR types to our org_type taxonomy
        type_lower = [t.lower() for t in ror_types]

        if "education" in type_lower:
            return "university"
        if "government" in type_lower:
            return "government"
        if "healthcare" in type_lower:
            return "hospital"
        if "company" in type_lower or "facility" in type_lower:
            return "company"
        if "nonprofit" in type_lower or "archive" in type_lower:
            return "ngo"

        return None

    def _build_match_result(
        self,
        fields: Dict[str, Any],
        suggested_type: Optional[str],
        base_score: float,
        affiliation_country_code: Optional[str],
    ) -> Optional[RorMatch]:
        """
        Build a RorMatch applying country filters and score adjustments.
        """
        affiliation_country = (
            affiliation_country_code.lower().strip() if affiliation_country_code else None
        )
        ror_country = (fields.get("ror_country_code") or "").lower()

        if affiliation_country:
            if not ror_country or ror_country != affiliation_country:
                return None
            adjusted_score = base_score
        else:
            boost = _PREFERRED_COUNTRY_BOOSTS.get(
                (fields.get("ror_country_code") or "").upper(), _OTHER_COUNTRY_PENALTY
            )
            adjusted_score = base_score + boost
            adjusted_score = max(0.0, min(1.0, adjusted_score))

        if adjusted_score < 0.92:
            return None

        return RorMatch(
            match_score=adjusted_score,
            suggested_org_type_from_ror=suggested_type,
            **fields,
        )

    def match_ror(
        self, affiliation_name: str, country_code: Optional[str] = None, threshold: float = 0.75
    ) -> Optional[RorMatch]:
        """
        Match an affiliation name against ROR organizations.

        Parameters
        ----------
        affiliation_name:
            The affiliation string to match.
        country_code:
            Optional ISO country code to narrow the search.
        threshold:
            Minimum fuzzy match score (0.0-1.0) to accept a match.

        Returns
        -------
        RorMatch or None
            Match result if found, None otherwise.
        """
        if not affiliation_name or not affiliation_name.strip():
            return None

        normalized_affiliation = self._normalize_name(affiliation_name)
        country_code_lower = country_code.lower().strip() if country_code else None
        cleaned_affiliation = _clean_match_text(affiliation_name)
        if not cleaned_affiliation:
            cleaned_affiliation = affiliation_name.lower().strip()

        # First, try exact match by normalized name
        candidates = self._by_normalized_name.get(normalized_affiliation, [])

        # Filter by country if provided
        if country_code_lower:
            candidates = [
                org
                for org in candidates
                if org.get("country", {}).get("country_code", "").lower() == country_code_lower
            ]

        # If exact match found, return it
        if candidates:
            for org in candidates:
                fields = self._extract_org_fields(org)
                suggested_type = self._suggest_org_type_from_ror_types(fields["ror_types"])
                match = self._build_match_result(
                    fields=fields,
                    suggested_type=suggested_type,
                    base_score=1.0,
                    affiliation_country_code=country_code_lower,
                )
                if match:
                    return match

        # If no exact match, try fuzzy matching
        if fuzz is None:
            LOGGER.warning("rapidfuzz not available, skipping fuzzy matching")
            return None

        # Get candidate organizations (optionally filtered by country)
        if country_code_lower and country_code_lower in self._by_country:
            search_space = self._by_country[country_code_lower]
        else:
            search_space = self.organizations

        # Use rapidfuzz to find best match
        # Create a list of strings to match against
        match_strings: List[str] = []
        org_list: List[Dict[str, Any]] = []
        for org in search_space:
            name = org.get("name", "")
            cleaned_name = _clean_match_text(name)
            if cleaned_name:
                match_strings.append(cleaned_name)
                org_list.append(org)
                # Also add aliases and acronyms
                for alias in org.get("aliases", []):
                    cleaned_alias = _clean_match_text(alias)
                    if cleaned_alias:
                        match_strings.append(cleaned_alias)
                        org_list.append(org)
                for acronym in org.get("acronyms", []):
                    cleaned_acronym = _clean_match_text(acronym)
                    if cleaned_acronym:
                        match_strings.append(cleaned_acronym)
                        org_list.append(org)

        if not match_strings:
            return None

        # Find best match using rapidfuzz
        result = process.extractOne(
            cleaned_affiliation,
            match_strings,
            scorer=fuzz.WRatio,
            score_cutoff=int(threshold * 100),
        )

        if result is None:
            return None

        matched_string, score, index = result
        score_normalized = score / 100.0  # Convert from 0-100 to 0-1

        # Get the organization that matched
        matched_org = org_list[index]
        fields = self._extract_org_fields(matched_org)
        suggested_type = self._suggest_org_type_from_ror_types(fields["ror_types"])

        return self._build_match_result(
            fields=fields,
            suggested_type=suggested_type,
            base_score=score_normalized,
            affiliation_country_code=country_code_lower,
        )


# Global ROR index (loaded once)
_ror_index: Optional[RorIndex] = None


def load_ror(path: str) -> RorIndex:
    """
    Load ROR dump from JSON file.

    Parameters
    ----------
    path:
        Path to the ROR JSON dump file.

    Returns
    -------
    RorIndex
        Loaded and indexed ROR data.
    """
    global _ror_index

    if _ror_index is not None:
        LOGGER.info("ROR index already loaded, reusing existing instance")
        return _ror_index

    LOGGER.info("Loading ROR dump from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, list):
        organizations = data
    elif isinstance(data, dict):
        organizations = list(data.values())
    else:
        raise ValueError(f"Unexpected ROR data format: {type(data)}")

    LOGGER.info("Loaded %d organizations from ROR dump", len(organizations))
    _ror_index = RorIndex(organizations)
    return _ror_index


def get_ror_index() -> Optional[RorIndex]:
    """
    Get the currently loaded ROR index.

    Returns
    -------
    RorIndex or None
        The loaded index, or None if not yet loaded.
    """
    return _ror_index


def match_ror(
    affiliation_name: str, country_code: Optional[str] = None, threshold: float = 0.75
) -> Optional[RorMatch]:
    """
    Match an affiliation against ROR (convenience function).

    Parameters
    ----------
    affiliation_name:
        The affiliation string to match.
    country_code:
        Optional ISO country code.
    threshold:
        Minimum fuzzy match score (0.0-1.0).

    Returns
    -------
    RorMatch or None
        Match result if found, None otherwise.
    """
    if _ror_index is None:
        raise RuntimeError("ROR index not loaded. Call load_ror() first.")
    return _ror_index.match_ror(affiliation_name, country_code, threshold)


