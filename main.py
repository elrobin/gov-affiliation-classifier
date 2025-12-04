"""
CLI utilities to classify affiliations as governmental or not using LM Studio.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Optional

import pandas as pd

from lm_client import LMStudioError, classify_affiliation, try_rule_based_classification
from ror_knowledge import RorMatch, load_ror, match_ror

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger("gov-affiliation-classifier")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify affiliations as governmental using LM Studio."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV with columns afid, affiliation, country_code.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the classified CSV will be saved.",
    )
    parser.add_argument(
        "--ror-path",
        default="v1.74-2025-11-24-ror-data/v1.74-2025-11-24-ror-data.json",
        help="Path to ROR JSON dump file (default: v1.74-2025-11-24-ror-data/v1.74-2025-11-24-ror-data.json)",
    )
    return parser.parse_args()


def _classify_row(row: pd.Series, ror_match: Optional[RorMatch] = None) -> Dict[str, Any]:
    """Helper to call LM Studio and capture errors for a DataFrame row."""
    affiliation = row.get("affiliation", "")
    country_code = row.get("country_code", "")
    try:
        return classify_affiliation(str(affiliation), str(country_code), ror_match=ror_match)
    except LMStudioError as err:
        LOGGER.error(
            "Failed to classify affiliation '%s' (%s): %s",
            affiliation,
            country_code,
            err,
        )
        return {
            "org_type": None,
            "gov_level": None,
            "gov_local_type": None,
            "mission_research_category": None,
            "mission_research": None,
            "confidence_org_type": None,  # Kept for CSV compatibility
            "confidence_gov_level": None,  # Kept for CSV compatibility
            "confidence_mission_research": None,  # Kept for CSV compatibility
            "rationale": f"Error: {err}",
        }


def _attach_ror_fields(result: Dict[str, Any], ror_match: Optional[RorMatch]) -> Dict[str, Any]:
    """
    Attach ROR fields to a classification result dictionary.

    Ensures all ROR columns are present, with None values when ROR is missing.
    """
    if ror_match is not None:
        result["ror_id"] = ror_match.ror_id
        result["ror_name"] = ror_match.ror_name
        result["ror_types"] = ror_match.ror_types
        result["ror_country_code"] = ror_match.ror_country_code
        result["ror_state"] = ror_match.ror_state
        result["ror_city"] = ror_match.ror_city
        result["ror_domains"] = ror_match.ror_domains
        result["ror_match_score"] = ror_match.match_score
        result["suggested_org_type_from_ror"] = ror_match.suggested_org_type_from_ror
    else:
        result["ror_id"] = None
        result["ror_name"] = None
        result["ror_types"] = None
        result["ror_country_code"] = None
        result["ror_state"] = None
        result["ror_city"] = None
        result["ror_domains"] = None
        result["ror_match_score"] = None
        result["suggested_org_type_from_ror"] = None

    return result


def main() -> None:
    args = _parse_args()

    LOGGER.info("Reading input CSV: %s", args.input)
    df = pd.read_csv(args.input)

    required_cols = {"afid", "affiliation", "country_code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Load ROR index
    ror_available = False
    LOGGER.info("Loading ROR knowledge base...")
    try:
        load_ror(args.ror_path)
        ror_available = True
        LOGGER.info("ROR knowledge base loaded successfully")
    except Exception as err:
        LOGGER.warning("Failed to load ROR dump: %s. Continuing without ROR matching.", err)

    LOGGER.info("Classifying %d rows...", len(df))
    
    # Match ROR for each row and classify
    ror_matches = []
    classification_results = []
    
    llm_calls = 0
    rule_based_calls = 0
    
    for idx, row in df.iterrows():
        affiliation = str(row.get("affiliation", ""))
        country_code = str(row.get("country_code", ""))
        
        # Try to match ROR if available
        ror_match = None
        ror_info_dict: Optional[Dict[str, Any]] = None
        ror_match_is_valid = False
        if ror_available:
            try:
                ror_match = match_ror(affiliation, country_code if country_code else None)
                if ror_match:
                    ror_match_is_valid = True
                    ror_info_dict = {
                        "ror_types": ror_match.ror_types,
                        "ror_name": ror_match.ror_name,
                    }
            except Exception as err:
                LOGGER.debug("ROR matching failed for '%s': %s", affiliation, err)
        
        ror_matches.append(ror_match)
        
        # Try rule-based classification first (fast track)
        if ror_match_is_valid:
            rule_based_result = try_rule_based_classification(
                affiliation=affiliation,
                country_code=country_code if country_code else None,
                ror_info=ror_info_dict,
            )
        else:
            rule_based_result = try_rule_based_classification(
                affiliation=affiliation,
                country_code=country_code if country_code else None,
                ror_info=None,
            )
        
        if rule_based_result is not None:
            # Use rule-based result, skip LLM call
            rule_based_calls += 1
            result_dict = _attach_ror_fields(rule_based_result, ror_match)
            classification_results.append(result_dict)
        else:
            # Use LLM for classification
            llm_calls += 1
            result = _classify_row(row, ror_match=ror_match)
            result_dict = _attach_ror_fields(result, ror_match)
            classification_results.append(result_dict)
    
    LOGGER.info(
        "Classification complete: %d rule-based, %d LLM calls",
        rule_based_calls,
        llm_calls,
    )
    
    # Convert to DataFrame
    results = pd.DataFrame(classification_results)

    # Standardize column names expected by the user.
    df["org_type"] = results["org_type"]
    df["gov_level"] = results["gov_level"]
    df["gov_local_type"] = results["gov_local_type"]
    df["mission_research_category"] = results["mission_research_category"]
    df["mission_research"] = results["mission_research"]
    df["confidence_org_type"] = results["confidence_org_type"]
    df["confidence_gov_level"] = results["confidence_gov_level"]
    df["confidence_mission_research"] = results["confidence_mission_research"]
    df["rationale"] = results["rationale"]

    # Add ROR fields from the enriched results dicts
    df["ror_id"] = results["ror_id"]
    df["ror_name"] = results["ror_name"]
    df["ror_types"] = results["ror_types"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else (x or "")
    )
    df["ror_country_code"] = results["ror_country_code"]
    df["ror_state"] = results["ror_state"]
    df["ror_city"] = results["ror_city"]
    df["ror_domains"] = results["ror_domains"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else (x or "")
    )
    df["ror_match_score"] = results["ror_match_score"]
    df["suggested_org_type_from_ror"] = results["suggested_org_type_from_ror"]

    LOGGER.info("Writing output CSV: %s", args.output)
    df.to_csv(args.output, index=False)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

