"""
CLI utilities to classify affiliations as governmental or not using LM Studio.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from lm_client import LMStudioError, classify_affiliation, classify_affiliations_batch, try_rule_based_classification
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


def _process_batch(
    batch_df: pd.DataFrame,
    ror_cache: Dict[Tuple[str, Optional[str]], Optional[RorMatch]],
    ror_available: bool,
    batch_size: int = 25,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Process a batch of affiliations: apply ROR matching, rule-based classification, and LLM batch.
    
    Returns:
        Tuple of (results list, rule_based_count, llm_batch_count)
    """
    results = []
    rule_based_count = 0
    llm_items = []
    
    # Step 1: ROR matching and rule-based classification
    for idx, row in batch_df.iterrows():
        affiliation = str(row.get("affiliation", ""))
        country_code = str(row.get("country_code", ""))
        afid = row.get("afid")
        
        # Check ROR cache
        cache_key = (affiliation, country_code if country_code else None)
        if cache_key in ror_cache:
            ror_match = ror_cache[cache_key]
        else:
            # Match ROR if available
            ror_match = None
            if ror_available:
                try:
                    ror_match = match_ror(affiliation, country_code if country_code else None)
                except Exception as err:
                    LOGGER.debug("ROR matching failed for '%s': %s", affiliation, err)
            ror_cache[cache_key] = ror_match
        
        ror_info_dict: Optional[Dict[str, Any]] = None
        ror_match_is_valid = False
        if ror_match:
            ror_match_is_valid = True
            ror_info_dict = {
                "ror_types": ror_match.ror_types,
                "ror_name": ror_match.ror_name,
            }
        
        # Try rule-based classification
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
            # Rule-based classification succeeded
            rule_based_count += 1
            result_dict = _attach_ror_fields(rule_based_result, ror_match)
            result_dict["afid"] = afid
            results.append((idx, result_dict))
        else:
            # Need LLM classification - add to batch
            llm_items.append({
                "id": str(afid),
                "affiliation": affiliation,
                "country_code": country_code if country_code else None,
                "ror_match": ror_match,
                "original_idx": idx,
            })
    
    # Step 2: Batch LLM call for unresolved items
    if llm_items:
        try:
            llm_results = classify_affiliations_batch(llm_items)
            for llm_result in llm_results:
                # Find the original item by id
                result_id = llm_result.get("id")
                original_item = next((item for item in llm_items if str(item["id"]) == str(result_id)), None)
                if original_item:
                    ror_match = original_item.get("ror_match")
                    result_dict = _attach_ror_fields(llm_result, ror_match)
                    result_dict["afid"] = original_item["id"]
                    # Remove "id" field to keep only "afid" (same structure as individual results)
                    result_dict.pop("id", None)
                    results.append((original_item["original_idx"], result_dict))
                else:
                    LOGGER.warning("Could not find original item for LLM result id '%s'", result_id)
        except LMStudioError as err:
            LOGGER.error("Batch LLM call failed: %s. Processing items individually.", err)
            # Fallback: process individually
            for item in llm_items:
                try:
                    result = classify_affiliation(
                        item["affiliation"],
                        item["country_code"] or "",
                        ror_match=item.get("ror_match"),
                    )
                    result_dict = _attach_ror_fields(result, item.get("ror_match"))
                    result_dict["afid"] = item["id"]
                    results.append((item["original_idx"], result_dict))
                except LMStudioError as err2:
                    LOGGER.error("Individual LLM call also failed for '%s': %s", item["affiliation"], err2)
                    error_result = {
                        "afid": item["id"],
                        "org_type": None,
                        "gov_level": None,
                        "gov_local_type": None,
                        "mission_research_category": None,
                        "mission_research": None,
                        "confidence_org_type": None,
                        "confidence_gov_level": None,
                        "confidence_mission_research": None,
                        "rationale": f"Error: {err2}",
                    }
                    result_dict = _attach_ror_fields(error_result, item.get("ror_match"))
                    results.append((item["original_idx"], result_dict))
    
    return results, rule_based_count, 1 if llm_items else 0


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

    LOGGER.info("Classifying %d rows with batching...", len(df))
    
    # ROR cache: (affiliation, country_code) -> RorMatch
    ror_cache: Dict[Tuple[str, Optional[str]], Optional[RorMatch]] = {}
    
    # Process in batches
    batch_size = 15
    all_results = []
    total_rule_based = 0
    total_llm_batches = 0
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        LOGGER.info("Processing batch %d-%d of %d...", batch_start + 1, batch_end, len(df))
        
        batch_results, rule_based_count, llm_batch_count = _process_batch(
            batch_df, ror_cache, ror_available, batch_size
        )
        
        all_results.extend(batch_results)
        total_rule_based += rule_based_count
        total_llm_batches += llm_batch_count
    
    LOGGER.info(
        "Classification complete: %d rule-based, %d LLM batch calls",
        total_rule_based,
        total_llm_batches,
    )
    
    # Sort results by original index and convert to DataFrame
    all_results.sort(key=lambda x: x[0])  # Sort by original index
    classification_results = [result[1] for result in all_results]
    results = pd.DataFrame(classification_results)

    # Check for duplicate afids and handle them (keep the last occurrence)
    if results["afid"].duplicated().any():
        duplicate_count = results["afid"].duplicated().sum()
        LOGGER.warning(
            "Found %d duplicate afid(s) in results. Keeping the last occurrence for each afid.",
            duplicate_count
        )
        results = results.drop_duplicates(subset=["afid"], keep="last")

    # Merge results back to original dataframe by afid
    results_with_afid = results.set_index("afid")
    
    # Standardize column names expected by the user.
    df["org_type"] = df["afid"].map(results_with_afid["org_type"])
    df["gov_level"] = df["afid"].map(results_with_afid["gov_level"])
    df["gov_local_type"] = df["afid"].map(results_with_afid["gov_local_type"])
    df["mission_research_category"] = df["afid"].map(results_with_afid["mission_research_category"])
    df["mission_research"] = df["afid"].map(results_with_afid["mission_research"])
    df["confidence_org_type"] = df["afid"].map(results_with_afid["confidence_org_type"])
    df["confidence_gov_level"] = df["afid"].map(results_with_afid["confidence_gov_level"])
    df["confidence_mission_research"] = df["afid"].map(results_with_afid["confidence_mission_research"])
    df["rationale"] = df["afid"].map(results_with_afid["rationale"])

    # Add ROR fields from the enriched results dicts
    df["ror_id"] = df["afid"].map(results_with_afid["ror_id"])
    df["ror_name"] = df["afid"].map(results_with_afid["ror_name"])
    df["ror_types"] = df["afid"].map(results_with_afid["ror_types"]).apply(
        lambda x: ", ".join(x) if isinstance(x, list) else (x or "")
    )
    df["ror_country_code"] = df["afid"].map(results_with_afid["ror_country_code"])
    df["ror_state"] = df["afid"].map(results_with_afid["ror_state"])
    df["ror_city"] = df["afid"].map(results_with_afid["ror_city"])
    df["ror_domains"] = df["afid"].map(results_with_afid["ror_domains"]).apply(
        lambda x: ", ".join(x) if isinstance(x, list) else (x or "")
    )
    df["ror_match_score"] = df["afid"].map(results_with_afid["ror_match_score"])
    df["suggested_org_type_from_ror"] = df["afid"].map(results_with_afid["suggested_org_type_from_ror"])

    # Validate for NaN in mission_research_category or mission_research before writing CSV
    nan_mask = df["mission_research_category"].isna() | df["mission_research"].isna()
    if nan_mask.any():
        nan_count = nan_mask.sum()
        LOGGER.warning(
            "Found %d row(s) with NaN in mission_research_category or mission_research. Reprocessing individually.",
            nan_count
        )
        
        # Get rows with NaN and reprocess them
        nan_rows = df[nan_mask].copy()
        ror_cache: Dict[Tuple[str, Optional[str]], Optional[RorMatch]] = {}
        
        for idx, row in nan_rows.iterrows():
            affiliation = str(row.get("affiliation", ""))
            country_code = str(row.get("country_code", ""))
            afid = row.get("afid")
            
            # Check ROR cache or match ROR
            cache_key = (affiliation, country_code if country_code else None)
            if cache_key in ror_cache:
                ror_match = ror_cache[cache_key]
            else:
                ror_match = None
                if ror_available:
                    try:
                        ror_match = match_ror(affiliation, country_code if country_code else None)
                    except Exception as err:
                        LOGGER.debug("ROR matching failed for '%s': %s", affiliation, err)
                ror_cache[cache_key] = ror_match
            
            # Reprocess individually
            try:
                result = classify_affiliation(affiliation, country_code, ror_match=ror_match)
                result_dict = _attach_ror_fields(result, ror_match)
                
                # Update the row in the dataframe
                df.loc[idx, "org_type"] = result_dict.get("org_type")
                df.loc[idx, "gov_level"] = result_dict.get("gov_level")
                df.loc[idx, "gov_local_type"] = result_dict.get("gov_local_type")
                df.loc[idx, "mission_research_category"] = result_dict.get("mission_research_category")
                df.loc[idx, "mission_research"] = result_dict.get("mission_research")
                df.loc[idx, "confidence_org_type"] = result_dict.get("confidence_org_type")
                df.loc[idx, "confidence_gov_level"] = result_dict.get("confidence_gov_level")
                df.loc[idx, "confidence_mission_research"] = result_dict.get("confidence_mission_research")
                df.loc[idx, "rationale"] = result_dict.get("rationale")
                
                # Update ROR fields
                if ror_match is not None:
                    df.loc[idx, "ror_id"] = ror_match.ror_id
                    df.loc[idx, "ror_name"] = ror_match.ror_name
                    df.loc[idx, "ror_types"] = ", ".join(ror_match.ror_types) if ror_match.ror_types else None
                    df.loc[idx, "ror_country_code"] = ror_match.ror_country_code
                    df.loc[idx, "ror_state"] = ror_match.ror_state
                    df.loc[idx, "ror_city"] = ror_match.ror_city
                    df.loc[idx, "ror_domains"] = ", ".join(ror_match.ror_domains) if ror_match.ror_domains else None
                    df.loc[idx, "ror_match_score"] = ror_match.match_score
                    df.loc[idx, "suggested_org_type_from_ror"] = ror_match.suggested_org_type_from_ror
                else:
                    df.loc[idx, "ror_id"] = None
                    df.loc[idx, "ror_name"] = None
                    df.loc[idx, "ror_types"] = None
                    df.loc[idx, "ror_country_code"] = None
                    df.loc[idx, "ror_state"] = None
                    df.loc[idx, "ror_city"] = None
                    df.loc[idx, "ror_domains"] = None
                    df.loc[idx, "ror_match_score"] = None
                    df.loc[idx, "suggested_org_type_from_ror"] = None
                
                LOGGER.info("Successfully reprocessed afid '%s'", afid)
            except LMStudioError as err3:
                LOGGER.error("Failed to reprocess afid '%s' individually: %s", afid, err3)
                # Keep the NaN values - they will be written to CSV as NaN

    # Final validation: check if any NaN remain
    final_nan_mask = df["mission_research_category"].isna() | df["mission_research"].isna()
    if final_nan_mask.any():
        final_nan_count = final_nan_mask.sum()
        LOGGER.warning(
            "After reprocessing, %d row(s) still have NaN in mission_research_category or mission_research.",
            final_nan_count
        )

    # Remove columns that are no longer used analytically before writing CSV
    columns_to_drop = [
        "confidence_org_type",
        "confidence_gov_level",
        "confidence_mission_research",
        "rationale",
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)

    LOGGER.info("Writing output CSV: %s", args.output)
    df.to_csv(args.output, index=False)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

