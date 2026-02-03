"""
CLI utilities to classify affiliations as governmental or not using LM Studio.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from lm_client import (
    LMStudioError,
    classify_affiliation,
    classify_affiliations_batch,
    get_classifier,
    set_default_backend,
    try_rule_based_classification,
    _is_technical_error,
)
from ror_knowledge import RorMatch, load_ror, match_ror

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger("gov-affiliation-classifier")

# Control flag for individual fallback processing
# Default: False for production runs (faster, no reprocessing)
# Set to True for debugging to enable individual reprocessing on technical errors
ENABLE_INDIVIDUAL_FALLBACK = False


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
    parser.add_argument(
        "--enable-individual-fallback",
        action="store_true",
        help="Enable individual reprocessing on technical errors (slower, for debugging only)",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "local"],
        default="local",
        help="LLM backend: 'gemini' (Google API) or 'local' (LM Studio / OpenAI-compatible). Default: local.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Google Gemini API key (or set GEMINI_API_KEY). Used when --backend=gemini.",
    )
    parser.add_argument(
        "--local-url",
        default=None,
        help="Base URL for local LLM (e.g. http://localhost:1234/v1). Used when --backend=local. (Or set LM_STUDIO_BASE_URL.)",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Model name for local backend (or set LM_STUDIO_MODEL_NAME).",
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


def _ensure_no_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any remaining NaN values in analytical fields with 'unknown' or appropriate defaults.
    This is a safety net to ensure no NaN values are written to the CSV.
    """
    analytical_fields = [
        "org_type",
        "gov_level",
        "gov_local_type",
        "mission_research_category",
        "mission_research",
    ]
    
    for field in analytical_fields:
        if field in df.columns:
            nan_count = df[field].isna().sum()
            if nan_count > 0:
                LOGGER.warning(
                    "Found %d NaN values in '%s'. Replacing with 'unknown'.",
                    nan_count,
                    field
                )
                if field == "mission_research":
                    # mission_research should be 0 or 1, use 0 for unknown
                    df[field] = df[field].fillna(0)
                else:
                    df[field] = df[field].fillna("unknown")
    
    return df


def _merge_results_by_afid(results_list: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge results by afid, prioritizing complete results over those with 'unknown' values.
    
    Priority order:
    1. Complete result (no 'unknown' in analytical fields)
    2. Result with some 'unknown' values
    3. Empty/technical result (should not exist)
    
    If multiple complete results exist for same afid, use the last one (most recent).
    """
    from collections import defaultdict
    
    # Group by afid
    afid_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, result_dict in results_list:
        afid = result_dict.get("afid")
        if afid:
            afid_groups[str(afid)].append((idx, result_dict))
    
    merged_results = []
    for afid, group in afid_groups.items():
        if len(group) == 1:
            # Single result, use it
            merged_results.append((group[0][0], group[0][1]))
        else:
            # Multiple results, select best one
            # Score: count of non-unknown values in analytical fields
            analytical_fields = ["org_type", "gov_level", "gov_local_type", "mission_research_category"]
            
            def score_result(result_dict: Dict[str, Any]) -> int:
                score = 0
                for field in analytical_fields:
                    value = result_dict.get(field)
                    if value and value != "unknown" and value != "non_applicable":
                        score += 1
                return score
            
            # Sort by score (descending), then by index (descending) to prefer most recent
            sorted_group = sorted(group, key=lambda x: (score_result(x[1]), -x[0]), reverse=True)
            best_result = sorted_group[0]
            merged_results.append((best_result[0], best_result[1]))
            
            if len(group) > 1:
                LOGGER.debug(
                    "Merged %d results for afid '%s', selected result with score %d",
                    len(group),
                    afid,
                    score_result(best_result[1])
                )
    
    return merged_results


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
        except Exception as err:
            is_technical = _is_technical_error(err)
            if is_technical:
                LOGGER.error(
                    "Batch LLM call failed with technical error: %s. Error type: %s",
                    err,
                    type(err).__name__
                )
            else:
                LOGGER.warning(
                    "Batch LLM call failed with semantic/parsing issue: %s. Using 'unknown' values.",
                    err
                )
            
            # Only reprocess individually if it's a technical error AND fallback is enabled
            if is_technical and ENABLE_INDIVIDUAL_FALLBACK:
                LOGGER.info("Individual fallback enabled. Reprocessing items individually.")
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
                    except Exception as err2:
                        is_technical_2 = _is_technical_error(err2)
                        if is_technical_2:
                            LOGGER.error(
                                "Individual LLM call also failed with technical error for '%s': %s",
                                item["affiliation"],
                                err2
                            )
                        else:
                            LOGGER.warning(
                                "Individual LLM call failed with semantic issue for '%s': %s",
                                item["affiliation"],
                                err2
                            )
                        # Create result with "unknown" values (not None)
                        error_result = {
                            "afid": item["id"],
                            "org_type": "unknown",
                            "gov_level": "unknown",
                            "gov_local_type": "unknown",
                            "mission_research_category": "unknown",
                            "mission_research": 0,
                            "confidence_org_type": None,
                            "confidence_gov_level": None,
                            "confidence_mission_research": None,
                            "rationale": f"Error: {err2}",
                        }
                        result_dict = _attach_ror_fields(error_result, item.get("ror_match"))
                        results.append((item["original_idx"], result_dict))
            else:
                # Fallback disabled or non-technical error: mark all items as "unknown"
                LOGGER.info(
                    "Individual fallback disabled or non-technical error. Marking %d items as 'unknown'.",
                    len(llm_items)
                )
                for item in llm_items:
                    unknown_result = {
                        "afid": item["id"],
                        "org_type": "unknown",
                        "gov_level": "unknown",
                        "gov_local_type": "unknown",
                        "mission_research_category": "unknown",
                        "mission_research": 0,
                        "confidence_org_type": None,
                        "confidence_gov_level": None,
                        "confidence_mission_research": None,
                        "rationale": f"Batch error: {err}",
                    }
                    result_dict = _attach_ror_fields(unknown_result, item.get("ror_match"))
                    results.append((item["original_idx"], result_dict))
    
    return results, rule_based_count, 1 if llm_items else 0


def run_classification_pipeline(
    df: pd.DataFrame,
    ror_path: Optional[str] = None,
    batch_size: int = 8,
    enable_individual_fallback: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de clasificación sobre un DataFrame.

    Usa el backend por defecto (establecido con set_default_backend).
    Requiere columnas: afid, affiliation, country_code.

    Returns
    -------
    pd.DataFrame
        DataFrame original enriquecido con columnas de clasificación y ROR.
    """
    required_cols = {"afid", "affiliation", "country_code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ror_available = False
    if ror_path:
        try:
            load_ror(ror_path)
            ror_available = True
            LOGGER.info("ROR knowledge base loaded successfully")
        except Exception as err:
            LOGGER.warning("Failed to load ROR dump: %s. Continuing without ROR matching.", err)

    ror_cache: Dict[Tuple[str, Optional[str]], Optional[RorMatch]] = {}
    all_results: List[Tuple[int, Dict[str, Any]]] = []
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

    merged_results = _merge_results_by_afid(all_results)
    merged_results.sort(key=lambda x: x[0])
    classification_results = [result[1] for result in merged_results]
    results = pd.DataFrame(classification_results)

    results_with_afid = results.set_index("afid")
    df["org_type"] = df["afid"].map(results_with_afid["org_type"])
    df["gov_level"] = df["afid"].map(results_with_afid["gov_level"])
    df["gov_local_type"] = df["afid"].map(results_with_afid["gov_local_type"])
    df["mission_research_category"] = df["afid"].map(results_with_afid["mission_research_category"])
    df["mission_research"] = df["afid"].map(results_with_afid["mission_research"])
    df["confidence_org_type"] = df["afid"].map(results_with_afid["confidence_org_type"])
    df["confidence_gov_level"] = df["afid"].map(results_with_afid["confidence_gov_level"])
    df["confidence_mission_research"] = df["afid"].map(results_with_afid["confidence_mission_research"])
    df["rationale"] = df["afid"].map(results_with_afid["rationale"])
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
    df = _ensure_no_nan(df)

    columns_to_drop = [
        "confidence_org_type",
        "confidence_gov_level",
        "confidence_mission_research",
        "rationale",
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)
    return df


def main() -> None:
    global ENABLE_INDIVIDUAL_FALLBACK
    args = _parse_args()

    ENABLE_INDIVIDUAL_FALLBACK = args.enable_individual_fallback
    if ENABLE_INDIVIDUAL_FALLBACK:
        LOGGER.info("Individual fallback ENABLED (debugging mode - slower)")
    else:
        LOGGER.info("Individual fallback DISABLED (production mode - faster)")

    # Configurar backend
    backend_type = args.backend
    if backend_type == "gemini":
        config = {"api_key": args.gemini_api_key}
    else:
        config = {
            "base_url": args.local_url,
            "model_name": args.local_model,
        }
    backend = get_classifier(backend_type, config)
    set_default_backend(backend)
    LOGGER.info("Using backend: %s", backend_type)

    LOGGER.info("Reading input CSV: %s", args.input)
    df = pd.read_csv(args.input)

    out_df = run_classification_pipeline(
        df,
        ror_path=args.ror_path,
        enable_individual_fallback=ENABLE_INDIVIDUAL_FALLBACK,
    )

    LOGGER.info("Writing output CSV: %s", args.output)
    out_df.to_csv(args.output, index=False)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

