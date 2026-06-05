from __future__ import annotations

# Release documentation:
# Computes claim-registry values for model size effect.
#
# Reads: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Writes: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Backs paper claim registry entries for model size effect.

from pathlib import Path
import numpy as np

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._model_size_primary import (
    load_paired_model_size_data,
    per_field_model_size_deltas,
    per_variant_cross_model_kappa,
)
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso


CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()

INPUT_FILES = [
    "data/raw_responses/methodology_1k_a/results.jsonl",
    "data/raw_responses/methodology_1k_b/results.jsonl",
    "data/raw_responses/methodology_1k_c/results.jsonl",
    "data/raw_responses/methodology_5k_a_subset500/results.jsonl",
    "data/raw_responses/methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/methodology_5k_audit_c/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_a/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_b/results.jsonl",
    "data/raw_responses/paired_gold_methodology_1k_c/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_a/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/paired_gold_methodology_5k_audit_c/results.jsonl",
    "src/schema/fields.py",
]

# Keep outputs strictly to currently retained Batch-4 claims.
ALLOWED_OUTPUT_KEYS = {
    "aki_present_delta_collapsed",
    "aki_present_delta_full",
    "cardiac_rehab_delta_collapsed",
    "cardiac_rehab_delta_full",
    "cognitive_impairment_delta_collapsed",
    "cognitive_impairment_delta_full",
    "discharge_delayed_reason_delta_collapsed",
    "discharge_delayed_reason_delta_full",
    "dnr_dni_delta_collapsed",
    "dnr_dni_delta_full",
    "financial_hardship_delta_collapsed",
    "financial_hardship_delta_full",
    "goals_of_care_delta_collapsed",
    "goals_of_care_delta_full",
    "home_health_delta_collapsed",
    "home_health_delta_full",
    "hospital_acquired_complication_delta_collapsed",
    "hospital_acquired_complication_delta_full",
    "model_size_perfield_delta_collapsed",
    "model_size_perfield_delta_tristate",
    "n_fields_negative_delta_full",
    "palliative_care_delta_collapsed",
    "palliative_care_delta_full",
    "shock_present_delta_collapsed",
    "shock_present_delta_full",
    "social_support_absent_delta_collapsed",
    "social_support_absent_delta_full",
    "aa_model_size_kappa_tristate",
    "aa_model_size_kappa_collapsed",
    "bb_model_size_kappa_tristate",
    "bb_model_size_kappa_collapsed",
    "cc_model_size_kappa_tristate",
    "cc_model_size_kappa_collapsed",
}

FIELD_KEY_MAP = {
    "cardiac_rehab_referred": "cardiac_rehab",
    "aki_present": "aki_present",
    "palliative_care_consult": "palliative_care",
    "home_health_ordered": "home_health",
    "hospital_acquired_complication": "hospital_acquired_complication",
    "shock_present": "shock_present",
    "cognitive_impairment": "cognitive_impairment",
    "goals_of_care_flag": "goals_of_care",
    "dnr_dni_documented": "dnr_dni",
    "social_support_absent": "social_support_absent",
    "financial_hardship": "financial_hardship",
    "discharge_delayed_reason": "discharge_delayed_reason",
}


def compute_all() -> dict:
    require_input_files(INPUT_FILES)
    data = load_paired_model_size_data(base_rate_threshold=10)
    full_delta_pp, collapsed_delta_pp = per_field_model_size_deltas(data)
    per_variant_kappa = per_variant_cross_model_kappa(data)
    included_fields = data.included_fields

    full_values = np.asarray([full_delta_pp[f] for f in included_fields], dtype=np.float64)
    collapsed_values = np.asarray([collapsed_delta_pp[f] for f in included_fields], dtype=np.float64)
    perfield_full = float(np.median(full_values))
    perfield_collapsed = float(np.median(collapsed_values))
    n_negative = int(np.sum(full_values < 0.0))

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)
    claims: dict[str, dict] = {}

    for field, key_prefix in FIELD_KEY_MAP.items():
        if field not in full_delta_pp or field not in collapsed_delta_pp:
            continue
        full_key = f"{key_prefix}_delta_full"
        collapsed_key = f"{key_prefix}_delta_collapsed"
        if full_key in ALLOWED_OUTPUT_KEYS:
            claims[full_key] = claim_entry(
                value=full_delta_pp[field],
                format_default=".2f",
                unit="pp",
                description=f"{field} model-size delta (full TriState), full minus small kappa",
                sample="methodology_1500",
                computed_at=timestamp,
                receipt=receipt,
            )
        if collapsed_key in ALLOWED_OUTPUT_KEYS:
            claims[collapsed_key] = claim_entry(
                value=collapsed_delta_pp[field],
                format_default=".2f",
                unit="pp",
                description=f"{field} model-size delta (collapsed), full minus small kappa",
                sample="methodology_1500",
                computed_at=timestamp,
                receipt=receipt,
            )

    if "model_size_perfield_delta_tristate" in ALLOWED_OUTPUT_KEYS:
        claims["model_size_perfield_delta_tristate"] = claim_entry(
            value=perfield_full,
            format_default=".2f",
            unit="pp",
            description=(
                "Per-field median model-size delta across TriState fields: median over fields "
                "of (full_kappa - small_kappa), full TriState; "
                "fields with <10 yes-votes in either model are excluded"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )
    if "model_size_perfield_delta_collapsed" in ALLOWED_OUTPUT_KEYS:
        claims["model_size_perfield_delta_collapsed"] = claim_entry(
            value=perfield_collapsed,
            format_default=".2f",
            unit="pp",
            description=(
                "Per-field median model-size delta across TriState fields: median over fields "
                "of (full_kappa - small_kappa), collapsed; "
                "fields with <10 yes-votes in either model are excluded"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )
    if "n_fields_negative_delta_full" in ALLOWED_OUTPUT_KEYS:
        claims["n_fields_negative_delta_full"] = claim_entry(
            value=n_negative,
            format_default="d",
            description=(
                "Number of base-rate-included TriState fields with negative model-size delta "
                "at full TriState"
            ),
            sample="methodology_1500",
            computed_at=timestamp,
            receipt=receipt,
        )

    for variant in ("A", "B", "C"):
        tri, collapsed = per_variant_kappa[variant]
        key_prefix = variant.lower() * 2
        tri_key = f"{key_prefix}_model_size_kappa_tristate"
        collapsed_key = f"{key_prefix}_model_size_kappa_collapsed"
        if tri_key in ALLOWED_OUTPUT_KEYS:
            claims[tri_key] = claim_entry(
                value=float(tri),
                format_default=".4f",
                description=(
                    f"Within-variant cross-model median kappa under TriState for variant {variant} "
                    "(small vs full on same 1500 notes; median over base-rate-included TriState fields)"
                ),
                sample="methodology_1500",
                computed_at=timestamp,
                receipt=receipt,
            )
        if collapsed_key in ALLOWED_OUTPUT_KEYS:
            claims[collapsed_key] = claim_entry(
                value=float(collapsed),
                format_default=".4f",
                description=(
                    f"Within-variant cross-model median kappa under binary collapse for variant {variant} "
                    "(small vs full on same 1500 notes; median over base-rate-included TriState fields)"
                ),
                sample="methodology_1500",
                computed_at=timestamp,
                receipt=receipt,
            )
    return claims


def main() -> int:
    new_claims = compute_all()
    n = merge_into_claims_json(CLAIMS_PATH, new_claims)
    print(f"Updated {n} claims in {CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
