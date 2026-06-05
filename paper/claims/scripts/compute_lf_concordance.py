from __future__ import annotations

# Release documentation:
# Computes claim-registry values for lf concordance.
#
# Reads: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Writes: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Backs paper claim registry entries for lf concordance.

from pathlib import Path

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._lf_concordance_primary import (
    PATTERNS_DIR,
    build_pooled_context,
    compute_aki_claim_metrics,
    compute_icd_target_rates,
)
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()
REPO = Path(__file__).resolve().parents[3]

TARGETS = {
    "cardiac_hf": ("admission_reason_tags", "cardiac_hf"),
    "trauma_fracture": ("admission_reason_tags", "trauma_fracture"),
}

INPUT_FILES = [
    "data/raw_responses/methodology_1k_a/results.jsonl",
    "data/raw_responses/methodology_1k_b/results.jsonl",
    "data/raw_responses/methodology_1k_c/results.jsonl",
    "data/raw_responses/methodology_5k_a_subset500/results.jsonl",
    "data/raw_responses/methodology_5k_audit_b/results.jsonl",
    "data/raw_responses/methodology_5k_audit_c/results.jsonl",
    "data/raw_responses/production_v1/results.jsonl",
    "data/raw_responses/extended_5k_b/results.jsonl",
    "data/raw_responses/extended_5k_c/results.jsonl",
    "data/splits/methodology_1k.csv",
    "data/splits/methodology_5k_audit_500.csv",
    "data/splits/extended_5k.csv",
    "src/labeling_functions/icd_lf.py",
    "src/labeling_functions/regex_lf.py",
    "src/labeling_functions/base.py",
    "src/labeling_functions/section_parser.py",
    "src/db/connection.py",
    "src/db/queries.py",
]
INPUT_FILES.extend(
    sorted(
        str(path.relative_to(REPO))
        for path in PATTERNS_DIR.glob("*.yaml")
        if path.is_file()
    )
)


def compute_all() -> dict:
    require_input_files(INPUT_FILES)
    ctx = build_pooled_context()
    aki_values = compute_aki_claim_metrics(ctx)
    target_values = compute_icd_target_rates(ctx, TARGETS)

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)

    claims = {
        "aki_llm_a_prevalence_pct": claim_entry(
            value=aki_values["aki_llm_a_prevalence_pct"],
            format_default=".2f",
            unit="%",
            description="AKI positive prevalence, LLM variant A",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aki_llm_c_prevalence_pct": claim_entry(
            value=aki_values["aki_llm_c_prevalence_pct"],
            format_default=".2f",
            unit="%",
            description="AKI positive prevalence, LLM variant C",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aki_icd_lf_prevalence_pct": claim_entry(
            value=aki_values["aki_icd_lf_prevalence_pct"],
            format_default=".2f",
            unit="%",
            description="AKI positive prevalence, ICD LF",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aki_regex_lf_prevalence_pct": claim_entry(
            value=aki_values["aki_regex_lf_prevalence_pct"],
            format_default=".2f",
            unit="%",
            description="AKI positive prevalence, regex LF",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aki_llm_vs_icd_kappa_a": claim_entry(
            value=aki_values["aki_llm_vs_icd_kappa_a"],
            format_default=".4f",
            description="Cohen kappa between LLM A and ICD LF for AKI positive signal",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aki_llm_vs_icd_kappa_b": claim_entry(
            value=aki_values["aki_llm_vs_icd_kappa_b"],
            format_default=".4f",
            description="Cohen kappa between LLM B and ICD LF for AKI positive signal",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aki_icd_only_no_llm_count": claim_entry(
            value=aki_values["aki_icd_only_no_llm_count"],
            format_default=",d",
            description="Count where ICD LF is positive and all LLM variants are negative for AKI",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aki_all_signals_negative_count": claim_entry(
            value=aki_values["aki_all_signals_negative_count"],
            format_default=",d",
            description="Count where all five AKI signals are negative",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "aki_all_llm_positive_no_icd_count": claim_entry(
            value=aki_values["aki_all_llm_positive_no_icd_count"],
            format_default=",d",
            description="Count where all LLM variants are positive for AKI and ICD LF is negative",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "cross_variant_pooled_n": claim_entry(
            value=aki_values["cross_variant_pooled_n"],
            format_default=",d",
            description="Shared pooled denominator for cross-variant analyses (A/B/C pooled)",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "cardiac_hf_p_llm_yes_given_icd": claim_entry(
            value=target_values["cardiac_hf_p_llm_yes_given_icd"],
            format_default=".4f",
            description="Empirical conditional rate P(LLM yes | ICD LF positive) for admission_reason_tags::cardiac_hf",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "cardiac_hf_p_icd_given_llm_yes": claim_entry(
            value=target_values["cardiac_hf_p_icd_given_llm_yes"],
            format_default=".4f",
            description="Empirical conditional rate P(ICD LF positive | LLM yes) for admission_reason_tags::cardiac_hf",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "trauma_fracture_p_llm_yes_given_icd": claim_entry(
            value=target_values["trauma_fracture_p_llm_yes_given_icd"],
            format_default=".4f",
            description="Empirical conditional rate P(LLM yes | ICD LF positive) for admission_reason_tags::trauma_fracture",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
        "trauma_fracture_p_icd_given_llm_yes": claim_entry(
            value=target_values["trauma_fracture_p_icd_given_llm_yes"],
            format_default=".4f",
            description="Empirical conditional rate P(ICD LF positive | LLM yes) for admission_reason_tags::trauma_fracture",
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        ),
    }

    return claims


def main() -> int:
    new_claims = compute_all()
    n = merge_into_claims_json(CLAIMS_PATH, new_claims)
    print(f"Updated {n} claims in {CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
