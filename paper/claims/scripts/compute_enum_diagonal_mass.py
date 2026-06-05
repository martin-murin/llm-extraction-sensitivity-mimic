from __future__ import annotations

# Release documentation:
# Computes claim-registry values for enum diagonal mass.
#
# Reads: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Writes: data/raw_responses/methodology_1k_a/results.jsonl, data/raw_responses/methodology_1k_b/results.jsonl, data/raw_responses/methodology_1k_c/results.jsonl, data/raw_responses/methodology_5k_a_subset500/results.jsonl, data/raw_responses/methodology_5k_audit_b/results.jsonl, data/raw_responses/methodology_5k_audit_c/results.jsonl.
# Backs paper claim registry entries for enum diagonal mass.

from pathlib import Path

from paper.claims.scripts._common import claim_entry, require_input_files
from paper.claims.scripts._receipt import build_receipt, merge_into_claims_json, now_utc_iso
from src.paper_figures._s07_enum_pairwise_common import (
    PAIR_KEYS,
    EnumPairFigConfig,
    _compute_confusion,
    _load_pooled_enum_frame,
)

CLAIMS_PATH = Path(__file__).resolve().parent.parent / "claims.json"
SCRIPT_PATH = Path(__file__).resolve()

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
]


def _pair_diagonal_pct(frame, labels: tuple[str, ...], left: str, right: str) -> float:
    counts, _rates, total = _compute_confusion(frame, labels, left, right)
    if total <= 0:
        return float("nan")
    return float(counts.trace() / total * 100.0)


def _field_claims(
    *,
    field_prefix: str,
    cfg: EnumPairFigConfig,
    timestamp: str,
    receipt: dict,
) -> dict[str, dict]:
    frame = _load_pooled_enum_frame(cfg)
    out: dict[str, dict] = {}
    for left, right in PAIR_KEYS:
        pair_key = f"{left}{right}".lower()
        key = f"enum_{field_prefix}_{pair_key}_diagonal_pct"
        out[key] = claim_entry(
            value=_pair_diagonal_pct(frame, cfg.labels, left, right),
            format_default=".1f",
            unit="%",
            description=(
                f"Diagonal mass (%) for enum field '{cfg.field}' in cross-variant pair {left}-{right} "
                "on pooled A/B/C sample"
            ),
            sample="methodology_6500_pooled",
            computed_at=timestamp,
            receipt=receipt,
        )
    return out


def compute_all() -> dict:
    require_input_files(INPUT_FILES)

    timestamp = now_utc_iso()
    receipt = build_receipt(SCRIPT_PATH, "compute_all", INPUT_FILES)

    mental_cfg = EnumPairFigConfig(
        figure_name="paper_fig_S07a_mental_status_confusion",
        field="mental_status",
        pretty_name="mental_status",
        labels=("intact", "mild_impairment", "confused_delirious", "not_documented"),
    )
    functional_cfg = EnumPairFigConfig(
        figure_name="paper_fig_S07b_functional_status_confusion",
        field="functional_status",
        pretty_name="functional_status",
        labels=("independent", "assisted", "dependent", "not_documented"),
    )
    discharge_cfg = EnumPairFigConfig(
        figure_name="paper_fig_S07c_discharge_condition_confusion",
        field="discharge_condition_category",
        pretty_name="discharge_condition_category",
        labels=("stable", "improved", "unchanged", "deteriorated", "expired", "not_documented"),
    )

    claims: dict[str, dict] = {}
    claims.update(
        _field_claims(
            field_prefix="mental_status",
            cfg=mental_cfg,
            timestamp=timestamp,
            receipt=receipt,
        )
    )
    claims.update(
        _field_claims(
            field_prefix="functional_status",
            cfg=functional_cfg,
            timestamp=timestamp,
            receipt=receipt,
        )
    )
    claims.update(
        _field_claims(
            field_prefix="discharge_condition",
            cfg=discharge_cfg,
            timestamp=timestamp,
            receipt=receipt,
        )
    )
    return claims


def main() -> int:
    new_claims = compute_all()
    n = merge_into_claims_json(CLAIMS_PATH, new_claims)
    print(f"Updated {n} claims in {CLAIMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

