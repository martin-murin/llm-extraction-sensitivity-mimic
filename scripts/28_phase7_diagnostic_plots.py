"""Phase 7 prep diagnostic plot builder.

Reads: local kappa JSON reports across refinement/holdout/1k/5k-audit, `data/raw_responses/**`, and `data/methodology_5k/predictions.parquet`.
Writes: `docs/figures/28_kappa_with_ci_across_samples.png`, `28_snorkel_probability_distributions.png`, and `28_baserate_stability_4way.png`.
Paper role: final pre-production diagnostic layer; informs methodology/limitations but is not itself a final named paper figure.
Usage: `python scripts/28_phase7_diagnostic_plots.py` unless argparse help says otherwise.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from src.labeling_functions.icd_lf import ICD_LF_SPECS
from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.diagnostic_plots import (
    TRISTATE_FIELDS,
    plot_baserate_stability_grid,
    plot_kappa_with_bootstrap_ci,
    plot_snorkel_probability_distributions,
)
from src.utils.threeway_kappa import cohen_kappa_safe

TRISTATE_TO_INT = {"yes": 1, "no": -1, "not_documented": 0}

ENUM_FIELDS = {"functional_status", "mental_status", "discharge_condition_category"}

BASELINE_SAMPLES: dict[str, tuple[str, str, str]] = {
    "refinement_150": ("refinement_v1_a", "refinement_v1_b", "refinement_v1_c"),
    "holdout_150": ("holdout_v1_a", "holdout_v1_b", "holdout_v1_c"),
    "methodology_1k": ("methodology_1k_a", "methodology_1k_b", "methodology_1k_c"),
    "methodology_5k_audit_500": (
        "methodology_5k_a_subset500",
        "methodology_5k_audit_b",
        "methodology_5k_audit_c",
    ),
}

STABILITY_TRISTATE_FIELDS = [
    "shock_present",
    "infection_as_trigger",
    "aki_present",
    "substance_use_active",
    "fall_risk_documented",
    "cognitive_impairment",
    "goals_of_care_flag",
    "palliative_care_consult",
    "dnr_dni_documented",
    "home_health_ordered",
    "cardiac_rehab_referred",
    "unresolved_diagnosis_at_discharge",
]


@dataclass
class SampleVotes:
    a: dict[int, dict[str, Any]]
    b: dict[int, dict[str, Any]]
    c: dict[int, dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_results_by_hadm(run_id: str) -> dict[int, dict[str, Any]]:
    path = Path("data/raw_responses") / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not bool(row.get("parse_ok")):
            continue
        features = row.get("features_json")
        if not isinstance(features, dict):
            continue
        out[int(row["hadm_id"])] = features
    return out


def _encode_label(field_key: str, features: dict[str, Any]) -> int | str:
    if field_key.startswith("admission_reason_tags::"):
        tag = field_key.split("::", maxsplit=1)[1]
        tags = features.get("admission_reason_tags")
        if isinstance(tags, list):
            return int(tag in set(str(x) for x in tags))
        return 0

    value = features.get(field_key)
    if field_key in TRISTATE_FIELDS:
        label = str(value)
        return TRISTATE_TO_INT.get(label, 0)

    if field_key == "dominant_admission_reason":
        label = str(value)
        return label if label in ADMISSION_REASON_TAGS else "other"

    if field_key in ENUM_FIELDS:
        return str(value)

    return str(value)


def _aligned_votes(
    field_key: str,
    sample_votes: SampleVotes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = sorted(set(sample_votes.a) & set(sample_votes.b) & set(sample_votes.c))
    if not shared:
        return np.asarray([]), np.asarray([]), np.asarray([])

    a_vals = [_encode_label(field_key, sample_votes.a[hadm]) for hadm in shared]
    b_vals = [_encode_label(field_key, sample_votes.b[hadm]) for hadm in shared]
    c_vals = [_encode_label(field_key, sample_votes.c[hadm]) for hadm in shared]

    # For non-integer labels, map to deterministic ints for kappa.
    if any(not isinstance(v, (int, np.integer)) for v in a_vals + b_vals + c_vals):
        labels = sorted({str(v) for v in a_vals + b_vals + c_vals})
        mapping = {label: idx for idx, label in enumerate(labels)}
        a_arr = np.asarray([mapping[str(v)] for v in a_vals], dtype=np.int64)
        b_arr = np.asarray([mapping[str(v)] for v in b_vals], dtype=np.int64)
        c_arr = np.asarray([mapping[str(v)] for v in c_vals], dtype=np.int64)
        return a_arr, b_arr, c_arr

    return (
        np.asarray(a_vals, dtype=np.int64),
        np.asarray(b_vals, dtype=np.int64),
        np.asarray(c_vals, dtype=np.int64),
    )


def _kappa_mean(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (
        cohen_kappa_safe(a, b) + cohen_kappa_safe(a, c) + cohen_kappa_safe(b, c)
    ) / 3.0


def _bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    n_iter: int,
    seed: int,
) -> tuple[float, float, float]:
    if a.size == 0:
        return 0.0, 0.0, 0.0

    base = _kappa_mean(a, b, c)
    rng = np.random.default_rng(seed)
    scores = np.zeros(n_iter, dtype=np.float64)
    n = a.size
    for idx in range(n_iter):
        sample_idx = rng.integers(0, n, size=n)
        scores[idx] = _kappa_mean(a[sample_idx], b[sample_idx], c[sample_idx])

    lo = float(np.percentile(scores, 2.5))
    hi = float(np.percentile(scores, 97.5))
    return base, lo, hi


def _load_well_supported_field_keys(sidecar_paths: list[Path]) -> list[str]:
    keys: set[str] | None = None
    for path in sidecar_paths:
        payload = _load_json(path)
        results = payload.get("kappa_results")
        if not isinstance(results, dict):
            raise ValueError(f"Invalid sidecar kappa_results in {path}")
        current = {
            key
            for key, metrics in results.items()
            if isinstance(metrics, dict) and not bool(metrics.get("low_base_rate_flag", False))
        }
        keys = current if keys is None else (keys & current)
    if keys is None:
        return []
    return sorted(keys)


def _load_regex_fields(patterns_dir: Path) -> set[str]:
    fields: set[str] = set()
    for path in sorted(patterns_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            field_name = payload.get("field_name")
            if isinstance(field_name, str) and field_name:
                fields.add(field_name)
    return fields


def _load_phase5_1k_audit_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _collect_baserates(
    sample_votes: dict[str, SampleVotes],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for sample_name, votes in sample_votes.items():
        out[sample_name] = {}
        for variant, data in [("A", votes.a), ("B", votes.b), ("C", votes.c)]:
            out[sample_name][variant] = {}
            if not data:
                continue
            n = len(data)
            for field in STABILITY_TRISTATE_FIELDS:
                yes = 0
                no = 0
                nd = 0
                for features in data.values():
                    value = str(features.get(field, "not_documented"))
                    if value == "yes":
                        yes += 1
                    elif value == "no":
                        no += 1
                    else:
                        nd += 1
                out[sample_name][variant][field] = {
                    "yes": yes / n,
                    "no": no / n,
                    "not_documented": nd / n,
                }
    return out


def _write_figures_readme(path: Path) -> None:
    lines = [
        "# Figures Index",
        "",
        "## Phase 5.1",
        "",
        (
            "- `22_tristate_baserates_1k.png`: Stacked per-field TriState base-rate "
            "comparison (A/B/C) on methodology_1k."
        ),
        (
            "- `22_pairwise_kappa_heatmap_1k.png`: Heatmap of pairwise and mean kappa "
            "for well-supported fields on methodology_1k."
        ),
        (
            "- `22_disagreement_outlier_breakdown_1k.png`: Top-20 disagreement fields "
            "with outlier-pattern breakdown."
        ),
        (
            "- `22_kappa_across_sample_sizes.png`: Field-level kappa trajectories across "
            "refinement, holdout, and methodology_1k samples."
        ),
        "",
        "## Phase 7 Prep",
        "",
        (
            "- `28_kappa_with_ci_across_samples.png`: Field-level kappa means with "
            "bootstrap 95% CIs across refinement, holdout, methodology_1k, and "
            "methodology_5k_audit_500."
        ),
        (
            "- `28_snorkel_probability_distributions.png`: Snorkel positive-probability "
            "histograms by target-anchor group (ICD, regex, single-LF fallback)."
        ),
        (
            "- `28_baserate_stability_4way.png`: Variant-wise TriState base-rate "
            "stability grid across four samples."
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 7 diagnostic plots.")
    parser.add_argument("--kappa-refinement", default="codex_outputs/16c_iter2_kappa.md.json")
    parser.add_argument(
        "--kappa-holdout",
        default="codex_outputs/21_holdout_kappa_report.md.json",
    )
    parser.add_argument(
        "--kappa-1k",
        default="codex_outputs/22_methodology_1k_kappa_report.md.json",
    )
    parser.add_argument(
        "--kappa-5k-audit",
        default="codex_outputs/26_methodology_5k_audit_kappa_report.md.json",
    )
    parser.add_argument(
        "--predictions-parquet",
        default="data/methodology_5k/predictions.parquet",
    )
    parser.add_argument(
        "--audit-corpus",
        default="data/optimization/audit_corpus_methodology_5k_audit.jsonl",
    )
    parser.add_argument("--patterns-dir", default="src/labeling_functions/patterns")
    parser.add_argument("--figures-dir", default="docs/figures")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Kappa with CI across four samples.
    sidecars = [
        Path(args.kappa_refinement),
        Path(args.kappa_holdout),
        Path(args.kappa_1k),
        Path(args.kappa_5k_audit),
    ]
    field_keys = _load_well_supported_field_keys(sidecars)

    sample_votes = {
        sample: SampleVotes(
            a=_load_results_by_hadm(run_ids[0]),
            b=_load_results_by_hadm(run_ids[1]),
            c=_load_results_by_hadm(run_ids[2]),
        )
        for sample, run_ids in BASELINE_SAMPLES.items()
    }

    sample_order = [
        "refinement_150",
        "holdout_150",
        "methodology_1k",
        "methodology_5k_audit_500",
    ]
    sample_labels = ["refinement 150", "holdout 150", "1k", "5k-audit 500"]

    field_series: dict[str, dict[str, dict[str, float]]] = {}
    for field_idx, field_key in enumerate(field_keys):
        field_series[field_key] = {}
        for sample_idx, sample in enumerate(sample_order):
            a, b, c = _aligned_votes(field_key, sample_votes[sample])
            mean, ci_low, ci_high = _bootstrap_ci(
                a,
                b,
                c,
                n_iter=max(10, int(args.n_bootstrap)),
                seed=int(args.seed) + (sample_idx * 1_000) + field_idx,
            )
            field_series[field_key][sample] = {
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

    plot_kappa_with_bootstrap_ci(
        field_series=field_series,
        sample_order=sample_order,
        sample_labels=sample_labels,
        output_path=figures_dir / "28_kappa_with_ci_across_samples.png",
    )

    # 2) Snorkel probability distributions by target type.
    predictions = pd.read_parquet(Path(args.predictions_parquet))
    icd_target_values = {
        str(spec.get("target_value"))
        for spec in ICD_LF_SPECS
        if spec.get("target_field") == "admission_reason_tags"
    }
    regex_fields = _load_regex_fields(Path(args.patterns_dir))
    plot_snorkel_probability_distributions(
        predictions=predictions,
        icd_target_values=icd_target_values,
        regex_fields=regex_fields,
        output_path=figures_dir / "28_snorkel_probability_distributions.png",
    )

    # 3) Base-rate stability grid across four samples.
    rates_by_sample = _collect_baserates(sample_votes)
    plot_baserate_stability_grid(
        rates_by_sample=rates_by_sample,
        sample_order=sample_order,
        variant_order=["A", "B", "C"],
        fields=STABILITY_TRISTATE_FIELDS,
        output_path=figures_dir / "28_baserate_stability_4way.png",
    )

    _write_figures_readme(figures_dir / "README.md")

    # small summary log
    summary_path = Path("codex_outputs/28_phase7_plot_generation.md")
    summary_path.write_text(
        "\n".join(
            [
                "# Phase 7 Plot Generation",
                "",
                f"Generated at: {datetime.now(tz=UTC).isoformat()}",
                f"Well-supported fields plotted with CI: {len(field_series)}",
                f"Bootstrap iterations per field/sample: {max(10, int(args.n_bootstrap))}",
                f"Predictions rows (Snorkel): {len(predictions):,}",
                f"Regex fields detected: {len(regex_fields)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote plot-generation summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
