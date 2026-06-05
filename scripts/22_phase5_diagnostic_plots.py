"""Phase 5 methodology-1k diagnostic plot builder.

Reads: `data/raw_responses/methodology_1k_{a,b,c}/results.jsonl`, kappa JSON reports for refinement/holdout/1k, and `data/optimization/audit_corpus_methodology_1k.jsonl`.
Writes: `docs/figures/22_tristate_baserates_1k.png`, `22_pairwise_kappa_heatmap_1k.png`, `22_disagreement_outlier_breakdown_1k.png`, `22_kappa_across_sample_sizes.png`, plus `docs/figures/README.md`.
Paper role: development diagnostic layer for methodology readiness; not a final named paper figure.
Usage: `python scripts/22_phase5_diagnostic_plots.py` unless argparse help says otherwise.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Any

from src import config
from src.utils.diagnostic_plots import (
    plot_disagreement_outlier_breakdown,
    plot_pairwise_kappa_heatmap,
    plot_sample_size_kappa_comparison,
    plot_tristate_baserates,
)

TRISTATE_FIELDS = [
    "shock_present",
    "infection_as_trigger",
    "aki_present",
    "lives_alone",
    "social_support_absent",
    "financial_hardship",
    "substance_use_active",
    "fall_risk_documented",
    "cognitive_impairment",
    "goals_of_care_flag",
    "palliative_care_consult",
    "dnr_dni_documented",
    "home_health_ordered",
    "cardiac_rehab_referred",
    "discharge_delayed_reason",
    "hospital_acquired_complication",
    "unresolved_diagnosis_at_discharge",
]


def _load_results_features(run_id: str) -> list[dict[str, Any]]:
    path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    features: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if bool(row.get("parse_ok")) and isinstance(row.get("features_json"), dict):
            features.append(row["features_json"])
    return features


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_figure_readme(path: Path) -> None:
    lines = [
        "# Figures Index",
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 5 diagnostic visualizations.")
    parser.add_argument(
        "--run-ids",
        nargs=3,
        default=["methodology_1k_a", "methodology_1k_b", "methodology_1k_c"],
    )
    parser.add_argument(
        "--kappa-1k",
        default="codex_outputs/22_methodology_1k_kappa_report.md.json",
    )
    parser.add_argument(
        "--kappa-refinement",
        default="codex_outputs/16c_iter2_kappa.md.json",
    )
    parser.add_argument(
        "--kappa-holdout",
        default="codex_outputs/21_holdout_kappa_report.md.json",
    )
    parser.add_argument(
        "--audit-corpus",
        default="data/optimization/audit_corpus_methodology_1k.jsonl",
    )
    parser.add_argument("--figures-dir", default="docs/figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    run_a, run_b, run_c = args.run_ids
    extractions = {
        "A": _load_results_features(run_a),
        "B": _load_results_features(run_b),
        "C": _load_results_features(run_c),
    }
    kappa_1k = _load_json(Path(args.kappa_1k))
    kappa_refinement = _load_json(Path(args.kappa_refinement))
    kappa_holdout = _load_json(Path(args.kappa_holdout))
    audit_rows = _load_jsonl(Path(args.audit_corpus))

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_tristate_baserates(
        extractions=extractions,
        fields=TRISTATE_FIELDS,
        output_path=figures_dir / "22_tristate_baserates_1k.png",
    )
    plot_pairwise_kappa_heatmap(
        kappa_data=kappa_1k,
        output_path=figures_dir / "22_pairwise_kappa_heatmap_1k.png",
    )
    plot_disagreement_outlier_breakdown(
        corpus_records=audit_rows,
        output_path=figures_dir / "22_disagreement_outlier_breakdown_1k.png",
    )
    plot_sample_size_kappa_comparison(
        refinement_kappa=kappa_refinement,
        holdout_kappa=kappa_holdout,
        methodology_1k_kappa=kappa_1k,
        output_path=figures_dir / "22_kappa_across_sample_sizes.png",
    )

    _write_figure_readme(figures_dir / "README.md")
    print(f"Wrote Phase 5 diagnostic figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
