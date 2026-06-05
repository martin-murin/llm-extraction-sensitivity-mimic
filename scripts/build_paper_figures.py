#!/usr/bin/env python3
from __future__ import annotations

# Release documentation:
# Runs staged pipeline step `build_paper_figures.py`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs all main and supplement paper figures.
# Usage: `python scripts/build_paper_figures.py` unless the script's argparse help says otherwise.

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIGURES: dict[str, tuple[str, str]] = {
    "01": ("fig01_per_field_full_vs_collapsed", "main"),
    "02": ("fig02_cross_prompt_grouped_bars", "main"),
    "03": ("fig03_kappa_tristate_collapsed", "main"),
    "04": ("fig04_tag_prevalence_three_panels", "main"),
    "05a": ("fig05a_dominant_admission_three_matrices", "main"),
    "05b": ("fig05b_dominant_admission_triangulated", "main"),
    "S01a": ("figS01a_llm_vs_icd_concordance", "supplement"),
    "S01b": ("figS01b_llm_vs_regex_concordance", "supplement"),
    "S01c": ("figS01c_aki_five_signal_concordance", "supplement"),
    "S02": ("figS02_per_variant_collapse", "supplement"),
    "S03ab": ("figS03_admission_tag_47_cross_variant_ab", "supplement"),
    "S03ac": ("figS03_admission_tag_47_cross_variant_ac", "supplement"),
    "S03bc": ("figS03_admission_tag_47_cross_variant_bc", "supplement"),
    "S03aa": ("figS03_admission_tag_47_model_size_aa", "supplement"),
    "S03bb": ("figS03_admission_tag_47_model_size_bb", "supplement"),
    "S03cc": ("figS03_admission_tag_47_model_size_cc", "supplement"),
    "S04": ("figS04_disagreement_decomposition_dual", "supplement"),
    "S05": ("figS05_refinement_holdout_generalization", "supplement"),
    "S06": ("figS06_sample_size_stability_forest", "supplement"),
    "S07a": ("figS07a_mental_status_confusion", "supplement"),
    "S07b": ("figS07b_functional_status_confusion", "supplement"),
    "S07c": ("figS07c_discharge_condition_confusion", "supplement"),
}


def _build_one(fig_key: str) -> None:
    module_name, _group = FIGURES[fig_key]
    module = importlib.import_module(f"src.paper_figures.{module_name}")
    if not hasattr(module, "build"):
        raise RuntimeError(f"Module {module_name} has no build()")
    module.build()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-quality paper figures.")
    parser.add_argument("--fig", action="append", help="Figure key(s), e.g. --fig 03 --fig S02")
    parser.add_argument("--main", action="store_true", help="Build main paper figures only")
    parser.add_argument(
        "--supplement", action="store_true", help="Build supplement figures only"
    )
    parser.add_argument("--list", action="store_true", help="List figure registry")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        for key, (module, group) in FIGURES.items():
            print(f"{key:>3}  {group:<10}  src.paper_figures.{module}")
        return 0

    selected: list[str]
    if args.fig:
        requested = [x.strip() for x in args.fig]
        unknown = [x for x in requested if x not in FIGURES]
        if unknown:
            raise ValueError(f"Unknown figure key(s): {unknown}")
        selected = requested
    elif args.main and not args.supplement:
        selected = [k for k, (_m, grp) in FIGURES.items() if grp == "main"]
    elif args.supplement and not args.main:
        selected = [k for k, (_m, grp) in FIGURES.items() if grp == "supplement"]
    else:
        selected = list(FIGURES.keys())

    for key in selected:
        print(f"[build] {key}")
        _build_one(key)

    print(f"Built {len(selected)} figure module(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
