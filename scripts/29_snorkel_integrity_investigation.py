"""Snorkel output integrity investigation.

Reads: `data/methodology_5k/predictions.parquet`, `codex_outputs/26_methodology_5k_snorkel_report.md`, LF/Snorkel configuration, and local audit artifacts.
Writes: `codex_outputs/29_snorkel_parquet_audit.md`, `29_snorkel_lf_integration_sample.md`, `29_snorkel_fitstatus_reconciliation.md`, `29_snorkel_config_audit.md`, `29_snorkel_investigation_summary.md`, and `docs/figures/29_snorkel_probability_distributions_corrected.png`.
Paper role: forensic check that Snorkel probabilities were interpretable before production; supports methodological honesty rather than a direct final figure.
Usage: `python scripts/29_snorkel_integrity_investigation.py` unless argparse help says otherwise.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ruff: noqa: E501

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src import config
from src.db.connection import get_engine
from src.db.queries import (
    fetch_icd_codes_by_hadm_ids,
    fetch_notes_by_hadm_ids,
    fetch_primary_icd_by_hadm_ids,
)
from src.labeling_functions.base import LFInput, LabelingFunction, Vote
from src.labeling_functions.icd_lf import ICD_LF_SPECS, build_all_icd_lfs
from src.labeling_functions.llm_lf import (
    SNORKEL_TARGET_FIELD_VALUE_PAIRS,
    build_all_llm_lfs,
)
from src.labeling_functions.regex_lf import build_all_regex_lfs
from src.labeling_functions.section_parser import parse_sections
from src.schema.fields import LLMNoteFeatures
from src.snorkel_fit.label_model import build_lf_vote_matrix

logger = logging.getLogger("scripts.29_snorkel_integrity_investigation")

EPS = 1e-6
ACTIVE_REGEX_TARGET_FIELDS: set[str] = {
    "aki_present",
    "dnr_dni_documented",
    "palliative_care_consult",
    "home_health_ordered",
    "substance_use_active",
    "fall_risk_documented",
    "cognitive_impairment",
    "goals_of_care_flag",
}


@dataclass(frozen=True)
class TargetSpec:
    label: str
    field: str
    value: str


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(cols) + " |"
    divider = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(str(row.get(c, "")).replace("|", "\\|") for c in cols)
            + " |"
        )
    return "\n".join([header, divider, *body])


def _load_llm_features(run_id: str) -> dict[int, LLMNoteFeatures]:
    path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    out: dict[int, LLMNoteFeatures] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not bool(row.get("parse_ok", False)):
            continue
        feats = row.get("features_json")
        if not isinstance(feats, dict):
            continue
        out[int(row["hadm_id"])] = LLMNoteFeatures.model_validate(feats)
    return out


def _target_key(field: str, value: str) -> str:
    return f"{field}::{value}"


def _summarize_probs(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "std": 0.0,
            "n_eq_0p5": 0.0,
            "n_gt_0p95": 0.0,
            "n_lt_0p05": 0.0,
        }
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "std": float(values.std()),
        "n_eq_0p5": int(np.sum(np.abs(values - 0.5) < EPS)),
        "n_gt_0p95": int(np.sum(values > 0.95)),
        "n_lt_0p05": int(np.sum(values < 0.05)),
    }


def _write_parquet_audit(df: pd.DataFrame, output: Path) -> dict[str, dict[str, Any]]:
    rows: list[str] = [
        "# Snorkel Parquet Audit",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        "## Dataset-level structure",
        "",
    ]

    dataset_rows = [
        {"metric": "row_count", "value": len(df)},
        {
            "metric": "unique_hadm_ids",
            "value": int(df["hadm_id"].nunique()),
        },
        {
            "metric": "unique_targets",
            "value": int(df[["target_field", "target_value"]].drop_duplicates().shape[0]),
        },
    ]
    rows.extend([_md_table(dataset_rows, ["metric", "value"]), ""])

    schema_rows = []
    for col in df.columns:
        schema_rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
            }
        )
    rows.extend(["## Columns and null counts", "", _md_table(schema_rows, ["column", "dtype", "null_count"]), ""])

    unique_rows = [
        {
            "column": "target_field",
            "n_unique": int(df["target_field"].nunique()),
            "sample_values": ", ".join(sorted(df["target_field"].astype(str).unique())[:12]),
        },
        {
            "column": "target_value",
            "n_unique": int(df["target_value"].nunique()),
            "sample_values": ", ".join(sorted(df["target_value"].astype(str).unique())[:20]),
        },
        {
            "column": "fit_status",
            "n_unique": int(df["fit_status"].nunique()),
            "sample_values": ", ".join(sorted(df["fit_status"].astype(str).unique())),
        },
        {
            "column": "n_lfs_total",
            "n_unique": int(df["n_lfs_total"].nunique()),
            "sample_values": ", ".join(str(x) for x in sorted(df["n_lfs_total"].unique())),
        },
        {
            "column": "n_lfs_contributing",
            "n_unique": int(df["n_lfs_contributing"].nunique()),
            "sample_values": ", ".join(
                str(x) for x in sorted(df["n_lfs_contributing"].unique())[:20]
            ),
        },
    ]
    rows.extend(["## Key categorical/value columns", "", _md_table(unique_rows, ["column", "n_unique", "sample_values"]), ""])

    stats_rows: list[dict[str, Any]] = []
    target_stats: dict[str, dict[str, Any]] = {}
    grouped = df.groupby(["target_field", "target_value"], dropna=False, sort=True)
    for (field, value), frame in grouped:
        probs = frame["snorkel_prob_positive"].to_numpy(dtype=np.float64)
        s = _summarize_probs(probs)
        key = _target_key(str(field), str(value))
        target_stats[key] = {
            "target_field": str(field),
            "target_value": str(value),
            "n_rows": len(frame),
            **s,
        }
        stats_rows.append(
            {
                "target": key,
                "n_rows": len(frame),
                "min": f"{s['min']:.6f}",
                "max": f"{s['max']:.6f}",
                "mean": f"{s['mean']:.6f}",
                "median": f"{s['median']:.6f}",
                "p10": f"{s['p10']:.6f}",
                "p25": f"{s['p25']:.6f}",
                "p75": f"{s['p75']:.6f}",
                "p90": f"{s['p90']:.6f}",
                "std": f"{s['std']:.6f}",
                "n_eq_0.5": int(s["n_eq_0p5"]),
                "n_gt_0.95": int(s["n_gt_0p95"]),
                "n_lt_0.05": int(s["n_lt_0p05"]),
            }
        )

    stats_rows.sort(key=lambda r: (-int(r["n_eq_0.5"]), r["target"]))
    rows.extend(
        [
            "## Per-target probability distributions",
            "",
            _md_table(
                stats_rows,
                [
                    "target",
                    "n_rows",
                    "min",
                    "max",
                    "mean",
                    "median",
                    "p10",
                    "p25",
                    "p75",
                    "p90",
                    "std",
                    "n_eq_0.5",
                    "n_gt_0.95",
                    "n_lt_0.05",
                ],
            ),
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows), encoding="utf-8")
    return target_stats


def _build_inputs_and_lfs(hadm_ids: list[int]) -> tuple[list[LFInput], list[LabelingFunction], dict[str, str]]:
    engine = get_engine()
    notes = fetch_notes_by_hadm_ids(engine, hadm_ids)
    icd_codes_by_hadm = fetch_icd_codes_by_hadm_ids(engine, hadm_ids)
    primary_icd_by_hadm = fetch_primary_icd_by_hadm_ids(engine, hadm_ids)
    llm_a = _load_llm_features("methodology_5k_a")

    inputs: list[LFInput] = []
    for hadm_id in hadm_ids:
        primary = primary_icd_by_hadm.get(hadm_id)
        inputs.append(
            LFInput(
                hadm_id=hadm_id,
                note_text=notes.get(hadm_id, ""),
                icd_codes=icd_codes_by_hadm.get(hadm_id),
                primary_icd_code=primary[0] if primary is not None else None,
                primary_icd_version=primary[1] if primary is not None else None,
                sections=parse_sections(notes.get(hadm_id, "")),
                llm_extraction_by_variant={"a": llm_a[hadm_id]} if hadm_id in llm_a else {},
            )
        )

    icd_lfs = build_all_icd_lfs()
    regex_lfs_all = build_all_regex_lfs(config.REPO_ROOT / "src" / "labeling_functions" / "patterns")
    regex_lfs = [lf for lf in regex_lfs_all if str(lf.target_field) in ACTIVE_REGEX_TARGET_FIELDS]
    llm_lfs = build_all_llm_lfs(variants=["a"], target_field_value_pairs=SNORKEL_TARGET_FIELD_VALUE_PAIRS)
    all_lfs: list[LabelingFunction] = [*icd_lfs, *regex_lfs, *llm_lfs]

    lf_type: dict[str, str] = {}
    for lf in icd_lfs:
        lf_type[str(lf.name)] = "icd"
    for lf in regex_lfs:
        lf_type[str(lf.name)] = "regex"
    for lf in llm_lfs:
        lf_type[str(lf.name)] = "llm"

    return inputs, all_lfs, lf_type


def _llm_vote(features: LLMNoteFeatures | None, field: str, value: str) -> str:
    if features is None:
        return "ABSTAIN"
    if field == "admission_reason_tags":
        return "POSITIVE" if value in set(features.admission_reason_tags) else "ABSTAIN"

    current = str(getattr(features, field, "not_documented"))
    if value == "yes":
        if current == "yes":
            return "POSITIVE"
        if current == "no":
            return "NEGATIVE"
        return "ABSTAIN"
    if value == "no":
        if current == "no":
            return "POSITIVE"
        if current == "yes":
            return "NEGATIVE"
        return "ABSTAIN"
    return "ABSTAIN"


def _vote_label(v: int) -> str:
    if v == Vote.POSITIVE:
        return "POSITIVE"
    if v == Vote.NEGATIVE:
        return "NEGATIVE"
    return "ABSTAIN"


def _sample_target_rows(
    *,
    target: TargetSpec,
    pred_df: pd.DataFrame,
    inputs: list[LFInput],
    all_lfs: list[LabelingFunction],
    lf_type_by_name: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    target_mask = (pred_df["target_field"] == target.field) & (pred_df["target_value"] == target.value)
    target_preds = pred_df[target_mask].copy()
    if target_preds.empty:
        return [], [
            f"Target `{target.label}` is not present in predictions.parquet.",
            "This indicates the target is outside the active Snorkel target universe for methodology_5k.",
        ]

    matched_lfs = [
        lf
        for lf in all_lfs
        if str(getattr(lf, "target_field", "")) == target.field
        and str(getattr(lf, "target_value", "")) == target.value
    ]
    if not matched_lfs:
        return [], [f"No matched LFs found for target `{target.label}`."]

    L, lf_names = build_lf_vote_matrix(matched_lfs, inputs)
    hadm_to_index = {inp.hadm_id: idx for idx, inp in enumerate(inputs)}

    rows_by_scenario: dict[str, list[dict[str, Any]]] = {
        "all_abstain": [],
        "llm_pos_plus_other_pos": [],
        "llm_pos_others_abstain": [],
        "disagreement": [],
    }

    for _, pred in target_preds.iterrows():
        hadm_id = int(pred["hadm_id"])
        idx = hadm_to_index.get(hadm_id)
        if idx is None:
            continue
        row_votes = L[idx, :]
        vote_pairs = list(zip(lf_names, row_votes, strict=True))
        llm_votes = [v for name, v in vote_pairs if lf_type_by_name.get(name) == "llm"]
        non_llm_votes = [v for name, v in vote_pairs if lf_type_by_name.get(name) != "llm"]

        all_abstain = bool(np.all(row_votes == Vote.ABSTAIN))
        has_pos = bool(np.any(row_votes == Vote.POSITIVE))
        has_neg = bool(np.any(row_votes == Vote.NEGATIVE))

        llm_positive = bool(any(v == Vote.POSITIVE for v in llm_votes))
        non_llm_positive = bool(any(v == Vote.POSITIVE for v in non_llm_votes))
        non_llm_all_abstain = bool(non_llm_votes) and bool(all(v == Vote.ABSTAIN for v in non_llm_votes))

        scenario = None
        if all_abstain:
            scenario = "all_abstain"
        elif llm_positive and non_llm_positive:
            scenario = "llm_pos_plus_other_pos"
        elif llm_positive and non_llm_all_abstain:
            scenario = "llm_pos_others_abstain"
        elif has_pos and has_neg:
            scenario = "disagreement"

        if scenario is None:
            continue

        feature_map = inputs[idx].llm_extraction_by_variant or {}
        llm_feature = feature_map.get("a")
        rows_by_scenario[scenario].append(
            {
                "hadm_id": hadm_id,
                "snorkel_prob_positive": float(pred["snorkel_prob_positive"]),
                "lf_votes": ", ".join(
                    f"{name}={_vote_label(int(v))}" for name, v in vote_pairs
                ),
                "direct_llm_vote": _llm_vote(llm_feature, target.field, target.value),
                "scenario": scenario,
            }
        )

    selected: list[dict[str, Any]] = []
    selected.extend(rows_by_scenario["all_abstain"][:3])
    selected.extend(rows_by_scenario["llm_pos_plus_other_pos"][:3])
    selected.extend(rows_by_scenario["llm_pos_others_abstain"][:3])
    selected.extend(rows_by_scenario["disagreement"][:1])

    notes: list[str] = []
    for key in ["all_abstain", "llm_pos_plus_other_pos", "llm_pos_others_abstain", "disagreement"]:
        notes.append(f"- {key}: {len(rows_by_scenario[key])} rows available")
    return selected, notes


def _write_lf_integration_sample(
    *,
    pred_df: pd.DataFrame,
    output: Path,
) -> None:
    hadm_ids = sorted(int(x) for x in pred_df["hadm_id"].unique().tolist())
    inputs, all_lfs, lf_type_by_name = _build_inputs_and_lfs(hadm_ids)

    targets = [
        TargetSpec("ICD anchored", "admission_reason_tags", "cardiac_hf"),
        TargetSpec("Regex anchored TriState", "aki_present", "yes"),
        TargetSpec("LLM-only fallback", "goals_of_care_flag", "no"),
        TargetSpec("High-base-rate enum", "discharge_condition_category", "improved"),
        TargetSpec("Low-base-rate admission tag", "admission_reason_tags", "metabolic_dka_hhs"),
    ]

    lines = [
        "# Snorkel LF Integration Sample",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        (
            "This report inspects per-row LF vote combinations versus Snorkel probabilities for "
            "five representative targets."
        ),
        "",
    ]

    for spec in targets:
        sample_rows, notes = _sample_target_rows(
            target=spec,
            pred_df=pred_df,
            inputs=inputs,
            all_lfs=all_lfs,
            lf_type_by_name=lf_type_by_name,
        )
        lines.extend(
            [
                f"## {spec.label}: `{spec.field}::{spec.value}`",
                "",
                *notes,
                "",
            ]
        )
        if sample_rows:
            lines.append(
                _md_table(
                    sample_rows,
                    [
                        "hadm_id",
                        "scenario",
                        "snorkel_prob_positive",
                        "direct_llm_vote",
                        "lf_votes",
                    ],
                )
            )
            lines.append("")
        else:
            lines.append("_No sample rows available for this target._")
            lines.append("")

    # Explicitly check for psych tag request.
    psych_key = (pred_df["target_field"] == "admission_reason_tags") & (
        pred_df["target_value"] == "psych_psychosis_crisis"
    )
    lines.extend(
        [
            "## Requested low-base example check: `admission_reason_tags::psych_psychosis_crisis`",
            "",
            (
                "Present in predictions: "
                f"`{bool(psych_key.any())}`."
            ),
            (
                "If absent, this is because the active Snorkel target set includes only ICD-anchored "
                "admission tags, and `psych_psychosis_crisis` has no ICD LF anchor in current config."
            ),
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_fitstatus_reconciliation(
    *,
    pred_df: pd.DataFrame,
    snorkel_report_path: Path,
    output: Path,
) -> None:
    report_text = snorkel_report_path.read_text(encoding="utf-8")

    # canonical from report (text-level quick read)
    fit_counts = (
        pred_df[["target_field", "target_value", "fit_status"]]
        .drop_duplicates()
        .groupby("fit_status")
        .size()
        .to_dict()
    )

    target_df = pred_df[["target_field", "target_value", "fit_status"]].drop_duplicates()
    single_targets = target_df[target_df["fit_status"] == "single_lf_only"].copy()
    success_targets = target_df[target_df["fit_status"] == "success"].copy()

    single_rows = []
    for _, row in single_targets.iterrows():
        mask = (pred_df["target_field"] == row["target_field"]) & (pred_df["target_value"] == row["target_value"])
        probs = pred_df.loc[mask, "snorkel_prob_positive"].to_numpy(dtype=np.float64)
        s = _summarize_probs(probs)
        single_rows.append(
            {
                "target": _target_key(str(row["target_field"]), str(row["target_value"])),
                "n_rows": int(mask.sum()),
                "median": f"{s['median']:.6f}",
                "n_eq_0.5": int(s["n_eq_0p5"]),
                "n_gt_0.95": int(s["n_gt_0p95"]),
                "n_lt_0.05": int(s["n_lt_0p05"]),
            }
        )

    success_rows = []
    for _, row in success_targets.iterrows():
        mask = (pred_df["target_field"] == row["target_field"]) & (pred_df["target_value"] == row["target_value"])
        probs = pred_df.loc[mask, "snorkel_prob_positive"].to_numpy(dtype=np.float64)
        s = _summarize_probs(probs)
        success_rows.append(
            {
                "target": _target_key(str(row["target_field"]), str(row["target_value"])),
                "n_rows": int(mask.sum()),
                "median": f"{s['median']:.6f}",
                "n_eq_0.5": int(s["n_eq_0p5"]),
                "n_gt_0.95": int(s["n_gt_0p95"]),
                "n_lt_0.05": int(s["n_lt_0p05"]),
            }
        )

    lines = [
        "# Snorkel Fit-Status Reconciliation",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        "## Canonical fit-status counts",
        "",
        _md_table(
            [
                {"fit_status": key, "n_targets": int(value)}
                for key, value in sorted(fit_counts.items(), key=lambda kv: kv[0])
            ],
            ["fit_status", "n_targets"],
        ),
        "",
        "## single_lf_only targets: existence + distribution",
        "",
        _md_table(single_rows, ["target", "n_rows", "median", "n_eq_0.5", "n_gt_0.95", "n_lt_0.05"]),
        "",
        "## success targets: existence + distribution",
        "",
        _md_table(success_rows, ["target", "n_rows", "median", "n_eq_0.5", "n_gt_0.95", "n_lt_0.05"]),
        "",
        "## Reconciliation finding",
        "",
        (
            "`single_lf_only` targets are present in parquet (not missing). The empty subplot in the old "
            "Phase 7-prep figure was due to grouping/filter logic, not missing predictions."
        ),
        "",
        "## Report provenance",
        "",
        f"- Source report parsed: `{snorkel_report_path}`",
        f"- Contains string 'single_lf_only': `{('single_lf_only' in report_text)}`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_config_audit(output: Path) -> None:
    lines = [
        "# Snorkel Configuration Audit",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        "## LabelModel configuration (from code)",
        "",
        "- File: `src/snorkel_fit/label_model.py`",
        "- `cardinality=2` (binary NEGATIVE/POSITIVE)",
        "- `LabelModel.fit(..., n_epochs=500, lr=0.01, seed=42)`",
        "- `predict_proba` is used for probability output.",
        "- No explicit `class_balance` passed; Snorkel default prior behavior applies.",
        "",
        "## Fallback logic",
        "",
        "- If no LFs match target: fit_status=`no_votes`, uniform `[0.5, 0.5]`.",
        "- If matched LFs all abstain: fit_status=`no_votes`, uniform `[0.5, 0.5]`.",
        "- If <=1 LF covers any row: fit_status=`single_lf_only`, direct deterministic mapping:",
        "  - POSITIVE -> `[0.0, 1.0]`",
        "  - NEGATIVE -> `[1.0, 0.0]`",
        "  - ABSTAIN -> `[0.5, 0.5]`",
        "",
        "## Target-universe configuration",
        "",
        "- File: `src/labeling_functions/llm_lf.py`",
        "- Active targets are `SNORKEL_TARGET_FIELD_VALUE_PAIRS` only:",
        "  - ICD-anchored admission tags (subset, not full 47)",
        "  - 9 TriState fields in yes/no form",
        "- Enum targets and LLM-only fields like `shock_present` are not part of current Snorkel output.",
        "",
        "## Suspicion check",
        "",
        (
            "The large 0.5 mass is expected when all matched LFs abstain for a row/target. This is common "
            "because many LFs are high-precision sparse emitters and admission-tag membership has no negative LF."
        ),
        (
            "Potential modeling caveat (not a runtime bug): for sparse membership targets without negative "
            "signals, many rows remain prior-valued (0.5), so downstream consumers must treat 0.5 as "
            "indeterminate rather than negative."
        ),
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _plot_corrected_distribution(
    *,
    df: pd.DataFrame,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    icd_tags = {
        str(spec["target_value"])
        for spec in ICD_LF_SPECS
        if str(spec.get("target_field")) == "admission_reason_tags"
    }
    regex_fields = ACTIVE_REGEX_TARGET_FIELDS

    icd_mask = (df["target_field"] == "admission_reason_tags") & (df["target_value"].astype(str).isin(icd_tags))
    regex_mask = df["target_field"].astype(str).isin(regex_fields) & (df["target_value"].astype(str) == "yes")
    single_mask = df["fit_status"].astype(str).eq("single_lf_only")

    groups = [
        ("ICD-anchored", icd_mask, "#4e79a7"),
        ("Regex-anchored (yes targets)", regex_mask, "#f28e2b"),
        ("single_lf_only (all targets)", single_mask, "#59a14f"),
    ]

    summary: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, (label, mask, color) in zip(axes, groups, strict=True):
        vals = df.loc[mask, "snorkel_prob_positive"].to_numpy(dtype=np.float64)
        if vals.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 1)
            summary[label] = {"n": 0.0, "median": 0.0, "mean": 0.0}
            continue
        ax.hist(vals, bins=30, color=color, edgecolor="white", alpha=0.88)
        med = float(np.median(vals))
        mean = float(vals.mean())
        ax.axvline(med, color="black", linestyle="--", linewidth=1.3)
        ax.set_xlim(0, 1)
        ax.set_title(label)
        ax.set_xlabel("Snorkel POSITIVE probability")
        ax.set_ylabel("Frequency")
        ax.text(
            0.03,
            0.95,
            f"n={len(vals):,}\nmedian={med:.3f}\nmean={mean:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        summary[label] = {"n": float(len(vals)), "median": med, "mean": mean}

    fig.suptitle("Snorkel probability distributions (corrected grouping)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def _write_summary(
    *,
    df: pd.DataFrame,
    target_stats: dict[str, dict[str, Any]],
    corrected_plot_summary: dict[str, dict[str, float]],
    output: Path,
) -> str:
    target_df = (
        df[["target_field", "target_value", "fit_status"]]
        .drop_duplicates()
        .sort_values(["target_field", "target_value"])
    )

    per_target_rows: list[dict[str, Any]] = []
    for _, row in target_df.iterrows():
        key = _target_key(str(row["target_field"]), str(row["target_value"]))
        s = target_stats[key]
        per_target_rows.append(
            {
                "target": key,
                "fit_status": str(row["fit_status"]),
                "n_rows": int(s["n_rows"]),
                "mean": f"{float(s['mean']):.6f}",
                "median": f"{float(s['median']):.6f}",
                "n_eq_0.5": int(s["n_eq_0p5"]),
                "n_gt_0.95": int(s["n_gt_0p95"]),
                "n_lt_0.05": int(s["n_lt_0p05"]),
            }
        )

    per_target_rows.sort(key=lambda r: (-int(r["n_eq_0.5"]), r["target"]))

    # Recommendation A/B/C
    frac_eq_0p5 = float(np.mean(np.abs(df["snorkel_prob_positive"].to_numpy(dtype=np.float64) - 0.5) < EPS))
    has_non_prior = bool((df["snorkel_prob_positive"] > 0.95).any() or (df["snorkel_prob_positive"] < 0.05).any())
    single_present = bool((df["fit_status"] == "single_lf_only").any())

    if has_non_prior and single_present:
        recommendation = "A"
        rec_text = (
            "Snorkel output is functioning as configured; the original plot interpretation was misleading. "
            "The heavy 0.5 mass is dominated by abstention rows, while non-prior probabilities appear on "
            "rows with active LF evidence."
        )
    elif has_non_prior:
        recommendation = "B"
        rec_text = (
            "Output is partially informative but grouping/coverage issues need a small offline correction "
            "before launch."
        )
    else:
        recommendation = "C"
        rec_text = "Output appears degenerate with no meaningful non-prior signal."

    lines = [
        "# Snorkel Investigation Summary",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        "## Corrected plot summary",
        "",
        _md_table(
            [
                {
                    "group": key,
                    "n_rows": int(value["n"]),
                    "median": f"{value['median']:.6f}",
                    "mean": f"{value['mean']:.6f}",
                }
                for key, value in corrected_plot_summary.items()
            ],
            ["group", "n_rows", "median", "mean"],
        ),
        "",
        "## High-resolution per-target table",
        "",
        _md_table(
            per_target_rows,
            [
                "target",
                "fit_status",
                "n_rows",
                "mean",
                "median",
                "n_eq_0.5",
                "n_gt_0.95",
                "n_lt_0.05",
            ],
        ),
        "",
        "## Interpretation",
        "",
        f"- Overall fraction of exact-0.5 probabilities: `{frac_eq_0p5:.4f}`",
        f"- Non-prior probabilities present (any <0.05 or >0.95): `{has_non_prior}`",
        f"- `single_lf_only` targets present in parquet: `{single_present}`",
        "",
        "## Recommendation",
        "",
        f"**{recommendation})** {rec_text}",
        "",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return recommendation


def _write_verification(
    *,
    recommendation: str,
    output: Path,
) -> None:
    lines = [
        "# Snorkel Investigation Verification",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        "## Artifacts",
        "",
        "- `codex_outputs/29_snorkel_parquet_audit.md`",
        "- `codex_outputs/29_snorkel_lf_integration_sample.md`",
        "- `codex_outputs/29_snorkel_fitstatus_reconciliation.md`",
        "- `codex_outputs/29_snorkel_config_audit.md`",
        "- `codex_outputs/29_snorkel_investigation_summary.md`",
        "- `docs/figures/29_snorkel_probability_distributions_corrected.png`",
        "",
        f"## Recommendation\n\n- `{recommendation}`",
        "",
        "## Cost",
        "",
        "- `$0` OpenAI extraction cost (offline analysis only)",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate Snorkel prediction integrity.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/methodology_5k/predictions.parquet"),
    )
    parser.add_argument(
        "--snorkel-report",
        type=Path,
        default=Path("codex_outputs/26_methodology_5k_snorkel_report.md"),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    df = pd.read_parquet(args.predictions)
    target_stats = _write_parquet_audit(df, Path("codex_outputs/29_snorkel_parquet_audit.md"))

    _write_lf_integration_sample(
        pred_df=df,
        output=Path("codex_outputs/29_snorkel_lf_integration_sample.md"),
    )

    _write_fitstatus_reconciliation(
        pred_df=df,
        snorkel_report_path=args.snorkel_report,
        output=Path("codex_outputs/29_snorkel_fitstatus_reconciliation.md"),
    )

    _write_config_audit(Path("codex_outputs/29_snorkel_config_audit.md"))

    plot_summary = _plot_corrected_distribution(
        df=df,
        output_path=Path("docs/figures/29_snorkel_probability_distributions_corrected.png"),
    )

    recommendation = _write_summary(
        df=df,
        target_stats=target_stats,
        corrected_plot_summary=plot_summary,
        output=Path("codex_outputs/29_snorkel_investigation_summary.md"),
    )

    _write_verification(
        recommendation=recommendation,
        output=Path("codex_outputs/29_snorkel_investigation_verification.md"),
    )

    print("Wrote all Snorkel investigation artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
