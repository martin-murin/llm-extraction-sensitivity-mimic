from __future__ import annotations

# Release documentation:
# Provides shared helpers/configuration for publication figure modules.
#
# Reads: data/raw_responses/paired_gold_methodology_1k_{a, data/raw_responses/paired_gold_methodology_5k_audit_{a, codex_outputs/55_paired_extraction_summary.md, codex_outputs/55_paired_accuracy.md, data/raw_responses/methodology_1k_{a, data/raw_responses/methodology_5k_a_subset500/results.jsonl.
# Writes: data/raw_responses/paired_gold_methodology_1k_{a, data/raw_responses/paired_gold_methodology_5k_audit_{a, codex_outputs/55_paired_extraction_summary.md, codex_outputs/55_paired_accuracy.md, data/raw_responses/methodology_1k_{a, data/raw_responses/methodology_5k_a_subset500/results.jsonl.
# Supports publication figure generation.

# ruff: noqa: UP033

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.schema.disease_specific.aki_v2_fields import AKIV2NoteFeatures
from src.schema.disease_specific.hf_v2_fields import HFV2NoteFeatures
from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw_responses"
OPT = REPO / "data" / "optimization"
SPLITS = REPO / "data" / "splits"
CODEX = REPO / "codex_outputs"
PROD_PARQUET = REPO / "data" / "production" / "parquet" / "production_v1_features.parquet"

TRISTATE_DOMAIN = {"yes", "no", "not_documented"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_results(run_id: str) -> pd.DataFrame:
    """Read one extraction run and flatten parse_ok rows.

    Returns DataFrame with columns:
    - hadm_id
    - run_id
    - parse_ok
    - input_tokens/output_tokens/latency_seconds
    - flattened `features_json.*` fields for parse_ok rows
    """
    path = RAW / run_id / "results.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return pd.DataFrame()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        base = {
            "hadm_id": int(row["hadm_id"]),
            "run_id": run_id,
            "parse_ok": bool(row.get("parse_ok", False)),
            "input_tokens": int(row.get("input_tokens", 0) or 0),
            "output_tokens": int(row.get("output_tokens", 0) or 0),
            "latency_seconds": float(row.get("latency_seconds", 0.0) or 0.0),
        }
        if base["parse_ok"] and isinstance(row.get("features_json"), dict):
            features = dict(row["features_json"])
            base.update(features)
        parsed_rows.append(base)
    return pd.DataFrame(parsed_rows)


def _variant_frame(run_map: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant, run_id in run_map.items():
        frame = _read_results(run_id)
        if frame.empty:
            continue
        frame = frame[frame["parse_ok"]].copy()
        frame["variant"] = variant
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["hadm_id"] = pd.to_numeric(out["hadm_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["hadm_id"]).copy()
    out["hadm_id"] = out["hadm_id"].astype("int64")
    return out


def _parse_markdown_table(md_text: str) -> pd.DataFrame:
    """Parse first markdown table in text into DataFrame."""
    lines = md_text.splitlines()
    header_idx = -1
    for i in range(len(lines) - 1):
        if lines[i].startswith("|") and lines[i + 1].startswith("|") and "---" in lines[i + 1]:
            header_idx = i
            break
    if header_idx < 0:
        raise ValueError("No markdown table found")

    header = [x.strip() for x in lines[header_idx].strip().strip("|").split("|")]
    rows: list[list[str]] = []
    j = header_idx + 2
    while j < len(lines) and lines[j].startswith("|"):
        rows.append([x.strip() for x in lines[j].strip().strip("|").split("|")])
        j += 1
    return pd.DataFrame(rows, columns=header)


def _clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("`", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    cleaned = cleaned.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _literal_values(annotation: Any) -> set[str] | None:
    origin = get_origin(annotation)
    if origin is Literal:
        return {arg for arg in get_args(annotation) if isinstance(arg, str)}
    args = get_args(annotation)
    if not args:
        return None
    out: set[str] = set()
    saw_literal = False
    for arg in args:
        if arg is type(None):
            continue
        sub = _literal_values(arg)
        if sub is None:
            return None
        out |= sub
        saw_literal = True
    return out if saw_literal else None


@lru_cache(maxsize=None)
def get_tristate_field_list(scope: Literal["base", "hf_v2", "aki_v2", "all"] = "base") -> list[str]:
    """Return canonical TriState fields.

    Source schema modules:
    - src/schema/fields.py (base)
    - src/schema/disease_specific/hf_v2_fields.py
    - src/schema/disease_specific/aki_v2_fields.py
    """

    def tri_fields(model: Any) -> set[str]:
        out: set[str] = set()
        for name, field in model.model_fields.items():
            values = _literal_values(field.annotation)
            if values == TRISTATE_DOMAIN:
                out.add(name)
        return out

    base = tri_fields(LLMNoteFeatures)
    hf = tri_fields(HFV2NoteFeatures)
    aki = tri_fields(AKIV2NoteFeatures)
    if scope == "base":
        return sorted(base)
    if scope == "hf_v2":
        return sorted(hf)
    if scope == "aki_v2":
        return sorted(aki)
    return sorted(base | hf | aki)


@lru_cache(maxsize=None)
def get_admission_tag_vocabulary() -> list[str]:
    """Return canonical 47-tag admission vocabulary.

    Source:
    - src/schema/vocabulary.py
    """
    return list(ADMISSION_REASON_TAGS)


@lru_cache(maxsize=None)
def load_paired_gold_extractions() -> pd.DataFrame:
    """Load paired gold extraction runs.

    Source files:
    - data/raw_responses/paired_gold_methodology_1k_{a,b,c}/results.jsonl
    - data/raw_responses/paired_gold_methodology_5k_audit_{a,b,c}/results.jsonl

    Corresponding reports:
    - codex_outputs/55_paired_extraction_summary.md
    - codex_outputs/55_paired_accuracy.md
    """
    run_map = {
        "a_1k": "paired_gold_methodology_1k_a",
        "b_1k": "paired_gold_methodology_1k_b",
        "c_1k": "paired_gold_methodology_1k_c",
        "a_500": "paired_gold_methodology_5k_audit_a",
        "b_500": "paired_gold_methodology_5k_audit_b",
        "c_500": "paired_gold_methodology_5k_audit_c",
    }
    frames: list[pd.DataFrame] = []
    for key, run_id in run_map.items():
        frame = _read_results(run_id)
        if frame.empty:
            continue
        frame = frame[frame["parse_ok"]].copy()
        variant = key.split("_")[0].upper()
        sample = "methodology_1k" if key.endswith("1k") else "methodology_5k_audit_500"
        frame["variant"] = variant
        frame["sample"] = sample
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["hadm_id"] = pd.to_numeric(out["hadm_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["hadm_id"]).copy()
    out["hadm_id"] = out["hadm_id"].astype("int64")
    return out


@lru_cache(maxsize=None)
def load_paired_full_extractions(model_size: Literal["small", "full"] = "small") -> pd.DataFrame:
    """Load paired 1500-note extractions as long frame with variant column.

    Source files:
    - Small model:
      - data/raw_responses/methodology_1k_{a,b,c}/results.jsonl
      - data/raw_responses/methodology_5k_a_subset500/results.jsonl
      - data/raw_responses/methodology_5k_audit_{b,c}/results.jsonl
    - Full model reference:
      - data/raw_responses/paired_gold_methodology_1k_{a,b,c}/results.jsonl
      - data/raw_responses/paired_gold_methodology_5k_audit_{a,b,c}/results.jsonl
    """
    if model_size == "small":
        run_map = {
            "A": ("methodology_1k_a", "methodology_5k_a_subset500"),
            "B": ("methodology_1k_b", "methodology_5k_audit_b"),
            "C": ("methodology_1k_c", "methodology_5k_audit_c"),
        }
    else:
        run_map = {
            "A": ("paired_gold_methodology_1k_a", "paired_gold_methodology_5k_audit_a"),
            "B": ("paired_gold_methodology_1k_b", "paired_gold_methodology_5k_audit_b"),
            "C": ("paired_gold_methodology_1k_c", "paired_gold_methodology_5k_audit_c"),
        }

    frames: list[pd.DataFrame] = []
    for variant, (run_1k, run_500) in run_map.items():
        left = _read_results(run_1k)
        right = _read_results(run_500)
        combo = pd.concat([left, right], ignore_index=True)
        combo = combo[combo["parse_ok"]].copy()
        combo["variant"] = variant
        combo["hadm_id"] = pd.to_numeric(combo["hadm_id"], errors="coerce").astype("Int64")
        combo = combo.dropna(subset=["hadm_id"]).copy()
        combo["hadm_id"] = combo["hadm_id"].astype("int64")
        combo = combo.drop_duplicates(subset=["hadm_id"], keep="last")
        frames.append(combo)
    out = pd.concat(frames, ignore_index=True)
    return out


@lru_cache(maxsize=None)
def load_methodology_1k_extractions() -> pd.DataFrame:
    """Load three-variant methodology_1k extractions."""
    run_map = {"A": "methodology_1k_a", "B": "methodology_1k_b", "C": "methodology_1k_c"}
    return _variant_frame(run_map)


@lru_cache(maxsize=None)
def load_methodology_5k_audit_extractions() -> pd.DataFrame:
    """Load three-variant methodology 5k audit subset extractions.

    Source files:
    - data/raw_responses/methodology_5k_a_subset500/results.jsonl (A)
    - data/raw_responses/methodology_5k_audit_b/results.jsonl (B)
    - data/raw_responses/methodology_5k_audit_c/results.jsonl (C)

    Report references:
    - codex_outputs/26_methodology_5k_audit_kappa_report.md
    - codex_outputs/92_disagreement_collapse_decomposition.md
    """
    run_map = {
        "A": "methodology_5k_a_subset500",
        "B": "methodology_5k_audit_b",
        "C": "methodology_5k_audit_c",
    }
    return _variant_frame(run_map)


@lru_cache(maxsize=None)
def load_refinement_extractions() -> pd.DataFrame:
    """Load three-variant refinement_150 extractions."""
    run_map = {"A": "refinement_v1_a", "B": "refinement_v1_b", "C": "refinement_v3_c"}
    return _variant_frame(run_map)


@lru_cache(maxsize=None)
def load_holdout_extractions() -> pd.DataFrame:
    """Load three-variant holdout_150 extractions."""
    run_map = {"A": "holdout_v1_a", "B": "holdout_v1_b", "C": "holdout_v1_c"}
    return _variant_frame(run_map)


@lru_cache(maxsize=None)
def load_extended_5k_extractions() -> pd.DataFrame:
    """Load extended 5k extractions and production A subset.

    Source files:
    - data/raw_responses/extended_5k_b/results.jsonl
    - data/raw_responses/extended_5k_c/results.jsonl
    - data/raw_responses/production_v1/results.jsonl (subset to split extended_5k.csv)
    """
    run_map = {"B": "extended_5k_b", "C": "extended_5k_c"}
    bc = _variant_frame(run_map)

    a = _read_results("production_v1")
    a = a[a["parse_ok"]].copy()
    split = pd.read_csv(SPLITS / "extended_5k.csv")
    hadm_ids = set(pd.to_numeric(split["hadm_id"], errors="coerce").dropna().astype("int64"))
    a["hadm_id"] = pd.to_numeric(a["hadm_id"], errors="coerce").astype("Int64")
    a = a.dropna(subset=["hadm_id"]).copy()
    a["hadm_id"] = a["hadm_id"].astype("int64")
    a = a[a["hadm_id"].isin(hadm_ids)].copy()
    a["variant"] = "A"

    out = pd.concat([a, bc], ignore_index=True)
    return out


@lru_cache(maxsize=None)
def load_production_v1() -> pd.DataFrame:
    """Load production feature parquet.

    Source file:
    - data/production/parquet/production_v1_features.parquet
    """
    return pd.read_parquet(PROD_PARQUET)


@lru_cache(maxsize=None)
def load_reasoning_on_off_extractions() -> pd.DataFrame:
    """Load paired reasoning ON/OFF outputs for variant A.

    Source files:
    - data/raw_responses/reasoning_on_methodology_1k_a/results.jsonl
    - data/raw_responses/methodology_1k_a/results.jsonl

    Report references:
    - codex_outputs/56_reasoning_comparison.md
    - codex_outputs/91_reasoning_collapse.md
    """
    on_df = _read_results("reasoning_on_methodology_1k_a")
    off_df = _read_results("methodology_1k_a")
    on_df = on_df[on_df["parse_ok"]].copy()
    off_df = off_df[off_df["parse_ok"]].copy()

    on_df["hadm_id"] = pd.to_numeric(on_df["hadm_id"], errors="coerce").astype("Int64")
    off_df["hadm_id"] = pd.to_numeric(off_df["hadm_id"], errors="coerce").astype("Int64")
    on_df = on_df.dropna(subset=["hadm_id"]).copy()
    off_df = off_df.dropna(subset=["hadm_id"]).copy()
    on_df["hadm_id"] = on_df["hadm_id"].astype("int64")
    off_df["hadm_id"] = off_df["hadm_id"].astype("int64")

    common = sorted(set(on_df["hadm_id"]) & set(off_df["hadm_id"]))
    on_df = on_df[on_df["hadm_id"].isin(common)].copy()
    off_df = off_df[off_df["hadm_id"].isin(common)].copy()

    on_cols = {c: f"{c}_on" for c in on_df.columns if c != "hadm_id"}
    off_cols = {c: f"{c}_off" for c in off_df.columns if c != "hadm_id"}
    on_df = on_df.rename(columns=on_cols)
    off_df = off_df.rename(columns=off_cols)
    merged = on_df.merge(off_df, on="hadm_id", how="inner")
    return merged


@lru_cache(maxsize=None)
def load_audit_corpus_extended_5k() -> pd.DataFrame:
    """Load canonical extended 5k audit corpus JSONL.

    Source:
    - data/optimization/audit_corpus_extended_5k.jsonl

    Report references:
    - codex_outputs/46_extended_audit_clusters.md
    - codex_outputs/93_audit_clusters_collapse.md
    """
    rows = _read_jsonl(OPT / "audit_corpus_extended_5k.jsonl")
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def load_paired_collapse_per_field_table() -> pd.DataFrame:
    """Load parsed table from Prompt 33 paired collapse per-field report.

    Source:
    - codex_outputs/91_paired_collapse_per_field.md
    """
    text = (CODEX / "91_paired_collapse_per_field.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| field |"):
            start = i
            break
    if start is None:
        raise ValueError("Could not find per-field table in 91_paired_collapse_per_field.md")
    tbl = _parse_markdown_table("\n".join(lines[start:]))

    numeric_cols = [
        "nano_kappa_full",
        "nano_kappa_collapsed",
        "gold_kappa_full",
        "gold_kappa_collapsed",
        "delta_full_pp",
        "delta_collapsed_pp",
        "collapse_effect_on_delta_pp",
    ]
    for col in numeric_cols:
        if col in tbl.columns:
            tbl[col] = _clean_numeric(tbl[col])
    return tbl


@lru_cache(maxsize=None)
def load_paired_collapse_accuracy_table() -> pd.DataFrame:
    """Load parsed table from Prompt 33 per-variant collapse accuracy report.

    Source:
    - codex_outputs/91_paired_collapse_accuracy.md
    """
    text = (CODEX / "91_paired_collapse_accuracy.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| variant |"):
            start = i
            break
    if start is None:
        raise ValueError("Could not find accuracy table in 91_paired_collapse_accuracy.md")
    tbl = _parse_markdown_table("\n".join(lines[start:]))

    numeric_cols = [
        "n_overlap",
        "agreement_full_pct",
        "agreement_collapsed_pct",
        "kappa_full",
        "kappa_collapsed",
        "delta_kappa_collapse_minus_full",
    ]
    for col in numeric_cols:
        if col in tbl.columns:
            tbl[col] = _clean_numeric(tbl[col])
    return tbl


@lru_cache(maxsize=None)
def load_cross_variant_kappa_collapse_table() -> pd.DataFrame:
    """Load pairwise collapsed/full kappa table from Task 2 report.

    Source:
    - codex_outputs/92_cross_variant_kappa_collapse.md
    """
    text = (CODEX / "92_cross_variant_kappa_collapse.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| pair |"):
            start = i
            break
    if start is None:
        raise ValueError("Could not find pairwise table in 92_cross_variant_kappa_collapse.md")
    tbl = _parse_markdown_table("\n".join(lines[start:]))
    for col in ["median_kappa_full", "median_kappa_collapsed", "median_delta"]:
        if col in tbl.columns:
            tbl[col] = _clean_numeric(tbl[col])
    return tbl


@lru_cache(maxsize=None)
def load_markdown_table(path: str, header_startswith: str) -> pd.DataFrame:
    """Generic markdown table loader for figure modules.

    Args:
        path: repo-relative path to markdown report
        header_startswith: header prefix, e.g. "| field |"
    """
    text = (REPO / path).read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header_startswith):
            start = i
            break
    if start is None:
        raise ValueError(f"Could not find table '{header_startswith}' in {path}")
    return _parse_markdown_table("\n".join(lines[start:]))


@lru_cache(maxsize=None)
def load_refinement_holdout_generalization() -> tuple[pd.DataFrame, dict[str, float]]:
    """Load refinement->holdout generalization metrics from Phase 4 sidecars.

    Source files:
    - codex_outputs/16c_iter2_kappa.md.json (refinement)
    - codex_outputs/21_holdout_kappa_report.md.json (holdout)

    Report references:
    - codex_outputs/21_refinement_vs_holdout_comparison.md
    - codex_outputs/21_phase4_verification.md

    Returns:
    - per_variant DataFrame with columns:
      variant, refinement_kappa, holdout_kappa, delta_pp, n_fields
      where per-variant kappa is median across filtered fields of:
      A: mean(kappa_A_B, kappa_A_C)
      B: mean(kappa_A_B, kappa_B_C)
      C: mean(kappa_A_C, kappa_B_C)
    - summary dict with:
      refinement_filtered_median, holdout_filtered_median, delta_pp, n_shared_filtered
    """

    ref_path = CODEX / "16c_iter2_kappa.md.json"
    hold_path = CODEX / "21_holdout_kappa_report.md.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    hold = json.loads(hold_path.read_text(encoding="utf-8"))

    ref_results = ref.get("kappa_results", {})
    hold_results = hold.get("kappa_results", {})
    if not isinstance(ref_results, dict) or not isinstance(hold_results, dict):
        raise ValueError("Invalid kappa sidecar format for refinement/holdout.")

    shared = sorted(set(ref_results.keys()) & set(hold_results.keys()))
    filtered_keys: list[str] = []
    for key in shared:
        ref_row = ref_results[key]
        hold_row = hold_results[key]
        if bool(ref_row.get("low_base_rate_flag", False)):
            continue
        if bool(hold_row.get("low_base_rate_flag", False)):
            continue
        filtered_keys.append(key)

    def _variant_metric(
        row: dict[str, Any],
        variant: str,
    ) -> float:
        k_ab = float(row["kappa_A_B"])
        k_ac = float(row["kappa_A_C"])
        k_bc = float(row["kappa_B_C"])
        if variant == "A":
            return 0.5 * (k_ab + k_ac)
        if variant == "B":
            return 0.5 * (k_ab + k_bc)
        return 0.5 * (k_ac + k_bc)

    rows: list[dict[str, Any]] = []
    for variant in ["A", "B", "C"]:
        ref_vals = [_variant_metric(ref_results[key], variant) for key in filtered_keys]
        hold_vals = [_variant_metric(hold_results[key], variant) for key in filtered_keys]
        ref_med = float(pd.Series(ref_vals, dtype="float64").median())
        hold_med = float(pd.Series(hold_vals, dtype="float64").median())
        rows.append(
            {
                "variant": variant,
                "refinement_kappa": ref_med,
                "holdout_kappa": hold_med,
                "delta_pp": (hold_med - ref_med) * 100.0,
                "n_fields": len(filtered_keys),
            }
        )

    per_variant = pd.DataFrame(rows)
    summary = {
        "refinement_filtered_median": float(
            ref.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
        ),
        "holdout_filtered_median": float(
            hold.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0)
        ),
        "n_shared_filtered": float(len(filtered_keys)),
    }
    summary["delta_pp"] = (
        summary["holdout_filtered_median"] - summary["refinement_filtered_median"]
    ) * 100.0
    return per_variant, summary


@lru_cache(maxsize=None)
def load_kappa_stability_five_samples(
    n_bootstrap: int = 5000,
    seed: int = 1106,
) -> pd.DataFrame:
    """Load five-sample filtered median kappa stability with bootstrap CIs.

    Source files:
    - codex_outputs/16c_iter2_kappa.md.json
    - codex_outputs/21_holdout_kappa_report.md.json
    - codex_outputs/methodology_1k_kappa_current.md.json
      (fallback: codex_outputs/22_methodology_1k_kappa_report.md.json)
    - codex_outputs/methodology_5k_audit_kappa_current.md.json
      (fallback: codex_outputs/26_methodology_5k_audit_kappa_report.md.json)
    - codex_outputs/extended_5k_kappa_current.md.json
      (fallback: codex_outputs/46_extended_kappa_report.md.json)

    Notes:
    - Point estimate uses sidecar `kappa_summary_filtered.overall_median_kappa`.
    - CI is bootstrap over filtered per-field `kappa_mean` values.
    """

    sample_specs: list[tuple[str, str, Path]] = [
        ("refinement_150", "Refinement 150", CODEX / "16c_iter2_kappa.md.json"),
        ("holdout_150", "Holdout 150", CODEX / "21_holdout_kappa_report.md.json"),
    ]

    def _prefer_current(current_name: str, fallback_name: str) -> Path:
        current = CODEX / current_name
        if current.exists():
            return current
        return CODEX / fallback_name

    sample_specs.extend(
        [
            (
                "methodology_1k",
                "Methodology 1k",
                _prefer_current(
                    "methodology_1k_kappa_current.md.json",
                    "22_methodology_1k_kappa_report.md.json",
                ),
            ),
            (
                "methodology_5k_audit_500",
                "Methodology 5k-audit 500",
                _prefer_current(
                    "methodology_5k_audit_kappa_current.md.json",
                    "26_methodology_5k_audit_kappa_report.md.json",
                ),
            ),
            (
                "extended_5k",
                "Extended 5k",
                _prefer_current(
                    "extended_5k_kappa_current.md.json",
                    "46_extended_kappa_report.md.json",
                ),
            ),
        ]
    )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for key, label, path in sample_specs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        kappa_results = payload.get("kappa_results", {})
        if not isinstance(kappa_results, dict):
            raise ValueError(f"Invalid sidecar format: {path}")

        filtered_vals: list[float] = []
        for row in kappa_results.values():
            if bool(row.get("low_base_rate_flag", False)):
                continue
            filtered_vals.append(float(row["kappa_mean"]))
        if not filtered_vals:
            raise ValueError(f"No filtered fields found for sample {key}")

        arr = np.asarray(filtered_vals, dtype=np.float64)
        point = float(payload.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0))
        n = int(arr.size)
        idx = rng.integers(0, n, size=(int(n_bootstrap), n))
        sampled = arr[idx]
        boot_meds = np.median(sampled, axis=1)
        ci_low = float(np.percentile(boot_meds, 2.5))
        ci_high = float(np.percentile(boot_meds, 97.5))

        rows.append(
            {
                "sample_key": key,
                "sample_label": label,
                "n_notes": int(payload.get("intersection_parsed_all_three", 0)),
                "n_fields_filtered": n,
                "median_kappa_filtered": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows)
