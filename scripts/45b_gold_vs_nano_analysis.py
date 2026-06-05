"""
Analyzes larger-model proxy/gold versus nano extraction outputs.

Reads: codex_outputs/22_methodology_1k_kappa_report.md.json, codex_outputs/26_methodology_5k_audit_kappa_report.md.json, codex_outputs/45_gold_vs_nano_accuracy.md, codex_outputs/45_prompt_framing_vs_model_size.md, codex_outputs/45_gold_kappa_report.md.json.
Writes: codex_outputs/22_methodology_1k_kappa_report.md.json, codex_outputs/26_methodology_5k_audit_kappa_report.md.json, codex_outputs/45_gold_vs_nano_accuracy.md, codex_outputs/45_prompt_framing_vs_model_size.md, codex_outputs/45_gold_kappa_report.md.json.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/45b_gold_vs_nano_analysis.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

from src import config
from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.threeway_kappa import (
    cohen_kappa_safe,
    count_positive_admission_tag,
    count_positive_enum,
    count_positive_tristate,
    encode_admission_reason_tags,
    encode_tristate,
    low_base_rate_flag,
    percent_agreement,
)

TRISTATE_FIELDS: set[str] = {
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
}
ENUM_FIELDS: set[str] = {"functional_status", "mental_status", "discharge_condition_category"}
COUNT_FIELDS: set[str] = {"new_meds_started_count", "meds_stopped_count"}
SKIP_FIELDS: set[str] = {"primary_diagnosis_text", "reasoning", *COUNT_FIELDS}
PAIR_KEYS: list[tuple[str, str]] = [("A", "B"), ("A", "C"), ("B", "C")]


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        values = [str(row.get(col, "")).replace("|", "\\|").replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *lines])


def _load_results(run_id: str) -> dict[str, Any]:
    results_path = config.RAW_RESPONSES_DIR / run_id / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    parsed: dict[int, dict[str, Any]] = {}
    attempted = 0
    input_tokens = 0
    output_tokens = 0
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            attempted += 1
            payload = json.loads(line)
            input_tokens += int(payload.get("input_tokens", 0) or 0)
            output_tokens += int(payload.get("output_tokens", 0) or 0)
            if bool(payload.get("parse_ok", False)) and isinstance(
                payload.get("features_json"), dict
            ):
                parsed[int(payload["hadm_id"])] = payload["features_json"]
    metadata_path = config.RAW_RESPONSES_DIR / run_id / "run_metadata.json"
    total_cost_usd = 0.0
    if metadata_path.exists():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        total_cost_usd = float(loaded.get("total_cost_usd", 0.0) or 0.0)
    return {
        "run_id": run_id,
        "attempted": attempted,
        "parsed_n": len(parsed),
        "parsed": parsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": total_cost_usd,
    }


def _field_names_for_kappa() -> list[str]:
    return [
        field
        for field in LLMNoteFeatures.model_fields
        if field not in SKIP_FIELDS and field != "admission_reason_tags"
    ]


def _encode_categorical_pair(
    left_values: list[Any],
    right_values: list[Any],
) -> tuple[list[int], list[int]]:
    categories = sorted({str(v) for v in left_values} | {str(v) for v in right_values})
    mapping = {value: idx for idx, value in enumerate(categories)}
    left_encoded = [int(mapping[str(v)]) for v in left_values]
    right_encoded = [int(mapping[str(v)]) for v in right_values]
    return left_encoded, right_encoded


def _pairwise_gold_vs_nano_rows(
    *,
    gold_label: str,
    gold_parsed: dict[int, dict[str, Any]],
    nano_run_id: str,
    nano_parsed: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    overlap = sorted(set(gold_parsed).intersection(nano_parsed))
    if not overlap:
        return [], 0

    rows: list[dict[str, Any]] = []
    fields = _field_names_for_kappa()
    for field in fields:
        gold_values = [gold_parsed[hadm_id].get(field, "not_documented") for hadm_id in overlap]
        nano_values = [nano_parsed[hadm_id].get(field, "not_documented") for hadm_id in overlap]

        if field in TRISTATE_FIELDS:
            gold_encoded = [encode_tristate(str(v)) for v in gold_values]
            nano_encoded = [encode_tristate(str(v)) for v in nano_values]
            n_positive_total = count_positive_tristate(
                [str(v) for v in gold_values]
            ) + count_positive_tristate([str(v) for v in nano_values])
            field_group = "tristate"
        elif field in ENUM_FIELDS:
            gold_encoded, nano_encoded = _encode_categorical_pair(gold_values, nano_values)
            n_positive_total = count_positive_enum(
                [str(v) for v in gold_values]
            ) + count_positive_enum([str(v) for v in nano_values])
            field_group = "enum"
        else:
            gold_encoded, nano_encoded = _encode_categorical_pair(gold_values, nano_values)
            n_positive_total = 0
            field_group = "categorical"

        kappa = cohen_kappa_safe(gold_encoded, nano_encoded)
        agreement = percent_agreement(gold_encoded, nano_encoded)
        rows.append(
            {
                "comparison": f"{gold_label}_vs_{nano_run_id}",
                "gold_variant": gold_label,
                "nano_run_id": nano_run_id,
                "field": field,
                "field_group": field_group,
                "n_overlap": len(overlap),
                "kappa": float(kappa),
                "pct_agreement": float(agreement * 100.0),
                "low_base_rate_flag": bool(low_base_rate_flag(int(n_positive_total))),
            }
        )

    for tag in ADMISSION_REASON_TAGS:
        field_key = f"admission_reason_tags::{tag}"
        gold_tag_values: list[int] = []
        nano_tag_values: list[int] = []
        gold_tag_lists: list[list[str]] = []
        nano_tag_lists: list[list[str]] = []
        for hadm_id in overlap:
            gold_tags = list(gold_parsed[hadm_id].get("admission_reason_tags", []))
            nano_tags = list(nano_parsed[hadm_id].get("admission_reason_tags", []))
            gold_tag_lists.append(gold_tags)
            nano_tag_lists.append(nano_tags)
            gold_tag_values.append(encode_admission_reason_tags(gold_tags)[tag])
            nano_tag_values.append(encode_admission_reason_tags(nano_tags)[tag])
        n_positive_total = count_positive_admission_tag(
            gold_tag_lists, tag
        ) + count_positive_admission_tag(nano_tag_lists, tag)
        rows.append(
            {
                "comparison": f"{gold_label}_vs_{nano_run_id}",
                "gold_variant": gold_label,
                "nano_run_id": nano_run_id,
                "field": field_key,
                "field_group": "admission_reason_tag",
                "n_overlap": len(overlap),
                "kappa": float(cohen_kappa_safe(gold_tag_values, nano_tag_values)),
                "pct_agreement": float(percent_agreement(gold_tag_values, nano_tag_values) * 100.0),
                "low_base_rate_flag": bool(low_base_rate_flag(int(n_positive_total))),
            }
        )

    return rows, len(overlap)


def _gold_cross_variant_kappas(
    parsed_a: dict[int, dict[str, Any]],
    parsed_b: dict[int, dict[str, Any]],
    parsed_c: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    overlap = sorted(set(parsed_a).intersection(parsed_b).intersection(parsed_c))
    if not overlap:
        return [], {
            "n_overlap": 0,
            "overall_median_kappa": 0.0,
            "overall_median_kappa_filtered": 0.0,
        }

    by_variant = {"A": parsed_a, "B": parsed_b, "C": parsed_c}
    rows: list[dict[str, Any]] = []

    for field in _field_names_for_kappa():
        encoded: dict[str, list[int]] = {}
        n_positive_total = 0
        field_group = "categorical"

        if field in TRISTATE_FIELDS:
            field_group = "tristate"
            for variant in ["A", "B", "C"]:
                values = [
                    str(by_variant[variant][hadm_id].get(field, "not_documented"))
                    for hadm_id in overlap
                ]
                encoded[variant] = [encode_tristate(v) for v in values]
                n_positive_total += count_positive_tristate(values)
        elif field in ENUM_FIELDS:
            field_group = "enum"
            all_values = {
                str(by_variant[variant][hadm_id].get(field, "not_documented"))
                for variant in ["A", "B", "C"]
                for hadm_id in overlap
            }
            mapping = {value: idx for idx, value in enumerate(sorted(all_values))}
            for variant in ["A", "B", "C"]:
                values = [
                    str(by_variant[variant][hadm_id].get(field, "not_documented"))
                    for hadm_id in overlap
                ]
                encoded[variant] = [int(mapping[v]) for v in values]
                n_positive_total += count_positive_enum(values)
        else:
            all_values = {
                str(by_variant[variant][hadm_id].get(field))
                for variant in ["A", "B", "C"]
                for hadm_id in overlap
            }
            mapping = {value: idx for idx, value in enumerate(sorted(all_values))}
            for variant in ["A", "B", "C"]:
                values = [str(by_variant[variant][hadm_id].get(field)) for hadm_id in overlap]
                encoded[variant] = [int(mapping[v]) for v in values]

        kappa_a_b = cohen_kappa_safe(encoded["A"], encoded["B"])
        kappa_a_c = cohen_kappa_safe(encoded["A"], encoded["C"])
        kappa_b_c = cohen_kappa_safe(encoded["B"], encoded["C"])
        row = {
            "field": field,
            "field_group": field_group,
            "n_overlap": len(overlap),
            "kappa_A_B": float(kappa_a_b),
            "kappa_A_C": float(kappa_a_c),
            "kappa_B_C": float(kappa_b_c),
            "kappa_mean": float(np.mean([kappa_a_b, kappa_a_c, kappa_b_c])),
            "low_base_rate_flag": bool(low_base_rate_flag(int(n_positive_total))),
        }
        rows.append(row)

    for tag in ADMISSION_REASON_TAGS:
        field_key = f"admission_reason_tags::{tag}"
        encoded = {"A": [], "B": [], "C": []}
        n_positive_total = 0
        for variant in ["A", "B", "C"]:
            tag_lists: list[list[str]] = []
            tag_values: list[int] = []
            for hadm_id in overlap:
                tags = list(by_variant[variant][hadm_id].get("admission_reason_tags", []))
                tag_lists.append(tags)
                tag_values.append(encode_admission_reason_tags(tags)[tag])
            encoded[variant] = tag_values
            n_positive_total += count_positive_admission_tag(tag_lists, tag)
        kappa_a_b = cohen_kappa_safe(encoded["A"], encoded["B"])
        kappa_a_c = cohen_kappa_safe(encoded["A"], encoded["C"])
        kappa_b_c = cohen_kappa_safe(encoded["B"], encoded["C"])
        rows.append(
            {
                "field": field_key,
                "field_group": "admission_reason_tag",
                "n_overlap": len(overlap),
                "kappa_A_B": float(kappa_a_b),
                "kappa_A_C": float(kappa_a_c),
                "kappa_B_C": float(kappa_b_c),
                "kappa_mean": float(np.mean([kappa_a_b, kappa_a_c, kappa_b_c])),
                "low_base_rate_flag": bool(low_base_rate_flag(int(n_positive_total))),
            }
        )

    kappas_all = np.asarray([float(row["kappa_mean"]) for row in rows], dtype=np.float64)
    kappas_filtered = np.asarray(
        [float(row["kappa_mean"]) for row in rows if not bool(row["low_base_rate_flag"])],
        dtype=np.float64,
    )
    summary = {
        "n_overlap": len(overlap),
        "overall_median_kappa": float(np.median(kappas_all)) if kappas_all.size else 0.0,
        "overall_median_kappa_filtered": float(np.median(kappas_filtered))
        if kappas_filtered.size
        else 0.0,
        "n_fields_total": int(kappas_all.size),
        "n_fields_filtered": int(kappas_filtered.size),
    }
    return rows, summary


def _load_kappa_sidecar(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sidecar: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("kappa_results"), dict):
        raise RuntimeError(f"Invalid sidecar payload: {path}")
    return payload


def _nanomedian(payload: dict[str, Any]) -> float:
    return float(payload.get("kappa_summary_filtered", {}).get("overall_median_kappa", 0.0))


def _write_accuracy_report(
    *,
    overlap_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    topline_rows = _pair_topline_rows(pair_rows)
    grouped = _group_pair_rows(pair_rows)

    worst_rows: list[dict[str, Any]] = []
    for comparison, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda r: float(r["kappa"]))
        for row in ordered[:5]:
            worst_rows.append(
                {
                    "comparison": comparison,
                    "field": row["field"],
                    "kappa": f"{float(row['kappa']):.4f}",
                    "pct_agreement": f"{float(row['pct_agreement']):.2f}",
                    "low_base_rate_flag": row["low_base_rate_flag"],
                }
            )

    lines = [
        "# Gold vs Nano Accuracy (Prompt 26)",
        "",
        "## Overlap coverage by run pair",
        _markdown_table(
            overlap_rows,
            [
                "gold_variant",
                "gold_run_id",
                "nano_run_id",
                "gold_parsed_n",
                "nano_parsed_n",
                "n_overlap",
            ],
        ),
        "",
        "## Per-pair topline metrics",
        _markdown_table(
            topline_rows,
            [
                "comparison",
                "n_overlap_notes",
                "n_fields",
                "median_kappa_all",
                "median_kappa_filtered",
                "mean_pct_agreement",
            ],
        ),
        "",
        "## Lowest-kappa fields (top 5 per pair)",
        _markdown_table(
            worst_rows,
            ["comparison", "field", "kappa", "pct_agreement", "low_base_rate_flag"],
        ),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _group_pair_rows(pair_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        grouped.setdefault(str(row["comparison"]), []).append(row)
    return grouped


def _pair_topline_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_pair_rows(pair_rows)
    topline_rows: list[dict[str, Any]] = []
    for comparison, rows in sorted(grouped.items()):
        kappas = np.asarray([float(r["kappa"]) for r in rows], dtype=np.float64)
        kappas_filtered = np.asarray(
            [float(r["kappa"]) for r in rows if not bool(r["low_base_rate_flag"])],
            dtype=np.float64,
        )
        agreements = np.asarray([float(r["pct_agreement"]) for r in rows], dtype=np.float64)
        topline_rows.append(
            {
                "comparison": comparison,
                "n_overlap_notes": int(rows[0]["n_overlap"]),
                "n_fields": int(kappas.size),
                "median_kappa_all": float(np.median(kappas)) if kappas.size else 0.0,
                "median_kappa_filtered": float(np.median(kappas_filtered))
                if kappas_filtered.size
                else 0.0,
                "mean_pct_agreement": float(np.mean(agreements)) if agreements.size else 0.0,
            }
        )
    return topline_rows


def _write_figures(
    *,
    overlap_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    gold_kappa_rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    topline = _pair_topline_rows(pair_rows)
    if topline:
        x = [row["comparison"] for row in topline]
        y = [float(row["median_kappa_filtered"]) for row in topline]
        plt.figure(figsize=(16, 5))
        plt.bar(x, y)
        plt.ylim(0.0, 1.0)
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Median kappa (filtered)")
        plt.title("Gold vs Nano Median Kappa by Pair")
        path = out_dir / "45_gold_vs_nano_median_kappa.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        written.append(str(path))

    if gold_kappa_rows:
        kappas = np.asarray([float(row["kappa_mean"]) for row in gold_kappa_rows], dtype=np.float64)
        plt.figure(figsize=(8, 5))
        plt.hist(kappas, bins=20)
        plt.xlim(0.0, 1.0)
        plt.xlabel("Field kappa mean")
        plt.ylabel("Count")
        plt.title("Gold Cross-Variant Kappa Distribution")
        path = out_dir / "45_gold_cross_variant_kappa_hist.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        written.append(str(path))

    if overlap_rows:
        gold_labels = sorted({str(row["gold_variant"]) for row in overlap_rows})
        nano_runs = sorted({str(row["nano_run_id"]) for row in overlap_rows})
        matrix = np.zeros((len(gold_labels), len(nano_runs)), dtype=np.float64)
        gidx = {label: idx for idx, label in enumerate(gold_labels)}
        nidx = {label: idx for idx, label in enumerate(nano_runs)}
        for row in overlap_rows:
            matrix[gidx[str(row["gold_variant"])], nidx[str(row["nano_run_id"])]] = float(
                row["n_overlap"]
            )
        plt.figure(figsize=(max(8, len(nano_runs) * 1.2), 4))
        im = plt.imshow(matrix, aspect="auto")
        plt.colorbar(im, label="Overlap notes")
        plt.xticks(range(len(nano_runs)), nano_runs, rotation=75, ha="right")
        plt.yticks(range(len(gold_labels)), gold_labels)
        plt.title("Gold vs Nano Overlap Matrix")
        path = out_dir / "45_gold_vs_nano_overlap_matrix.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        written.append(str(path))

    return written


def _write_model_size_report(
    *,
    gold_summary: dict[str, Any],
    nano_1k: dict[str, Any],
    nano_5k_audit: dict[str, Any],
    gold_kappa_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    topline_rows = [
        {
            "dataset": "gold_1k_full_model",
            "run_ids": "gold_v1_a,gold_v1_b,gold_v1_c",
            "n_overlap": gold_summary["n_overlap"],
            "median_kappa_filtered": f"{gold_summary['overall_median_kappa_filtered']:.4f}",
        },
        {
            "dataset": "methodology_1k_nano",
            "run_ids": ",".join(nano_1k.get("run_ids", [])),
            "n_overlap": int(nano_1k.get("intersection_parsed_all_three", 0)),
            "median_kappa_filtered": f"{_nanomedian(nano_1k):.4f}",
        },
        {
            "dataset": "methodology_5k_audit_500_nano",
            "run_ids": ",".join(nano_5k_audit.get("run_ids", [])),
            "n_overlap": int(nano_5k_audit.get("intersection_parsed_all_three", 0)),
            "median_kappa_filtered": f"{_nanomedian(nano_5k_audit):.4f}",
        },
    ]

    gold_by_field = {str(row["field"]): row for row in gold_kappa_rows}
    nano_1k_fields = dict(nano_1k.get("kappa_results", {}))
    nano_5k_fields = dict(nano_5k_audit.get("kappa_results", {}))
    shared = sorted(set(gold_by_field) & set(nano_1k_fields) & set(nano_5k_fields))

    delta_rows: list[dict[str, Any]] = []
    for field in shared:
        gold_k = float(gold_by_field[field]["kappa_mean"])
        n1_k = float(nano_1k_fields[field]["kappa_mean"])
        n5_k = float(nano_5k_fields[field]["kappa_mean"])
        delta_rows.append(
            {
                "field": field,
                "gold_full_kappa_mean": f"{gold_k:.4f}",
                "nano_1k_kappa_mean": f"{n1_k:.4f}",
                "nano_5k_audit_kappa_mean": f"{n5_k:.4f}",
                "delta_gold_minus_nano1k_pp": f"{(gold_k - n1_k) * 100.0:.2f}",
                "delta_gold_minus_nano5k_audit_pp": f"{(gold_k - n5_k) * 100.0:.2f}",
            }
        )
    delta_rows.sort(key=lambda row: float(row["delta_gold_minus_nano1k_pp"]))

    lines = [
        "# Prompt Framing vs Model Size (Gold vs Nano)",
        "",
        "## Filtered median kappa comparison",
        _markdown_table(topline_rows, ["dataset", "run_ids", "n_overlap", "median_kappa_filtered"]),
        "",
        "## Shared-field kappa deltas (gold full-model minus nano)",
        _markdown_table(
            delta_rows[:25],
            [
                "field",
                "gold_full_kappa_mean",
                "nano_1k_kappa_mean",
                "nano_5k_audit_kappa_mean",
                "delta_gold_minus_nano1k_pp",
                "delta_gold_minus_nano5k_audit_pp",
            ],
        ),
        "",
        "## Notes",
        (
            "- Nano baselines are sourced from prior sidecars: "
            "`22_methodology_1k_kappa_report.md.json` and "
            "`26_methodology_5k_audit_kappa_report.md.json`."
        ),
        "- Gold uses the same prompt/schema family but swaps model tier to `gpt-5.4-2026-03-05`.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_gold_cross_sidecar(
    *,
    run_ids: list[str],
    kappa_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "run_ids": run_ids,
        "split": "gold_1k",
        "intersection_parsed_all_three": int(summary["n_overlap"]),
        "kappa_results": {
            str(row["field"]): {
                "kappa_A_B": float(row["kappa_A_B"]),
                "kappa_A_C": float(row["kappa_A_C"]),
                "kappa_B_C": float(row["kappa_B_C"]),
                "kappa_mean": float(row["kappa_mean"]),
                "low_base_rate_flag": bool(row["low_base_rate_flag"]),
            }
            for row in kappa_rows
        },
        "kappa_summary_all": {
            "overall_median_kappa": float(summary["overall_median_kappa"]),
            "n_fields": int(summary["n_fields_total"]),
        },
        "kappa_summary_filtered": {
            "overall_median_kappa": float(summary["overall_median_kappa_filtered"]),
            "n_fields_included": int(summary["n_fields_filtered"]),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Prompt 26 gold-vs-nano agreement.")
    parser.add_argument("--gold-run-ids", nargs=3, default=["gold_v1_a", "gold_v1_b", "gold_v1_c"])
    parser.add_argument(
        "--nano-run-ids",
        nargs="+",
        default=[
            "production_v1",
            "methodology_1k_a",
            "methodology_1k_b",
            "methodology_1k_c",
            "methodology_5k_a",
            "methodology_5k_audit_b",
            "methodology_5k_audit_c",
        ],
    )
    parser.add_argument(
        "--nano-kappa-1k-sidecar",
        default="codex_outputs/22_methodology_1k_kappa_report.md.json",
    )
    parser.add_argument(
        "--nano-kappa-5k-audit-sidecar",
        default="codex_outputs/26_methodology_5k_audit_kappa_report.md.json",
    )
    parser.add_argument("--accuracy-output", default="codex_outputs/45_gold_vs_nano_accuracy.md")
    parser.add_argument("--size-output", default="codex_outputs/45_prompt_framing_vs_model_size.md")
    parser.add_argument(
        "--gold-kappa-sidecar", default="codex_outputs/45_gold_kappa_report.md.json"
    )
    parser.add_argument("--figures-dir", default="docs/figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    gold_loaded = [_load_results(run_id) for run_id in args.gold_run_ids]
    gold_parsed = {
        "A": gold_loaded[0]["parsed"],
        "B": gold_loaded[1]["parsed"],
        "C": gold_loaded[2]["parsed"],
    }

    nano_loaded = [_load_results(run_id) for run_id in args.nano_run_ids]
    nano_by_run = {item["run_id"]: item for item in nano_loaded}

    pair_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    gold_run_map = {"A": args.gold_run_ids[0], "B": args.gold_run_ids[1], "C": args.gold_run_ids[2]}

    for gold_label in ["A", "B", "C"]:
        for nano_run_id, nano_item in nano_by_run.items():
            rows, n_overlap = _pairwise_gold_vs_nano_rows(
                gold_label=gold_label,
                gold_parsed=gold_parsed[gold_label],
                nano_run_id=nano_run_id,
                nano_parsed=nano_item["parsed"],
            )
            pair_rows.extend(rows)
            overlap_rows.append(
                {
                    "gold_variant": gold_label,
                    "gold_run_id": gold_run_map[gold_label],
                    "nano_run_id": nano_run_id,
                    "gold_parsed_n": len(gold_parsed[gold_label]),
                    "nano_parsed_n": int(nano_item["parsed_n"]),
                    "n_overlap": int(n_overlap),
                }
            )

    gold_kappa_rows, gold_kappa_summary = _gold_cross_variant_kappas(
        gold_parsed["A"],
        gold_parsed["B"],
        gold_parsed["C"],
    )
    _write_gold_cross_sidecar(
        run_ids=list(args.gold_run_ids),
        kappa_rows=gold_kappa_rows,
        summary=gold_kappa_summary,
        output_path=Path(args.gold_kappa_sidecar),
    )

    _write_accuracy_report(
        overlap_rows=overlap_rows,
        pair_rows=pair_rows,
        output_path=Path(args.accuracy_output),
    )

    nano_1k = _load_kappa_sidecar(Path(args.nano_kappa_1k_sidecar))
    nano_5k = _load_kappa_sidecar(Path(args.nano_kappa_5k_audit_sidecar))
    _write_model_size_report(
        gold_summary=gold_kappa_summary,
        nano_1k=nano_1k,
        nano_5k_audit=nano_5k,
        gold_kappa_rows=gold_kappa_rows,
        output_path=Path(args.size_output),
    )
    figure_paths = _write_figures(
        overlap_rows=overlap_rows,
        pair_rows=pair_rows,
        gold_kappa_rows=gold_kappa_rows,
        out_dir=Path(args.figures_dir),
    )

    print(f"Wrote {args.accuracy_output}")
    print(f"Wrote {args.size_output}")
    print(f"Wrote {args.gold_kappa_sidecar}")
    for figure_path in figure_paths:
        print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
