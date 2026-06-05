from __future__ import annotations

# Release documentation:
# Validates LLM features against ICD-derived signals.
#
# Reads: data/production/parquet/production_v1_features.parquet, codex_outputs/57_icd_validation_summary.md.
# Writes: data/production/parquet/production_v1_features.parquet, codex_outputs/57_icd_validation_summary.md, docs/figures/57_icd_validation_breakdown.png.
# Backs ICD/LLM concordance supplement claims.
# Usage: `python scripts/57_icd_validation.py` unless the script's argparse help says otherwise.

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.connection import get_engine
from src.db.queries import fetch_icd_codes_by_hadm_ids

TARGET_SPECS: list[dict[str, Any]] = [
    {
        "key": "aki_present",
        "label": "aki_present",
        "field_type": "tristate",
        "field": "aki_present",
        "tag": None,
        "icd10_patterns": ["N17"],
        "icd9_patterns": ["584"],
    },
    {
        "key": "admission_reason_tags::cardiac_hf",
        "label": "admission_reason_tags::cardiac_hf",
        "field_type": "tag",
        "field": "admission_reason_tags",
        "tag": "cardiac_hf",
        "icd10_patterns": ["I50"],
        "icd9_patterns": ["428"],
    },
    {
        "key": "admission_reason_tags::sepsis_bacteremia",
        "label": "admission_reason_tags::sepsis_bacteremia",
        "field_type": "tag",
        "field": "admission_reason_tags",
        "tag": "sepsis_bacteremia",
        "icd10_patterns": ["A40", "A41", "R65.2"],
        "icd9_patterns": ["038", "995.91", "995.92"],
    },
    {
        "key": "admission_reason_tags::respiratory_infection",
        "label": "admission_reason_tags::respiratory_infection",
        "field_type": "tag",
        "field": "admission_reason_tags",
        "tag": "respiratory_infection",
        "icd10_patterns": ["J12", "J13", "J14", "J15", "J16", "J17", "J18", "J20", "J21", "J22"],
        "icd9_patterns": ["480", "481", "482", "483", "484", "485", "486", "487"],
    },
    {
        "key": "admission_reason_tags::hepatic_failure_cirrhosis",
        "label": "admission_reason_tags::hepatic_failure_cirrhosis",
        "field_type": "tag",
        "field": "admission_reason_tags",
        "tag": "hepatic_failure_cirrhosis",
        "icd10_patterns": ["K70", "K71", "K72", "K73", "K74"],
        "icd9_patterns": ["571", "572", "573"],
    },
]


@dataclass(frozen=True)
class Metrics:
    target: str
    n_total: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision_proxy: float
    recall_proxy: float
    f1_proxy: float
    llm_unique_capture_count: int
    llm_unique_capture_rate_all: float
    llm_unique_capture_rate_of_llm_pos: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICD-vs-LLM proxy validation on production outputs."
    )
    parser.add_argument("--run-id", default="production_v1")
    parser.add_argument(
        "--features-parquet",
        default="data/production/parquet/production_v1_features.parquet",
    )
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--output",
        default="codex_outputs/57_icd_validation_summary.md",
    )
    parser.add_argument(
        "--figure",
        default="docs/figures/57_icd_validation_breakdown.png",
    )
    return parser.parse_args()


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(cols) + " |"
    divider = "|" + "|".join(["---"] * len(cols)) + "|"
    body: list[str] = []
    for row in rows:
        body.append(
            "| " + " | ".join(str(row.get(c, "")).replace("|", "\\|") for c in cols) + " |"
        )
    return "\n".join([header, divider, *body])


def _normalize(code: str) -> str:
    return code.strip().upper()


def _matches(code: str, patterns: list[str]) -> bool:
    normalized_code = _normalize(code)
    for pattern in patterns:
        normalized_pattern = _normalize(pattern)
        if "." in normalized_pattern:
            if normalized_code == normalized_pattern:
                return True
            continue
        if normalized_code.startswith(normalized_pattern):
            return True
    return False


def _extract_llm_positive(frame: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    if spec["field_type"] == "tristate":
        return (
            frame[str(spec["field"])]
            .fillna("not_documented")
            .astype(str)
            .eq("yes")
            .to_numpy()
        )

    tag = str(spec["tag"])

    def has_tag(value: Any) -> bool:
        if isinstance(value, list):
            return tag in {str(v) for v in value}
        if isinstance(value, np.ndarray):
            return tag in {str(v) for v in value.tolist()}
        return False

    return frame["admission_reason_tags"].apply(has_tag).to_numpy(dtype=bool)


def _iter_chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_icd_presence(hadm_ids: list[int], chunk_size: int) -> dict[str, np.ndarray]:
    engine = get_engine()
    pos_map = {spec["key"]: np.zeros(len(hadm_ids), dtype=bool) for spec in TARGET_SPECS}
    index_by_hadm = {hadm_id: idx for idx, hadm_id in enumerate(hadm_ids)}

    for chunk in _iter_chunks(hadm_ids, chunk_size):
        code_map = fetch_icd_codes_by_hadm_ids(engine, chunk)
        for hadm_id in chunk:
            idx = index_by_hadm[hadm_id]
            codes = code_map.get(hadm_id, [])
            if not codes:
                continue
            for spec in TARGET_SPECS:
                key = str(spec["key"])
                icd10_patterns = list(spec["icd10_patterns"])
                icd9_patterns = list(spec["icd9_patterns"])
                matched = False
                for code, version in codes:
                    if int(version) == 10 and _matches(str(code), icd10_patterns):
                        matched = True
                        break
                    if int(version) == 9 and _matches(str(code), icd9_patterns):
                        matched = True
                        break
                if matched:
                    pos_map[key][idx] = True

    return pos_map


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _compute_metrics(
    target: str,
    llm_pos: np.ndarray,
    icd_pos: np.ndarray,
) -> Metrics:
    tp = int(np.sum(llm_pos & icd_pos))
    fp = int(np.sum(llm_pos & ~icd_pos))
    fn = int(np.sum(~llm_pos & icd_pos))
    tn = int(np.sum(~llm_pos & ~icd_pos))
    total = int(llm_pos.size)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    llm_pos_count = int(np.sum(llm_pos))
    llm_unique_count = fp
    llm_unique_rate_all = _safe_div(llm_unique_count, total)
    llm_unique_rate_llm = _safe_div(llm_unique_count, llm_pos_count)

    return Metrics(
        target=target,
        n_total=total,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision_proxy=precision,
        recall_proxy=recall,
        f1_proxy=f1,
        llm_unique_capture_count=llm_unique_count,
        llm_unique_capture_rate_all=llm_unique_rate_all,
        llm_unique_capture_rate_of_llm_pos=llm_unique_rate_llm,
    )


def _plot_breakdown(metrics: list[Metrics], output_path: Path) -> None:
    labels = [m.target for m in metrics]
    totals = np.array([m.n_total for m in metrics], dtype=float)

    tp = np.array([m.tp for m in metrics], dtype=float) / totals
    fp = np.array([m.fp for m in metrics], dtype=float) / totals
    fn = np.array([m.fn for m in metrics], dtype=float) / totals
    tn = np.array([m.tn for m in metrics], dtype=float) / totals

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(labels))

    ax.bar(x, tp, label="LLM+ / ICD+", color="#2a9d8f")
    ax.bar(x, fp, bottom=tp, label="LLM+ / ICD-", color="#f4a261")
    ax.bar(x, fn, bottom=tp + fp, label="LLM- / ICD+", color="#e76f51")
    ax.bar(x, tn, bottom=tp + fp + fn, label="LLM- / ICD-", color="#6c757d")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate over production notes")
    ax.set_title("ICD validation: 4-cell LLM vs ICD breakdown")
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    features_path = Path(args.features_parquet)
    output_path = Path(args.output)
    figure_path = Path(args.figure)

    if not features_path.exists():
        raise FileNotFoundError(f"Missing production features parquet: {features_path}")

    df = pd.read_parquet(features_path)
    if "parse_ok" in df.columns:
        df = df[df["parse_ok"].fillna(False)]
    if "hadm_id" not in df.columns:
        raise ValueError("production parquet must include hadm_id")

    df = df.drop_duplicates(subset=["hadm_id"], keep="first").reset_index(drop=True)
    hadm_ids = [int(v) for v in df["hadm_id"].tolist()]

    icd_presence = _build_icd_presence(hadm_ids=hadm_ids, chunk_size=int(args.chunk_size))

    metrics: list[Metrics] = []
    for spec in TARGET_SPECS:
        key = str(spec["key"])
        llm_pos = _extract_llm_positive(df, spec)
        icd_pos = icd_presence[key]
        metrics.append(_compute_metrics(target=key, llm_pos=llm_pos, icd_pos=icd_pos))

    _plot_breakdown(metrics, figure_path)

    rows_main: list[dict[str, Any]] = []
    for m in metrics:
        rows_main.append(
            {
                "target": m.target,
                "n_total": m.n_total,
                "llm_pos_icd_pos": m.tp,
                "llm_pos_icd_absent": m.fp,
                "llm_notpos_icd_pos": m.fn,
                "llm_notpos_icd_absent": m.tn,
                "precision_proxy": f"{m.precision_proxy:.4f}",
                "recall_proxy": f"{m.recall_proxy:.4f}",
                "f1_proxy": f"{m.f1_proxy:.4f}",
                "llm_unique_capture_count": m.llm_unique_capture_count,
                "llm_unique_capture_rate_all": f"{m.llm_unique_capture_rate_all * 100:.2f}%",
                "llm_unique_capture_rate_of_llm_pos": (
                    f"{m.llm_unique_capture_rate_of_llm_pos * 100:.2f}%"
                ),
            }
        )

    rows_notes: list[str] = [
        "# ICD Validation Summary",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        "Proxy setup: ICD presence is treated as a practical reference signal (not gold truth).",
        "",
        f"- Source production parsed notes: **{len(df):,}**",
        f"- Figure: `{figure_path}`",
        "",
        "## 4-cell agreement breakdown + proxy metrics",
        "",
        _md_table(
            rows_main,
            [
                "target",
                "n_total",
                "llm_pos_icd_pos",
                "llm_pos_icd_absent",
                "llm_notpos_icd_pos",
                "llm_notpos_icd_absent",
                "precision_proxy",
                "recall_proxy",
                "f1_proxy",
                "llm_unique_capture_count",
                "llm_unique_capture_rate_all",
                "llm_unique_capture_rate_of_llm_pos",
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            "- `llm_pos_icd_absent` may represent false positives OR clinically "
            "valid captures uncoded in billing ICD."
        ),
        (
            "- `llm_notpos_icd_pos` may represent LLM misses OR admission focus "
            "differences versus coding practice."
        ),
        "- Proxy metrics are for consistency checking only, not final clinical accuracy claims.",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows_notes), encoding="utf-8")
    print(f"Wrote ICD validation summary to {output_path}")
    print(f"Wrote figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
