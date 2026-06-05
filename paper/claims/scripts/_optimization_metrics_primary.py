from __future__ import annotations

# Release documentation:
# Provides shared helpers for claim-registry recomputation.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports paper claim recomputation and receipt verification.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.optimization.audit_corpus import (
    has_any_disagreement,
    should_include_record,
    summarize_disagreement_pattern,
)
from src.optimization.pattern_clustering import cluster_corpus
from src.schema.vocabulary import ADMISSION_REASON_TAGS
from src.utils.threeway_kappa import (
    count_positive_admission_tag,
    count_positive_enum,
    count_positive_tristate,
)

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw_responses"
OPT_LOGS = REPO / "logs" / "optimization"

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
ENUM_FIELDS = ["functional_status", "mental_status", "discharge_condition_category"]


@dataclass(frozen=True)
class OptimizationPrimaryMetrics:
    iterations_applied: int
    initial_cluster_disagreements: int
    final_cluster_disagreements: int
    reduction_pct: float
    final_run_ids: dict[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_variant_features(run_id: str) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(RAW / run_id / "results.jsonl")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if bool(row.get("parse_ok")) and isinstance(row.get("features_json"), dict):
            out[int(row["hadm_id"])] = row["features_json"]
    return out


def _normalized_vote(value: Any) -> str:
    if value is None:
        return "not_documented"
    token = str(value).strip()
    return token if token else "not_documented"


def _disagreement_summary(votes_by_hadm: dict[int, dict[str, str]]) -> tuple[int, str]:
    disagreements: list[dict[str, Any]] = []
    for hadm_id, votes in votes_by_hadm.items():
        if len({votes["a"], votes["b"], votes["c"]}) <= 1:
            continue
        disagreements.append({"hadm_id": hadm_id, "votes": votes})
    return len(disagreements), summarize_disagreement_pattern(disagreements)


def _build_records(final_run_ids: dict[str, str]) -> list[dict[str, Any]]:
    parsed_by_variant = {
        "a": _load_variant_features(final_run_ids["a"]),
        "b": _load_variant_features(final_run_ids["b"]),
        "c": _load_variant_features(final_run_ids["c"]),
    }
    hadm_ids = sorted(
        set(parsed_by_variant["a"])
        .intersection(parsed_by_variant["b"])
        .intersection(parsed_by_variant["c"])
    )
    if not hadm_ids:
        raise ValueError("No shared hadm_id across final optimization run_ids.")

    records: list[dict[str, Any]] = []

    for field in TRISTATE_FIELDS:
        values = {
            variant: [
                _normalized_vote(parsed_by_variant[variant][hadm_id].get(field, "not_documented"))
                for hadm_id in hadm_ids
            ]
            for variant in ("a", "b", "c")
        }
        vote_tuples = list(zip(values["a"], values["b"], values["c"], strict=True))
        if not has_any_disagreement(vote_tuples):
            continue
        votes_by_hadm = {
            hadm_id: {"a": values["a"][idx], "b": values["b"][idx], "c": values["c"][idx]}
            for idx, hadm_id in enumerate(hadm_ids)
        }
        disagreement_count, summary = _disagreement_summary(votes_by_hadm)
        n_positive_total = (
            count_positive_tristate(values["a"])
            + count_positive_tristate(values["b"])
            + count_positive_tristate(values["c"])
        )
        if should_include_record(n_positive_total, include_low_base_rate=False):
            records.append(
                {
                    "field": field,
                    "target_value": None,
                    "field_type": "tristate",
                    "disagreement_count": disagreement_count,
                    "disagreement_pattern_summary": summary,
                    "n_positive_total": n_positive_total,
                }
            )

    for field in ENUM_FIELDS + ["dominant_admission_reason"]:
        values = {
            variant: [
                _normalized_vote(parsed_by_variant[variant][hadm_id].get(field, "not_documented"))
                for hadm_id in hadm_ids
            ]
            for variant in ("a", "b", "c")
        }
        vote_tuples = list(zip(values["a"], values["b"], values["c"], strict=True))
        if not has_any_disagreement(vote_tuples):
            continue
        votes_by_hadm = {
            hadm_id: {"a": values["a"][idx], "b": values["b"][idx], "c": values["c"][idx]}
            for idx, hadm_id in enumerate(hadm_ids)
        }
        disagreement_count, summary = _disagreement_summary(votes_by_hadm)
        n_positive_total = (
            count_positive_enum(values["a"])
            + count_positive_enum(values["b"])
            + count_positive_enum(values["c"])
        )
        if should_include_record(n_positive_total, include_low_base_rate=False):
            records.append(
                {
                    "field": field,
                    "target_value": None,
                    "field_type": "enum" if field in ENUM_FIELDS else "dominant_admission_reason",
                    "disagreement_count": disagreement_count,
                    "disagreement_pattern_summary": summary,
                    "n_positive_total": n_positive_total,
                }
            )

    for tag in ADMISSION_REASON_TAGS:
        values: dict[str, list[str]] = {"a": [], "b": [], "c": []}
        lists_by_variant: dict[str, list[list[str]]] = {"a": [], "b": [], "c": []}
        for variant in ("a", "b", "c"):
            for hadm_id in hadm_ids:
                tags = list(parsed_by_variant[variant][hadm_id].get("admission_reason_tags", []))
                lists_by_variant[variant].append(tags)
                values[variant].append("present" if tag in set(tags) else "absent")
        vote_tuples = list(zip(values["a"], values["b"], values["c"], strict=True))
        if not has_any_disagreement(vote_tuples):
            continue
        votes_by_hadm = {
            hadm_id: {"a": values["a"][idx], "b": values["b"][idx], "c": values["c"][idx]}
            for idx, hadm_id in enumerate(hadm_ids)
        }
        disagreement_count, summary = _disagreement_summary(votes_by_hadm)
        n_positive_total = (
            count_positive_admission_tag(lists_by_variant["a"], tag)
            + count_positive_admission_tag(lists_by_variant["b"], tag)
            + count_positive_admission_tag(lists_by_variant["c"], tag)
        )
        if should_include_record(n_positive_total, include_low_base_rate=False):
            records.append(
                {
                    "field": "admission_reason_tags",
                    "target_value": tag,
                    "field_type": "admission_tag",
                    "disagreement_count": disagreement_count,
                    "disagreement_pattern_summary": summary,
                    "n_positive_total": n_positive_total,
                }
            )

    return records


def _load_iteration_payloads() -> list[dict[str, Any]]:
    paths = sorted(OPT_LOGS.glob("iteration_*.json"))
    if not paths:
        raise FileNotFoundError("No optimization iteration logs found under logs/optimization.")
    return [_read_json(path) for path in paths]


def compute_optimization_primary_metrics() -> OptimizationPrimaryMetrics:
    payloads = _load_iteration_payloads()
    applied_payloads = [p for p in payloads if bool(p.get("applied"))]
    if not applied_payloads:
        raise ValueError("No applied optimization iterations found in logs/optimization.")

    iterations_applied = len(applied_payloads)
    initial = int(payloads[0]["cluster_targeted"]["total_disagreement_count"])
    final_payload = max(applied_payloads, key=lambda p: int(p.get("iteration", 0)))
    final_run_ids_raw = final_payload.get("run_ids")
    if not isinstance(final_run_ids_raw, dict):
        raise ValueError("Final optimization payload is missing run_ids map.")

    final_run_ids = {
        "a": str(final_run_ids_raw["a"]),
        "b": str(final_run_ids_raw["b"]),
        "c": str(final_run_ids_raw["c"]),
    }
    target_variant = str(final_payload["cluster_targeted"]["affected_variant"]).lower()

    records = _build_records(final_run_ids)
    clusters = cluster_corpus(records)
    final = int(
        sum(
            cluster.total_disagreement_count
            for cluster in clusters
            if cluster.affected_variant == target_variant
        )
    )
    reduction = (100.0 * (initial - final) / initial) if initial > 0 else 0.0
    return OptimizationPrimaryMetrics(
        iterations_applied=iterations_applied,
        initial_cluster_disagreements=initial,
        final_cluster_disagreements=final,
        reduction_pct=float(reduction),
        final_run_ids=final_run_ids,
    )
