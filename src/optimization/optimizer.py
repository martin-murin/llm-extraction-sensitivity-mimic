from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src import config
from src.llm.client import LLMClient
from src.llm.extractor import check_prompt_vocabulary_sync, extract_content_from_raw_response
from src.optimization.pattern_clustering import PatternCluster


@dataclass
class IterationLog:
    iteration: int
    cluster_id: str
    targeted_variant: str | None
    applied: bool
    guard_results: dict[str, Any] = field(default_factory=dict)
    kappa_filtered_before: float | None = None
    kappa_filtered_after: float | None = None
    delta_pp: float | None = None
    cluster_kappa_before: float | None = None
    cluster_kappa_after: float | None = None
    cluster_delta_pp: float | None = None
    effectiveness: str | None = None
    cost_usd: float = 0.0
    optimizer_model_snapshot: str | None = None
    request_hash: str | None = None


def select_next_cluster(
    clusters: list[PatternCluster],
    history: list[IterationLog],
    min_disagreements: int = 50,
) -> PatternCluster | None:
    """Select the largest eligible cluster.

    Eligibility: affected_variant is set AND
    total_disagreement_count > min_disagreements.

    Note: previously-addressed clusters remain eligible if substantial residual
    disagreements remain.
    """
    eligible = [
        cluster
        for cluster in clusters
        if cluster.affected_variant is not None
        and cluster.total_disagreement_count > min_disagreements
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda cluster: cluster.total_disagreement_count)


def compute_cluster_kappa(
    cluster: PatternCluster,
    kappa_results: dict[str, dict[str, float]],
) -> float:
    kappas: list[float] = []
    for member in cluster.member_fields:
        key = str(member.get("field", ""))
        target_value = member.get("target_value")
        if target_value not in (None, "", "None"):
            key = f"{key}::{target_value}"

        row = kappa_results.get(key)
        if row is None:
            continue
        if "kappa_mean" not in row:
            continue
        kappas.append(float(row["kappa_mean"]))

    if not kappas:
        return 0.0
    return float(sum(kappas) / len(kappas))


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = []
    for row in rows:
        rendered = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *lines])


def _load_meta_prompt_template() -> str:
    template_path = (
        config.REPO_ROOT / "src" / "optimization" / "meta_prompts" / "optimizer_v1.md"
    )
    if not template_path.exists():
        raise FileNotFoundError(f"Missing meta prompt template: {template_path}")
    return template_path.read_text(encoding="utf-8")


def build_revision_request(cluster: PatternCluster, current_variant_text: str) -> str:
    template = _load_meta_prompt_template()

    member_rows = []
    for member in cluster.member_fields:
        field_name = str(member["field"])
        target_value = member.get("target_value")
        if target_value in (None, "", "None"):
            full_name = field_name
        else:
            full_name = f"{field_name}::{target_value}"
        member_rows.append(
            {
                "field": full_name,
                "kappa_mean": f"{float(member['kappa_mean']):.4f}",
                "disagreement_count": int(member["disagreement_count"]),
                "n_positive_total": int(member["n_positive_total"]),
            }
        )

    member_fields_table = _markdown_table(
        member_rows,
        ["field", "kappa_mean", "disagreement_count", "n_positive_total"],
    )

    example_lines: list[str] = []
    for idx, example in enumerate(cluster.representative_examples, start=1):
        hadm_id = int(example.get("hadm_id", 0))
        votes = example.get("votes", {})
        outlier_variant = str(example.get("outlier_variant") or "").lower()
        outlier_reasoning = str(example.get("outlier_reasoning_excerpt", "")).strip()
        consensus_reasoning = example.get("consensus_reasoning_excerpts", {})
        consensus_chunks = []
        if isinstance(consensus_reasoning, dict):
            for variant, excerpt in sorted(consensus_reasoning.items()):
                consensus_chunks.append(f"{variant}: {str(excerpt).strip()}")
        consensus_block = "\n".join(consensus_chunks) if consensus_chunks else "(none)"
        example_lines.extend(
            [
                f"## Example {idx}",
                f"- hadm_id: {hadm_id}",
                f"- votes: {json.dumps(votes, ensure_ascii=True)}",
                f"- outlier_variant: {outlier_variant or 'none'}",
                f"- outlier_reasoning_excerpt: {outlier_reasoning}",
                f"- consensus_reasoning_excerpts:\n{consensus_block}",
                "",
            ]
        )

    replacements = {
        "{cluster_label}": cluster.cluster_label,
        "{affected_variant}": str(cluster.affected_variant or ""),
        "{total_disagreement_count}": str(cluster.total_disagreement_count),
        "{member_fields_table}": member_fields_table,
        "{representative_examples_block}": "\n".join(example_lines).strip(),
        "{current_variant_text}": current_variant_text,
    }

    request = template
    for placeholder, value in replacements.items():
        request = request.replace(placeholder, value)
    return request


def _optimizer_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "OptimizerRevisionResponse",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "revised_prompt": {"type": "string"},
                    "rationale": {"type": "string"},
                    "self_assessment": {"type": "string"},
                },
                "required": ["revised_prompt", "rationale", "self_assessment"],
                "additionalProperties": False,
            },
        },
    }


def call_optimizer(
    meta_prompt: str,
    optimizer_model: str,
    llm_client: LLMClient,
) -> tuple[str, dict[str, Any]]:
    async def _call() -> tuple[str, dict[str, Any]]:
        response = await llm_client.chat(
            messages=[
                {"role": "system", "content": "You are a prompt optimization assistant."},
                {"role": "user", "content": meta_prompt},
            ],
            response_format=_optimizer_response_format(),
            max_completion_tokens=12000,
            model=optimizer_model,
            temperature=0.0,
        )

        if hasattr(response, "model_dump"):
            raw_response = dict(response.model_dump(mode="json"))
        else:
            raw_response = dict(response)

        content = extract_content_from_raw_response(raw_response)
        if not content:
            raise RuntimeError("Optimizer returned empty content.")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            payload = json.loads(content[start : end + 1])
        revised_prompt = str(payload.get("revised_prompt", "")).strip()
        if not revised_prompt:
            raise RuntimeError("Optimizer returned empty revised_prompt.")

        usage = raw_response.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}

        metadata = {
            "model_snapshot": str(raw_response.get("model", optimizer_model)),
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "raw_response": raw_response,
            "rationale": str(payload.get("rationale", "")),
            "self_assessment": str(payload.get("self_assessment", "")),
        }
        return revised_prompt, metadata

    return asyncio.run(_call())


def _contains_any(haystack_lower: str, options: list[str]) -> bool:
    return any(option.lower() in haystack_lower for option in options)


def run_guards(
    revised_prompt: str,
    original_prompt: str,
) -> tuple[bool, dict[str, Any]]:
    guard_results: dict[str, Any] = {}

    try:
        check_prompt_vocabulary_sync(revised_prompt)
        guard_results["vocabulary_sync"] = {"passed": True, "missing_tags": []}
    except Exception as exc:  # pragma: no cover - exact exception type not important for guard
        guard_results["vocabulary_sync"] = {"passed": False, "error": str(exc)}

    has_placeholder = "{{REASONING_INSTRUCTIONS}}" in revised_prompt
    guard_results["reasoning_placeholder_present"] = {"passed": has_placeholder}

    revised_lower = revised_prompt.lower()
    revised_normalized = revised_lower.replace("`", "")
    tri_state_passed = (
        "yes" in revised_lower and "no" in revised_lower and "not_documented" in revised_lower
    )
    edge_header_passed = "edge cases" in revised_lower
    expired_passed = _contains_any(revised_lower, ["expired patients", "expired patient"])
    redacted_passed = _contains_any(
        revised_lower,
        ["redacted notes", "heavily redacted", "redacted"],
    )
    transfer_passed = _contains_any(
        revised_lower,
        ["transfer admissions", "transfer admission"],
    )
    hospice_passed = _contains_any(revised_lower, ["hospice admissions", "hospice admission"])
    cardinality_passed = (
        "admission_reason_tags" in revised_normalized
        and "dominant_admission_reason" in revised_normalized
        and _contains_any(
            revised_normalized,
            [
                "must appear in admission_reason_tags",
                "must be one of the tags",
                "must be one of the tags in admission_reason_tags",
                "must be one of the tags from q1",
                "must be from your q1 list",
                "must be in the list",
                "one of the tags in your q1 answer",
                "pick exactly one tag from your q1 list",
            ],
        )
        and _contains_any(
            revised_normalized,
            [
                "admission_reason_tags is never empty",
                "admission_reason_tags must never be empty",
                "admission_reason_tags must be non-empty",
                "must be non-empty",
                "non-empty list",
                "at least one tag required",
                "q1 answer has at least one tag",
            ],
        )
    )

    frozen_checks = {
        "three_valued_logic": tri_state_passed,
        "edge_cases_header": edge_header_passed,
        "edge_case_expired": expired_passed,
        "edge_case_redacted": redacted_passed,
        "edge_case_transfer": transfer_passed,
        "edge_case_hospice": hospice_passed,
        "cardinality_constraints": cardinality_passed,
    }
    guard_results["frozen_content"] = {
        "passed": all(frozen_checks.values()),
        "checks": frozen_checks,
    }

    edit_distance = 1.0 - SequenceMatcher(None, original_prompt, revised_prompt).ratio()
    edit_distance_passed = edit_distance <= 0.25
    guard_results["edit_distance"] = {
        "passed": edit_distance_passed,
        "distance": edit_distance,
        "threshold": 0.25,
    }

    all_passed = all(
        [
            guard_results["vocabulary_sync"]["passed"],
            guard_results["reasoning_placeholder_present"]["passed"],
            guard_results["frozen_content"]["passed"],
            guard_results["edit_distance"]["passed"],
        ]
    )
    guard_results["all_passed"] = all_passed
    return all_passed, guard_results


def apply_revision(variant_letter: str, revised_prompt: str, prompts_dir: Path) -> Path:
    variant = variant_letter.lower().strip()
    if variant not in {"a", "b", "c"}:
        raise ValueError(f"Unsupported variant letter: {variant_letter}")

    path = prompts_dir / f"variant_{variant}.md"
    path.write_text(revised_prompt, encoding="utf-8")
    return path


def build_request_hash(meta_prompt: str, cluster: PatternCluster, current_variant_text: str) -> str:
    payload = {
        "meta_prompt": meta_prompt,
        "cluster": {
            "cluster_id": cluster.cluster_id,
            "cluster_label": cluster.cluster_label,
            "affected_variant": cluster.affected_variant,
            "total_disagreement_count": cluster.total_disagreement_count,
            "member_fields": cluster.member_fields,
            "representative_examples": cluster.representative_examples,
        },
        "current_variant_text": current_variant_text,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_iteration_sequence(
    clusters: list[PatternCluster],
    original_prompt: str,
    revision_provider: Callable[[PatternCluster, int], str],
    apply_callback: Callable[[PatternCluster, str], None],
    *,
    max_iterations: int = 5,
    min_disagreements: int = 50,
    cluster_delta_provider: Callable[[PatternCluster, int], float] | None = None,
    kappa_plateau_pp: float = 2.0,
) -> list[IterationLog]:
    history: list[IterationLog] = []
    consecutive_failures = 0

    for iteration in range(1, max_iterations + 1):
        cluster = select_next_cluster(
            clusters,
            history,
            min_disagreements=min_disagreements,
        )
        if cluster is None:
            break
        revised_prompt = revision_provider(cluster, iteration)
        passed, guard_results = run_guards(revised_prompt, original_prompt)
        applied = False
        effectiveness: str | None = None
        cluster_delta_pp: float | None = None
        if passed:
            apply_callback(cluster, revised_prompt)
            applied = True
            consecutive_failures = 0
            if cluster_delta_provider is not None:
                cluster_delta_pp = float(cluster_delta_provider(cluster, iteration))
                if cluster_delta_pp < kappa_plateau_pp:
                    effectiveness = "addressed_but_ineffective"
                else:
                    effectiveness = "successful"
            else:
                effectiveness = "successful"
        else:
            consecutive_failures += 1

        history.append(
            IterationLog(
                iteration=iteration,
                cluster_id=cluster.cluster_id,
                targeted_variant=cluster.affected_variant,
                applied=applied,
                guard_results=guard_results,
                effectiveness=effectiveness,
                cluster_delta_pp=cluster_delta_pp,
            )
        )

        if consecutive_failures >= 2:
            break

    return history
