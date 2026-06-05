"""Phase 3f autonomous prompt-optimization loop.

Reads: `src/schema/prompts`, `data/optimization/audit_corpus_v*.jsonl`, prior run/kappa artifacts, and OpenAI configuration.
Writes: `logs/optimization/iteration_*.json`, updated prompt variants during the loop, `data/optimization/audit_corpus_v*.jsonl`, and `codex_outputs/18_optimization_loop_summary.md`.
Paper role: backs optimization-loop manuscript claims, including iteration count and disagreement-cluster reduction.
Usage: `python scripts/16_run_optimization_loop.py --resume --max-iterations 5 --budget-cap-usd 50.0` for the documented resumed loop.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
from datetime import UTC, datetime
from difflib import unified_diff
from typing import Any

from src import config
from src.llm.client import LLMClient
from src.optimization.optimizer import (
    IterationLog,
    apply_revision,
    build_request_hash,
    build_revision_request,
    call_optimizer,
    compute_cluster_kappa,
    run_guards,
    select_next_cluster,
)
from src.optimization.pattern_clustering import PatternCluster, cluster_corpus

logger = logging.getLogger("scripts.16_run_optimization_loop")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autonomous prompt optimization loop.")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--budget-cap-usd", type=float, default=50.0)
    parser.add_argument("--soft-target-usd", type=float, default=40.0)
    parser.add_argument("--kappa-plateau-pp", type=float, default=2.0)
    parser.add_argument("--prompts-dir", default="src/schema/prompts")
    parser.add_argument("--corpus", default="data/optimization/audit_corpus_v1.jsonl")
    parser.add_argument("--logs-dir", default="logs/optimization")
    parser.add_argument("--output-summary", default="codex_outputs/18_optimization_loop_summary.md")
    parser.add_argument(
        "--baseline-run-ids",
        nargs=3,
        default=["refinement_v1_a", "refinement_v1_b", "refinement_v1_c"],
        metavar=("RUN_A", "RUN_B", "RUN_C"),
    )
    parser.add_argument("--optimizer-run-id", default="optimization_loop_v1")
    parser.add_argument("--optimizer-input-price-per-million", type=float, default=5.0)
    parser.add_argument("--optimizer-output-price-per-million", type=float, default=15.0)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


def _run_command(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=config.REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return output.strip()


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing audit corpus: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_last_cost_for_run(run_id: str) -> float:
    log_path = config.LOGS_DIR / "runs" / f"{run_id}_cost.json"
    if not log_path.exists():
        return 0.0
    last = ""
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = line.strip()
    if not last:
        return 0.0
    payload = json.loads(last)
    return float(payload.get("total_cost_usd", 0.0))


def _parse_summary_table(report_text: str, heading: str) -> dict[str, str]:
    lines = report_text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == heading and idx + 3 < len(lines):
            header_line = lines[idx + 1]
            value_line = lines[idx + 3]
            headers = [part.strip() for part in header_line.strip().strip("|").split("|")]
            values = [part.strip() for part in value_line.strip().strip("|").split("|")]
            if len(headers) != len(values):
                continue
            return {headers[i]: values[i] for i in range(len(headers))}
    raise RuntimeError(f"Could not parse summary heading: {heading}")


def _parse_kappa_medians(report_path: Path) -> tuple[float, float]:
    text = report_path.read_text(encoding="utf-8")
    all_table = _parse_summary_table(text, "## Kappa summary (all fields)")
    filtered_table = _parse_summary_table(
        text,
        "## Kappa summary (filtered, excluding low-base-rate fields)",
    )
    return (
        float(all_table["overall_median_kappa"]),
        float(filtered_table["overall_median_kappa"]),
    )


def _load_kappa_sidecar(report_path: Path) -> dict[str, dict[str, float]]:
    sidecar_path = Path(f"{report_path}.json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Missing kappa sidecar JSON: {sidecar_path}")

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    raw = payload.get("kappa_results", {})
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid kappa sidecar structure: {sidecar_path}")

    kappa_results: dict[str, dict[str, float]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        kappa_results[key] = {
            "kappa_mean": float(value.get("kappa_mean", 0.0)),
            "kappa_A_B": float(value.get("kappa_A_B", 0.0)),
            "kappa_A_C": float(value.get("kappa_A_C", 0.0)),
            "kappa_B_C": float(value.get("kappa_B_C", 0.0)),
        }
    return kappa_results


def _extract_guard_status(guard_results: dict[str, Any]) -> str:
    if not guard_results:
        return "n/a"
    parts = []
    for key in [
        "vocabulary_sync",
        "reasoning_placeholder_present",
        "frozen_content",
        "edit_distance",
    ]:
        entry = guard_results.get(key, {})
        passed = bool(entry.get("passed", False))
        parts.append(f"{key}={'PASS' if passed else 'FAIL'}")
    return ", ".join(parts)


def _load_iteration_history(
    logs_dir: Path,
) -> tuple[list[IterationLog], list[Path], dict[str, str]]:
    history: list[IterationLog] = []
    paths: list[Path] = []
    latest_run_ids: dict[str, str] = {}
    latest_run_ids_any: dict[str, str] = {}

    for path in sorted(logs_dir.glob("iteration_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        paths.append(path)
        history.append(
            IterationLog(
                iteration=int(payload.get("iteration", 0)),
                cluster_id=str(payload.get("cluster_targeted", {}).get("cluster_id", "")),
                targeted_variant=payload.get("cluster_targeted", {}).get("affected_variant"),
                applied=bool(payload.get("applied", False)),
                guard_results=dict(payload.get("guard_outcomes", {})),
                kappa_filtered_before=(
                    float(payload["kappa_filtered_before"])
                    if payload.get("kappa_filtered_before") is not None
                    else None
                ),
                kappa_filtered_after=(
                    float(payload["kappa_filtered_after"])
                    if payload.get("kappa_filtered_after") is not None
                    else None
                ),
                delta_pp=(
                    float(payload["kappa_delta_pp"])
                    if payload.get("kappa_delta_pp") is not None
                    else None
                ),
                cluster_kappa_before=(
                    float(payload["cluster_kappa_before"])
                    if payload.get("cluster_kappa_before") is not None
                    else None
                ),
                cluster_kappa_after=(
                    float(payload["cluster_kappa_after"])
                    if payload.get("cluster_kappa_after") is not None
                    else None
                ),
                cluster_delta_pp=(
                    float(payload["cluster_delta_pp"])
                    if payload.get("cluster_delta_pp") is not None
                    else None
                ),
                effectiveness=(
                    str(payload.get("effectiveness"))
                    if payload.get("effectiveness") is not None
                    else None
                ),
                cost_usd=float(payload.get("iteration_cost_usd", 0.0)),
                optimizer_model_snapshot=(
                    str(payload.get("optimizer_model_snapshot"))
                    if payload.get("optimizer_model_snapshot")
                    else None
                ),
                request_hash=(
                    str(payload.get("request_hash"))
                    if payload.get("request_hash")
                    else None
                ),
            )
        )
        run_ids = payload.get("run_ids", {})
        if isinstance(run_ids, dict):
            for variant in ["a", "b", "c"]:
                value = run_ids.get(variant)
                if value:
                    latest_run_ids_any[variant] = str(value)
                    if bool(payload.get("applied", False)):
                        latest_run_ids[variant] = str(value)

    if not latest_run_ids:
        latest_run_ids = dict(latest_run_ids_any)

    history.sort(key=lambda item: item.iteration)
    paths.sort()
    return history, paths, latest_run_ids


def _detect_latest_corpus_path() -> tuple[Path | None, int]:
    opt_dir = config.DATA_DIR / "optimization"
    best_path: Path | None = None
    best_version = 0
    for candidate in opt_dir.glob("audit_corpus_v*.jsonl"):
        matched = re.search(r"audit_corpus_v(\d+)\.jsonl$", candidate.name)
        if not matched:
            continue
        version = int(matched.group(1))
        if version > best_version:
            best_version = version
            best_path = candidate
    return best_path, best_version


def _corpus_version(path: Path) -> int:
    matched = re.search(r"audit_corpus_v(\d+)\.jsonl$", path.name)
    if matched:
        return int(matched.group(1))
    return 1


def _discover_latest_gpt54_snapshot(client: LLMClient) -> str:
    async def _discover() -> str:
        page = await client.client.models.list()
        model_ids: list[str] = [str(item.id) for item in getattr(page, "data", [])]

        while True:
            has_next_attr = getattr(page, "has_next_page", None)
            has_next = bool(has_next_attr()) if callable(has_next_attr) else bool(has_next_attr)
            if not has_next:
                break
            page = await page.get_next_page()
            model_ids.extend(str(item.id) for item in getattr(page, "data", []))

        candidates = [
            model_id
            for model_id in model_ids
            if model_id.startswith("gpt-5.4")
            and "mini" not in model_id
            and "nano" not in model_id
            and "pro" not in model_id
        ]
        if not candidates:
            raise RuntimeError("No eligible gpt-5.4 full snapshot found in models endpoint.")

        def sort_key(model_id: str) -> tuple[int, str]:
            matched = re.search(r"(\d{4})-(\d{2})-(\d{2})", model_id)
            if matched:
                y, m, d = (int(matched.group(1)), int(matched.group(2)), int(matched.group(3)))
                return (y * 10000 + m * 100 + d, model_id)
            return (0, model_id)

        candidates.sort(key=sort_key)
        return candidates[-1]

    import asyncio

    return asyncio.run(_discover())


def _snapshot_original_prompts(prompts_dir: Path, originals_dir: Path) -> dict[str, Path]:
    originals_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, Path] = {}
    for variant in ["a", "b", "c"]:
        src = prompts_dir / f"variant_{variant}.md"
        if not src.exists():
            raise FileNotFoundError(f"Missing prompt file: {src}")
        dst = originals_dir / src.name
        shutil.copy2(src, dst)
        snapshots[variant] = dst
    return snapshots


def _restore_original_prompts(prompts_dir: Path, snapshots: dict[str, Path]) -> None:
    for variant, src in snapshots.items():
        dst = prompts_dir / f"variant_{variant}.md"
        shutil.copy2(src, dst)


def _build_diff(original_path: Path, revised_path: Path, label: str) -> str:
    original_lines = original_path.read_text(encoding="utf-8").splitlines()
    revised_lines = revised_path.read_text(encoding="utf-8").splitlines()
    diff = unified_diff(
        original_lines,
        revised_lines,
        fromfile=f"{label}_original",
        tofile=f"{label}_final",
        lineterm="",
    )
    return "\n".join(diff)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    lines: list[str] = []
    for row in rows:
        vals = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider, *lines])


def _remaining_substantial_clusters(
    corpus_path: Path,
    min_disagreements: int,
) -> list[PatternCluster]:
    records = _load_corpus(corpus_path)
    clusters = cluster_corpus(records)
    return [
        cluster
        for cluster in clusters
        if cluster.affected_variant is not None
        and cluster.total_disagreement_count > min_disagreements
    ]


def main() -> int:
    args = _parse_args()
    config.load_env()

    prompts_dir = Path(args.prompts_dir)
    logs_dir = Path(args.logs_dir)
    originals_dir = logs_dir / "originals"
    output_summary_path = Path(args.output_summary)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    baseline_run_ids = {
        "a": args.baseline_run_ids[0],
        "b": args.baseline_run_ids[1],
        "c": args.baseline_run_ids[2],
    }

    optimizer_client = LLMClient(
        run_id=args.optimizer_run_id,
        max_budget_usd=args.budget_cap_usd,
        input_price_per_million=args.optimizer_input_price_per_million,
        output_price_per_million=args.optimizer_output_price_per_million,
    )
    optimizer_model_snapshot = _discover_latest_gpt54_snapshot(optimizer_client)
    logger.info("Using optimizer model snapshot: %s", optimizer_model_snapshot)

    snapshots: dict[str, Path]
    history: list[IterationLog]
    iteration_log_paths: list[Path]
    current_run_ids = dict(baseline_run_ids)
    current_corpus_path = Path(args.corpus)
    current_corpus_version = _corpus_version(current_corpus_path)
    start_iteration = 1

    if args.resume:
        if not originals_dir.exists():
            raise FileNotFoundError(
                f"Resume requested but originals snapshot directory is missing: {originals_dir}"
            )
        snapshots = {
            variant: originals_dir / f"variant_{variant}.md"
            for variant in ["a", "b", "c"]
        }
        for variant, path in snapshots.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Resume requested but missing snapshot for variant {variant}: {path}"
                )

        loaded_history, loaded_paths, latest_run_ids = _load_iteration_history(logs_dir)
        if not loaded_history:
            raise RuntimeError("Resume requested but no iteration logs were found.")

        current_run_ids.update(latest_run_ids)

        latest_corpus_path, latest_version = _detect_latest_corpus_path()
        if latest_corpus_path is None:
            raise RuntimeError("Resume requested but no audit corpus v*.jsonl files were found.")
        current_corpus_path = latest_corpus_path
        current_corpus_version = latest_version

        max_applied_iteration = max(
            (entry.iteration for entry in loaded_history if entry.applied),
            default=0,
        )
        start_iteration = max_applied_iteration + 1
        history = [entry for entry in loaded_history if entry.iteration < start_iteration]
        iteration_log_paths = [
            path
            for path in loaded_paths
            if int(path.stem.split("_")[1]) < start_iteration
        ]
    else:
        snapshots = _snapshot_original_prompts(prompts_dir, originals_dir)
        history = []
        iteration_log_paths = []

    baseline_kappa_report = config.CODEX_OUTPUTS_DIR / "16c_baseline_kappa.md"
    _run_command(
        [
            sys.executable,
            "scripts/14_threeway_kappa.py",
            "--run-ids",
            baseline_run_ids["a"],
            baseline_run_ids["b"],
            baseline_run_ids["c"],
            "--output",
            str(baseline_kappa_report),
        ]
    )
    baseline_unfiltered_median, baseline_filtered_median = _parse_kappa_medians(
        baseline_kappa_report
    )

    current_kappa_report = config.CODEX_OUTPUTS_DIR / "16c_current_kappa.md"
    _run_command(
        [
            sys.executable,
            "scripts/14_threeway_kappa.py",
            "--run-ids",
            current_run_ids["a"],
            current_run_ids["b"],
            current_run_ids["c"],
            "--output",
            str(current_kappa_report),
        ]
    )

    latest_kappa_report_path = current_kappa_report
    _latest_unfiltered_median, latest_filtered_median = _parse_kappa_medians(
        current_kappa_report
    )
    latest_kappa_results = _load_kappa_sidecar(current_kappa_report)

    total_cost_usd = float(sum(entry.cost_usd for entry in history))
    any_applied_this_invocation = False
    stopping_reason = "max_iterations_reached"

    consecutive_failures = 0
    for trailing_failures, entry in enumerate(reversed(history), start=1):
        if entry.applied:
            break
        consecutive_failures = trailing_failures

    if total_cost_usd >= args.soft_target_usd:
        logger.warning(
            "Soft target already exceeded before this invocation: %.2f >= %.2f",
            total_cost_usd,
            args.soft_target_usd,
        )

    for iteration in range(start_iteration, args.max_iterations + 1):
        if total_cost_usd >= args.budget_cap_usd:
            stopping_reason = "budget_cap_reached"
            break

        clusters = cluster_corpus(_load_corpus(current_corpus_path))
        cluster = select_next_cluster(clusters, history, min_disagreements=50)
        if cluster is None:
            stopping_reason = "no_remaining_clusters_above_threshold"
            break
        if cluster.affected_variant is None:
            stopping_reason = "selected_cluster_not_targetable"
            break

        variant = cluster.affected_variant
        prompt_path = prompts_dir / f"variant_{variant}.md"
        current_variant_text = prompt_path.read_text(encoding="utf-8")
        meta_prompt = build_revision_request(cluster, current_variant_text)
        request_hash = build_request_hash(meta_prompt, cluster, current_variant_text)

        optimizer_metadata: dict[str, Any] = {}
        guard_results: dict[str, Any] = {}
        applied = False
        effectiveness: str | None = None
        cluster_kappa_before: float | None = None
        cluster_kappa_after: float | None = None
        cluster_delta_pp: float | None = None

        kappa_before = latest_filtered_median
        kappa_after = latest_filtered_median
        delta_pp: float | None = None
        iteration_cost = 0.0

        try:
            revised_prompt, optimizer_metadata = call_optimizer(
                meta_prompt=meta_prompt,
                optimizer_model=optimizer_model_snapshot,
                llm_client=optimizer_client,
            )
            optimizer_call_cost = (
                (int(optimizer_metadata.get("input_tokens", 0)) / 1_000_000)
                * args.optimizer_input_price_per_million
                + (int(optimizer_metadata.get("output_tokens", 0)) / 1_000_000)
                * args.optimizer_output_price_per_million
            )
            iteration_cost += optimizer_call_cost
            total_cost_usd += optimizer_call_cost

            guard_passed, guard_results = run_guards(
                revised_prompt=revised_prompt,
                original_prompt=current_variant_text,
            )

            if guard_passed:
                apply_revision(variant, revised_prompt, prompts_dir)
                applied = True
                any_applied_this_invocation = True
                consecutive_failures = 0

                cluster_kappa_before = compute_cluster_kappa(cluster, latest_kappa_results)

                next_run_id = f"refinement_v{iteration + 1}_{variant}"
                coverage_output = config.CODEX_OUTPUTS_DIR / f"05_{next_run_id}_coverage.md"
                _run_command(
                    [
                        sys.executable,
                        "scripts/05_run_smoke_coverage.py",
                        "--run-id",
                        next_run_id,
                        "--split",
                        "refinement",
                        "--variant",
                        variant,
                        "--include-reasoning",
                        "--output",
                        str(coverage_output),
                    ]
                )
                extraction_cost = _load_last_cost_for_run(next_run_id)
                iteration_cost += extraction_cost
                total_cost_usd += extraction_cost
                current_run_ids[variant] = next_run_id

                kappa_report_path = config.CODEX_OUTPUTS_DIR / f"16c_iter{iteration}_kappa.md"
                _run_command(
                    [
                        sys.executable,
                        "scripts/14_threeway_kappa.py",
                        "--run-ids",
                        current_run_ids["a"],
                        current_run_ids["b"],
                        current_run_ids["c"],
                        "--output",
                        str(kappa_report_path),
                    ]
                )
                latest_kappa_report_path = kappa_report_path
                _latest_unfiltered_median, latest_filtered_median = _parse_kappa_medians(
                    latest_kappa_report_path
                )
                latest_kappa_results = _load_kappa_sidecar(latest_kappa_report_path)
                kappa_after = latest_filtered_median
                delta_pp = (kappa_after - kappa_before) * 100.0

                cluster_kappa_after = compute_cluster_kappa(cluster, latest_kappa_results)
                cluster_delta_pp = (cluster_kappa_after - cluster_kappa_before) * 100.0
                if cluster_delta_pp < args.kappa_plateau_pp:
                    effectiveness = "addressed_but_ineffective"
                else:
                    effectiveness = "successful"

                next_corpus_version = current_corpus_version + 1
                next_corpus_path = (
                    config.DATA_DIR
                    / "optimization"
                    / f"audit_corpus_v{next_corpus_version}.jsonl"
                )
                _run_command(
                    [
                        sys.executable,
                        "scripts/15_build_audit_corpus.py",
                        "--run-ids",
                        current_run_ids["a"],
                        current_run_ids["b"],
                        current_run_ids["c"],
                        "--output",
                        str(next_corpus_path),
                    ]
                )
                current_corpus_path = next_corpus_path
                current_corpus_version = next_corpus_version
            else:
                consecutive_failures += 1
        except Exception as exc:  # pragma: no cover
            guard_results = {"all_passed": False, "runtime_error": str(exc)}
            consecutive_failures += 1

        log_entry = IterationLog(
            iteration=iteration,
            cluster_id=cluster.cluster_id,
            targeted_variant=variant,
            applied=applied,
            guard_results=guard_results,
            kappa_filtered_before=kappa_before,
            kappa_filtered_after=kappa_after,
            delta_pp=delta_pp,
            cluster_kappa_before=cluster_kappa_before,
            cluster_kappa_after=cluster_kappa_after,
            cluster_delta_pp=cluster_delta_pp,
            effectiveness=effectiveness,
            cost_usd=iteration_cost,
            optimizer_model_snapshot=str(
                optimizer_metadata.get("model_snapshot", optimizer_model_snapshot)
            ),
            request_hash=request_hash,
        )
        history.append(log_entry)

        iteration_payload = {
            "iteration": iteration,
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "cluster_targeted": {
                "cluster_id": cluster.cluster_id,
                "cluster_label": cluster.cluster_label,
                "affected_variant": cluster.affected_variant,
                "total_disagreement_count": cluster.total_disagreement_count,
                "member_fields": cluster.member_fields,
            },
            "optimizer_model_snapshot": log_entry.optimizer_model_snapshot,
            "request_hash": request_hash,
            "meta_prompt_sha256": hashlib.sha256(meta_prompt.encode("utf-8")).hexdigest(),
            "optimizer_response": optimizer_metadata.get("raw_response", {}),
            "guard_outcomes": guard_results,
            "applied": applied,
            "kappa_filtered_before": kappa_before,
            "kappa_filtered_after": kappa_after,
            "kappa_delta_pp": delta_pp,
            "cluster_kappa_before": cluster_kappa_before,
            "cluster_kappa_after": cluster_kappa_after,
            "cluster_delta_pp": cluster_delta_pp,
            "effectiveness": effectiveness,
            "run_ids": current_run_ids,
            "iteration_cost_usd": iteration_cost,
            "total_cost_usd": total_cost_usd,
        }
        iteration_log_path = logs_dir / f"iteration_{iteration}.json"
        _write_json(iteration_log_path, iteration_payload)
        iteration_log_paths.append(iteration_log_path)

        if total_cost_usd >= args.soft_target_usd:
            logger.warning(
                "Soft target reached (%.2f >= %.2f). Continuing until hard stop.",
                total_cost_usd,
                args.soft_target_usd,
            )

        if consecutive_failures >= 2:
            stopping_reason = "two_consecutive_failed_iterations"
            break
        if total_cost_usd >= args.budget_cap_usd:
            stopping_reason = "budget_cap_reached"
            break

        remaining = _remaining_substantial_clusters(current_corpus_path, 50)
        if not remaining:
            stopping_reason = "no_remaining_clusters_above_threshold"
            break

    if not args.resume and not any_applied_this_invocation:
        _restore_original_prompts(prompts_dir, snapshots)

    history = sorted(history, key=lambda item: item.iteration)
    iteration_rows = [
        {
            "iteration": entry.iteration,
            "cluster_id": entry.cluster_id,
            "targeted_variant": entry.targeted_variant or "",
            "applied": "Y" if entry.applied else "N",
            "guard_outcomes": _extract_guard_status(entry.guard_results),
            "kappa_before": f"{(entry.kappa_filtered_before or 0.0):.4f}",
            "kappa_after": f"{(entry.kappa_filtered_after or 0.0):.4f}",
            "kappa_delta_pp": "" if entry.delta_pp is None else f"{entry.delta_pp:.2f}",
            "cluster_kappa_before": (
                "" if entry.cluster_kappa_before is None else f"{entry.cluster_kappa_before:.4f}"
            ),
            "cluster_kappa_after": (
                "" if entry.cluster_kappa_after is None else f"{entry.cluster_kappa_after:.4f}"
            ),
            "cluster_delta_pp": (
                "" if entry.cluster_delta_pp is None else f"{entry.cluster_delta_pp:.2f}"
            ),
            "effectiveness": entry.effectiveness or "",
            "iteration_cost_usd": f"{entry.cost_usd:.4f}",
        }
        for entry in history
    ]

    diff_sections: list[str] = []
    changed_variants: list[str] = []
    for variant in ["a", "b", "c"]:
        original_path = snapshots[variant]
        current_path = prompts_dir / f"variant_{variant}.md"
        if original_path.read_text(encoding="utf-8") == current_path.read_text(encoding="utf-8"):
            continue
        changed_variants.append(variant)
        diff_text = _build_diff(original_path, current_path, f"variant_{variant}.md")
        diff_sections.extend(
            [
                f"### variant_{variant}.md diff",
                "```diff",
                diff_text if diff_text.strip() else "(no textual diff)",
                "```",
                "",
            ]
        )

    final_unfiltered_median, final_filtered_median = _parse_kappa_medians(latest_kappa_report_path)

    summary_lines = [
        "# Optimization Loop Summary",
        "",
        "## Run metadata",
        _markdown_table(
            [
                {
                    "timestamp_utc": datetime.now(tz=UTC).isoformat(),
                    "optimizer_model_snapshot": optimizer_model_snapshot,
                    "iterations_attempted": len(iteration_rows),
                    "total_cost_usd": f"{total_cost_usd:.4f}",
                    "stopping_reason": stopping_reason,
                    "resume_mode": str(args.resume),
                }
            ],
            [
                "timestamp_utc",
                "optimizer_model_snapshot",
                "iterations_attempted",
                "total_cost_usd",
                "stopping_reason",
                "resume_mode",
            ],
        ),
        "",
        "## Per-iteration table",
        _markdown_table(
            iteration_rows,
            [
                "iteration",
                "cluster_id",
                "targeted_variant",
                "applied",
                "guard_outcomes",
                "kappa_before",
                "kappa_after",
                "kappa_delta_pp",
                "cluster_kappa_before",
                "cluster_kappa_after",
                "cluster_delta_pp",
                "effectiveness",
                "iteration_cost_usd",
            ],
        ),
        "",
        "## Final kappa summary",
        _markdown_table(
            [
                {
                    "baseline_filtered_median": f"{baseline_filtered_median:.4f}",
                    "final_filtered_median": f"{final_filtered_median:.4f}",
                    "filtered_delta_pp": (
                        f"{(final_filtered_median - baseline_filtered_median) * 100.0:.2f}"
                    ),
                    "baseline_unfiltered_median": f"{baseline_unfiltered_median:.4f}",
                    "final_unfiltered_median": f"{final_unfiltered_median:.4f}",
                    "unfiltered_delta_pp": (
                        f"{(final_unfiltered_median - baseline_unfiltered_median) * 100.0:.2f}"
                    ),
                    "latest_kappa_report": str(latest_kappa_report_path),
                }
            ],
            [
                "baseline_filtered_median",
                "final_filtered_median",
                "filtered_delta_pp",
                "baseline_unfiltered_median",
                "final_unfiltered_median",
                "unfiltered_delta_pp",
                "latest_kappa_report",
            ],
        ),
        "",
        "## Final variant file changes",
    ]

    if diff_sections:
        summary_lines.extend(diff_sections)
    else:
        summary_lines.extend(["_No variant file changes were applied._", ""])

    summary_lines.extend(
        [
            "## Traceability paths",
            _markdown_table(
                [
                    {
                        "original_snapshots_dir": str(originals_dir),
                        "iteration_logs_dir": str(logs_dir),
                        "iteration_logs": ", ".join(
                            str(path) for path in iteration_log_paths
                        ),
                        "final_corpus": str(current_corpus_path),
                        "changed_variants": (
                            ", ".join(changed_variants) if changed_variants else "(none)"
                        ),
                    }
                ],
                [
                    "original_snapshots_dir",
                    "iteration_logs_dir",
                    "iteration_logs",
                    "final_corpus",
                    "changed_variants",
                ],
            ),
            "",
        ]
    )

    output_summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote optimization summary to {output_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
