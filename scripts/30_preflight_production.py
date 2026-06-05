"""
Runs preflight checks before production extraction.

Reads: configs/production.yaml, data/raw_responses, src/schema/prompts/variant_a.md, data/splits/SPLITS_MANIFEST.json, data/splits, codex_outputs/30_preflight_production.md.
Writes: data/raw_responses, data/splits/SPLITS_MANIFEST.json, data/splits, codex_outputs/30_preflight_production.md, data/raw_responses/{args.run_id}.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/30_preflight_production.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime

from openai import AsyncOpenAI
from sqlalchemy import text

from src import config
from src.db.connection import discover_schemas, get_engine
from src.db.queries import count_notes
from src.utils.threeway_kappa import file_sha256


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | WARN
    detail: str


def _markdown_table(rows: list[CheckResult]) -> str:
    header = "| check | status | detail |"
    divider = "|---|---|---|"
    lines = [header, divider]
    for row in rows:
        detail = row.detail.replace("|", "\\|")
        lines.append(f"| {row.name} | {row.status} | {detail} |")
    return "\n".join(lines)


def _check_configuration() -> list[CheckResult]:
    results: list[CheckResult] = []
    prod_path = Path("configs/production.yaml")
    if not prod_path.exists():
        return [CheckResult("configuration", "FAIL", "configs/production.yaml missing")]

    import yaml  # type: ignore[import-untyped]

    payload = yaml.safe_load(prod_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return [CheckResult("configuration", "FAIL", "Invalid YAML payload in production.yaml")]

    concurrency_raw = payload.get("max_concurrent_requests")
    concurrency = int(concurrency_raw) if isinstance(concurrency_raw, int) else None
    retries = int(config.MAX_RETRIES)
    results.append(
        CheckResult(
            "config.max_concurrent_requests",
            "PASS" if concurrency == 8 else "FAIL",
            f"observed={concurrency}, expected=8",
        )
    )
    results.append(
        CheckResult(
            "llm.max_retries",
            "PASS" if retries == 5 else "FAIL",
            f"observed={retries}, expected=5",
        )
    )
    return results


async def _api_key_validity_test() -> CheckResult:
    if not config.SETTINGS.openai_api_key:
        return CheckResult("api_key_validity", "FAIL", "OPENAI_API_KEY is not set")

    client = AsyncOpenAI(api_key=config.SETTINGS.openai_api_key)
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.MODEL_ID,
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=8,
                temperature=0.0,
                timeout=20.0,
            ),
            timeout=30.0,
        )
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        return CheckResult(
            "api_key_validity",
            "PASS",
            (
                "Chat completion succeeded "
                f"(prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens})"
            ),
        )
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return CheckResult(
            "api_key_validity",
            "FAIL",
            f"OpenAI call failed: {type(exc).__name__}: {exc}",
        )


def _check_database() -> list[CheckResult]:
    if not config.SETTINGS.mimic_pg_uri:
        return [CheckResult("database", "FAIL", "MIMIC_PG_URI is not set")]

    results: list[CheckResult] = []
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")).scalar_one()
        schemas = discover_schemas(engine)
        discharge_table = schemas.get("discharge_notes")
        if not discharge_table:
            results.append(
                CheckResult(
                    "db.discharge_table",
                    "FAIL",
                    "discharge notes table not found",
                )
            )
            return results
        results.append(
            CheckResult(
                "db.discharge_table",
                "PASS",
                f"resolved table: {discharge_table}",
            )
        )

        n_notes = count_notes(engine)
        status = "PASS" if 331_000 <= n_notes <= 332_500 else "FAIL"
        results.append(
            CheckResult(
                "db.discharge_note_count",
                status,
                f"observed={n_notes:,}; expected approximately 331,793",
            )
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        results.append(
            CheckResult(
                "database",
                "FAIL",
                f"DB check failed: {type(exc).__name__}: {exc}",
            )
        )
    return results


def _check_disk_space(target_dir: Path, min_free_gb: float = 4.0) -> CheckResult:
    base = target_dir if target_dir.exists() else target_dir.parent
    usage = shutil.disk_usage(base)
    free_gb = usage.free / (1024**3)
    status = "PASS" if free_gb >= min_free_gb else "FAIL"
    return CheckResult(
        "disk_space",
        status,
        f"free={free_gb:.2f} GB in {base} (required >= {min_free_gb:.1f} GB)",
    )


def _check_run_id_collision(run_id: str) -> CheckResult:
    run_dir = Path("data/raw_responses") / run_id
    if not run_dir.exists():
        return CheckResult("run_id_collision", "PASS", f"{run_dir} does not exist")
    if run_dir.is_dir() and not any(run_dir.iterdir()):
        return CheckResult(
            "run_id_collision",
            "PASS",
            f"{run_dir} exists but is empty (safe to launch fresh run)",
        )

    has_results = (run_dir / "results.jsonl").exists()
    has_meta = (run_dir / "run_metadata.json").exists()
    if has_results and has_meta:
        return CheckResult(
            "run_id_collision",
            "PASS",
            f"{run_dir} exists with checkpoint artifacts (resume mode valid)",
        )

    return CheckResult(
        "run_id_collision",
        "FAIL",
        (
            f"{run_dir} exists but checkpoint is incomplete "
            f"(results.jsonl={has_results}, run_metadata.json={has_meta})"
        ),
    )


def _check_zombie_processes() -> CheckResult:
    try:
        completed = subprocess.run(
            ["ps", "-ef"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        return CheckResult("zombie_processes", "WARN", f"Could not inspect process list: {exc}")

    lines = [
        line
        for line in completed.stdout.splitlines()
        if "python" in line and "scripts/05_run_smoke_coverage.py" in line
    ]
    if not lines:
        return CheckResult("zombie_processes", "PASS", "No existing batch-runner process found")

    return CheckResult(
        "zombie_processes",
        "WARN",
        f"Found {len(lines)} existing runner process(es); verify before launch",
    )


def _check_snapshot_integrity() -> list[CheckResult]:
    mapping = [
        (
            Path("src/schema/prompts/variant_a.md"),
            Path("snapshots/phase6_locked/prompts/variant_a.md"),
        ),
        (Path("src/schema/vocabulary.py"), Path("snapshots/phase6_locked/vocabulary.py")),
        (Path("src/schema/fields.py"), Path("snapshots/phase6_locked/fields.py")),
    ]
    results: list[CheckResult] = []
    for live, snap in mapping:
        if not live.exists() or not snap.exists():
            results.append(
                CheckResult(
                    f"snapshot.{live.name}",
                    "FAIL",
                    f"Missing file(s): {live} or {snap}",
                )
            )
            continue
        live_hash = file_sha256(live)
        snap_hash = file_sha256(snap)
        status = "PASS" if live_hash == snap_hash else "FAIL"
        results.append(
            CheckResult(
                f"snapshot.{live.name}",
                status,
                f"live={live_hash[:12]}..., snapshot={snap_hash[:12]}...",
            )
        )
    return results


def _check_splits_manifest() -> list[CheckResult]:
    manifest_path = Path("data/splits/SPLITS_MANIFEST.json")
    if not manifest_path.exists():
        return [CheckResult("splits_manifest", "FAIL", f"Missing {manifest_path}")]

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = payload.get("checksums_sha256")
    if not isinstance(checksums, dict):
        return [CheckResult("splits_manifest", "FAIL", "Missing checksums_sha256 map")]

    results: list[CheckResult] = []
    tracked_keys = sorted(k for k in checksums if isinstance(k, str) and k.endswith(".csv"))
    for key in tracked_keys:
        file_path = Path("data/splits") / key
        if not file_path.exists():
            results.append(CheckResult(f"split.{key}", "FAIL", "File missing"))
            continue
        observed = file_sha256(file_path)
        expected = str(checksums[key])
        status = "PASS" if observed == expected else "FAIL"
        results.append(
            CheckResult(
                f"split.{key}",
                status,
                f"observed={observed[:12]}..., expected={expected[:12]}...",
            )
        )

    # Non-tracked CSV files are warnings for visibility but do not block launch.
    csv_files = sorted(Path("data/splits").glob("*.csv"))
    tracked_set = set(tracked_keys)
    extras = [p.name for p in csv_files if p.name not in tracked_set]
    if extras:
        results.append(
            CheckResult(
                "splits_manifest_extras",
                "WARN",
                f"Untracked split CSVs present: {', '.join(extras)}",
            )
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-flight checks for production_v1 launch.")
    parser.add_argument("--run-id", default="production_v1")
    parser.add_argument("--output", default="codex_outputs/30_preflight_production.md")
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.load_env()

    results: list[CheckResult] = []
    results.extend(_check_configuration())
    results.append(asyncio.run(_api_key_validity_test()))
    results.extend(_check_database())
    results.append(_check_disk_space(Path(f"data/raw_responses/{args.run_id}"), args.min_free_gb))
    results.append(_check_run_id_collision(args.run_id))
    results.append(_check_zombie_processes())
    results.extend(_check_snapshot_integrity())
    results.extend(_check_splits_manifest())

    has_fail = any(item.status == "FAIL" for item in results)
    banner = "PASS" if not has_fail else "FAIL"

    lines = [
        "# Production Pre-flight Checklist",
        "",
        f"_Generated at {datetime.now(tz=UTC).isoformat()}_",
        "",
        f"## Banner: {banner}",
        "",
        _markdown_table(results),
        "",
        "## Exit code behavior",
        "- Exit code 0 only when no check is in FAIL status.",
        "- WARN checks are non-blocking but should be reviewed before launch.",
        "",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote pre-flight report to {output_path}")

    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
