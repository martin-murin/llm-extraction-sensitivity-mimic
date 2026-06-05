"""Mid-flight production extraction audit hook.

Reads: ongoing `data/raw_responses/production_v1/results.jsonl` or another configured run plus 5k baseline distribution artifacts.
Writes: timestamped `codex_outputs/audit_hook_*.md` drift reports.
Paper role: production-run QA guardrail; audit-hook findings informed production QA and limitations but are not a direct final figure.
Usage: `python scripts/33_audit_hooks_production.py` during or after production extraction.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from src.schema.vocabulary import ADMISSION_REASON_TAGS

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


def _load_features(run_id: str) -> list[dict[str, Any]]:
    path = Path("data/raw_responses") / run_id / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not bool(row.get("parse_ok")):
            continue
        feats = row.get("features_json")
        if isinstance(feats, dict):
            out.append(feats)
    return out


def _distribution(values: list[str]) -> dict[str, float]:
    if not values:
        return {}
    total = len(values)
    out: dict[str, float] = {}
    for value in values:
        out[value] = out.get(value, 0.0) + 1.0
    for key in list(out.keys()):
        out[key] = out[key] / total
    return out


def _markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(cols) + " |"
    divider = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")).replace("|", "\\|") for c in cols) + " |")
    return "\n".join([header, divider, *body])


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production mid-flight audit hooks.")
    parser.add_argument("--run-id", default="production_v1")
    parser.add_argument("--baseline-run-id", default="methodology_5k_a")
    parser.add_argument("--drift-pp", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    current = _load_features(args.run_id)
    baseline = _load_features(args.baseline_run_id)
    if not current:
        raise RuntimeError(f"No parsed features in run_id={args.run_id}")
    if not baseline:
        raise RuntimeError(f"No parsed features in baseline_run_id={args.baseline_run_id}")

    flagged: list[dict[str, Any]] = []

    for field in TRISTATE_FIELDS + ENUM_FIELDS + ["dominant_admission_reason"]:
        cur_vals = [str(row.get(field, "not_documented")) for row in current]
        base_vals = [str(row.get(field, "not_documented")) for row in baseline]
        cur_dist = _distribution(cur_vals)
        base_dist = _distribution(base_vals)

        for value in sorted(set(cur_dist) | set(base_dist)):
            cur_rate = cur_dist.get(value, 0.0)
            base_rate = base_dist.get(value, 0.0)
            delta_pp = (cur_rate - base_rate) * 100.0
            if abs(delta_pp) > args.drift_pp:
                flagged.append(
                    {
                        "field": field,
                        "value": value,
                        "baseline_rate": _fmt_pct(base_rate),
                        "current_rate": _fmt_pct(cur_rate),
                        "delta_pp": f"{delta_pp:+.2f}",
                    }
                )

    # Admission tags: compare tag presence rates.
    for tag in ADMISSION_REASON_TAGS:
        cur_rate = (
            sum(
                1
                for row in current
                if tag in set(row.get("admission_reason_tags", []))
            )
            / len(current)
        )
        base_rate = (
            sum(
                1
                for row in baseline
                if tag in set(row.get("admission_reason_tags", []))
            )
            / len(baseline)
        )
        delta_pp = (cur_rate - base_rate) * 100.0
        if abs(delta_pp) > args.drift_pp:
            flagged.append(
                {
                    "field": "admission_reason_tags",
                    "value": tag,
                    "baseline_rate": _fmt_pct(base_rate),
                    "current_rate": _fmt_pct(cur_rate),
                    "delta_pp": f"{delta_pp:+.2f}",
                }
            )

    status = "PASS" if not flagged else "WARN"
    now = datetime.now(tz=UTC)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output)
        if args.output
        else Path("codex_outputs") / f"audit_hook_{timestamp}.md"
    )

    lines = [
        "# Production Audit Hook",
        "",
        f"_Generated at {now.isoformat()}_",
        "",
        f"## Status: {status}",
        "",
        f"- Current run: `{args.run_id}` (n={len(current):,} parsed notes)",
        f"- Baseline run: `{args.baseline_run_id}` (n={len(baseline):,} parsed notes)",
        f"- Drift threshold: `{args.drift_pp:.1f}` percentage points",
        "",
        "## Drift flags (> threshold)",
        "",
        _markdown_table(
            sorted(flagged, key=lambda row: abs(float(row["delta_pp"])), reverse=True),
            ["field", "value", "baseline_rate", "current_rate", "delta_pp"],
        ),
        "",
        "## Recommendation",
        "",
        "- Run this hook after 50k completed notes, then every additional 100k notes.",
        (
            "- WARN status does not auto-halt the extraction; use it as an "
            "early-signal review trigger."
        ),
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit-hook report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
