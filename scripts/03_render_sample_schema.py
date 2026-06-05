"""
Renders the structured extraction schema for prompt/API verification.

Reads: codex_outputs/03_schema_render.md.
Writes: codex_outputs/03_schema_render.md.
Backs the paper's extraction, QA, agreement, or reproducibility pipeline.
Usage: `python scripts/03_render_sample_schema.py` unless the script's argparse help says otherwise.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Any, get_args, get_origin

from src.schema.fields import LLMNoteFeatures
from src.schema.vocabulary import ADMISSION_REASON_TAGS, ADMISSION_REASON_TAG_DESCRIPTIONS


def _format_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        if hasattr(annotation, "__name__"):
            return str(annotation.__name__)
        return str(annotation)

    if origin is list:
        args = get_args(annotation)
        inner = _format_type(args[0]) if args else "Any"
        return f"list[{inner}]"

    if str(origin).endswith("Literal"):
        values = ", ".join(repr(v) for v in get_args(annotation))
        return f"Literal[{values}]"

    if origin in {tuple, dict}:
        args = ", ".join(_format_type(arg) for arg in get_args(annotation))
        return f"{origin.__name__}[{args}]"

    if str(origin).endswith("UnionType") or str(origin).endswith("Union"):
        args = " | ".join(_format_type(arg) for arg in get_args(annotation))
        return args

    args = ", ".join(_format_type(arg) for arg in get_args(annotation))
    name = getattr(origin, "__name__", str(origin))
    return f"{name}[{args}]" if args else name


def _markdown_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    if not rows:
        return "_No rows._"

    head = "| " + " | ".join(headers) + " |"
    divider = "|" + "|".join(["---"] * len(headers)) + "|"
    body = []
    for row in rows:
        rendered = []
        for header in headers:
            value = row.get(header, "")
            value = value.replace("\n", " ").replace("|", "\\|")
            rendered.append(value)
        body.append("| " + " | ".join(rendered) + " |")
    return "\n".join([head, divider, *body])


def build_example() -> LLMNoteFeatures:
    return LLMNoteFeatures(
        admission_reason_tags=["cardiac_hf", "renal_aki"],
        dominant_admission_reason="cardiac_hf",
        primary_diagnosis_text="Acute decompensated heart failure with cardiorenal syndrome",
        shock_present="no",
        infection_as_trigger="not_documented",
        aki_present="yes",
        functional_status="assisted",
        mental_status="intact",
        discharge_condition_category="improved",
        lives_alone="yes",
        social_support_absent="no",
        financial_hardship="not_documented",
        substance_use_active="no",
        fall_risk_documented="yes",
        cognitive_impairment="no",
        goals_of_care_flag="yes",
        palliative_care_consult="no",
        dnr_dni_documented="yes",
        new_meds_started_count=3,
        meds_stopped_count=1,
        home_health_ordered="yes",
        cardiac_rehab_referred="yes",
        discharge_delayed_reason="no",
        hospital_acquired_complication="no",
        unresolved_diagnosis_at_discharge="no",
        reasoning=(
            "Discharge summary attributes admission to volume-overload HF with concurrent AKI; "
            "clinical status improved after diuresis."
        ),
    )


def main() -> int:
    model = build_example()
    schema = LLMNoteFeatures.model_json_schema()
    schema_text = json.dumps(schema, indent=2)
    example_text = model.model_dump_json(indent=2)

    print(schema_text)
    print(example_text)

    field_rows: list[dict[str, str]] = []
    for field_name, field_info in LLMNoteFeatures.model_fields.items():
        field_rows.append(
            {
                "field": field_name,
                "type": _format_type(field_info.annotation),
                "required": "required" if field_info.is_required() else "optional",
                "description": field_info.description or "",
            }
        )

    vocab_rows = [
        {"tag": tag, "description": ADMISSION_REASON_TAG_DESCRIPTIONS[tag]}
        for tag in ADMISSION_REASON_TAGS
    ]

    output_path = Path("codex_outputs/03_schema_render.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# Schema Render",
                "",
                "## JSON Schema",
                "```json",
                schema_text,
                "```",
                "",
                "## Example serialized instance",
                "```json",
                example_text,
                "```",
                "",
                "## Field inventory",
                _markdown_table(field_rows, ["field", "type", "required", "description"]),
                "",
                "## Vocabulary listing",
                _markdown_table(vocab_rows, ["tag", "description"]),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote schema render to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
