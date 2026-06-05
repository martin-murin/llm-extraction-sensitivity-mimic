from __future__ import annotations

# Release documentation:
# Provides shared helpers/configuration for publication figure modules.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Supports publication figure generation.

from pathlib import Path

import matplotlib
# Force a non-interactive backend so figure builds work over SSH/headless shells.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from src.paper_figures import config

_FIELD_LABEL_OVERRIDES: dict[str, str] = {
    "aki_present": "AKI present",
    "lives_alone": "Lives alone",
    "social_support_absent": "Social support absent",
    "financial_hardship": "Financial hardship",
    "substance_use_active": "Substance use active",
    "fall_risk_documented": "Fall risk documented",
    "cognitive_impairment": "Cognitive impairment",
    "goals_of_care_flag": "Goals-of-care flag",
    "palliative_care_consult": "Palliative-care consult",
    "dnr_dni_documented": "DNR/DNI documented",
    "home_health_ordered": "Home health ordered",
    "cardiac_rehab_referred": "Cardiac rehab referred",
    "infection_as_trigger": "Infection as trigger",
    "hospital_acquired_complication": "Hospital-acquired complication",
    "unresolved_diagnosis_at_discharge": "Unresolved diagnosis at discharge",
    "shock_present": "Shock present",
    "discharge_delayed_reason": "Discharge delayed (non-medical)",
}


def apply_paper_style() -> None:
    """Apply global rcParams from config.py."""
    plt.rcParams.update(
        {
            "figure.dpi": config.DPI,
            "savefig.dpi": config.DPI,
            "font.family": config.FONT_FAMILY,
            "font.size": config.FONT_SIZE_BASE,
            "axes.titlesize": config.FONT_SIZE_TITLE,
            "axes.labelsize": config.FONT_SIZE_AXIS_LABEL,
            "xtick.labelsize": config.FONT_SIZE_TICK_LABEL,
            "ytick.labelsize": config.FONT_SIZE_TICK_LABEL,
            "legend.fontsize": config.FONT_SIZE_LEGEND,
            "axes.grid": True,
            "grid.alpha": config.GRID_ALPHA,
            "grid.linewidth": config.LINE_WIDTH,
            "axes.linewidth": config.LINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson confidence interval for a proportion."""
    if total <= 0:
        return (0.0, 0.0)
    # Current figure suite uses 95% intervals.
    _ = confidence
    z = 1.96
    phat = successes / total
    denom = 1 + (z * z) / total
    center = (phat + (z * z) / (2 * total)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) / total) + (z * z) / (4 * total * total))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def format_kappa_axis(ax: plt.Axes, ymin: float = -0.4, ymax: float = 1.0) -> None:
    """Standard axis formatting for kappa plots."""
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(np.ceil(ymin * 10) / 10, ymax + 0.001, 0.2))
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.7)


def add_significance_stars(ax: plt.Axes, x1: float, x2: float, y: float, p: float) -> None:
    """Add a simple significance bracket with stars based on p-value."""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "ns"

    ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], color="black", linewidth=0.8)
    ax.text(
        (x1 + x2) / 2,
        y + 0.012,
        stars,
        ha="center",
        va="bottom",
        fontsize=config.FONT_SIZE_ANNOTATION,
    )


def truncate_field_name(name: str, max_chars: int = 22) -> str:
    """Truncate long field names for readable axis labels."""
    if len(name) <= max_chars:
        return name
    return name[: max_chars - 1] + "…"


def humanize_field_id(field: str) -> str:
    """Convert schema field identifiers to consistent human-readable labels."""
    if field in _FIELD_LABEL_OVERRIDES:
        return _FIELD_LABEL_OVERRIDES[field]
    return field.replace("_", " ")


def save_figure(
    fig: plt.Figure,
    name: str,
    formats: list[str] | None = None,
    supplement: bool = False,
) -> list[Path]:
    """Save figure in configured formats and return output paths."""
    out_formats = formats or list(config.FORMATS)
    out_dir = config.SUPPLEMENT_DIR if supplement else config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for ext in out_formats:
        out_path = out_dir / f"{name}.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        outputs.append(out_path)
    return outputs
