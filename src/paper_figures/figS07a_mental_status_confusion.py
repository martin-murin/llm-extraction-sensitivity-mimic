from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS07a_mental_status_confusion`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Supplement Figure S7.
# Usage: `python -m src.paper_figures.figS07a_mental_status_confusion` or `python scripts/build_paper_figures.py`.

from matplotlib.figure import Figure

from src.paper_figures._s07_enum_pairwise_common import (
    EnumPairFigConfig,
    build_enum_pairwise_figure,
)


CFG = EnumPairFigConfig(
    figure_name="paper_fig_S07a_mental_status_confusion",
    field="mental_status",
    pretty_name="mental_status",
    labels=("intact", "mild_impairment", "confused_delirious", "not_documented"),
)


def build() -> Figure:
    return build_enum_pairwise_figure(CFG)


if __name__ == "__main__":
    build()
