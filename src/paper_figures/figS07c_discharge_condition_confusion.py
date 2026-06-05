from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS07c_discharge_condition_confusion`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Supplement Figure S7.
# Usage: `python -m src.paper_figures.figS07c_discharge_condition_confusion` or `python scripts/build_paper_figures.py`.

from matplotlib.figure import Figure

from src.paper_figures._s07_enum_pairwise_common import (
    EnumPairFigConfig,
    build_enum_pairwise_figure,
)


CFG = EnumPairFigConfig(
    figure_name="paper_fig_S07c_discharge_condition_confusion",
    field="discharge_condition_category",
    pretty_name="discharge_condition_category",
    labels=("stable", "improved", "unchanged", "deteriorated", "expired", "not_documented"),
)


def build() -> Figure:
    return build_enum_pairwise_figure(CFG)


if __name__ == "__main__":
    build()
