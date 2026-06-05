from __future__ import annotations

# Release documentation:
# Builds publication figure module `figS03_admission_tag_47_model_size_aa`.
#
# Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
# Writes: local reports/artifacts determined by CLI defaults.
# Backs Supplement Figure S3.
# Usage: `python -m src.paper_figures.figS03_admission_tag_47_model_size_aa` or `python scripts/build_paper_figures.py`.

import matplotlib.pyplot as plt

from src.paper_figures._s03_admission_tag_matrix_common import (
    compute_confusion,
    load_model_size_common_for_variant,
    order_tags_by_support,
    render_single_matrix,
)


def build() -> plt.Figure:
    df = load_model_size_common_for_variant("A")
    tags = order_tags_by_support(df, "small", "full")
    _counts, rates, _diag_mass, _n = compute_confusion(df, left_col="small", right_col="full", tags=tags)
    return render_single_matrix(
        figure_name="S3_admission-tag-47_model-size_AA",
        tags=tags,
        rates=rates,
    )


if __name__ == "__main__":
    build()

