# MIMIC Notes LLM Extraction Code Release

This repository contains the analysis code for the arXiv paper **Measuring the
sensitivity of LLM-based structured extraction from MIMIC-IV discharge summaries**,
`arXiv:2606.05970`.

This is a code-only GitHub repository prepared to satisfy the MIMIC/PhysioNet
Data Use Agreement requirement to contribute the code used to produce published
results to a repository open to the research community.

## Data Are Not Included

MIMIC-IV v3.1 and MIMIC-IV-Note v2.2 are **not included**. Credentialed users must
obtain those datasets from PhysioNet under the MIMIC Data Use Agreement and load
them into their own local environment. This repository also excludes all
data-derived artifacts: raw LLM responses, split CSVs, feature parquets, cached
embeddings, logs, generated figures, generated reports, and manuscript build
outputs.

## What This Repository Is and Is Not

This repository is the actual code layer behind the paper: extraction schemas and
prompts, database access helpers, split-building logic, extraction runners,
labeling-function and Snorkel analyses, agreement/kappa computations, figure code,
and claim recomputation scripts.

It is not a turnkey reproduction package. The data are restricted, the original
workflow was iterative, and credentialed users must regenerate local `data/*` and
`codex_outputs/*` artifacts themselves.

## Start Here

- `SCHEMA_ASSUMPTIONS.md`: exact database tables, columns, and PostgreSQL
  assumptions the code expects.
- `PIPELINE.md`: honest ordered map of the iterative project pipeline and how
  restricted local artifacts feed later analyses.
- `CLAIMS.md`: claim-to-script map and dependency audit for the paper
  figures and numeric claims.

Minimal configuration placeholders:

```bash
MIMIC_PG_URI=postgresql+psycopg://<USER>:<PASSWORD>@<HOST>:<PORT>/<DATABASE>
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
OPENAI_MODEL=<MODEL_SNAPSHOT>
```

## Main Code Areas

- `src/schema/` and `src/schema/prompts/`: extraction schema and prompt variants.
- `src/db/`: MIMIC database discovery and query helpers.
- `scripts/`: staged pipeline scripts used during split creation, extraction,
  QA, agreement analysis, paired/model-size/reasoning analysis, and figure builds.
- `src/labeling_functions/` and `src/snorkel_fit/`: weak-supervision and
  triangulation code.
- `src/paper_figures/`: publication figure builders.
- `paper/claims/`: numeric claim recomputation and receipt verification.
- `paper/sources/`: manuscript and supplement markdown sources with claim
  placeholders.

## Citations

Users should cite the official PhysioNet records for the exact versions accessed,
including:

- MIMIC-IV
- MIMIC-IV v3.1
- MIMIC-IV-Note v2.2
- PhysioNet

## License

Code license: MIT License. See `LICENSE`. This license applies only to the code
in this repository. It does not grant rights to MIMIC data or to any
data-derived artifact.
