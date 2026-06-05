"""
Provides shared helpers/configuration for publication figure modules.

Reads: MIMIC/Postgres, generated local artifacts, or in-memory inputs as configured.
Writes: docs/figures/paper, docs/figures/paper/supplement.
Supports publication figure generation.
"""

from pathlib import Path

# Output settings
DPI = 320
OUTPUT_DIR = Path("docs/figures/paper")
SUPPLEMENT_DIR = Path("docs/figures/paper/supplement")
FORMATS = ["png", "pdf"]

# Typography
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE_BASE = 9
FONT_SIZE_TITLE = 10
FONT_SIZE_AXIS_LABEL = 9
FONT_SIZE_TICK_LABEL = 8
FONT_SIZE_ANNOTATION = 7
FONT_SIZE_LEGEND = 8

# Page sizes (inches)
SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.2
FULL_PAGE_WIDTH = 7.2

# =============================================================================
# UNIFIED PAPER COLOR SYSTEM
# =============================================================================
# Each category gets a distinct palette. Use these semantic colors consistently.

# Category 1: Schema interpretation (full TriState vs collapsed)
COLOR_SCHEMA_TRISTATE = "#FF9966"
COLOR_SCHEMA_COLLAPSED = "#1B9E77"

# Category 2: Model size (small vs full)
COLOR_MODEL_SMALL = "#ffc064"
COLOR_MODEL_FULL = "#ff6353"

# Category 3: Prompt variant identity (A / B / C)
COLOR_VARIANT_A = "#0173B2"
COLOR_VARIANT_B = "#DE8F05"
COLOR_VARIANT_C = "#029E73"

# Variant-pair identity (derived colors)
COLOR_PAIR_AB = "#7AA6C0"
COLOR_PAIR_AC = "#7BCBA2"
COLOR_PAIR_BC = "#E5B66B"

# Category 4: Signal source (LLM / ICD LF / regex LF / Snorkel)
COLOR_SIGNAL_LLM = "#0173B2"
COLOR_SIGNAL_ICD_LF = "#A45EE5"
COLOR_SIGNAL_REGEX_LF = "#CCB22B"
COLOR_SIGNAL_SNORKEL = "#D55E00"

# Category 5: Sample identity progression
COLOR_SAMPLE_REFINEMENT = "#7FCDBB"
COLOR_SAMPLE_HOLDOUT = "#41B6C4"
COLOR_SAMPLE_METH_1K = "#1D91C0"
COLOR_SAMPLE_METH_5K_AUDIT = "#225EA8"
COLOR_SAMPLE_EXTENDED = "#0C2C84"
COLOR_SAMPLE_PRODUCTION = "#000000"

# Category 6: Disagreement composition
COLOR_DISAGREE_AGREEMENT = "#2ca02c"
COLOR_DISAGREE_SOFT_YES_VS_NOT = "#FFB000"
COLOR_DISAGREE_SOFT_NO_VS_NOT = "#FFD700"
COLOR_DISAGREE_HARD_YES_VS_NO = "#CC0000"
COLOR_DISAGREE_RESIDUAL_COLLAPSED = "#FF6B35"

# Quantitative colormaps
CMAP_AGREEMENT_HEATMAP = "Blues"
CMAP_DIVERGING_DELTA = "RdBu_r"

# Backward-compatibility aliases for older figure modules.
CMAP_SEQUENTIAL = CMAP_AGREEMENT_HEATMAP
CMAP_DIVERGING = CMAP_DIVERGING_DELTA
COLOR_FULL_TRISTATE = COLOR_SCHEMA_TRISTATE
COLOR_COLLAPSED = COLOR_SCHEMA_COLLAPSED
COLOR_AGREEMENT = COLOR_DISAGREE_AGREEMENT
COLOR_SOFT_DISAGREEMENT = COLOR_DISAGREE_SOFT_YES_VS_NOT
COLOR_HARD_DISAGREEMENT = COLOR_DISAGREE_HARD_YES_VS_NO
COLOR_SEM_TRISTATE = COLOR_SCHEMA_TRISTATE
COLOR_SEM_COLLAPSED = COLOR_SCHEMA_COLLAPSED
COLOR_SEM_MODEL_NANO = COLOR_MODEL_SMALL
COLOR_SEM_MODEL_FULL = COLOR_MODEL_FULL

# Reasoning
COLOR_REASONING_OFF = "#666666"
COLOR_REASONING_ON = COLOR_SIGNAL_LLM
COLOR_GOLD_FULL_MODEL = COLOR_MODEL_FULL

# Misc
LINE_WIDTH = 0.8
GRID_ALPHA = 0.3
ERROR_BAR_CAPSIZE = 2
