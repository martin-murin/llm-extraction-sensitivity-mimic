# Paper Figures Package

Persistent publication-figure code for the methodology paper.

## Structure
- `config.py`: global typography, colors, output directories, DPI.
- `data_loaders.py`: cached shared loaders (`functools.lru_cache`) for all figure data inputs.
- `plot_utils.py`: shared style/axis/save helpers.
- `fig*.py`: one module per figure with clear top-level `CFG` knobs.

## Build
- List registry:
```bash
python scripts/build_paper_figures.py --list
```
- Build all:
```bash
python scripts/build_paper_figures.py
```
- Build one:
```bash
python scripts/build_paper_figures.py --fig 01
```
- Build supplement only:
```bash
python scripts/build_paper_figures.py --supplement
```

## Add a new figure
1. Copy an existing `fig*.py` module.
2. Keep the structure: `CFG` dataclass -> `prepare_data()` -> `render()` -> `build()`.
3. Register the module key in `scripts/build_paper_figures.py` `FIGURES`.
4. Use `save_figure(fig, "paper_fig_XX_name", supplement=...)`.

## Iterate on cosmetics
1. Open the specific `fig*.py`.
2. Edit only the `CFG` block (sizes, colors, legend placement, sort mode).
3. Re-run standalone:
```bash
python -m src.paper_figures.fig01_per_field_full_vs_collapsed
```
4. Inspect `docs/figures/paper/*.png` and `*.pdf`.

## Main-paper numbering (current)
- `01`: per-field full-vs-collapsed deltas
- `02`: cross-prompt agreement across model size/reasoning
- `03`: pairwise kappa by field (TriState vs collapsed)
- `04`: tag prevalence (three panels)
- `05a` / `05b`: dominant admission confusion (matrix / triangulated)

## Supplement additions
- `S01a`: LLM vs ICD empirical conditional rates
- `S01b`: LLM vs regex empirical conditional rates
- `S01c`: AKI five-signal concordance (LLM A/B/C + ICD + regex; includes Cohen's kappa)
- `S05`: refinement-to-holdout firewall generalization
- `S06`: sample-size stability forest plot across five samples
- `S04`: disagreement decomposition (dual panel, category breakdown)

## Data caching behavior
`data_loaders.py` uses `@lru_cache(maxsize=None)`. In one Python process, each data source is read once and reused across figure modules.

## Known limitations
- `fig05b` triangulated 30x30 tiles is information-dense; readability can degrade in small print.
- Some context metrics (especially reasoning-ON comparisons) are computed on subset-overlap samples and explicitly annotated with sample sizes in the figure/subtitle.
