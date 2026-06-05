# Methods {#sec-methods}

## Cohort and splits {#sec-cohort-splits}

The study draws on MIMIC-IV v3.1 discharge summary notes \cite{johnson2023mimiciv,johnson2024mimiciv,johnson2023mimicnote,goldberger2000physionet} ({{production_n_admissions}} unique admissions). Each admission has exactly one discharge summary in the source data.

To enable a controlled methodology study while preserving generalization tests, disjoint splits of admissions were constructed. Notes were stratified by ICD-10 chapter so that clinical content was distributed comparably across splits. Random sampling within strata used a fixed seed. Each split is defined by a manifest of admission identifiers committed to the project repository, and each manifest carries a SHA-256 hash for integrity verification.

The following splits were used:

- a 200-note smoke set, used for an initial round of prompt revision before three-variant extraction was launched;
- a {{refinement_150_n}}-note refinement set, used for the initial three-variant extraction and the optimization loop;
- a {{holdout_150_n}}-note holdout set, used for single-touch generalization evaluation after the optimization loop completed (Section \ref{sec-optimization-loop});
- a {{methodology_1000_n}}-note validation set, used for cross-prompt agreement evaluation at intermediate scale and for selecting the production variant (Section \ref{sec-variant-selection-production});
- a {{extended_5k_n}}-note extended validation set, used to confirm that cross-prompt agreement is stable as the sample grows beyond the 1,000-note scale;
- a 500-note audit subset, drawn from the extended validation set, used for detailed disagreement audit;
- a {{methodology_1500_n}}-note paired sample, formed by the union of the 1,000-note validation set and the 500-note audit subset, used for the same-note model-size comparison (Section \ref{sec-model-size-comparison});
- the population extraction over the remaining admissions, used for downstream applications outside the scope of this work (Section \ref{sec-variant-selection-production}).

The smoke, refinement, and holdout sets were constructed together in a single stratified split build. Split-construction scripts for the larger samples exclude all earlier admission identifiers, and overlap between splits is checked at construction time. Split sizes and purposes are summarized in Table \ref{tbl:splits}.

The holdout set was treated as single-touch: it was used exactly once, to evaluate the cross-prompt agreement of the prompt set produced after the optimization loop (Section \ref{sec-optimization-loop}). No prompt edits, schema changes, or hyperparameter choices were made after consulting holdout results.

## Schema design {#sec-schema-design}

The extraction schema defines four feature categories: an admission-reason vocabulary, a set of TriState clinical flags, a set of enumerated single-choice fields, and a set of count and free-text fields.

The first category is the admission-reason vocabulary, a controlled set of 47 tags organized by organ system and clinical pattern. Each note requires two outputs: a non-empty subset of the vocabulary (the admission-reason tag set) and a single dominant tag, which must be a member of the returned set. The vocabulary was designed before three-variant prompt development and was not modified during the methodology study. The full vocabulary with tag definitions is provided in Table \ref{tbl:admission-tags}.

The second category is the TriState clinical flags: 17 fields representing the documentation status of specific clinical events, social determinants, and discharge planning items (Table \ref{tbl:tristate-fields}). Each field takes one of three values. The value is `yes` when the feature is affirmatively documented in the note, `no` when an explicit negation is present, and `not_documented` when the note is silent on the feature. This three-way distinction was adopted so that downstream consumers can separate active absence, in which the feature was considered and rejected, from passive absence, in which the note carries no information about the feature. The clinical motivation is the documentation-quality use case, where the presence or absence of an explicit negation can carry meaning beyond the presence or absence of a positive statement.

The third category is the enum fields: discrete-valued single-choice fields with closed value sets, covering mental status, functional status at discharge, discharge condition category, and others (Table \ref{tbl:enum-fields}).

The fourth category comprises integer count fields, such as the number of specialty consults, and a free-text evidence field populated by the model with quoted or paraphrased note content supporting the structured output. Count fields are part of the production extraction but were not analyzed in this work. The evidence field is retained for audit and qualitative review but is not consumed by any downstream metric here.

Every extracted record was validated against a Pydantic schema. The validation enforces that types match the schema, that all enum and tag values lie in their declared value sets, that the admission-reason tag set is non-empty, and that the dominant admission reason is a member of the returned tag set. Records failing validation were marked as parse failures and excluded from downstream aggregation.

## The three controlled comparisons {#sec-comparisons}

The methodology holds the extraction task and the source notes fixed and varies three configuration choices in turn: the prompt phrasing, the model size, and the granularity of the categorical schema. Each comparison is described below. The agreement metrics applied across all three are defined in Section \ref{sec-cross-prompt-metrics}.

### Prompt phrasing: three variants {#sec-three-prompt-variants}

Three prompt variants were developed to elicit the same structured output under different phrasings. Variant A is a detailed prose prompt, in which the task is described in continuous text with the schema appended as JSON-style examples. Variant B presents the task in an evidence-first structure, instructing the model to identify and quote relevant note passages for each schema field and then to produce the structured output keyed to those quotes. Variant C uses a questionnaire format, in which each schema field is presented as a separate narrow question with explicit options and the model answers field by field.

The variants were drafted and evaluated on the 200-note smoke set. A single hand-edit was applied across all three during smoke testing to address a vocabulary coverage gap, and no further variant-level changes were made before three-variant extraction was launched on the refinement set. The full prompt text for each variant is provided in supplementary material.

### Model size: paired re-extraction {#sec-model-size-comparison}

To compare the effect of model size on cross-prompt agreement against sample-draw variation, a paired re-extraction was performed using the full model. The same {{methodology_1500_n}} admissions covered by the 1,000-note validation set and the 500-note audit subset were re-extracted with all three prompt variants at the full model. Both model sizes are snapshots from a single vendor's product range: the small model is `gpt-5.4-nano-2026-03-17` and the full model is `gpt-5.4-2026-03-05`.

This design enables a same-note paired comparison: for every note in the paired sample, the small-model and full-model outputs are available across all three variants. Because the comparison is made on the same notes, any same-note difference reflects the change in model operating point rather than a difference in sample composition. The two model sizes are treated as two operating points to be compared, with neither taken as a reference standard for the other.

### Schema granularity: binary collapse {#sec-binary-collapse}

A post-hoc re-analysis evaluated the dependence of cross-prompt agreement on the granularity of the TriState schema. Under this re-analysis, each TriState value was re-mapped: `yes` was preserved as `yes`, and `no` and `not_documented` were merged into a single class labeled `not_yes`. The collapse is a re-labeling of the existing extractions and not a re-extraction; the prompts are unchanged and the model is not asked to categorize under a binary schema.

All cross-prompt agreement metrics (kappa, percent agreement, disagreement decomposition, per-pair confusion analysis) were recomputed on the re-mapped values. The binary-collapse analysis is reported alongside the full-TriState analysis throughout Section \ref{sec-results}. Whether cross-prompt agreement changes substantially under binary collapse, and how that change is distributed across fields, is a primary empirical finding of this work and is addressed in Section \ref{sec-results-collapse-structure}.

## Cross-prompt agreement metrics {#sec-cross-prompt-metrics}

Pairwise Cohen's kappa across prompt variants is a categorical-agreement metric in the inter-rater-reliability tradition \cite{mchugh2012kappa,uzuner2011i2b2}. Chance correction is the relevant property here: most TriState fields have skewed marginals, with `not_documented` dominating, so raw percent agreement would register as high on these fields for reasons of base rate alone, whereas kappa discounts the agreement expected from the marginals and isolates agreement on the categorization itself. It serves the same diagnostic role as the entropy-based prompt-sensitivity metrics formalized for general-domain classification \cite{errica2025quantifying,razavi2025benchmarking}, with the practical difference that it is interpretable at field-level resolution in a multi-field structured-extraction setting. The choice of pairwise kappa rather than per-input prediction entropy follows from the design of the present study. Entropy-based prompt-sensitivity metrics estimate a distribution over outputs and therefore need a large ensemble of rephrasings to be meaningful; this study uses three deliberately distinct prompt phrasings, too few to estimate an output distribution but well suited to pairwise agreement, which is defined for as few as two raters.

Cross-prompt agreement was measured as the pairwise Cohen's kappa between variant outputs on the same note, computed independently per schema field and per variant pair. For TriState fields, kappa was computed treating the three values (`yes`, `no`, `not_documented`) as categorical labels with no implied ordering. The three values are not an ordered scale: `not_documented` is an evidence-availability state rather than a midpoint between `yes` and `no`, so an ordinal treatment would impose a ranking the schema does not carry. For enum fields, kappa was computed over the declared value set. For the dominant admission reason, kappa was computed over the 47-tag vocabulary.

Two multi-class admission-reason quantities are reported. Cross-variant agreement on the single-choice primary admission reason is the exact-match rate on the dominant tag, equivalent to the diagonal mass of a $47\times47$ confusion matrix with one label per note. Agreement on the multi-label admission-reason tag set, where a single-label confusion matrix does not apply, is summarized by admission-tag set agreement, the mean per-note Jaccard index. For note $i$ with variant tag sets $A_i$ and $B_i$, per-note agreement is $|A_i \cap B_i| / |A_i \cup B_i|$, defined as 1 when both sets are empty, and the reported set agreement is the mean of this quantity across notes. A value of one indicates identical tag sets on every note; lower values indicate divergence in which tags were assigned. Because each prompt variant is instructed to include the primary admission reason in the tag set, set agreement is bounded below by dominant-tag exact-match agreement and therefore primarily reflects consistency on the secondary, non-dominant tags. This set agreement is a descriptive measure and is not chance-corrected; it is reported as a percentage and is not placed on the same scale as the chance-corrected Cohen's kappa used for the TriState and enum fields.

For each TriState field, the per-field kappa values are reported alongside a filtered-median summary across those fields. The filter excludes fields with low positive-class base rate, defined as fewer than 10 total positive votes across all variants on the sample at hand. In the model-size comparison the filter is applied jointly to both model sizes, so a field is included only if it passes the threshold at the small model and at the full model. This filter is applied because Cohen's kappa is sensitive to the marginal distribution of the categorical labels. When the positive class is very rare, the point estimate is dominated by base-rate variance and is no longer a reliable measure of agreement structure.

Several kappa-related quantities are referenced in subsequent sections, and the following notation is used throughout. The quantity $\kappa_{p,f}$ denotes Cohen's kappa for variant pair $p \in {\text{A-B}, \text{A-C}, \text{B-C}}$ on field $f$. The atomic unit of measurement is therefore a single $\kappa$ value for one (variant-pair, field) cell. With 17 TriState fields and three variant pairs there are up to 51 such cells per model size before base-rate filtering, and 48 cells after filtering on the paired model-size sample (16 fields $\times$ 3 pairs).

The model-size comparison is summarized by a paired per-field statistic. For a single field, the median over the three variant pairs of $\kappa_{p,f}(\text{full}) - \kappa_{p,f}(\text{small})$ gives that field's model-size difference, $\Delta\kappa_f$. Taking the median of $\Delta\kappa_f$ over the filtered field set gives $\overline{\Delta\kappa}^{\,\mathrm{per\text{-}field}}$, the model-size difference experienced by the typical field. Because each $\Delta\kappa_f$ is formed on the same notes before aggregation, this median-of-differences is the paired quantity used for every between-model comparison in this work. Separately, the pooled median $\bar{\kappa}$ is the median taken over all (variant-pair, field) cells at one model size at once, treating those cells as a single flat population; it is reported per model size as a benchmark-anchor level (small and full, under the TriState schema and under collapse), so that the agreement of each configuration can be situated against external work. The unpaired difference of the two pooled medians is not reported as the model-size statistic, because the field at the median of the small-model distribution need not be the field at the median of the full-model distribution, so their difference does not correspond to any field's experience.

## Labeling function ensemble {#sec-lf-ensemble}

Independent labeling functions were developed for a subset of schema targets to provide signal beyond the LLM extraction. Three labeling-function families were used. The complete labeling-function definitions are provided in Table~\ref{tbl:icd-lfs} and Table~\ref{tbl:regex-lfs}.

ICD labeling functions (15 in total) map admission ICD-10 codes to specific schema targets. Each ICD labeling function is associated with one target and emits a positive vote when any code in its target's code list appears in the admission's structured diagnosis codes, and an abstain vote otherwise. ICD labeling functions cover targets where ICD coverage is expected, such as cardiac heart failure, sepsis, AKI, and hepatic failure. The full set of ICD anchors with their code lists is given in Table \ref{tbl:icd-lfs}.

Regex labeling functions (8 in total) match curated regular expressions against the note text. Each regex labeling function emits a positive vote on match and abstains otherwise. The pattern set was bootstrapped from the refinement set and is listed in Table \ref{tbl:regex-lfs}.

LLM labeling functions (96 in total) are derived from the three-variant extraction. They cover 32 target field-value pairs: 14 admission-tag membership targets and 9 TriState fields each split into a `yes` and a `no` target ($14 + 9 \times 2 = 32$). For each target field-value pair the three variant outputs each contribute one labeling function, yielding $3 \times 32 = 96$ LLM-based votes per note.

Labeling-function votes were integrated using a Snorkel LabelModel fit per target \cite{ratner2017snorkel}. The LabelModel infers per-labeling-function accuracies from cross-LF agreement patterns and produces a probabilistic consensus label for each note. The label-model output was used as the aggregated reference signal for two purposes in this work: selecting the production prompt variant (Section \ref{sec-variant-selection-production}), and clustering disagreement patterns for the optimization loop (Section \ref{sec-optimization-loop}).

Embedding-based labeling functions were also explored but produced overlapping signal and noise distributions on the refinement set and were not included in the final ensemble.

## Autonomous optimization loop {#sec-optimization-loop}

A concern during prompt development was that the three variants might produce systematically different outputs on subsets of targets, in the sense that one variant might consistently disagree with the consensus of the others in a way attributable to its specific phrasing rather than to the content of the note. To detect and address such systematic divergence, an automated optimization procedure was developed. The procedure is described in its general form below; in practice it was applied only to variant C, which on the initial three-variant extraction showed clustered disagreement against the consensus of variants A and B on several targets.

The procedure is:

1. Compute three-variant labeling-function votes on the refinement set and identify clusters of systematic disagreement. A cluster is a target-and-pattern combination where one variant systematically diverges from the other two, for example a case in which variant C asserts `no` on cognitive_impairment where A and B assert `not_documented`.

2. For each cluster, construct a prompt for a full-model rewrite step. The full model is given the current prompt of the diverging variant, a sample of disagreement examples, and instructions to revise the prompt to address the cluster while preserving schema structure.

3. The revised prompt is accepted only if four deterministic guards pass: the revised prompt produces output of the same schema cardinality, with all required fields and no unexpected fields; all enum and tag values remain within the declared value sets; no field is renamed; and no field is removed. Failed rewrites are discarded.

4. The accepted prompt is re-extracted on the refinement set, and the new cluster disagreement rate is measured. The loop continues to the next cluster.

5. Iteration stops when any one of the following holds: a maximum of 5 iterations is reached; the per-iteration improvement in cluster disagreement rate falls below 2 percentage points; no remaining cluster exceeds a residual disagreement volume of 50 cases; or two consecutive iterations produce no improvement.

The loop ran for {{optimization_loop_iterations}} iterations on variant C, reducing the targeted cluster disagreement count from {{optimization_loop_initial_cluster_disagreements}} to {{optimization_loop_final_cluster_disagreements}} (a {{optimization_loop_cluster_reduction_pct}}% reduction). The post-optimization prompts for all three variants were then locked and evaluated on the holdout set (Section \ref{sec-cohort-splits}) as a single-touch generalization test.

## Variant selection and production extraction {#sec-variant-selection-production}

After holdout validation, a single prompt variant was selected for the population-scale extraction. The selection used the 1,000-note validation set. For each variant, the LLM extraction output was compared to the Snorkel-aggregated reference signal across the labeling-function targets. The agreement metric was the mean weighted agreement between the variant's per-target output and the per-target Snorkel posterior, weighted by per-target support. For this metric, TriState `not_documented` was treated as an abstain vote on the corresponding labeling-function target.

The metric is one of several reasonable choices for variant selection. It measures agreement with the consensus label produced by the full labeling-function ensemble, of which the LLM variant is one input, which introduces a degree of self-consistency in the score. The selection nevertheless served the practical purpose of choosing a variant. Cross-prompt agreement metrics are pairwise across all three variants and therefore do not depend on which one is designated as production. Variant A scored highest in the metric and was selected.

All extractions were produced with deterministic decoding: temperature was set to zero and no reasoning mode was enabled, for both model sizes and all three prompt variants. Cross-prompt and cross-model disagreement reported in this work is therefore not attributable to sampling stochasticity in the decoding step; it reflects the model's response to the prompt and schema inputs themselves.

A population-scale extraction using variant A and the small model was subsequently performed on the {{production_n_admissions}} admissions for downstream applications outside the scope of this work.
