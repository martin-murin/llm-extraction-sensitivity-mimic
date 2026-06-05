# Labeling Function Complementarity

This section describes the independent labeling functions integrated with the LLM extractions (Section~\ref{sec-lf-ensemble}) and reports the per-target concordance between LLM and labeling-function signals (Section~\ref{sec-results-lf-llm}). The ensemble combines two families of labeling functions: ICD-10-CM-based anchors (Table~\ref{tbl:icd-lfs}) and regex-based anchors over the note text (Table~\ref{tbl:regex-lfs}). Figures~\ref{fig:lf-icd-concordance}, \ref{fig:lf-regex-concordance}, and \ref{fig:aki-five-signal} report the cross-tabulation of LLM and ICD signals on 15 ICD-anchored targets, the same on 9 regex-anchored targets, and a detailed five-signal pool analysis on `aki_present` that exposes the coverage asymmetries between billing-code and discharge-narrative content. The asymmetries are clinically interpretable rather than purely noisy (Section~\ref{sec-discussion-implications}).

## LLM vs ICD Concordance

![**Cross-tabulation of LLM-positive and ICD-labeling-function-positive on 15 ICD-anchored targets.** For each target, the figure reports prevalence of LLM-positive across variants A, B, and C alongside the ICD-anchor prevalence and the variant-vs-ICD Cohen's kappa. The relationship between LLM and ICD signals is not uniform across targets: LLM and ICD capture overlapping but distinct content per target.](../figures/supplement/paper_fig_S01a_llm_vs_icd_concordance.pdf){#fig:lf-icd-concordance width=\linewidth}

## LLM vs Regex Concordance

![**Cross-tabulation of LLM-positive and regex-labeling-function-positive on 9 regex-anchored targets.** Regex labeling functions have high specificity but limited coverage compared to LLM extraction; the figure quantifies this gap per target. Regex prevalence on `aki_present` is {{aki_regex_lf_prevalence_pct:.1f}}\%, substantially below the LLM and ICD prevalence on the same target.](../figures/supplement/paper_fig_S01b_llm_vs_regex_concordance.pdf){#fig:lf-regex-concordance width=\linewidth}

## Five-Signal AKI Pool Analysis

![**Joint distribution of five independent signals on `aki_present`.** Three subpanels report the joint distribution of $\{\text{LLM-A}, \text{LLM-B}, \text{LLM-C}, \text{ICD anchor}, \text{regex anchor}\}$ from complementary perspectives: signal-firing patterns, pairwise agreement structure, and the marginal asymmetry between signal sources. The pool contains {{aki_all_signals_negative_count}} notes where no signal fires, {{aki_icd_only_no_llm_count}} notes where ICD fires but no LLM variant does, and {{aki_all_llm_positive_no_icd_count}} notes where all three LLM variants fire but ICD does not. The asymmetry between ICD-only-no-LLM and all-LLM-no-ICD cases is informative about each signal's coverage characteristics. Pooled cross-variant sample, $n = {{cross_variant_pooled_n}}$.](../figures/supplement/paper_fig_S01c_aki_five_signal_concordance.pdf){#fig:aki-five-signal width=\linewidth}
