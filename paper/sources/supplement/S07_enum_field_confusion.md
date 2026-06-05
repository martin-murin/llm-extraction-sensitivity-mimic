# Enum Field Confusion

This section reports the cross-variant confusion matrices on the three enumerated-value-set fields described in Section~\ref{sec-results-enum}: mental status, functional status at discharge, and discharge condition category. Table~\ref{tbl:enum-fields} lists the value sets and per-value definitions for each enum field, and the two count fields included in the schema for completeness. Figure~\ref{fig:enum-mental-status}, Figure~\ref{fig:enum-functional-status}, and Figure~\ref{fig:enum-discharge-condition} report the variant-pair confusion structure on the pooled $n = {{cross_variant_pooled_n}}$-note cross-variant sample for each enum field. Functional status at discharge is the most consistent enum field across variants; discharge condition category shows the largest residual variant-pair differences.

## Mental Status Confusion

![**Cross-variant confusion matrices on mental status.** Three sub-matrices for variant pairs A-B, A-C, and B-C. Diagonal mass is {{enum_mental_status_ab_diagonal_pct:.1f}}\% (A-B), {{enum_mental_status_ac_diagonal_pct:.1f}}\% (A-C), and {{enum_mental_status_bc_diagonal_pct:.1f}}\% (B-C). Notable off-diagonal mass between `mild_impairment` and `confused_delirious`, and between `not_documented` and `intact`. Pooled cross-variant sample, $n = {{cross_variant_pooled_n}}$.](../figures/supplement/paper_fig_S07a_mental_status_confusion.pdf){#fig:enum-mental-status width=\linewidth}

## Functional Status Confusion

![**Cross-variant confusion matrices on functional status at discharge.** Three sub-matrices for variant pairs A-B, A-C, and B-C. Diagonal mass is {{enum_functional_status_ab_diagonal_pct:.1f}}\% (A-B), {{enum_functional_status_ac_diagonal_pct:.1f}}\% (A-C), and {{enum_functional_status_bc_diagonal_pct:.1f}}\% (B-C). The boundary between `dependent` and `assisted` is the only off-diagonal cluster at appreciable scale. Pooled cross-variant sample, $n = {{cross_variant_pooled_n}}$.](../figures/supplement/paper_fig_S07b_functional_status_confusion.pdf){#fig:enum-functional-status width=\linewidth}

## Discharge Condition Confusion

![**Cross-variant confusion matrices on discharge condition category.** Three sub-matrices for variant pairs A-B, A-C, and B-C. Diagonal mass is {{enum_discharge_condition_ab_diagonal_pct:.1f}}\% (A-B), {{enum_discharge_condition_ac_diagonal_pct:.1f}}\% (A-C), and {{enum_discharge_condition_bc_diagonal_pct:.1f}}\% (B-C). Off-diagonal mass concentrates between `unchanged` and the more clinically specific categories. Pooled cross-variant sample, $n = {{cross_variant_pooled_n}}$.](../figures/supplement/paper_fig_S07c_discharge_condition_confusion.pdf){#fig:enum-discharge-condition width=\linewidth}
