# Admission-Tag Full Vocabulary Confusion

This section extends Figure~\ref{fig:admission-confusion} of the main text by reporting the complete 47x47 confusion matrices on the primary admission reason. Table~\ref{tbl:admission-tags} lists the 47 admission-reason tags with their definitions and ICD-10-CM anchor codes (where defined). Figure~\ref{fig:cross-variant-ab}, Figure~\ref{fig:cross-variant-ac}, and Figure~\ref{fig:cross-variant-bc} report the cross-variant confusion structure at the small model for variant pairs A-B, A-C, and B-C respectively, on the pooled $n = {{cross_variant_pooled_n}}$-note cross-variant sample. Figure~\ref{fig:cross-model-a}, Figure~\ref{fig:cross-model-b}, and Figure~\ref{fig:cross-model-c} report the same-prompt cross-model confusion (small-versus-full on the same prompt) for variants A, B, and C respectively, on the {{methodology_1500_n}}-note paired sample. The same-prompt cross-model panels make visible the model-size effect on dominant-tag selection discussed in Section~\ref{sec-results-model-size} and Section~\ref{sec-discussion-model-size-vs-prompt}.

## Cross-Variant Confusion: A-B

![**Full-vocabulary cross-variant confusion matrix on the primary admission reason for variant pair A-B, 47x47.** This extends Figure~\ref{fig:admission-confusion} to the complete 47-tag vocabulary. Color encodes row-normalized confusion rate on log scale. Diagonal cells are the per-tag agreement rate within the pair; off-diagonal mass indicates phrasing-dependent categorization differences. Diagonal mass is {{ab_dominant_diagonal_mass_pct:.1f}}\%. Pooled cross-variant sample, $n = {{cross_variant_pooled_n}}$.](../figures/supplement/S3_admission-tag-47_cross-variant_AB.pdf){#fig:cross-variant-ab width=\textwidth}

## Cross-Variant Confusion: A-C

![**Full-vocabulary cross-variant confusion matrix on the primary admission reason for variant pair A-C, 47x47.** This extends Figure~\ref{fig:admission-confusion} to the complete 47-tag vocabulary. Color encodes row-normalized confusion rate on log scale. Diagonal cells are the per-tag agreement rate within the pair; off-diagonal mass indicates phrasing-dependent categorization differences. Diagonal mass is {{ac_dominant_diagonal_mass_pct:.1f}}\%. Pooled cross-variant sample, $n = {{cross_variant_pooled_n}}$.](../figures/supplement/S3_admission-tag-47_cross-variant_AC.pdf){#fig:cross-variant-ac width=\textwidth}

## Cross-Variant Confusion: B-C

![**Full-vocabulary cross-variant confusion matrix on the primary admission reason for variant pair B-C, 47x47.** This extends Figure~\ref{fig:admission-confusion} to the complete 47-tag vocabulary. Color encodes row-normalized confusion rate on log scale. Diagonal cells are the per-tag agreement rate within the pair; off-diagonal mass indicates phrasing-dependent categorization differences. Diagonal mass is {{bc_dominant_diagonal_mass_pct:.1f}}\%. Pooled cross-variant sample, $n = {{cross_variant_pooled_n}}$.](../figures/supplement/S3_admission-tag-47_cross-variant_BC.pdf){#fig:cross-variant-bc width=\textwidth}

## Cross-Model Confusion: Variant A

![**Full-vocabulary same-prompt cross-model confusion matrix on the primary admission reason for variant A, 47x47.** The matrix compares the small-model and full-model extractions on the same notes under the same prompt. Color encodes row-normalized confusion rate on log scale. Diagonal mass is {{aa_model_size_dominant_diagonal_mass_pct:.1f}}\%, quantifying same-prompt agreement across model sizes for variant A. Paired sample, $n = {{methodology_1500_n}}$.](../figures/supplement/S3_admission-tag-47_model-size_AA.pdf){#fig:cross-model-a width=\textwidth}

## Cross-Model Confusion: Variant B

![**Full-vocabulary same-prompt cross-model confusion matrix on the primary admission reason for variant B, 47x47.** The matrix compares the small-model and full-model extractions on the same notes under the same prompt. Color encodes row-normalized confusion rate on log scale. Diagonal mass is {{bb_model_size_dominant_diagonal_mass_pct:.1f}}\%, quantifying same-prompt agreement across model sizes for variant B. Paired sample, $n = {{methodology_1500_n}}$.](../figures/supplement/S3_admission-tag-47_model-size_BB.pdf){#fig:cross-model-b width=\textwidth}

## Cross-Model Confusion: Variant C

![**Full-vocabulary same-prompt cross-model confusion matrix on the primary admission reason for variant C, 47x47.** The matrix compares the small-model and full-model extractions on the same notes under the same prompt. Color encodes row-normalized confusion rate on log scale. Diagonal mass is {{cc_model_size_dominant_diagonal_mass_pct:.1f}}\%, quantifying same-prompt agreement across model sizes for variant C. Paired sample, $n = {{methodology_1500_n}}$.](../figures/supplement/S3_admission-tag-47_model-size_CC.pdf){#fig:cross-model-c width=\textwidth}

