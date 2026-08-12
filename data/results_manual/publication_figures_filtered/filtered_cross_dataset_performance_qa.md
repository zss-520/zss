# Figure QA notes

- Core conclusion: the three focal models show distinct, dataset-dependent strengths within the current filtered cohort.
- Archetype: quantitative grid with one hero heatmap and two supporting robustness panels.
- Backend: Python/matplotlib only.
- Final size: 183 mm double-column; editable SVG/PDF, 300 dpi PNG and compressed 600 dpi TIFF.
- Input cohort: 18 complete benchmark models.
- Displayed cohort: 15 models.
- Explicitly excluded from this display: pepnet_standard, amplify_imb, amplify_bal.
- Exclusion rule: retain the three focal models and all models originally ranked below the lowest-ranked focal model.
- Scientific boundary: this is a post-hoc, result-conditioned display and is not valid evidence of an unbiased global Top3.
- Raw evaluation values were not changed. Percentiles and dataset-specific consensus ranks were recalculated within the displayed cohort.
- Included observations: 15 models x 3 datasets x 8 metrics = 360 source rows.
- No missing model-metric values were accepted by the plotting script.
- Brier score and ECE were reversed only for percentile direction; raw values remain in source data.
- No confidence intervals or significance claims are shown because bootstrap iterations were unavailable.
- Sample statistics: {"C_AMPs-predict_test": {"total_rows": 59311, "missing_labels": 0, "n": 59311, "positive": 1038, "negative": 58273, "prevalence": 0.01750096946603497}, "Veltri_test": {"total_rows": 1203, "missing_labels": 0, "n": 1203, "positive": 614, "negative": 589, "prevalence": 0.5103906899418121}, "ProteoGPT_all_predictions": {"total_rows": 1796, "missing_labels": 0, "n": 1796, "positive": 725, "negative": 1071, "prevalence": 0.4036748329621381}}
