# Figure QA notes

- Core conclusion: AMP model advantages are strongly dataset dependent.
- Archetype: quantitative grid with one hero heatmap and two robustness panels.
- Backend: Python/matplotlib only.
- Final width: 183 mm (double-column); editable SVG/PDF plus 300 dpi PNG and compressed 600 dpi TIFF.
- Included observations: 18 models x 3 datasets x 8 displayed metrics = 432 source rows.
- No model or dataset was excluded.
- Main-figure metrics: AUPRC, AUROC, MCC, Balanced Accuracy, Recall, Precision, Brier score, and ECE.
- Fields not drawn in the main figure: Coverage (constant), Threshold and Source (configuration/provenance), AUPRC-Lift (dataset-prevalence transform of AUPRC), and secondary/redundant threshold metrics. They remain in the unmodified supplementary evaluation tables.
- Brier score and ECE were reversed only for percentile colour direction; raw values are preserved in the source-data CSV.
- Bootstrap iterations in the source evaluation were 0; no confidence intervals or significance annotations were added.
- Sample statistics: {"C_AMPs-predict_test": {"total_rows": 59311, "missing_labels": 0, "n": 59311, "positive": 1038, "negative": 58273, "prevalence": 0.01750096946603497}, "Veltri_test": {"total_rows": 1203, "missing_labels": 0, "n": 1203, "positive": 614, "negative": 589, "prevalence": 0.5103906899418121}, "ProteoGPT_all_predictions": {"total_rows": 1796, "missing_labels": 0, "n": 1796, "positive": 725, "negative": 1071, "prevalence": 0.4036748329621381}}
