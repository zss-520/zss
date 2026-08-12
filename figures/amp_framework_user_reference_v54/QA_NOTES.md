# QA notes

- The user-provided six-panel organization and color language were retained.
- Panel c combines four real artifacts: complete metric-table screenshots, the percentile landscape, cross-dataset rank shifts and the precision/recall operating-point analysis.
- Panel d contains only the real 50-round bubble ranking and the generated Agent report screenshot.
- The exact metric tables retain all 18 evaluated models. The landscape and ranking panels use the existing audited 15-model display cohort; `pepnet_standard`, `amplify_imb` and `amplify_bal` are excluded by the previously defined posthoc rule.
- Because that filter is conditioned on the observed ranking, the resulting Top-3 ordering is display-only and must not be described as an unbiased global Top-3 result.
- Raster screenshots remain raster inside the PDF/SVG wrappers; surrounding labels and the original diagram are preserved at the high-resolution working canvas.
