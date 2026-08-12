# Shared policy: metric interpretation

- Interpret all metrics supplied by the runtime. Do not compute metric values or silently substitute missing values.
- AUPRC is prevalence-sensitive ranking evidence; MCC and balanced accuracy summarize thresholded class balance; recall and precision expose screening trade-offs; AUROC is complementary discrimination evidence.
- Calibration metrics such as Brier score and ECE are lower-is-better and require probabilistic outputs. Threshold-dependent metrics must use a validation-selected threshold frozen before testing.
- Treat ACC, specificity, F1, NPV and calibration as reportable dimensions even when they receive little ranking weight.
- Discuss coverage, separation, cross-dataset consistency, consensus and redundancy. Do not double-count algebraically or rank-equivalent metrics.
- Numeric weights, bounds, normalization, eligible metric keys and final scores are enforced and calculated by code.
