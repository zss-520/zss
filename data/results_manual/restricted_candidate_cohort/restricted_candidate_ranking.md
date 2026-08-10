# Prespecified candidate-cohort single-model ranking

> This is a restricted-cohort comparison, not a global Top3 claim across all evaluated models.

- Candidate models: C_AMPs-predict, HMD-AMP, AMPsorter
- Datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, Veltri_test
- Dynamic weight rounds: 50
- Model-specific score bonus: disabled
- Post-hoc deletion based on observed rank: disabled

| Cohort rank | Model | Median score | Mean score | Score IQR | Top3 frequency |
|---:|---|---:|---:|---:|---:|
| 1 | HMD-AMP | 0.639769 | 0.632558 | 0.177236 | 100.0% |
| 2 | C_AMPs-predict | 0.513440 | 0.486902 | 0.200612 | 100.0% |
| 3 | AMPsorter | 0.345497 | 0.380540 | 0.211148 | 100.0% |

## Interpretation boundary

The three rows above are the complete ranking within the prespecified cohort. Models outside this cohort were not reclassified as failures and remain in the complete benchmark.
