# Benchmark Evidence Agent

Review the project-local 15-model benchmark evidence across 3 datasets
(C_AMPs-predict_test, Veltri_test, ProteoGPT_all_predictions) under a unified
15-field protocol. Confirm the single-model ranking (C_AMPs-predict > HMD-AMP
> AMPsorter) and explain score IQR, Top3 frequency and dataset-composition
sensitivity. Report per-dataset AUPRC, MCC, Recall and Precision for the top-3
models. All dataset gates (independence, homology, training-overlap) are closed.
State which performance claims are strongly supported versus which carry
residual calibration or prevalence sensitivity.
