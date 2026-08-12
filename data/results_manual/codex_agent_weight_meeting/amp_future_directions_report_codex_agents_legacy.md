# AMP benchmark: actual 50-round local multi-Agent result

> Legacy compact report retained for provenance. The canonical report is `amp_future_directions_report_codex_agents.md`.

## Result in one sentence

Across 50 blinded, Agent-derived metric-weight rounds, **pepnet_standard** ranked first by median weighted rank score; ranking stability is summarized by score IQR and Top-3 frequency rather than a single fixed weight vector.

## Accepted metric-weight consensus

| Metric | Initial Chief weight | Round-50 weight | 50-round median |
|---|---:|---:|---:|
| AUPRC | 0.226809 | 0.208301 | 0.207657 |
| MCC | 0.184663 | 0.167108 | 0.167337 |
| Recall | 0.132240 | 0.133358 | 0.133176 |
| Precision | 0.102500 | 0.102471 | 0.102683 |
| AUROC | 0.064384 | 0.066344 | 0.066466 |
| BalancedAccuracy | 0.059401 | 0.061008 | 0.060899 |
| F1-Score | 0.055569 | 0.055921 | 0.055978 |
| BrierScore | 0.047478 | 0.053813 | 0.053842 |
| Specificity | 0.039291 | 0.045888 | 0.045959 |
| ECE | 0.038871 | 0.045170 | 0.045335 |
| NPV | 0.031156 | 0.038475 | 0.038236 |
| ACC | 0.017638 | 0.022143 | 0.022175 |

## Model ranking

| Rank | Model | Median score | Mean rank | Top-3 frequency |
|---:|---|---:|---:|---:|
| 1 | pepnet_standard | 0.738863 | 2.68 | 70.0% |
| 2 | amplify_imb | 0.706146 | 3.58 | 66.0% |
| 3 | C_AMPs-predict | 0.697374 | 3.86 | 50.0% |
| 4 | HMD-AMP | 0.675815 | 4.54 | 30.0% |
| 5 | amplify_bal | 0.648560 | 5.74 | 16.0% |
| 6 | AMPsorter | 0.625123 | 6.04 | 30.0% |
| 7 | pepnet_fast | 0.596353 | 7.42 | 0.0% |
| 8 | macrel | 0.592317 | 7.10 | 0.0% |
| 9 | esm-AxP-GDL | 0.542104 | 8.82 | 12.0% |
| 10 | ascan2 | 0.491793 | 9.66 | 26.0% |
| 11 | lstm | 0.439487 | 11.74 | 0.0% |
| 12 | ai4amp | 0.430937 | 12.00 | 0.0% |
| 13 | iampcn | 0.429798 | 12.20 | 0.0% |
| 14 | apin | 0.426318 | 12.28 | 0.0% |
| 15 | amPEPpy | 0.388705 | 13.70 | 0.0% |
| 16 | ampir | 0.332724 | 14.78 | 0.0% |
| 17 | apex1.1 | 0.211610 | 16.86 | 0.0% |
| 18 | iamp-ca2l | 0.026120 | 18.00 | 0.0% |

## Interpretation boundary

Weights were selected before model identities and scores were revealed to the weight-setting Agents. However, the available datasets still have unresolved provenance, independence and homology gates, and the metric evidence is computed from stored test-like results. Therefore, these findings are suitable for exploratory model comparison and figure development, but not yet for a leakage-free formal benchmark claim.
