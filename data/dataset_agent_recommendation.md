# Dataset Agent 数据集推荐

- 推荐来源：**literature_global_meeting_consensus**
- 文献会议 Top 3 状态：**ready_for_acquisition**（3/3）
- 正式选集状态：**blocked_no_three_formally_eligible_meeting_datasets**
- 候选池规模：339
- 已审计本地候选：6
- 是否生成 Agent strategy：False

## 已有评测结果支持的互补三数据集

状态：**selected_pending_independence_and_homology_gates**。该组合由真实标签分布动态选择，不是固定名称模板；正式独立外测仍需训练重叠和同源性门禁。

1. **Veltri_test** — profile=balanced, positive_fraction=0.5103906899418121, ratio=0.9592833876221498
2. **C_AMPs-predict_test** — profile=imbalanced, positive_fraction=0.01750096946603497, ratio=0.017812709144886998
3. **ProteoGPT_all_predictions** — profile=imbalanced, positive_fraction=0.4036748329621381, ratio=0.676937441643324

## 优先下载与审计的 3 个候选

这些是下载和真实序列审计优先项，尚不是正式 benchmark 数据集。

1. **StarPep** —  (evidence score=27.0)
2. **DRAMP+UniProt ABP** —  (evidence score=26.0)
3. **dbAMP-derived binary set** —  (evidence score=19.0)

## 已审计本地候选

- **AntiBP2_数据集**: profile=balanced, ratio=0.7793895436687821, 10-50 aa=0.9966032608695652, formal blockers=citation_missing, independent_external_test_not_confirmed, label_definition_missing, license_missing, low_homology_report_missing, negative_sampling_strategy_missing, retrieval_date_missing, source_url_missing, training_references_missing, version_missing
- **Dataset_1_test1**: profile=imbalanced, ratio=0.676937441643324, 10-50 aa=1.0, formal blockers=citation_missing, class_length_distribution_mismatch, independent_external_test_not_confirmed, label_definition_missing, license_missing, low_homology_report_missing, negative_sampling_strategy_missing, retrieval_date_missing, sha256_missing, source_url_missing, training_references_missing, version_missing
- **sAMPpred-GAT_测试集**: profile=imbalanced, ratio=0.25806451612903225, 10-50 aa=1.0, formal blockers=independent_external_test_not_confirmed, label_definition_missing, license_missing, low_homology_report_missing, negative_sampling_strategy_missing, retrieval_date_missing, training_references_missing, version_missing
- **C_AMPs-predict_test**: profile=imbalanced, ratio=0.017812709144886998, 10-50 aa=1.0, formal blockers=independent_external_test_not_confirmed, label_definition_missing, license_missing, low_homology_report_missing, negative_sampling_strategy_missing, retrieval_date_missing, sha256_missing, training_references_missing, version_missing
- **ProteoGPT_all_predictions**: profile=imbalanced, ratio=0.676937441643324, 10-50 aa=1.0, formal blockers=class_length_distribution_mismatch, independent_external_test_not_confirmed, label_definition_missing, low_homology_report_missing, negative_sampling_strategy_missing, retrieval_date_missing, sha256_missing, training_references_missing
- **Veltri_test**: profile=balanced, ratio=0.9592833876221498, 10-50 aa=1.0, formal blockers=independent_external_test_not_confirmed, label_definition_missing, license_missing, low_homology_report_missing, negative_sampling_strategy_missing, retrieval_date_missing, sha256_missing, training_references_missing, version_missing
