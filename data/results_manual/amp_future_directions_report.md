# AMP 模型未来发展方向报告

> 本报告由本地确定性兜底逻辑生成：没有把评测数据发送到外部 LLM API。

## 当前评测概况

- 结果目录：`data\results_manual`
- 数据集数量：3
- 有效参评模型数量：18
- 实际指标：ACC, AUPRC, AUPRC-Lift, AUROC, BalancedAccuracy, BrierScore, Coverage, ECE, F1-Score, MCC, NPV, Precision, Recall, Specificity, Threshold
- 权重更新轮数：50
- 最终指标权重（50 轮中位数）：{'ACC': 0.07893657, 'AUPRC': 0.08183202, 'AUROC': 0.08186662, 'BalancedAccuracy': 0.07967132, 'BrierScore': 0.07879967, 'ECE': 0.08204952, 'F1-Score': 0.07943869, 'MCC': 0.08000517, 'NPV': 0.09110171, 'Precision': 0.08781125, 'Recall': 0.09018343, 'Specificity': 0.08830403}
- 模型特定优先加分：未启用
- 资源资格规则：measured_budget_gate（在性能评分前统一执行）
- 因实测资源超限排除：无
- 尚缺资源测量但暂时保留：18 个模型

## 跨数据集综合排名

| Rank | Model | Median score | Mean score | Score IQR | Top3 frequency | AUPRC | MCC | Recall | Precision | AUROC | Datasets |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | pepnet_standard | 0.7415 | 0.7414 | 0.0340 | 0.8000 | 0.8665 | 0.6994 | 0.8677 | 0.7039 | 0.9681 | 3 |
| 2 | HMD-AMP | 0.6831 | 0.6962 | 0.1302 | 0.5800 | 0.8179 | 0.6257 | 0.8837 | 0.6299 | 0.9564 | 3 |
| 3 | C_AMPs-predict | 0.6780 | 0.6690 | 0.0882 | 0.4800 | 0.9493 | 0.7875 | 0.7621 | 0.9513 | 0.9712 | 3 |
| 4 | amplify_imb | 0.6777 | 0.6926 | 0.1293 | 0.3400 | 0.9127 | 0.6129 | 0.9279 | 0.6025 | 0.9671 | 3 |
| 5 | amplify_bal | 0.6411 | 0.6248 | 0.0971 | 0.1400 | 0.8908 | 0.6441 | 0.8729 | 0.6466 | 0.9501 | 3 |
| 6 | AMPsorter | 0.6345 | 0.6604 | 0.1866 | 0.4600 | 0.8316 | 0.6616 | 0.8054 | 0.7200 | 0.9485 | 3 |
| 7 | pepnet_fast | 0.6059 | 0.6037 | 0.1000 | 0.0000 | 0.8641 | 0.6288 | 0.8631 | 0.6453 | 0.9442 | 3 |
| 8 | macrel | 0.5815 | 0.5732 | 0.0905 | 0.0000 | 0.8511 | 0.6431 | 0.7275 | 0.7360 | 0.9375 | 3 |
| 9 | esm-AxP-GDL | 0.5265 | 0.5474 | 0.1842 | 0.0000 | 0.8967 | 0.5146 | 0.9094 | 0.5428 | 0.9243 | 3 |
| 10 | ascan2 | 0.4906 | 0.4683 | 0.2164 | 0.2000 | 0.6562 | 0.5358 | 0.8838 | 0.5516 | 0.9083 | 3 |

## Top3 集成学习候选模型推荐

推荐 Top3：**pepnet_standard**, **HMD-AMP**, **C_AMPs-predict**。

| Rank | Model | 推荐理由 |
|---:|---|---|
| 1 | pepnet_standard | 50 轮中位综合分 0.7415、IQR 0.0340、进入 Top3 频率 0.8000；AUPRC 0.8665，MCC 0.6994，Recall 0.8677，Precision 0.7039；覆盖 3 个数据集。 |
| 2 | HMD-AMP | 50 轮中位综合分 0.6831、IQR 0.1302、进入 Top3 频率 0.5800；AUPRC 0.8179，MCC 0.6257，Recall 0.8837，Precision 0.6299；覆盖 3 个数据集。 |
| 3 | C_AMPs-predict | 50 轮中位综合分 0.6780、IQR 0.0882、进入 Top3 频率 0.4800；AUPRC 0.9493，MCC 0.7875，Recall 0.7621，Precision 0.9513；覆盖 3 个数据集。 |

### 为什么推荐集成学习

这组模型的互补性主要来自 Precision/Recall 取舍和跨数据集稳定性差异。AUPRC 更高的模型更适合做候选排序主干，Recall 更高的模型适合扩大候选召回，MCC/Precision 更高的模型适合在后处理阶段压低假阳性。单模型通常只能固定在一个阈值策略上，而 ensemble 可以把排序能力、召回能力和阈值决策拆开组合。

### 推荐集成策略

- **首选 rank averaging / soft voting**：当前已有多个模型的概率输出，最容易无训练集泄漏地实现，可先按跨数据集 AUPRC/MCC 作为权重做加权平均。
- **可选 stacking/meta-classifier**：如果后续有独立验证集，可用 Logistic Regression 或 LightGBM 作为二层模型；没有验证集时不建议在测试集上调参。
- **high-recall candidate union**：用于湿实验筛选前置阶段，先用高 Recall 模型并集扩大候选，再用高 Precision/MCC 模型二次排序。
- **不建议只做 hard voting**：hard voting 会丢失概率排序信息，对 AUPRC 和早期富集不友好。

## 下一步建议

1. 保留当前手动预测文件作为 raw evidence，并固定 `final_results_with_predictions.csv` 作为后续复现实验输入。
2. 为 Top3 候选构建独立验证集，避免在测试集上选择 ensemble 权重或阈值。
3. 优先报告 AUPRC、MCC、Recall、Precision 和覆盖率；如果用于湿实验筛选，额外加入 top-k enrichment 或 Recall@Precision。
4. 对高 Recall 但 Precision 较弱的模型做二阶段过滤，对高 Precision 但 Recall 较弱的模型做候选排序校准。
