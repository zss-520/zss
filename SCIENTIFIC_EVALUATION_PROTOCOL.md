# AMP Benchmark 科学评测协议

协议版本：2.0。

## 1. 数据集与独立性门禁

- 正式主榜固定使用三套外部测试集：1 套近似平衡、2 套不平衡程度不同的数据集。
- 三套数据集均须满足项目的序列要求：整体、正类和负类中 10–50 aa 的比例均不低于 80%；绝对长度为 5–100 aa；只允许 20 种标准氨基酸。
- 单套数据集内不得有重复序列或冲突标签，三套数据集之间不得共享序列。
- 每套数据集必须记录原始 URL、论文 DOI/PMID、版本、许可证、获取日期、原文件 SHA256、标准化文件 SHA256、正负样本来源和抽样规则。
- 必须与每个参评模型的训练/验证序列做精确重叠检查和不高于 40% 序列一致性的 CD-HIT/MMseqs2 审计。
- 模型论文自带的 train/validation/test/benchmark set 对该模型不构成独立外测。它可用于复现或其他模型的候选测试，但必须先通过逐模型泄漏审计。
- 数据集候选只有经过“生成清单 → 下载 → SHA256 → 解压 → 标准化 → 泄漏检查 → manifest”完整门禁后才能进入正式主榜。

## 2. 评测单位、纳入与覆盖率

- 独立评测单位是一条带二分类真值的肽序列；存在同源簇时，同源簇是 bootstrap 重采样单位。
- 输入须包含 `True_Label`/`label`/`target`/`class` 之一，以及每个参评模型的 `*_Prob` 概率列。
- 缺失、非数值、无穷或超出 `[0,1]` 的概率不参与该模型的指标计算，并报告 coverage 和无效行数。
- 若有 `Standard_ID`，报告重复 ID；若有 `Homology_Cluster`、`Sequence_Cluster`、`Cluster_ID` 或 `Family_ID`，自动使用簇 bootstrap。
- 正式运行中任何目标模型没有有效预测，或缺少概率列，均判为评测失败。

## 3. 主终点和次终点

- 唯一预注册主终点：AUPRC。分别在三套测试集上报告，不能用跨数据集平均值掩盖阳性率差异。
- 关键次终点：MCC，使用验证集冻结的阈值计算。
- 强制报告：Accuracy、Balanced Accuracy、Precision、Recall/Sensitivity、Specificity、NPV、F1、MCC、AUROC、AUPRC、AUPRC lift 和混淆矩阵。
- AUPRC/MCC/Recall/Precision 加权分仅用于探索性模型排序，不替代逐终点效应量和置信区间。

## 4. 阈值策略

- 正式主榜必须提供独立验证集预测，并只在验证集上选择 Max MCC 阈值；并列时依次优先 Recall 更高、距 0.5 更近者。
- 阈值一经选择即冻结；禁止在测试集调阈值。
- 固定阈值 0.5 仅作为诊断性对照。正式运行缺少验证集阈值时直接失败。
- Youden Index 仅可作为预先声明的敏感性分析，不与 Max MCC 混用挑选有利结果。

## 5. 校准和实际筛选效用

- 校准：Brier score、10 个等宽概率箱的 ECE/MCE，以及完整 calibration curve 数据。
- 排序效用：AUPRC 相对阳性率基线的 lift；top 1%、5%、10% 的 Precision、Recall、enrichment factor 和 number needed to test。
- 若输入含序列，报告 10–20、21–30、31–50、51–100 aa 分层指标，以检查长度依赖。
- 每套数据集单独报告阳性率，不能把 Accuracy 或 AUROC 作为不平衡场景的唯一结论。

## 6. 不确定性、模型比较与多重校正

- 默认 500 次确定性 bootstrap，种子 `20260714`，报告 95% 分位数置信区间。
- 有同源簇时按簇重采样；没有簇信息时按序列重采样，并在报告中明确降级。
- 模型比较只使用两模型共同有效的样本：
  - 对 AUPRC、MCC、Balanced Accuracy 和 Brier score 报告配对 bootstrap 差值及 95% CI；
  - 对冻结阈值后的错误差异执行连续性校正 McNemar 检验；
  - 对同一数据集内全部成对 McNemar 比较执行 Holm 家族错误率校正。
- 报告效应差值和区间，不用单独的显著性星号代替效果大小。

## 7. 鲁棒性与计算资源

- 三套数据集分别作为平衡、轻/中度不平衡和重度不平衡压力测试；低同源不是第四套可选数据集，而是三套数据集共同的硬门禁。
- 至少加入负样本构造敏感性分析，优先使用 AMPBenchmark 这类保留多种负样本抽样策略的资源。
- 从运行 manifest/SLURM 记录报告：状态、ExitCode、Elapsed、MaxRSS、序列吞吐量，以及 CPU/GPU 配置。
- 如果多个模型在一个 SLURM 作业中串行或并行执行，作业总时间只能标注为 job-level，不能伪装为单模型延迟。

## 8. Agent 文献会议评测

文献会议还须用独立人工金标准 `data/literature_agent_gold_labels.json` 评价搜索和筛选：有效模型检索召回率、有效模型保留率、误检识别召回率、误检泄漏率、筛选 precision/accuracy/MCC、排除理由可追踪率、主元数据字段准确率和最终部署污染率。运行：

```bash
python literature_agent_evaluation.py
```

输出 `data/literature_agent_evaluation.json`、`data/literature_agent_evaluation.md` 和逐模型决定表。

## 9. 每套数据集的输出

```text
final_results_with_predictions.csv
scientific_evaluation.json
scientific_evaluation.md
eval_result.json
evaluation_curves.png
critic_individual.md
```

可用 `SCIENTIFIC_BOOTSTRAP_ITERATIONS`、`SCIENTIFIC_BOOTSTRAP_SEED` 调整 bootstrap。`SCIENTIFIC_REQUIRE_VALIDATION_THRESHOLD=1` 可对其他调用路径启用正式阈值门禁；主 benchmark 已在代码中强制开启。
