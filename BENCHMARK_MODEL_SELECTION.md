# AMP Benchmark 分层选模协议

版本：1.0；近期窗口按运行年份动态计算。

## 为什么不能只按架构、IF 和引用量排序

影响因子和引用量衡量论文传播与历史影响，不等价于模型在统一外部测试集上的性能；同时会系统性低估刚发表的模型。因此正式模型组合采用四层选择：

1. **经典基线**：至少 3 个，要求有代码和可批量推理能力。
2. **近期 SOTA 候选**：至少 3 个，来自最近两年，要求有代码，并有独立或外部测试证据。
3. **架构代表**：补齐传统 ML、CNN、RNN、CNN+RNN、Transformer/PLM、GNN、ensemble/pipeline。
4. **证据排序补位**：剩余位置再综合复现性、引用量、IF 和文章影响分。

最终主榜最多 20 个模型。经典基线和近期候选是硬性配额，不能被纯 Top-N 排名挤掉。

## 经典基线锚点

当前内置锚点包括：AMP Scanner v2、Macrel、amPEPpy、AMPlify、Co-AMPpred、Deep-AmPEP30。只有当前证据中存在代码的模型才会实际入选。

- Macrel：2020 年、理化特征与传统机器学习工具，官方代码与 benchmark 可追溯：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7751412/>
- amPEPpy：便携式 AMP 预测工具：<https://pubmed.ncbi.nlm.nih.gov/33135060/>
- AMPlify：BiLSTM 与注意力经典深度学习模型，开源工具：<https://pmc.ncbi.nlm.nih.gov/articles/PMC8788131/>
- Deep-AmPEP30：短肽 CNN 基线：<https://pubmed.ncbi.nlm.nih.gov/32464552/>

## 近期 SOTA 候选观察表

截至 2026-07-14，内置检索观察表包括：

| 模型 | 年份 | 证据与代码 | 流程中的身份 |
|---|---:|---|---|
| CG-AMP | 2025 | <https://www.nature.com/articles/s41598-025-29666-z>；<https://github.com/ghli16/CG-AMP> | 有代码、报告两个独立测试集，近期 SOTA 候选 |
| deepAMPNet | 2024 | <https://pubmed.ncbi.nlm.nih.gov/39040937/>；<https://github.com/Iseeu233/deepAMPNet> | 结构与序列融合，近期 SOTA 候选 |
| UniproLcad | 2024 | <https://www.mdpi.com/2073-8994/16/4/464>；<https://github.com/harkic/UniproLcad> | 多 PLM 融合，近期 SOTA 候选 |
| PepNet | 2024 | <https://pmc.ncbi.nlm.nih.gov/articles/PMC11438969/> | 观察候选；缺少已核实本地代码时不能进入正式部署榜 |

观察表只用于提高检索召回率，不会绕过证据门禁。随着年份变化，流程同时会从证据池动态发现最近两年的其他模型。

## “SOTA”命名纪律

论文中的 `state-of-the-art`、`outperform` 只构成候选证据。正式报告在统一数据、目标模型预测列、验证集阈值、bootstrap CI 和配对检验完成前，不得称某模型为本 benchmark 的 SOTA。

## 输出与使用

文献流程在 `literature_deep_research_memory.json` 中生成：

- `final_deployment_models`：带 `benchmark_role` 的最终部署顺序；
- `benchmark_model_portfolio`：配额、角色计数、架构覆盖和缺口；
- `gaps`：经典基线、近期候选或架构覆盖不足时的待补模型清单。

模型入库菜单会显示角色和年份；正式评测菜单可以输入 `PORTFOLIO` 使用当前分层建议组合。每次运行的实际组合与缺口也会写入 run manifest。
