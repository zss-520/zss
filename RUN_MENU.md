# AMP Benchmark 统一运行菜单

## 三个常用增量入口

在项目根目录运行：

```powershell
python amp_benchmark_menu.py
```

- 主菜单 `9`：只读取文献会议产生的三数据集共识并更新候选审计排序，不下载数据，也不执行完整门禁；核验种子不能自行替代会议决定。
- 文献子菜单 `16`：读取已有 compact evidence/chunk summaries，重新调用文献会议并更新 memory；不重新检索论文、不抓全文、不做 GitHub/Qwen 联网补链。
- 文献子菜单 `17`：增量搜索最近两年最新模型（包含当年 online-first），完成多源检索、去重、证据抽取、会议和 memory 更新。

也可以直接运行等价命令：

```powershell
# 只更新数据集 Top 3 推荐
python dataset_recommendation_agent.py recommend

# 只重新运行文献会议，不重新搜索文献
python deep_research_literature_agent.py --resume-global-only --no-github-enrichment --resume-note meeting_only_no_search

# 搜索 2025–2026 最新模型并更新会议（以后年份请相应修改）
python deep_research_literature_agent.py --year-from 2025 --year-to 2026 --max-results 50 --max-queries 30 --citation-seed-limit 10 --force-github-enrichment
```

注意：`--use-existing-meeting` 会复用旧会议结论，不会真正重新开会；因此“只跑文献会议”必须使用 `--resume-global-only`，且不要同时传入 `--use-existing-meeting`。

数据集 Top 3 的唯一推荐来源现在是 `meeting_recommended_datasets`。正确顺序为：先运行文献子菜单 `16` 形成新会议共识，再运行主菜单 `9` 将会议候选送入 Dataset Recommendation Agent。Python 规则只负责补证据、下载审计和科学门禁，不再用固定数据集清单覆盖会议结论。

文献子菜单 15 会运行文献会议 Agent 评测，基于独立人工审定集计算有效模型保留率、误检模型发现率、错误泄漏率、排除理由可追溯率、原始论文元数据正确率和最终部署榜污染率；同时列出全部唯一检索模型的会议决定、有效/误检/暂缓/待复核数量及比例。结果写入 `data/literature_agent_evaluation.json`、`.md` 和 `data/literature_meeting_screening_decisions.csv`。

Windows 下双击 `run_menu.bat`，或在项目根目录运行：

```powershell
python amp_benchmark_menu.py
```

顶层菜单包含：

1. 运行正式 benchmark，可按注册表编号选择一个、多个或全部模型。
2. 新模型自动入库、下载、HPC 部署和 smoke test。
3. 进入文献更新子菜单，执行年份检索、GitHub 补链、Qwen 补漏或 memory 重建。
4. 基于最新一次 run 的 `eval_result.json` 生成研究发展建议报告。
5. 查看 `.env` 配置状态、模型 registry、数据集、最新 run 和结果文件；不会显示密钥内容。
6. 查看推荐运行顺序和安全提示。
7. 运行门禁、SLURM 判定、科学评测协议与 manifest 的自动化测试。
8. 由 Dataset Recommendation Agent 从证据池建立候选池，基于真实序列自动选择 1 个平衡集和 2 个不平衡集，再完成下载、SHA256、安全解压、标准化、泄漏检查并写入可复现 manifest。

文献子菜单 13 提供严格隔离的“LLM 提名 100 个模型”流程：先在不启用网页搜索的情况下分 5 批、每批约 20 个生成候选；随后使用 Crossref、OpenAlex、GitHub API 和本地官方来源 JIF 清单逐条核验。未核验候选只保存在 `data/llm_top_journal_model_nominations.*`，不能进入证据池；只有论文与 AMP 模型身份通过核验的记录才会写入 `data/evidence_pool.json` 并触发最终模型重排。

文献子菜单 14 是三个必审候选的科学证据门禁：`C_AMPs-predict`、`AMPSorter`（分类器，父模型为 `ProteoGPT`）和 `HMD-AMP`。种子文件只负责提供待核验声明；流程会重新核验 DOI/题名、出版社原文中的模型/代码/数据链接以及 GitHub API，全部通过后才进入证据池。用户报告的“人工评测最优”会保留为待审声明，只有关联同一数据划分下的评测结果和 run manifest 后才作为性能证据。

也可以直接运行：

```powershell
python scientific_model_evidence.py --stage all
```

正式 benchmark 选择模型时还可以输入 `PORTFOLIO`，使用文献流程生成的分层建议组合。该组合会优先保留经典基线和近期 SOTA 候选，再补齐架构覆盖；详见
[`BENCHMARK_MODEL_SELECTION.md`](BENCHMARK_MODEL_SELECTION.md)。

建议先选择“查看项目状态”。正式 benchmark 会强制要求模型同时满足
`HPC=ready` 与 `smoke=passed`。诊断性绕过需要在菜单中明确确认，不会静默放行。

正式 benchmark 还要求最近一次数据集准备门禁通过。菜单 8 按以下顺序执行：

```text
literature memory + evidence pool + local audited datasets
  -> data/dataset_candidate_pool.json
  -> Dataset Recommendation Agent constrained selection
  -> data/benchmark_strategy.agent.json（仅在三个数据集全部合格时生成）
  -> data/dataset_plan.json
  -> 下载或复用本地原始文件
  -> SHA256 官方校验或首轮 TOFU 基线（data/dataset_source_lock.json）
  -> 防路径穿越的安全解压
  -> combined_test.fasta + ground_truth.csv
  -> 测试集交叉重复、标签冲突和训练参考集泄漏检查
  -> data/dataset_manifests/{gate_id}.json
```

Dataset Agent 会把文献提名和正式入选严格分开：没有下载并读取真实序列的数据集只能进入 `acquisition_queue`，不能凭论文描述猜测类别比例或长度分布。只有存在恰好 1 个平衡和 2 个不平衡的完整合格组合时，才生成 `data/benchmark_strategy.agent.json`；否则菜单 8 停止在推荐报告，不会回退到人工清单。

直接运行 `dataset_gate.py` 且未提供 strategy 时仍保留本地 bootstrap，供开发测试使用；正式菜单流程只接受 Dataset Agent 生成的 strategy。
若清单包含 `leakage_reference_paths` 或 `training_reference_paths`，这些训练集/排除集会参与精确序列泄漏检查。
默认不同测试集之间出现相同序列即阻断；菜单中可显式允许，但重叠仍会写入 manifest 警告。

当前正式策略由 `data/benchmark_strategy.json` 预注册，并把该文件作为数据集白名单：只选择 3 个数据集，其中 1 个近似平衡集、2 个不平衡集。少数类/多数类样本数之比不低于 0.70 定义为近似平衡，否则定义为不平衡；两个不平衡集的少数类占比至少相差 0.10，以覆盖不同失衡强度。每个集合及每个类别至少 80% 的序列必须处于 10--50 aa，绝对长度限制为 5--100 aa。

科学选集门禁还要求：总样本数至少 500、每类至少 100；正负类中位长度差不超过 15 aa；来源 URL、论文引用、数据版本、许可证和官方 SHA256 可追溯；正类标签定义与负样本构造方法明确；测试集独立于所有参评模型；提供训练参考序列及不高于 40% 序列一致性的 CD-HIT/MMseqs2 低同源报告；集合内部去重、集合之间无交叉、无标签冲突。任何一项缺失都会写入 `dataset_selection_check` 或 `leakage_check`，正式 benchmark 不放行。

每次 benchmark 都创建独立目录：

```text
data/runs/{run_id}/
├── manifest.json
├── artifacts/
└── results/{dataset_name}/
```

`data/runs/latest.json` 指向最近一次运行。详细指标约定见
[`SCIENTIFIC_EVALUATION_PROTOCOL.md`](SCIENTIFIC_EVALUATION_PROTOCOL.md)。
