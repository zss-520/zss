# AMP Benchmark Literature Memory

Updated: 2026-07-03 13:57:30

## Latest Run

```json
{
  "time": "2026-07-03 13:57:30",
  "mode": "multi_source_global_meeting",
  "sources": [
    "all"
  ],
  "max_results": 10,
  "batch_size": 2,
  "paper_count": 783,
  "processed_this_run": 39,
  "evidence_batches": 65,
  "failed_evidence_batches": 0,
  "source_counts": {
    "crossref": 239,
    "europe_pmc": 177,
    "openalex": 246,
    "pubmed": 150,
    "pubmed_similar": 12,
    "semantic_scholar": 20,
    "semantic_scholar_citations": 4,
    "semantic_scholar_references": 3
  },
  "fetch_fulltext": true,
  "backsearch_models": true,
  "expand_citations": true,
  "evidence_compression": true,
  "chunk_target_size": 6,
  "max_chunks": 40,
  "max_chars_per_chunk": 60000,
  "compact_evidence_pool": "data\\compact_evidence_pool.json",
  "github_enrichment": true,
  "github_enrichment_file": "data\\github_missing_model_enrichment.json",
  "qwen_web_enrichment": true,
  "qwen_web_model": "qwen3.7-max",
  "qwen_web_enrichment_file": "data\\qwen_web_enrichment.json"
}
```

## Final Execution Decision / 最终执行决策

主榜先部署模型、推荐数据集和指标如下；候选模型与 Agent 讨论放在后文附录。

### 1. 最终先部署模型

|deployment_rank|deployment_tier|model_name|representation_category|architecture_category|task_type|code_repository_url|dataset_source_or_link|source_journal|citation_count|journal_impact_factor|article_impact_score|deployment_selection_score|deployment_status|deployment_reason|first_next_action|blocking_issues|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|core_main_benchmark_top10|AmPEP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|https://github.com/ShirleyWISiu/AmPEP|Collected from CAMP, APD, UniProt (3268 AMPs, 166791 non-AMPs); benchmarked on iAMPpred and iAMP-2L datasets||0.0|0.0|0.0|19.65|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"github_search_candidate_requires_manual_verification"<br>]|
|2|core_main_benchmark_top10|AntiBP2|traditional_physicochemical_statistical_features|machine_learning_models|antibacterial peptide prediction|not_reported_in_available_evidence|Antibacterial Peptide Database (APD)||0.0|0.0|0.0|19.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original paper not provided; no link available in this evidence"<br>]|
|3|core_main_benchmark_top10|PeptideRanker|structure_graph_representation|gnn_models|general peptide bioactivity prediction (including antimicrobial)|not_reported_in_available_evidence|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)||0.0|0.0|0.0|19.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口||
|4|core_main_benchmark_top10|WeightedEnsemble_L3 (Anti_Cp)|structure_graph_representation|gnn_models|antimicrobial peptide activity classification|https://github.com/xubocheng/Anti_Cp.git|https://github.com/xubocheng/Anti_Cp.git||0.0|0.0|0.0|19.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口||
|5|core_main_benchmark_top10|c_AMPs-prediction|protein_language_model_representation|rnn_lstm_dominant_models|AMP prediction|https://github.com/mayuefine/c_AMPs-prediction|https://github.com/mayuefine/c_AMPs-prediction||0.0|0.0|0.0|19.5|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_model_paper_uncertain",<br>"weights_not_reported"<br>]|
|6|core_main_benchmark_top10|AMPer|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|not_reported_in_available_evidence|known antimicrobial peptides (not further specified)||0.0|0.0|0.0|19.4|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original paper not provided; details rely on this reference"<br>]|
|7|core_main_benchmark_top10|Macrel|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_needed_for_architecture_code_weights"<br>]|
|8|core_main_benchmark_top10|APSvr.2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"github_search_candidate_requires_manual_verification"<br>]|
|9|core_main_benchmark_top10|AxPEP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|https://sourceforge.net/projects/axpep/|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口||
|10|core_main_benchmark_top10|AMP Scanner v2|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|https://github.com/dan-veltri/amp-scanner-v2|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_needed"<br>]|
|11|extended_deployment_pool_11_20|AMPlify_bal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_needed"<br>]|
|12|extended_deployment_pool_11_20|AMPlify_imbal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_needed"<br>]|
|13|extended_deployment_pool_11_20|Deep-AmPEP30|sequence_encoding_representation|cnn_dominant_models|AMP prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.5|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"github_search_candidate_requires_manual_verification"<br>]|
|14|extended_deployment_pool_11_20|RF-AmPEP30|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.5|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"No weights reported"<br>]|
|15|extended_deployment_pool_11_20|AMP MIC predictor (CNN/RNN)|sequence_encoding_representation|cnn_dominant_models|AMP prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.5|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口||
|16|extended_deployment_pool_11_20|iAMPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.1|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|
|17|extended_deployment_pool_11_20|AMPlify|sequence_encoding_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|https://github.com/bcgsc/AMPlify|UniProtKB/Swiss-Prot (used as mining source)||0.0|0.0|0.0|17.54|extended_pool_deploy_after_weight_and_data_check|经典 AMP 分类深度学习模型；有 GitHub，适合作为纯序列/RNN-Attention 基线。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"code_not_found"<br>]|
|18|extended_deployment_pool_11_20|CAMPR3(RF)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|16.4|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"needs original paper verification"<br>]|
|19|extended_deployment_pool_11_20|AntiBP|traditional_physicochemical_statistical_features|machine_learning_models|antibacterial peptide prediction|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|16.4|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"needs original paper verification"<br>]|
|20|extended_deployment_pool_11_20|ADAM|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|16.3|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"needs original paper verification"<br>]|

### 2. 推荐最合适的 3 个数据集

|dataset_rank|dataset_name|linked_model|recommended_role|dataset_source_or_link|why_selected|required_cleaning|status|
|---|---|---|---|---|---|---|---|
|1|iAMP-SeE Dataset / Zenodo|iAMP-SeE|primary benchmark candidate / 主测试集候选|需要从仓库/Zenodo/补充材料中确认|当前证据中来源最清楚，包含 DRAMP、dbAMP、CAMPr-4、AMPfun、ADAPTABLE、UniProt 负样本，并有 Zenodo 线索；最适合作为第一版 benchmark 的主数据集候选。|核查 Zenodo 文件；统一 FASTA/CSV 格式；确认正负标签；用 CD-HIT/MMseqs2 去冗余；过滤与模型训练集高度同源序列。|recommended_top3_dataset_needs_cleaning_and_version_lock|
|2|Co-AMPpred / DEEP-AmPEP30 derived dataset|Co-AMPpred|classic comparison set / 经典对照测试集|需要从仓库/Zenodo/补充材料中确认|Co-AMPpred 有 GitHub 和 DEEP-AmPEP30 衍生数据线索，适合做传统 ML 与经典 AMP benchmark 对照。|核查 GitHub 中正负样本数量、负样本来源和去重方式；排除训练集重叠；补充数据集版本记录。|recommended_top3_dataset_needs_cleaning_and_version_lock|
|3|AMP-BERT GitHub dataset|AMP-BERT|PLM reproduction set / PLM 模型复现测试集|需要从仓库/Zenodo/补充材料中确认|AMP-BERT 与 PLM 模型直接配套，GitHub 中有代码和数据线索，适合验证 PLM 路线和复现 AMP-BERT。|核查数据文件、标签列和训练/测试划分；拆分出外部测试集；做低同源过滤，避免 PLM 模型过拟合历史划分。|recommended_top3_dataset_needs_cleaning_and_version_lock|

### 3. 最终指标体系

### 主排名指标

|指标|权重|用途|
|---|---:|---|
|AUPRC|0.35|主指标；适合不平衡二分类，优于只看 AUROC。|
|MCC|0.3|综合 TP/TN/FP/FN，对类别不平衡更稳健。|
|Recall / Sensitivity|0.2|控制 AMP 漏检，适合发现任务。|
|Precision|0.15|控制假阳性，避免大量错误候选进入后续实验。|

### 强制报告指标

Accuracy, Specificity, AUROC, F1-score, Confusion Matrix

### 阈值与测试矩阵

- 阈值策略：在验证集上用 Max MCC 或 Max Youden Index 确定阈值；测试集禁止后验调阈值，禁止默认固定 0.5。
- 测试矩阵：1:1 balanced test, 1:10 mild imbalance test, 1:100 severe imbalance test, low-homology independent test
- 统计报告：95% bootstrap confidence interval, paired bootstrap or McNemar test for model comparison

## Agent Discussion Process

# 🧠 AMP 文献证据全局会议记录

## 📚 历史共识基线
```text
【现有记忆/精选模型摘要】:
Macrel, ACPred, AMPfun, AntiCP, AntiCP2.0, iAMPpred, HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP
```

## 🕵️ Agent 1 (Scout / Model-Dataset) 增量提案

### 一、证据池与召回概况
- Chunk summaries: 42
- Paper count: 783
- Source counts: {'crossref': 239, 'europe_pmc': 177, 'openalex': 246, 'pubmed': 150, 'pubmed_similar': 12, 'semantic_scholar': 20, 'semantic_scholar_citations': 4, 'semantic_scholar_references': 3}

### 二、模型与数据集初步提案
- 模型-数据集关系数量：3
- 数据集数量：4
- 优先模型数量：5
- 候选模型数量：43
- **Scout 增量提案**

基于 42 个 compact evidence chunk summaries 的全局整理，合并了重复模型和数据集，并进行了候选状态评估。

**新增/确认的可用于 benchmark 的模型**：
- Co-AMPpred (GBC + 特征选择，代码/数据公开)
- iAMPCN (CNN 两阶段，代码公开)
- ACEP (CNN-LSTM + 注意力，代码公开)
- Macrel (Random Forest，代码公开)
- Deep-AmPEP30 (CNN，权重待确认)
- RF-AmPEP30 (RF，权重待确认)
- Multi-label WKnn-MLR (传统 ML，无代码但方法清晰)
- AMP prediction server (biosino) (NNA 方法，webserver 可用)
- AMP Scanner v2 (APSvr.2) (webserver，广泛使用)

**需要降级或排除的模型**：
- 非 AMP 预测：ADMETlab 3, PeptideRanker, ACP-DL (抗癌肽)
- 生成式模型：cdGAN (设计模型，不直接用于 AMP 分类)
- 毒性/溶血/过敏原预测：HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP, AllerCatPro (非 AMP 分类，但保留作为相关任务)
- 仅有 webserver 且无代码/权重：ACPred, AMPfun, AntiCP, AntiCP2.0, iAMPpred, CAMPR3, DBAASP 等
- 名称不明确或仅有综述引用：ADAM, iAMP-2L, iAMPred, AmPEP 等

**拦截记录**：多模型因缺少代码、权重或仅 webserver 而无法直接纳入 benchmark，但保留为候选以备后续验证。

**建议**：
1. 优先验证 Co-AMPpred, iAMPCN, ACEP, Macrel 的代码和权重可用性。
2. 对 Deep-AmPEP30, RF-AmPEP30 等需确认原始论文和权重。
3. 对 webserver 类工具，考虑通过 API 或本地复现进行基准测试。
4. 补充缺失的模型原始论文信息，尤其是 ACPred, AMPfun 等。
5. 对多标签模型（如 Multi-label WKnn-MLR）评估其在 AMP 识别上的性能。
- ACEP paper DOI mismatch: need to locate correct paper.
- Are Deep-AmPEP30 and RF-AmPEP30 from the same AmPEP30 suite?
- Can we obtain weights for iAMPCN and ACEP?

### 三、模型分类梳理：数据/输入表示（Representation）
| 类别 | 类别特点 | 具体模型 | 每类代表模型 1-2 个 |
|:---|:---|:---|:---|
| 传统理化/统计特征为主 | 以全局理化性质、氨基酸组成/伪氨基酸组成、k-mer 统计等手工特征为主，不显式保留序列位置，只做整体向量化。 | Macrel, RF-AmPEP30, CAMPR34, CLASSAMP5, DBAASP6, AmPEP, CAMPR3, DBAASPv3.0, CAMPR3(RF), CAMPR3(SVM), MLAMP, AntiBP, AntiBP2, Multi-label weighted KNN-MLR model, ISCAPE, AMP Scanner v2, StackAMP, AMPlify_bal, AMPlify_imbal, AMPer, CAMP, AVPpred, MetaPepticon, Macrel (BigDataBiology), Macrel (MacReloader), Macrel (macrelay), AmPEP (amPEPpy), AmPEP (Ampep_Python), AmPEP (ShirleyWISiu), APD3 | Macrel, AmPEP |
| 纯序列/编码表示 | 直接对氨基酸序列做编码，如 one-hot、索引 embedding、PC6 理化编码、PseKRAAC 降维编码，或把序列转成小图像；不依赖大型 PLM，也不显式用 3D 结构。 | ACPred, AMPfun, AntiCP, AntiCP2.0, iAMPpred, HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP, AllerCatPro, AxPEP3, Deep-AmPEP30, ADAM, APSvr.2, DBAASP, BAGEL3, BACTIBASE, ADAM (prediction tool), ADMETlab 3, AMP MIC predictor (CNN/RNN), AxPEP, AMPlify, hydramp, PrefixProt, HydraAMP, Venomics artificial intelligence, hydramp (conda-feedstock), hydramp (pytorch port) | AMPlify, APEX |
| 蛋白语言模型（PLM）表示 | 使用预训练蛋白语言模型（BERT/T5/ESM/ProtT5/GPT-2 等）从序列生成高维 embedding，再接 CNN/MLP 等下游分类器。 | cdGAN, AMPGenix, PLUM, c_AMPs-prediction | c_AMPs-prediction, PLUM |
| 结构/图表示 | 将肽构造成图：节点可以是原子、残基或 k-mer；边来自共价键、空间距离或共现关系；有的结合预测 3D 结构，节点特征可叠加 ESM embedding。 | AMP prediction server (biosino), AMP-GSM, PeptideRanker, WeightedEnsemble_L3 (Anti_Cp), WeightedEnsemble_L3, Lab, Co-AMPpred | Lab, PeptideRanker |
| 多模态 / 混合表示 | 同时使用两种及以上类型的输入，如 one-hot 序列 + 大量理化特征，或 PLM embedding + 手工 PD 特征等。 | Deep learning hybrid model (unnamed) | Deep learning hybrid model (unnamed) |

### 四、Representation 每类代表模型选择依据
| 类别 | 代表模型 | 方法族 | 代码/工具链接 | 数据集线索 | 代表性理由 |
|:---|:---|:---|:---|:---|:---|
| 传统理化/统计特征为主 | Macrel | ML | not_reported_in_available_evidence | not_reported_in_available_evidence | 按用户指定的 传统理化/统计特征为主 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 传统理化/统计特征为主 | AmPEP | ML | https://github.com/ShirleyWISiu/AmPEP | Collected from CAMP, APD, UniProt (3268 AMPs, 166791 non-AMPs); benchmarked on iAMPpred and iAMP-2L datasets | 按用户指定的 传统理化/统计特征为主 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 纯序列/编码表示 | AMPlify | DL | https://github.com/bcgsc/AMPlify | UniProtKB/Swiss-Prot (used as mining source) | 按用户指定的 纯序列/编码表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 纯序列/编码表示 | APEX | DL | not_reported_in_available_evidence | not_reported_in_available_evidence (training data not described, in-house peptides mentioned) | 按用户指定的 纯序列/编码表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 蛋白语言模型（PLM）表示 | c_AMPs-prediction | DL | https://github.com/mayuefine/c_AMPs-prediction | https://github.com/mayuefine/c_AMPs-prediction | 按用户指定的 蛋白语言模型（PLM）表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 蛋白语言模型（PLM）表示 | PLUM | DL | https://github.com/priyamayur/PLUM | Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository | 按用户指定的 蛋白语言模型（PLM）表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 结构/图表示 | Lab |  | https://github.com/google-deepmind/lab | not_reported_in_available_evidence | 按用户指定的 结构/图表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 结构/图表示 | PeptideRanker | DL | not_reported_in_available_evidence | BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control) | 按用户指定的 结构/图表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 多模态 / 混合表示 | Deep learning hybrid model (unnamed) | deep learning (hybrid) | not_reported_in_available_evidence | not_reported_in_available_evidence | 按用户指定的 多模态 / 混合表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |

### 五、模型分类梳理：模型架构（Architecture）
| 类别 | 类别特点 | 具体模型 | 每类代表模型 1-2 个 |
|:---|:---|:---|:---|
| 机器学习模型 | 基于特征工程 + 传统分类器/回归器，如 Random Forest、SVM、LightGBM、逻辑回归等。 | Macrel, RF-AmPEP30, CAMPR34, CLASSAMP5, DBAASP6, AmPEP, CAMPR3, CAMPR3(RF), CAMPR3(SVM), MLAMP, AntiBP, AntiBP2, AMP prediction server (biosino), Multi-label weighted KNN-MLR model, ISCAPE, AMP Scanner v2, StackAMP, AMPlify_bal, AMPlify_imbal, CAMP, AVPpred, MetaPepticon, Macrel (BigDataBiology), Macrel (MacReloader), Macrel (macrelay), AmPEP (amPEPpy), AmPEP (Ampep_Python), AmPEP (ShirleyWISiu), APD3, AVCpred | Macrel, AmPEP |
| CNN 主导模型 | 一维/二维卷积是主干，负责自动学习局部 motif 与局部模式，有时堆叠多层 CNN（DenseNet、VGG、ResNet 等）。 | Deep learning hybrid model (unnamed), Deep-AmPEP30, cdGAN, AMP MIC predictor (CNN/RNN), hydramp, HydraAMP, Deep-AmPEP30 web server, iAMPCN | Deep-AmPEP30, iAMPCN |
| RNN/LSTM 主导模型 | 以（双向）LSTM/GRU 为主干，建模序列顺序依赖；注意力层通常作为辅助模块。 | AMPlify, AMPlify (AWS Amplify JS), AMPlify (AWS Amplify CLI), AMPlify (Jekyll AMP theme), c_AMPs-prediction, AMPlify GitHub | AMPlify, c_AMPs-prediction |
| CNN + RNN 混合模型 | 先用 CNN 抽局部 motif，再用 LSTM/BiLSTM 建模长程依赖，最后接全连接/分类器。 |  | 待定 |
| Transformer / LLM 主导模型 | 主干是多头自注意力/Transformer 模块（包括 GPT-2、BERT、ProtT5 等大模型），或在下游显式堆叠 Transformer block、cross-attention 作为核心特征提取器。 | AMPGenix, PLUM, ApexGO | AMPGenix, PLUM |
| 图神经网络（GNN）模型 | 将肽建模为图结构（原子/残基/k-mer 为节点，键/空间距离/共现为边），使用 GCN、GAT 等进行 message passing。 | AMP-GSM, PeptideRanker, WeightedEnsemble_L3 (Anti_Cp), WeightedEnsemble_L3, Lab, Co-AMPpred | Lab, PeptideRanker |
| 其他（多阶段流水线 / 集成框架） | 用多个模型串联或集成，或多模型 + 集成/堆叠。 | ACPred, AMPfun, AntiCP, AntiCP2.0, iAMPpred, HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP, AllerCatPro, AxPEP3, ADAM, APSvr.2, DBAASPv3.0, DBAASP, BAGEL3, BACTIBASE, ADAM (prediction tool), ADMETlab 3, AxPEP, PrefixProt, AMPer, Venomics artificial intelligence, hydramp (conda-feedstock), hydramp (pytorch port), APEX, MAPLE, AmPEP web server | APEX, ADAM (prediction tool) |

### 六、Architecture 每类代表模型选择依据
| 类别 | 代表模型 | 方法族 | 代码/工具链接 | 数据集线索 | 代表性理由 |
|:---|:---|:---|:---|:---|:---|
| 机器学习模型 | Macrel | ML | not_reported_in_available_evidence | not_reported_in_available_evidence | 按用户指定的 机器学习模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 机器学习模型 | AmPEP | ML | https://github.com/ShirleyWISiu/AmPEP | Collected from CAMP, APD, UniProt (3268 AMPs, 166791 non-AMPs); benchmarked on iAMPpred and iAMP-2L datasets | 按用户指定的 机器学习模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| CNN 主导模型 | Deep-AmPEP30 | DL | not_reported_in_available_evidence | not_reported_in_available_evidence | 按用户指定的 CNN 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| CNN 主导模型 | iAMPCN |  |  |  | 按用户指定的 CNN 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| RNN/LSTM 主导模型 | AMPlify | DL | https://github.com/bcgsc/AMPlify | UniProtKB/Swiss-Prot (used as mining source) | 按用户指定的 RNN/LSTM 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| RNN/LSTM 主导模型 | c_AMPs-prediction | DL | https://github.com/mayuefine/c_AMPs-prediction | https://github.com/mayuefine/c_AMPs-prediction | 按用户指定的 RNN/LSTM 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| Transformer / LLM 主导模型 | AMPGenix | DL | not_reported_in_available_evidence | not_reported_in_available_evidence | 按用户指定的 Transformer / LLM 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| Transformer / LLM 主导模型 | PLUM | DL | https://github.com/priyamayur/PLUM | Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository | 按用户指定的 Transformer / LLM 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 图神经网络（GNN）模型 | Lab |  | https://github.com/google-deepmind/lab | not_reported_in_available_evidence | 按用户指定的 图神经网络（GNN）模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 图神经网络（GNN）模型 | PeptideRanker | DL | not_reported_in_available_evidence | BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control) | 按用户指定的 图神经网络（GNN）模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 其他（多阶段流水线 / 集成框架） | APEX | DL | not_reported_in_available_evidence | not_reported_in_available_evidence (training data not described, in-house peptides mentioned) | 按用户指定的 其他（多阶段流水线 / 集成框架） 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |
| 其他（多阶段流水线 / 集成框架） | ADAM (prediction tool) |  | not_reported_in_available_evidence | not_reported_in_available_evidence | 按用户指定的 其他（多阶段流水线 / 集成框架） 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。 |

## 📐 Agent 2 (Metrics) 指标与测试集提案
- ## 初版指标提案

基于 compact evidence chunk summaries，AMP 二分类 benchmark 的初版指标组合如下：

- **核心指标**：AUROC、AUPRC、MCC、F1、Sensitivity (Recall)、Specificity、Precision、Accuracy。
- **阈值策略**：大部分模型默认使用 0.5 作为决策阈值。
- **数据集切分**：按模型各自论文的随机划分或留一法进行。
- **文献对标**：未见统一的多分布测试矩阵和同源泄漏控制。

## 修正说明（对齐旧项目纪律）

基于旧项目 `meeting_trace.md` 的纪律，对本提案进行以下修正：

1. **核心权重归一化**：不平衡 AMP 二分类中，优先考虑 AUPRC、MCC、Recall/Sensitivity、Precision。核心权重总和必须为 1.0。
   - 默认推荐：AUPRC 0.35, MCC 0.30, Recall/Sensitivity 0.20, Precision 0.15。
2. **强制报告但不赋权**：为了与文献对标，必须报告 ACC、Specificity、AUROC、F1，但不参与主权重排序。
3. **阈值策略**：不能默认 0.5。必须基于验证集，使用 Max MCC 或 Max Youden Index 确定每个模型的最佳阈值，并在测试集上固定使用。
4. **同源泄漏控制**：必须使用 CD-HIT 或 MMseqs2 进行序列去冗余（推荐 40% 或 70% 同一性），并使用 StratifiedGroupKFold 或按序列聚类分组交叉验证，避免训练/测试集同源泄漏。
5. **多分布测试矩阵**：为全面评估模型在不同类别不平衡条件下的表现，推荐构建以下测试集：
   - 1:1 平衡测试集
   - 1:10 轻度不平衡测试集
   - 1:100 重度不平衡测试集
   - 低同源独立集（与训练集序列同一性 < 40% 或来源于不同数据库）
6. **统计检验**：建议在最终排名时采用 McNemar 检验或 Wilcoxon 符号秩检验，评估模型间差异的显著性。

修正后的指标体系和实验设计更符合 AMP 领域基准测试的严谨性要求，同时兼顾了文献对标需求。
- All models must be evaluated under the same multi-distribution test matrix to ensure fair comparison.
- Thresholds must be individually optimized per model on a validation set, not defaulted to 0.5.
- Strict homology leakage control is required to avoid overoptimistic performance estimates.
- Core metrics (AUPRC, MCC, Recall, Precision) should be weighted to rank models, while other metrics are reported for reference.
- Statistical significance testing between models is recommended for final ranking.
- Should we require McNemar's test or Wilcoxon signed-rank test for pairwise model comparisons?
- How to handle models that output only class labels without probability scores (threshold optimization impossible)?

## ⚖️ Agent 3 (Critic) 深度质疑
- **Critic 深度审查 — 终审裁决**

审查对象：Model/ Dataset/ Metric 三线提案。

**总体评估**：提案方向正确，但证据链存在多处致命缺陷，模型清洗尚不彻底，数据集无法支撑独立 benchmark，代表模型选择存在严重逻辑矛盾。下文逐一列出必须执行的裁决与修正清单，零容忍原则下，三线结论均需大幅回退重梳。

---

### 1. 模型清洗与权重生死线

**致命缺陷**：
- `Co-AMPpred`, `iAMPCN`, `ACEP`, `Macrel` 四个模型被标记为 `benchmark_ready`，但无一满足“权重 + 批量推理脚本”两项硬性要求。
  - `Co-AMPpred`：GitHub 代码存在，但未见预训练权重文件，且未提供批量推理脚本。
  - `iAMPCN`：代码存在，权重未确认，缺少批量推理入口。
  - `ACEP`：论文 DOI 不匹配，无法确认正确来源，且权重未验证。
  - `Macrel`：仅有代码下载链接，无 explicit dataset，且为传统 RF 模型，无权重概念，但必须提供训练好的模型对象或可复现脚本。
- **裁决**：上述四个模型一律降级为 `needs_weight_check` 或 `needs_reproducible_script`，不得进入 `benchmark_ready` 列表。
- `Deep-AmPEP30`, `RF-AmPEP30` 等明确无权重报告，必须保持 `needs_weight_check`。
- `Multi-label WKnn-MLR` 无代码、无 web server，无法进行批量推理，**必须排除出 benchmark 候选**。
- `AMP prediction server (biosino)`, `AMP Scanner v2` 列为 webserver_only，但按纪律，webserver-only 工具只能在提供 API 且有严格速率限制的前提下作为辅助比较，绝不可进入主榜。
- `ACPred`, `AMPfun`, `AntiCP`, `AntiCP2.0`, `iAMPpred` 等既无代码又无原始论文，**必须立即排除**，不可保留为候选。

**跨界污染复查**：
- ACP-DL 虽在 Scout 中提及降级，但最终列表未体现，需确认是否已彻底排除（抗癌肽预测）。
- `cdGAN` 被列为生成式模型，且已 out_of_scope，但错误地出现在 architecture 代表模型 `Transformer / LLM 主导模型` 中，**必须移除**。
- `MultiPep` 未确认 AMP 分类，应排除或标记为 out_of_scope。

### 2. 数据集链致命漏洞

**仅有的“数据集”只是挂名，无独立测试集**：
- `model_dataset_links` 中仅有 `Co-AMPpred` 数据集和 `APD (May 2016)` 等，但它们均为训练集或来源数据库，**没有一套独立的、带有明确负样本的二分类测试集**。
- `APD3` 为 AMP 阳性数据库，不能作为测试集。`Wang et al. 2011` 数据集未公开，不可用。
- **负样本来源完全缺失**：Co-AMPpred 数据集虽包含正负样本，但负样本的具体来源、去冗余处理、是否含 ACP/AIP/AVP 等跨界肽均未说明，存在污染风险。
- 提案中提到了“多分布测试矩阵”，但未提供任何真实数据集的构建方案和链接，仅是空谈。

**裁决**：
- 任何 `benchmark_ready` 标记必须取消，直到每个模型都明确指定一个独立的、可下载的、包含正负样本的测试集，并且负样本来源经过审查。
- 必须构建至少一个基准测试集，例如从 APD3 取阳性，从 UniProt 随机抽取非抗菌肽，并经过 CD-HIT 去冗余。
- 对现有 `Co-AMPpred` 数据集，需立即核实负样本的组成和去冗余步骤。

### 3. 指标与实验设计

指标专家提案基本合理，ALIGN 了旧纪律，但缺少两个关键点：
- **置信区间**：必须要求所有核心指标报告 95% 置信区间（如通过 bootstrap），否则无法判断微小差异的显著性。
- **对无法输出概率的模型**：如某些仅输出类别标签的模型，应明确排除或降级处理，因为无法进行阈值优化，违反纪律。

### 4. 代表模型选择充满矛盾

代理模型分类存在严重不一致：
- `representation` 分类中，`传统理化/统计特征` 选用了 `Co-AMPpred` 和 `Macrel`，但 `ACEP` 同样使用传统特征（氨基酸组成+PSSM）却被忽略，却在 `architecture` 中作为 CNN+RNN 代表，造成认知割裂。
- `多模态 / 混合表示` 类仅选了 `AMP Scanner v2`，然而该模型为 webserver-only，未验证架构，不应作为代表。
- `结构/图表示` 类为空，而 `SSFGM-Model` 被归入 PLM 表示，但其 paper 未确认，不可靠。
- `cdGAN` 被错误放入 architecture 代表，必须移除，该位置应由真正的 Transformer 判别模型填补（如 AMP-BERT 等，但当前未提供）。

**裁决**：必须重新分配代表模型，确保每个类别至少有一个真实的、可复现的模型，且无重复归类。

### 5. 执行清单

1. 立即将所有 `benchmark_ready` 模型降级为 `needs_weight_check`，并制定权重验证 checklist。
2. 删除所有 webserver-only 且无 API 的模型，或标记为 `excluded`。
3. 清除 `Multi-label WKnn-MLR` 等无任何可执行代码的模型。
4. 重构数据集部分，要求提供至少一个完整测试集（含正负样本、去冗余、下载链接）。
5. 修正代表模型映射，确保 `cdGAN` 等不混入二分类架构代表。
6. 在指标政策中强制要求置信区间。
7. 排查所有命名混淆（如 CAMPR3 vs CAMPR34），统一为规范名称。

不执行以上条目，所有模型/数据集/指标结论均不可进入下一阶段。
- The entire benchmark pipeline is blocked until at least one model with full inference code, weights, and a clean test set is established.
- All model selection and ranking will be unreliable if homology leakage and negative sample contamination are not strictly controlled.
- The current dataset list cannot support the multi-distribution test matrix; a new dataset construction phase is required.
- Can we obtain weights for Co-AMPpred, iAMPCN, ACEP?
- What is the exact negative set composition for Co-AMPpred dataset?
- Is there any PLM-based AMP classifier with publicly available code and weights?
- How to construct an independent test set with verified negative peptides?
- How to handle models that cannot be executed locally (e.g., webserver-only)?
- What is the correct paper and code for SSFGM-Model?

## 🛡️ Agent 1 (Scout) 辩护与修正
- 接受 Critic 对跨界模型、生成式模型、纯工具管线、无权重模型的降级要求。
- 本轮按两套分类体系整理：`Representation` 用于理解输入表示，`Architecture` 用于安排复现路线和工程依赖。
- 不删除候选模型；但重复模型按 canonical model name 去重，并保留证据更强的一条记录。
- 对所有缺失数据集 URL 的模型保留 `dataset_followup_tasks`，后续继续查 full text、supplementary、GitHub README、Zenodo/Figshare/Dryad/DataCite。

## 🛡️ Agent 2 (Metrics) 辩护与修正
- 保留核心决策指标：AUPRC、MCC、Recall/Sensitivity、Precision。
- 为了和文献对标，ACC、Specificity、AUROC、F1 不参与主权重但必须报告。
- 对二分类 AMP benchmark，优先采用多分布测试矩阵：平衡、轻度不平衡、重度不平衡、低同源独立集。

## ⚖️ Agent 3 (Critic) 终审点评
- 模型端：允许保留全量候选，但进入主 benchmark 前必须通过纯 AMP 二分类、代码/权重、数据集、可批量推理四项核查。
- 分类端：Representation 和 Architecture 两套分类不得混在一起；同一个模型可以同时有一个表示类别和一个架构类别。
- 去重端：同名/别名模型只保留一条规范记录，禁止在代表模型中重复出现同一模型。
- 工程端：下一步应围绕每类代表模型先做仓库可运行性核查，再逐步扩展到全量候选。

## 📜 Final Consensus / 执行清单
1. 保留 `All Candidate Models` 作为全量情报池，不因证据弱而删除。
2. `Benchmark Ready Models` 只作为优先复现/评测队列，仍需执行权重和推理命令核查。
3. 按 `Representation` 和 `Architecture` 两套体系分别选择每类 1-2 个代表模型先跑通。
4. 所有模型表按 canonical model name 去重，避免 Co-AMPpred、AMP-BERT、CalcAMP 等重复行。
5. 数据集继续以 `Model-Dataset Links` 和 `Dataset Follow-up Tasks` 追踪，不再只看单个 dataset 字段。
6. 会议结论写入 memory.md，原始 Agent JSON 仍保存在 `data/deepseek_meeting_raw.jsonl`。

### evidence_compressor_agent

- **Role**: 按模型名称 / 主题 / 来源分块压缩 evidence
- **Status**: ok
- **Counts**: chunk_summary_count=42, paper_count=783, source_counts={'crossref': 239, 'europe_pmc': 177, 'openalex': 246, 'pubmed': 150, 'pubmed_similar': 12, 'semantic_scholar': 20, 'semantic_scholar_citations': 4, 'semantic_scholar_references': 3}
- **Discussion / key decisions**:
  - 每个 chunk 保留 PMID/PMCID/DOI/title/url/source 等可追溯证据。
  - 未发现数据集链接时不删除模型，而是记录 dataset_status 与 followup task。

### model_dataset_agent

- **Role**: 全局合并模型、数据集、代码仓库，并判断 benchmark 候选状态
- **Status**: ok
- **Counts**: models=43, all_candidate_models=43, benchmark_ready_models=5, repositories=11, datasets=4, dataset_links=2, model_dataset_links=3, dataset_followup_tasks=3, model_classification=10, representative_models_by_category=1, open_questions=8
- **Discussion / key decisions**:
  - ACEP paper DOI mismatch: need to locate correct paper.
  - Are Deep-AmPEP30 and RF-AmPEP30 from the same AmPEP30 suite?
  - Can we obtain weights for iAMPCN and ACEP?
  - What is the exact negative set for Co-AMPpred dataset?
  - Are there any PLM-based AMP classifiers with available code? (e.g., ProteoGPT, PepNet)

### metric_agent

- **Role**: 全局整理评价指标、外部验证、推荐 benchmark 指标
- **Status**: ok
- **Counts**: metrics=9, benchmark_implications=5, open_questions=5
- **Discussion / key decisions**:
  - All models must be evaluated under the same multi-distribution test matrix to ensure fair comparison.
  - Thresholds must be individually optimized per model on a validation set, not defaulted to 0.5.
  - Strict homology leakage control is required to avoid overoptimistic performance estimates.
  - Core metrics (AUPRC, MCC, Recall, Precision) should be weighted to rank models, while other metrics are reported for reference.
  - Statistical significance testing between models is recommended for final ranking.
  - Should we require McNemar's test or Wilcoxon signed-rank test for pairwise model comparisons?
  - How to handle models that output only class labels without probability scores (threshold optimization impossible)?
  - What is the acceptable minimum sequence identity for the independent test set (e.g., 40% vs 70%)?

### critic_agent

- **Role**: 审查重复模型、证据不足、链接不确定、数据集缺失
- **Status**: ok
- **Counts**: benchmark_implications=3, open_questions=7
- **Discussion / key decisions**:
  - The entire benchmark pipeline is blocked until at least one model with full inference code, weights, and a clean test set is established.
  - All model selection and ranking will be unreliable if homology leakage and negative sample contamination are not strictly controlled.
  - The current dataset list cannot support the multi-distribution test matrix; a new dataset construction phase is required.
  - Can we obtain weights for Co-AMPpred, iAMPCN, ACEP?
  - What is the exact negative set composition for Co-AMPpred dataset?
  - Is there any PLM-based AMP classifier with publicly available code and weights?
  - How to construct an independent test set with verified negative peptides?
  - How to handle models that cannot be executed locally (e.g., webserver-only)?

### chief_agent

- **Role**: 合并三位 Agent 输出为长期记忆 JSON；不得删除候选模型，只能分类
- **Status**: ok
- **Counts**: models=74, all_candidate_models=116, benchmark_ready_models=95, repositories=316, datasets=101, dataset_links=68, model_dataset_links=321, dataset_followup_tasks=238, model_classification=12, representative_models_by_category=21, metrics=42, papers=65, benchmark_implications=10, open_questions=175
- **Discussion / key decisions**:
  - `{
"topic": "Model inclusion criteria",
"decision": "Models must provide pre-trained weights (or reproducible training script) and a batch inference script to be considered for main benchmark. All current candidates downgraded to needs_weight_check or needs_verification.",
"reason": "Without verified weights and batch inference, fair automated evaluation is impossible.",
"evidence": "Critic review of provided repositories; none meet the full criteria."
}`
  - `{
"topic": "Dataset construction",
"decision": "No current dataset qualifies as an independent test set. A new benchmark dataset must be constructed from APD3 (positive) and UniProt (negative), deduplicated and with verified negative sample purity.",
"reason": "APD3 is positive-only; Co-AMPpred dataset requires audit; Wang et al. 2011 not public.",
"evidence": "Critic dataset quality review."
}`
  - `{
"topic": "Metric weighting",
"decision": "Core ranking metrics: AUPRC (0.35), MCC (0.30), Recall (0.20), Precision (0.15). Additionally report Accuracy, Specificity, AUROC, F1 without weight.",
"reason": "Imbalanced AMP data; AUPRC and MCC better reflect real-world performance.",
"evidence": "Literature and prior project discipline."
}`
  - `{
"topic": "Threshold optimization",
"decision": "Each model must determine its optimal threshold on validation set using Max MCC or Max Youden Index; default 0.5 is banned.",
"reason": "Balanced accuracy is critical; threshold should be optimized per model.",
"evidence": "Meeting trace discipline."
}`
  - `{
"topic": "Confidence intervals",
"decision": "95% confidence intervals via bootstrap must be reported for all core metrics.",
"reason": "To assess statistical significance of differences between models.",
"evidence": "Critic suggestion."
}`
  - `{
"question": "What is the deep learning hybrid model used in PMID 41731616? It is referenced as [7] but not described in the available evidence.",
"reason": "chunk_summary_uncertainty",
"next_action": "manual_or_followup_search"
}`
  - `{
"question": "ACPred original paper not found; only usage evidence from a benchmark study.",
"reason": "chunk_summary_uncertainty",
"next_action": "manual_or_followup_search"
}`
  - `{
"question": "All listed webservers lack original paper links, code, or dataset details in this chunk.",
"reason": "chunk_summary_uncertainty",
"next_action": "manual_or_followup_search"
}`

## Model Classification Overview

### 数据/输入表示（Representation）

|类别|类别特点|具体模型|每类代表模型 1-2 个|当前证据池命中数|
|---|---|---|---|---:|
|传统理化/统计特征为主|以全局理化性质、氨基酸组成/伪氨基酸组成、k-mer 统计等手工特征为主，不显式保留序列位置，只做整体向量化。|Macrel, RF-AmPEP30, CAMPR34, CLASSAMP5, DBAASP6, AmPEP, CAMPR3, DBAASPv3.0, CAMPR3(RF), CAMPR3(SVM), MLAMP, AntiBP, AntiBP2, Multi-label weighted KNN-MLR model, ISCAPE, AMP Scanner v2, StackAMP, AMPlify_bal, AMPlify_imbal, AMPer, CAMP, AVPpred, MetaPepticon, Macrel (BigDataBiology), Macrel (MacReloader), Macrel (macrelay), AmPEP (amPEPpy), AmPEP (Ampep_Python), AmPEP (ShirleyWISiu), APD3|Macrel, AmPEP|34|
|纯序列/编码表示|直接对氨基酸序列做编码，如 one-hot、索引 embedding、PC6 理化编码、PseKRAAC 降维编码，或把序列转成小图像；不依赖大型 PLM，也不显式用 3D 结构。|ACPred, AMPfun, AntiCP, AntiCP2.0, iAMPpred, HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP, AllerCatPro, AxPEP3, Deep-AmPEP30, ADAM, APSvr.2, DBAASP, BAGEL3, BACTIBASE, ADAM (prediction tool), ADMETlab 3, AMP MIC predictor (CNN/RNN), AxPEP, AMPlify, hydramp, PrefixProt, HydraAMP, Venomics artificial intelligence, hydramp (conda-feedstock), hydramp (pytorch port)|AMPlify, APEX|70|
|蛋白语言模型（PLM）表示|使用预训练蛋白语言模型（BERT/T5/ESM/ProtT5/GPT-2 等）从序列生成高维 embedding，再接 CNN/MLP 等下游分类器。|cdGAN, AMPGenix, PLUM, c_AMPs-prediction|c_AMPs-prediction, PLUM|4|
|结构/图表示|将肽构造成图：节点可以是原子、残基或 k-mer；边来自共价键、空间距离或共现关系；有的结合预测 3D 结构，节点特征可叠加 ESM embedding。|AMP prediction server (biosino), AMP-GSM, PeptideRanker, WeightedEnsemble_L3 (Anti_Cp), WeightedEnsemble_L3, Lab, Co-AMPpred|Lab, PeptideRanker|7|
|多模态 / 混合表示|同时使用两种及以上类型的输入，如 one-hot 序列 + 大量理化特征，或 PLM embedding + 手工 PD 特征等。|Deep learning hybrid model (unnamed)|Deep learning hybrid model (unnamed)|1|

### 模型架构（Architecture）

|类别|类别特点|具体模型|每类代表模型 1-2 个|当前证据池命中数|
|---|---|---|---|---:|
|机器学习模型|基于特征工程 + 传统分类器/回归器，如 Random Forest、SVM、LightGBM、逻辑回归等。|Macrel, RF-AmPEP30, CAMPR34, CLASSAMP5, DBAASP6, AmPEP, CAMPR3, CAMPR3(RF), CAMPR3(SVM), MLAMP, AntiBP, AntiBP2, AMP prediction server (biosino), Multi-label weighted KNN-MLR model, ISCAPE, AMP Scanner v2, StackAMP, AMPlify_bal, AMPlify_imbal, CAMP, AVPpred, MetaPepticon, Macrel (BigDataBiology), Macrel (MacReloader), Macrel (macrelay), AmPEP (amPEPpy), AmPEP (Ampep_Python), AmPEP (ShirleyWISiu), APD3, AVCpred|Macrel, AmPEP|32|
|CNN 主导模型|一维/二维卷积是主干，负责自动学习局部 motif 与局部模式，有时堆叠多层 CNN（DenseNet、VGG、ResNet 等）。|Deep learning hybrid model (unnamed), Deep-AmPEP30, cdGAN, AMP MIC predictor (CNN/RNN), hydramp, HydraAMP, Deep-AmPEP30 web server, iAMPCN|Deep-AmPEP30, iAMPCN|8|
|RNN/LSTM 主导模型|以（双向）LSTM/GRU 为主干，建模序列顺序依赖；注意力层通常作为辅助模块。|AMPlify, AMPlify (AWS Amplify JS), AMPlify (AWS Amplify CLI), AMPlify (Jekyll AMP theme), c_AMPs-prediction, AMPlify GitHub|AMPlify, c_AMPs-prediction|6|
|CNN + RNN 混合模型|先用 CNN 抽局部 motif，再用 LSTM/BiLSTM 建模长程依赖，最后接全连接/分类器。|||0|
|Transformer / LLM 主导模型|主干是多头自注意力/Transformer 模块（包括 GPT-2、BERT、ProtT5 等大模型），或在下游显式堆叠 Transformer block、cross-attention 作为核心特征提取器。|AMPGenix, PLUM, ApexGO|AMPGenix, PLUM|3|
|图神经网络（GNN）模型|将肽建模为图结构（原子/残基/k-mer 为节点，键/空间距离/共现为边），使用 GCN、GAT 等进行 message passing。|AMP-GSM, PeptideRanker, WeightedEnsemble_L3 (Anti_Cp), WeightedEnsemble_L3, Lab, Co-AMPpred|Lab, PeptideRanker|6|
|其他（多阶段流水线 / 集成框架）|用多个模型串联或集成，或多模型 + 集成/堆叠。|ACPred, AMPfun, AntiCP, AntiCP2.0, iAMPpred, HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP, AllerCatPro, AxPEP3, ADAM, APSvr.2, DBAASPv3.0, DBAASP, BAGEL3, BACTIBASE, ADAM (prediction tool), ADMETlab 3, AxPEP, PrefixProt, AMPer, Venomics artificial intelligence, hydramp (conda-feedstock), hydramp (pytorch port), APEX, MAPLE, AmPEP web server|APEX, ADAM (prediction tool)|61|

## Representative Models by Category

### 数据/输入表示（Representation）

|category_title|model_name|task_type|method_family|code_repository_url|web_server_url|dataset_source_or_link|source_pmid|source_doi|evidence_level|why_representative|
|---|---|---|---|---|---|---|---|---|---|---|
|传统理化/统计特征为主|Macrel|AMP prediction|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|40891852|10.1128/spectrum.01504-25|fulltext|按用户指定的 传统理化/统计特征为主 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|传统理化/统计特征为主|AmPEP|AMP prediction|ML|https://github.com/ShirleyWISiu/AmPEP|http://cbbio.cis.umac.mo/software/AmPEP/|Collected from CAMP, APD, UniProt (3268 AMPs, 166791 non-AMPs); benchmarked on iAMPpred and iAMP-2L datasets|29374199|10.1038/s41598-018-19752-w|fulltext|按用户指定的 传统理化/统计特征为主 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|纯序列/编码表示|AMPlify|antimicrobial peptide classification|DL|https://github.com/bcgsc/AMPlify|not_reported_in_available_evidence|UniProtKB/Swiss-Prot (used as mining source)|40100125|10.1093/nar/gki524|fulltext|按用户指定的 纯序列/编码表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|纯序列/编码表示|APEX|AMP prediction (MIC prediction)|DL|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (training data not described, in-house peptides mentioned)|39764027|10.1101/2024.12.17.628923|fulltext|按用户指定的 纯序列/编码表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|蛋白语言模型（PLM）表示|c_AMPs-prediction|AMP prediction|DL|https://github.com/mayuefine/c_AMPs-prediction|not_reported_in_available_evidence|https://github.com/mayuefine/c_AMPs-prediction|41164228|10.3389/fvets.2025.1689589|fulltext|按用户指定的 蛋白语言模型（PLM）表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|蛋白语言模型（PLM）表示|PLUM|antimicrobial peptide generation and classification|DL|https://github.com/priyamayur/PLUM|not_reported_in_available_evidence|Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository|42124643|10.64898/2026.02.21.707214|fulltext|按用户指定的 蛋白语言模型（PLM）表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|结构/图表示|Lab|||https://github.com/google-deepmind/lab||not_reported_in_available_evidence|||github_search|按用户指定的 结构/图表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|结构/图表示|PeptideRanker|general peptide bioactivity prediction (including antimicrobial)|DL|not_reported_in_available_evidence|http://bioware.ucd.ie/|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)|23056189|10.1371/journal.pone.0045012|fulltext|按用户指定的 结构/图表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|多模态 / 混合表示|Deep learning hybrid model (unnamed)|AMP prediction|deep learning (hybrid)|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|41731616|10.1186/s40168-025-02326-0|fulltext|按用户指定的 多模态 / 混合表示 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|

### 模型架构（Architecture）

|category_title|model_name|task_type|method_family|code_repository_url|web_server_url|dataset_source_or_link|source_pmid|source_doi|evidence_level|why_representative|
|---|---|---|---|---|---|---|---|---|---|---|
|机器学习模型|Macrel|AMP prediction|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|40891852|10.1128/spectrum.01504-25|fulltext|按用户指定的 机器学习模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|机器学习模型|AmPEP|AMP prediction|ML|https://github.com/ShirleyWISiu/AmPEP|http://cbbio.cis.umac.mo/software/AmPEP/|Collected from CAMP, APD, UniProt (3268 AMPs, 166791 non-AMPs); benchmarked on iAMPpred and iAMP-2L datasets|29374199|10.1038/s41598-018-19752-w|fulltext|按用户指定的 机器学习模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|CNN 主导模型|Deep-AmPEP30|AMP prediction|DL|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|34867843|not_reported_in_available_evidence|fulltext|按用户指定的 CNN 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|CNN 主导模型|iAMPCN||||||||moderate|按用户指定的 CNN 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|RNN/LSTM 主导模型|AMPlify|antimicrobial peptide classification|DL|https://github.com/bcgsc/AMPlify|not_reported_in_available_evidence|UniProtKB/Swiss-Prot (used as mining source)|40100125|10.1093/nar/gki524|fulltext|按用户指定的 RNN/LSTM 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|RNN/LSTM 主导模型|c_AMPs-prediction|AMP prediction|DL|https://github.com/mayuefine/c_AMPs-prediction|not_reported_in_available_evidence|https://github.com/mayuefine/c_AMPs-prediction|41164228|10.3389/fvets.2025.1689589|fulltext|按用户指定的 RNN/LSTM 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|Transformer / LLM 主导模型|AMPGenix|other|DL|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|40891852|10.1128/spectrum.01504-25|fulltext|按用户指定的 Transformer / LLM 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|Transformer / LLM 主导模型|PLUM|antimicrobial peptide generation and classification|DL|https://github.com/priyamayur/PLUM|not_reported_in_available_evidence|Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository|42124643|10.64898/2026.02.21.707214|fulltext|按用户指定的 Transformer / LLM 主导模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|图神经网络（GNN）模型|Lab|||https://github.com/google-deepmind/lab||not_reported_in_available_evidence|||github_search|按用户指定的 图神经网络（GNN）模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|图神经网络（GNN）模型|PeptideRanker|general peptide bioactivity prediction (including antimicrobial)|DL|not_reported_in_available_evidence|http://bioware.ucd.ie/|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)|23056189|10.1371/journal.pone.0045012|fulltext|按用户指定的 图神经网络（GNN）模型 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|其他（多阶段流水线 / 集成框架）|APEX|AMP prediction (MIC prediction)|DL|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (training data not described, in-house peptides mentioned)|39764027|10.1101/2024.12.17.628923|fulltext|按用户指定的 其他（多阶段流水线 / 集成框架） 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|
|其他（多阶段流水线 / 集成框架）|ADAM (prediction tool)|AMP prediction||not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|37523405|10.1371/journal.ppat.1011508|fulltext|按用户指定的 其他（多阶段流水线 / 集成框架） 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。|

## Final Deployment Model List

|deployment_rank|deployment_tier|model_name|canonical_name|representation_category|architecture_category|task_type|method_family|code_repository_url|web_server_url|dataset_source_or_link|source_journal|citation_count|journal_impact_factor|article_impact_score|deployment_selection_score|deployment_status|deployment_reason|first_next_action|blocking_issues|evidence_level|confidence|source_pmid|source_doi|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|core_main_benchmark_top10|AmPEP|AmPEP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|https://github.com/ShirleyWISiu/AmPEP|http://cbbio.cis.umac.mo/software/AmPEP/|Collected from CAMP, APD, UniProt (3268 AMPs, 166791 non-AMPs); benchmarked on iAMPpred and iAMP-2L datasets||0.0|0.0|0.0|19.65|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.95|29374199|10.1038/s41598-018-19752-w|
|2|core_main_benchmark_top10|AntiBP2|AntiBP2|traditional_physicochemical_statistical_features|machine_learning_models|antibacterial peptide prediction|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|Antibacterial Peptide Database (APD)||0.0|0.0|0.0|19.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original paper not provided; no link available in this evidence"<br>]|fulltext|0.9|20122190|10.1186/1471-2105-11-s1-s19|
|3|core_main_benchmark_top10|PeptideRanker|PeptideRanker|structure_graph_representation|gnn_models|general peptide bioactivity prediction (including antimicrobial)|DL|not_reported_in_available_evidence|http://bioware.ucd.ie/|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)||0.0|0.0|0.0|19.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口||fulltext|0.9|23056189|10.1371/journal.pone.0045012|
|4|core_main_benchmark_top10|WeightedEnsemble_L3 (Anti_Cp)|WeightedEnsemble_L3|structure_graph_representation|gnn_models|antimicrobial peptide activity classification|ML|https://github.com/xubocheng/Anti_Cp.git|not_reported_in_available_evidence|https://github.com/xubocheng/Anti_Cp.git||0.0|0.0|0.0|19.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口||fulltext|0.9|38266820|10.1016/j.jare.2024.01.023|
|5|core_main_benchmark_top10|c_AMPs-prediction|c_AMPs-prediction|protein_language_model_representation|rnn_lstm_dominant_models|AMP prediction|DL|https://github.com/mayuefine/c_AMPs-prediction|not_reported_in_available_evidence|https://github.com/mayuefine/c_AMPs-prediction||0.0|0.0|0.0|19.5|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_model_paper_uncertain",<br>"weights_not_reported"<br>]|fulltext|0.8|41164228|10.3389/fvets.2025.1689589|
|6|core_main_benchmark_top10|AMPer|AMPer|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|feature-engineering|not_reported_in_available_evidence|http://marray.cmdr.ubc.ca/cgi-bin/amp.pl|known antimicrobial peptides (not further specified)||0.0|0.0|0.0|19.4|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original paper not provided; details rely on this reference"<br>]|fulltext|0.7|23056189|10.1371/journal.pone.0045012|
|7|core_main_benchmark_top10|Macrel|Macrel|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_needed_for_architecture_code_weights"<br>]|fulltext|0.9|40891852|10.1128/spectrum.01504-25|
|8|core_main_benchmark_top10|APSvr.2|Antimicrobial Peptide Scanner v.2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|not_reported_in_available_evidence|https://aps.unmc.edu/prediction/predict|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|37523405|not_reported_in_available_evidence|
|9|core_main_benchmark_top10|AxPEP|AxPEP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction||https://sourceforge.net/projects/axpep/|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口||fulltext|0.9|41315055|10.1007/s00248-025-02620-2|
|10|core_main_benchmark_top10|AMP Scanner v2|AMP Scanner V2|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|https://github.com/dan-veltri/amp-scanner-v2|https://www.dveltri.com/ascan/v2/ascan.html|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|core_main_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_needed"<br>]|fulltext|0.9|[<br>"41315055",<br>"40891852"<br>]|[<br>"10.1007/s00248-025-02620-2",<br>"10.1128/spectrum.01504-25"<br>]|
|11|extended_deployment_pool_11_20|AMPlify_bal|AMPlify_bal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_needed"<br>]|fulltext|0.9|40891852|10.1128/spectrum.01504-25|
|12|extended_deployment_pool_11_20|AMPlify_imbal|AMPlify_imbal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.6|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_needed"<br>]|fulltext|0.9|40891852|10.1128/spectrum.01504-25|
|13|extended_deployment_pool_11_20|Deep-AmPEP30|Deep-AmPEP30|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.5|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.8|34867843|not_reported_in_available_evidence|
|14|extended_deployment_pool_11_20|RF-AmPEP30|RF-AmPEP30|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.5|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"No weights reported"<br>]|fulltext|0.8|34867843|not_reported_in_available_evidence|
|15|extended_deployment_pool_11_20|AMP MIC predictor (CNN/RNN)|AMP-MIC-predictor-CNN-RNN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.5|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口||fulltext|0.8|37938588|10.1038/s41467-023-42434-9|
|16|extended_deployment_pool_11_20|iAMPpred|iAMPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|18.1|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.7|not_reported_in_available_evidence|not_reported_in_available_evidence|
|17|extended_deployment_pool_11_20|AMPlify|AMPlify|sequence_encoding_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|https://github.com/bcgsc/AMPlify|not_reported_in_available_evidence|UniProtKB/Swiss-Prot (used as mining source)||0.0|0.0|0.0|17.54|extended_pool_deploy_after_weight_and_data_check|经典 AMP 分类深度学习模型；有 GitHub，适合作为纯序列/RNN-Attention 基线。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"code_not_found"<br>]|fulltext|0.9|40100125|10.1093/nar/gki524|
|18|extended_deployment_pool_11_20|CAMPR3(RF)|CAMPR3(RF)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|16.4|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"needs original paper verification"<br>]|review|0.5|28203715|10.1093/bioinformatics/btx081|
|19|extended_deployment_pool_11_20|AntiBP|AntiBP|traditional_physicochemical_statistical_features|machine_learning_models|antibacterial peptide prediction|ML|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|16.4|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"needs original paper verification"<br>]|review|0.5|28203715|10.1093/bioinformatics/btx081|
|20|extended_deployment_pool_11_20|ADAM|ADAM|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|not_applicable|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||0.0|0.0|0.0|16.3|extended_pool_deploy_after_weight_and_data_check|证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。|检查仓库是否有可批量推理脚本；确认预训练权重或训练复现脚本；下载并标准化对应数据集；封装为统一 predict_proba(input_fasta) 接口|[<br>"needs original paper verification"<br>]|review|0.4|28203715|10.1093/bioinformatics/btx081|

## Final Recommended Datasets

|dataset_rank|dataset_name|linked_model|recommended_role|dataset_source_or_link|why_selected|required_cleaning|source_pmid|source_doi|evidence_level|status|
|---|---|---|---|---|---|---|---|---|---|---|
|1|iAMP-SeE Dataset / Zenodo|iAMP-SeE|primary benchmark candidate / 主测试集候选|需要从仓库/Zenodo/补充材料中确认|当前证据中来源最清楚，包含 DRAMP、dbAMP、CAMPr-4、AMPfun、ADAPTABLE、UniProt 负样本，并有 Zenodo 线索；最适合作为第一版 benchmark 的主数据集候选。|核查 Zenodo 文件；统一 FASTA/CSV 格式；确认正负标签；用 CD-HIT/MMseqs2 去冗余；过滤与模型训练集高度同源序列。||||recommended_top3_dataset_needs_cleaning_and_version_lock|
|2|Co-AMPpred / DEEP-AmPEP30 derived dataset|Co-AMPpred|classic comparison set / 经典对照测试集|需要从仓库/Zenodo/补充材料中确认|Co-AMPpred 有 GitHub 和 DEEP-AmPEP30 衍生数据线索，适合做传统 ML 与经典 AMP benchmark 对照。|核查 GitHub 中正负样本数量、负样本来源和去重方式；排除训练集重叠；补充数据集版本记录。|||moderate|recommended_top3_dataset_needs_cleaning_and_version_lock|
|3|AMP-BERT GitHub dataset|AMP-BERT|PLM reproduction set / PLM 模型复现测试集|需要从仓库/Zenodo/补充材料中确认|AMP-BERT 与 PLM 模型直接配套，GitHub 中有代码和数据线索，适合验证 PLM 路线和复现 AMP-BERT。|核查数据文件、标签列和训练/测试划分；拆分出外部测试集；做低同源过滤，避免 PLM 模型过拟合历史划分。||||recommended_top3_dataset_needs_cleaning_and_version_lock|

## Final Metrics Plan

### 主排名指标

|指标|权重|用途|
|---|---:|---|
|AUPRC|0.35|主指标；适合不平衡二分类，优于只看 AUROC。|
|MCC|0.3|综合 TP/TN/FP/FN，对类别不平衡更稳健。|
|Recall / Sensitivity|0.2|控制 AMP 漏检，适合发现任务。|
|Precision|0.15|控制假阳性，避免大量错误候选进入后续实验。|

### 强制报告指标

Accuracy, Specificity, AUROC, F1-score, Confusion Matrix

### 阈值与测试矩阵

- 阈值策略：在验证集上用 Max MCC 或 Max Youden Index 确定阈值；测试集禁止后验调阈值，禁止默认固定 0.5。
- 测试矩阵：1:1 balanced test, 1:10 mild imbalance test, 1:100 severe imbalance test, low-homology independent test
- 统计报告：95% bootstrap confidence interval, paired bootstrap or McNemar test for model comparison

## All Candidate Models

|model_name|canonical_name|representation_category|architecture_category|task_type|method_family|source_pmid|source_doi|code_repository_url|web_server_url|dataset_source_or_link|benchmark_candidate|blocking_issues|evidence_level|confidence|chunk_id|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Co-AMPpred|Co-AMPpred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|34330209|10.1186/s12859-021-04305-2|https://github.com/onkarS23/CoAMPpred|not_reported_in_available_evidence|https://github.com/onkarS23/CoAMPpred (contains training and test data from DEEP-AmPEP30)|True|[<br>"pre-trained weights not confirmed; may require training from scratch"<br>]|fulltext|0.9|model_2020_peptidomics|
|CTCM-Neo & ConformaX-PEP framework|CTCM-Neo & ConformaX-PEP|protein_language_model_representation|gnn_models|antimicrobial peptide classification (antimalarial)|DL|41859462|10.3389/fcimb.2026.1707267|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (likely derived from APD3)|True|[<br>"no code repository link",<br>"no full text available",<br>"antimalarial-specific may limit general AMP benchmark"<br>]|abstract|0.6|model_2020_peptidomics|
|A-CaMP|A-CaMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification / anti-cancer peptide prediction|sequence alignment-based / fingerprinting|31870207|10.1080/07391102.2019.1708796|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, no dataset",<br>"task boundary unclear (also predicts anticancer peptides)"<br>]|fulltext|0.8|model_a_camp|
|PCSPred|PCSPred|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|40781463|10.1109/NEleX59773.2023.10421222|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no_code_available",<br>"no_full_text",<br>"no_dataset_details"<br>]|abstract|0.6|model_aagp|
|iAMPCN|iAMPCN|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|37369638|10.1093/bib/bbad240|https://github.com/joy50706/iAMPCN|not_reported_in_available_evidence|Integrated from multiple databases (APD3, dbAMP, DRAMP, etc.) and UniProt for negatives.|True|[<br>"original_model_article_not_this_one",<br>"dataset_not_specified"<br>]|fulltext|0.95|model_amplify|
|SSFGM-Model|SSFGM-Model|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide classification|DL|40462515|10.1186/s12864-020-06978-0|https://github.com/ggcameronnogg/SSFGM-Model|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"evidence only from abstract; full text mismatch suspected",<br>"pre-trained weights not confirmed",<br>"dataset not reported"<br>]|abstract|0.8|model_acep|
|ACEP|ACEP|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP recognition|DL|40462515|10.1186/s12864-020-06978-0|https://github.com/Fuhaoyi/ACEP|not_reported_in_available_evidence|APD database (mentioned in fulltext)|True|[<br>"pre-trained weights not confirmed"<br>]|fulltext|0.9|model_acep|
|ACP-DL|ACP-DL|traditional_physicochemical_statistical_features|cnn_dominant_models|anticancer peptide prediction|deep learning|34880291|10.1038/s41598-021-02703-3|https://github.com/haichengyi/ACP-DL|https://anticancer.pythonanywhere.com/|not_reported_in_available_evidence|False|[<br>"targets anticancer peptides, not antimicrobial peptides"<br>]|repository|0.7|model_acp_dl|
|MultiPep|MultiPep|sequence_encoding_representation|cnn_dominant_models|multi-label peptide bioactivity classification (potential AMP prediction)|DL|34909478|10.1093/biomethods/bpab021|not_reported_in_available_evidence|not_reported_in_available_evidence|multiple public databases (not specified in abstract)|True|[<br>"The abstract does not explicitly list AMP among the 20 bioactivity classes; needs full-text verification."<br>]|abstract|0.5|model_acp_ope|
|iAMP-2L|iAMP-2L|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original code not available",<br>"Web server not reported in evidence"<br>]|fulltext|0.6|model_ann_based_amp_prediction_model_ref_4|
|iAMPred|iAMPred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, review only"<br>]|review|0.5|model_adam|
|AmPEP|AmPEP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, review only"<br>]|review|0.5|model_adam|
|AntiBP2|AntiBP2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antibacterial peptide prediction|web-server|37914524|10.24272/j.issn.2095-8137.2023.246|not_reported_in_available_evidence|https://webs.iiitd.edu.in/raghava/antibp2/|not_reported_in_available_evidence|True|[<br>"needs_original_publication"<br>]|fulltext|0.5|model_amppred|
|CAMPR3|CAMPR3|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, webserver only",<br>"non-reproducible locally"<br>]|review|0.5|model_adam|
|ADAM|ADAM|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|http://bioinformatics.cs.ntou.edu.tw/ADAM|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.6|model_adam|
|DBAASP|DBAASP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide activity prediction|web-server|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|model_ampir|
|MLAMP|MLAMP|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original code not available"<br>]|fulltext|0.6|model_ann_based_amp_prediction_model_ref_4|
|CAMP|CAMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, webserver only",<br>"only described in review"<br>]|review|0.5|model_adam|
|ClassAMP|ClassAMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|web-server|37914524|10.24272/j.issn.2095-8137.2023.246|not_reported_in_available_evidence|http://www.bicnirrh.res.in/classamp/|not_reported_in_available_evidence|True|[<br>"needs_original_publication"<br>]|fulltext|0.5|model_amppred|
|AVPpred|AVPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|unknown|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, review only"<br>]|review|0.5|model_adam|
|AMPER|AMPER|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|unknown|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, review only"<br>]|review|0.5|model_adam|
|EFC-FCBF|EFC-FCBF|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|AMP prediction|feature-engineering|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, review only"<br>]|review|0.5|model_adam|
|AMPlify|AMPlify|sequence_encoding_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|35078402|10.1186/s12864-022-08310-4|https://github.com/bcgsc/AMPlify|not_reported_in_available_evidence|Training data from APD, CAMP, etc. (details in paper); test set from Bullfrog genome and other sources.|True|[<br>"preprint not yet peer-reviewed, full text verification needed"<br>]|fulltext|0.95|model_amplify|
|E-CLEAP|E-CLEAP|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38722967|10.1371/journal.pone.0300125|https://github.com/Wangsicheng52/E-CLEAP|not_reported_in_available_evidence|https://github.com/Wangsicheng52/E-CLEAP|True|[<br>"incomplete evidence",<br>"source paper unknown"<br>]|fulltext|0.95|model_amp_scanner|
|UniproLcad|UniproLcad|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/harkic/UniproLcad|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.7|model_adam|
|TriStack|TriStack|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/hjy23/TriStack|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.7|model_adam|
|iAMP-DL|iAMP-DL|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/mldlproject/2022-iAMP-DL|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.7|model_adam|
|amp-gan|amp-gan|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://gitlab.com/vail-uvm/amp-gan|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.7|model_adam|
|AVPIden|AVPIden|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|http://awi.cuhk.edu.cn/AVPIden/|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.6|model_adam|
|antibp|antibp|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|http://www.imtech.res.in/raghava/antibp/|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.6|model_adam|
|ampsphere|ampsphere|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction / database|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|https://ampsphere.big-data-biology.org/|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.6|model_amp_gan|
|hydramp|hydramp|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|https://hydramp.mimuw.edu.pl|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.6|model_adam|
|AMPDiscover|AF-QSAM AMPDiscover|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML|34081438|10.1021/acs.jcim.1c00251|not_reported_in_available_evidence|https://biocom-ampdiscover.cicese.mx/|not_reported_in_available_evidence|True|[<br>"No code repository available; only web server provided."<br>]|abstract|0.9|model_af_qsam|
|ESM2-AFPpred|ESM2-AFPpred|protein_language_model_representation|machine_learning_models|AMP prediction / antimicrobial peptide classification|DL|35724626|10.1093/bib/bbac226|https://github.com/DongYin521/AFP_DL|not_reported_in_available_evidence|DRAMP and APD3 databases (no direct download link provided in evidence)|True|[<br>"specific to antifungal peptides; removed from main AMP benchmark",<br>"no pre-trained weights"<br>]|fulltext|0.95|model_afp_dl|
|ANIA|ANIA|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction|DL|41664908|10.1093/bib/bbag023|https://github.com/SilverGojo4/ANIA.|https://biomics.lab.nycu.edu.tw/ANIA/|DBAASP, dbAMP, DRAMP|True|[<br>"regression task (MIC), not binary AMP classification"<br>]|fulltext|0.95|model_ai4afp|
|AI4AFP|AI4AFP|protein_language_model_representation|cnn_dominant_models|antimicrobial peptide classification|ML/DL|42146199|10.1021/acsomega.6c00049|not_reported_in_available_evidence|https://axp.iis.sinica.edu.tw/AI4AFP|CAMP, DRAMP, YADAMP, SATPdb, DBAASP (AFPs); UniProtKB/Swiss-Prot (non-AMPs); DBAASP (hemolysis data)|True|[<br>"specific to antifungal peptides; removed from main AMP benchmark",<br>"no code repository"<br>]|fulltext|0.9|model_ai4afp|
|AI4AMP|AI4AMP|traditional_physicochemical_statistical_features|cnn_rnn_hybrid_models|antimicrobial peptide classification|DL|34783578|10.1128/msystems.00299-21|https://github.com/LinTzuTang/AI4AMP_predictor|http://symbiosis.iis.sinica.edu.tw/PC_6/|not_reported_in_available_evidence|True|[<br>"no code or data link",<br>"only mentioned in review"<br>]|fulltext|0.95|model_ai4amp|
|Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships|Sparse NN AMP model|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|27870247|10.1002/minf.201600029|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||abstract|0.7|model_ai4amp|
|SAMP|SAMP|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|39573886|10.1093/bfgp/elae046|https://github.com/wan-mlab/SAMP|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.95|model_ai4amp|
|DL-QSARES|DL-QSARES|traditional_physicochemical_statistical_features|cnn_dominant_models|antifungal peptide prediction/design|DL|39921483|10.1002/advs.202412488|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"code not available",<br>"only abstract evidence"<br>]|abstract|0.5|model_ai4amp|
|AI4AVP|AI4AVP|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|37626205|10.1109/JBHI.2021.3130825|https://github.com/LinTzuTang/AI4AVP_predictor|http://axp.iis.sinica.edu.tw/AI4AVP/|https://github.com/LinTzuTang/AI4AVP_predictor (datasets from APD3, DRAMP, YADAMP, DBAASP, CAMP, AVPdb, UniProt/SwissProt)|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|model_ai4avp|
|PepForge|PepForge|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|DL|39705302|10.64898/2026.05.29.728379|https://github.com/wqx1999/PepForge|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|abstract|0.9|model_al_omari_2024_amp_prediction_model|
|Al-Omari 2024 AMP prediction model|Al-Omari 2024 AMP prediction model|traditional_physicochemical_statistical_features|cnn_dominant_models|antimicrobial peptide classification|DL|39705302|10.1371/journal.pone.0315477|not_reported_in_available_evidence|not_reported_in_available_evidence|https://dbaasp.org|True|[<br>"Code not available"<br>]|fulltext|0.8|model_al_omari_2024_amp_prediction_model|
|BBATProt|BBATProt|protein_language_model_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|41212592|10.1093/bib/bbaf593|https://github.com/Xukai-YE/BBATProt|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.9|model_al_omari_2024_amp_prediction_model|
|AMAP|AMAP|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original code not available"<br>]|fulltext|0.6|model_ann_based_amp_prediction_model_ref_4|
|AMP|AMP Ensemble Model|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|ML/DL|38972032|10.1007/s12539-024-00640-z|https://github.com/researchprotein/amp|http://amp.denglab.org|https://github.com/researchprotein/amp|True||abstract|0.8|model_amp|
|Deep-AmPEP30|Deep-AmPEP30|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP prediction|DL|32464552|10.1016/j.omtn.2020.05.006|not_reported_in_available_evidence|https://cbbio.cis.um.edu.mo/AxPEP|Benchmark dataset of 188 samples (balanced); training set of 1529 positive samples from AMP databases (AmPEP, etc.)|True|[<br>"code not available",<br>"no dataset link provided"<br>]|fulltext|0.95|model_amp_toxicity_predictor|
|EBAMP|EBAMP|sequence_encoding_representation|transformer_llm_dominant_models|antimicrobial peptide design|DL|40906555|10.1016/j.celrep.2025.116215|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code or web server available",<br>"method details not fully described"<br>]|abstract|0.5|model_amp|
|DLFea4AMPGen|DLFea4AMPGen|traditional_physicochemical_statistical_features|cnn_dominant_models|antimicrobial peptide design|DL|41093853|10.1002/adma.202307680|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code or web server available"<br>]|abstract|0.5|model_amp|
|AMP-BERT|AMP-BERT|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction|DL|36461699|10.1002/pro.4529|https://github.com/GIST-CSBL/AMP-BERT.|not_reported_in_available_evidence|https://github.com/GIST-CSBL/AMP-BERT.|True||fulltext|0.95|model_amp_bert|
|COMDEL|COMDEL|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39234615|10.1016/j.apsb.2024.05.003|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.9|model_amp_bert|
|C. acnes-targeted AMP generation pipeline (activity classifier)|Dong2024_AMP_activity_classifier|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|38402320|10.1038/s41598-024-55205-3|not_reported_in_available_evidence|not_reported_in_available_evidence|https://dbaasp.org/|True|[<br>"No code or web server available",<br>"Focused on C. acnes, not general AMP prediction",<br>"Not intended as a standalone benchmark model"<br>]|fulltext|0.8|model_amp_bert|
|BERT-based AMP recognition model|Zhang2021_BERT_AMP|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|34037687|10.1093/bib/bbab200|not_reported_in_available_evidence|not_reported_in_available_evidence|Six AMP datasets (not specified in abstract) and a new constructed AMP dataset|True|[<br>"No code or web server available",<br>"Fulltext not available; evidence from abstract only",<br>"Dataset details unclear"<br>]|abstract|0.7|model_amp_bert|
|AmpGPT2|AmpGPT2|protein_language_model_representation|transformer_llm_dominant_models|other|DL|42174216|10.1038/s44259-026-00218-3|https://imigitlab.uni-muenster.de/heiderlab/ampgpt2|not_reported_in_available_evidence|COMPASS database (https://compass.imi.uni-muenster.de)|True|[<br>"Not a direct AMP activity classifier; requires external classifier for evaluation."<br>]|fulltext|0.95|model_amp_capsnet|
|AMP-CapsNet|AMP-CapsNet|structure_graph_representation|gnn_models|AMP prediction|DL|41654884|10.1186/s44342-026-00067-6|not_reported_in_available_evidence|not_reported_in_available_evidence|derived from UniProt and previous study [31]; positive: 1085 AMPs, negative: 1316 non-AMPs|True|[<br>"No code or model weights publicly available",<br>"Dataset not independently accessible"<br>]|fulltext|0.9|model_amp_capsnet|
|deepAMP|deepAMP|protein_language_model_representation|transformer_llm_dominant_models|other|DL|41753681|10.3390/microorganisms14020394|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original paper not in this batch; details sparse",<br>"No code availability reported"<br>]|fulltext|0.5|model_amp_capsnet|
|AMP-RL|AMP-RL|protein_language_model_representation|transformer_llm_dominant_models|AMP generation and optimization|DL|37992451|10.1016/j.sbi.2023.102733|https://github.com/GIST-CSBL/AMP-RL.|not_reported_in_available_evidence|PeptideAtlas, DBAASP v3 (no direct links provided)|True||fulltext|0.9|model_amp_designer|
|PepCVAE|PepCVAE|sequence_encoding_representation|cnn_dominant_models|AMP generation|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|model_amp_designer|
|PrefixProt|PrefixProt|sequence_encoding_representation|cnn_dominant_models|AMP generation / protein design|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|model_amp_designer|
|MoFormer|MoFormer|sequence_encoding_representation|transformer_llm_dominant_models|AMP generation / multi-objective optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|model_amp_designer|
|HMAMP|HMAMP|sequence_encoding_representation|cnn_dominant_models|AMP generation / multi-objective optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|model_amp_designer|
|AMP-Designer|AMP-Designer|protein_language_model_representation|transformer_llm_dominant_models|AMP generation / optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|model_amp_designer|
|AMP-MIC|AMP-MIC|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|29679519|10.1002/cmdc.201800204|https://github.com/jkwang93/AMP-Designer|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Paper title/abstract conflict with fulltext; model named AMP-MIC is part of AMP-Designer, not a standalone AMP prediction model; needs verification of original publication."<br>]|fulltext|0.7|model_amp_designer|
|AP_Sin|AP_Sin|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38416364|10.1038/s41467-018-03746-3|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"无代码仓库，训练数据未公开"<br>]|fulltext|0.7|model_amp_detector|
|AMP-Detector|AMP-Detector|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|DL/ML|39201537|10.3389/fmicb.2018.00323|not_reported_in_available_evidence|not_reported_in_available_evidence|Peptide Atlas (used for discovery)|True|[<br>"无代码仓库，训练数据描述不完整"<br>]|fulltext|0.7|model_amp_detector|
|AMP-RNNpro|AMP-RNNpro|traditional_physicochemical_statistical_features|rnn_lstm_dominant_models|AMP identification|ML/DL|38839785|10.1038/s41598-024-63461-6|not_reported_in_available_evidence|http://13.126.159.30/|not_reported_in_available_evidence (combined dataset from XUAMP, DBAASP, LAMP, DRAMP)|True|[<br>"No code repository; web server only, may not be suitable for large-scale offline benchmarking."<br>]|fulltext|0.9|model_amp_rnnpro|
|AMP-Distillation|AMP-Distillation|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction|DL|42155201|10.1016/j.compbiolchem.2026.109129|not_reported_in_available_evidence|not_reported_in_available_evidence|APD3 and DADP databases, CD-HIT deduplication|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|abstract|0.75|model_amp_distillation|
|iAMP-SeE|iAMP-SeE|protein_language_model_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|41913931|10.7717/peerj.20978|https://github.com/cqw0715/iAMP-SeE.git|not_reported_in_available_evidence|Dataset 1: DRAMP, dbAMP, CAMPr-4, AMPfun, ADAPTABLE (positive), UniProt (negative); Dataset 2: from deep-AMPpred (Zhao et al. 2024); Zenodo data: https://doi.org/10.5281/zenodo.17398951|True||fulltext|0.95|model_amp_distillation|
|STAMP|STAMP|sequence_encoding_representation|machine_learning_models|AMP activity prediction (MIC prediction)|ML/DL|42155201|10.64898/2026.05.28.728246|not_reported_in_available_evidence|not_reported_in_available_evidence|Used three benchmark datasets including two previously published and a new curated dataset from DBAASP|True|[<br>"No code available in abstract"<br>]|abstract|0.7|model_amp_distillation|
|CF-AMP prediction|CF-AMP prediction|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|42020672|10.1101/2022.11.16.516845|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code or data availability",<br>"Preprint, not peer-reviewed",<br>"Only abstract evidence"<br>]|abstract|0.5|model_amp_dualtransnet|
|AMP-DualTransnet|AMP-DualTransnet|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction|DL|42020672|10.1016/j.nexres.2026.101536|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract or full text",<br>"No code/data",<br>"Journal article with limited info"<br>]|abstract|0.3|model_amp_dualtransnet|
|AMP-FreqNet|AMP-FreqNet|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL||10.1145/3766671.3766835|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|model_amp_freqnet|
|Collaborative Filtering and Link Prediction model|Unnamed AMP prediction model (Medvedeva et al. 2023)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.1021/acs.jcim.3c00137|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|model_amp_freqnet|
|Predictive and Interpretable ML Models|Unnamed AMP prediction models (acsomega 2024)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.1021/acsomega.3c08676.s001|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract, full text, or code available; only title evidence; possibly a supporting information file"<br>]|metadata|0.3|model_amp_freqnet|
|AMP prediction ML model|Unnamed AMP prediction model (Ahmad & Garg 2024)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.54985/peeref.2405p7278831|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|model_amp_freqnet|
|GAC-BiTCNN-AMP|GAC-BiTCNN-AMP|protein_language_model_representation|cnn_dominant_models|AMP prediction|DL|41844874|10.1038/s41598-026-43370-6|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (likely dbAMP 3.0 or similar, no explicit URL)|True|[<br>"code not reported"<br>]|fulltext|0.9|model_ampgan|
|CVAE-BIO|CVAE-BIO|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML/DL|41849223|10.1093/bib/bbag115|https://github.com/scan2030|not_reported_in_available_evidence|APD3 (http://aps.unmc.edu/)|True|[<br>"code availability unclear"<br>]|fulltext|0.85|model_amp_gan|
|AMPGAN|AMPGAN|sequence_encoding_representation|cnn_dominant_models|AMP generation / prediction|DL|41463765|10.3390/antibiotics14121263|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.35|model_amp_gan|
|Macrel|Macrel|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|model_ampir|
|iAMPpred|iAMPpred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|41463765|10.3390/antibiotics14121263|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.35|model_amp_gan|
|AMP-GPT|AMP-GPT|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide generation|DL|40193623|10.1038/s44386-026-00045-6|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code or trained model weights provided",<br>"Training data details missing"<br>]|fulltext|0.85|model_amp_gpt|
|MCL-AMP|MCL-AMP|protein_language_model_representation|cnn_dominant_models|AMP prediction|DL|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"code not available",<br>"training data not reported",<br>"no external test details"<br>]|fulltext|0.7|model_amp_scanner|
|MAPLE|MAPLE|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|39792442|10.1021/acs.jcim.4c01913|https://github.com/Harkool/MAPLE|not_reported_in_available_evidence|Benchmark dataset: integrated from dbAMP, DBAASP, APD3, DRAMP, etc. (no single download link); 25,507 AMPs and 72,606 non-AMPs. Independent validation set: 24,582 AMPs, 36,653 non-AMPs.|True||fulltext|0.9|model_amp_gpt|
|PepVAE|PepVAE|sequence_encoding_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|34659152|10.3389/fmicb.2021.725727|not_reported_in_available_evidence|not_reported_in_available_evidence|https://github.com/zswitten/Antimicrobial-Peptides|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|model_amp_prediction_by_svm_lz_complexity|
|LMPred|LMPred|sequence_encoding_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|36699381|10.1101/2020.07.12.199554v3|https://github.com/williamdee1/LMPred_AMP_Prediction|not_reported_in_available_evidence|https://github.com/williamdee1/LMPred_AMP_Prediction|True|[<br>"review-level evidence",<br>"needs original paper verification"<br>]|fulltext|0.95|model_amp_prediction_by_svm_lz_complexity|
|AMP prediction SVM-LZ|AMP prediction by SVM-LZ complexity|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML|25802839|10.1093/nar/gkn823|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code or model provided",<br>"Full-text cache mismatch (APD2 database article)"<br>]|abstract|0.6|model_amp_prediction_by_svm_lz_complexity|
|DDM|DDM|protein_language_model_representation|transformer_llm_dominant_models|AMP classification|DL|41692989|10.1093/bioinformatics/btag077|https://github.com/kww567upup/DDM|not_reported_in_available_evidence|https://github.com/kww567upup/DDM (data provided in repository)|True||fulltext|0.95|model_amp_rnnpro|
|UniAMP|UniAMP|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction|DL|39799358|10.1186/s12859-025-06033-3|not_reported_in_available_evidence|https://amp.starhelix.cn|not_reported_in_available_evidence (dataset constructed from public AMP databases, no direct download link)|True|[<br>"No code repository found, only web server; reproducibility may be limited."<br>]|fulltext|0.9|model_amp_rnnpro|
|AMP Scanner|AMP Scanner|sequence_encoding_representation|cnn_rnn_hybrid_models|AMP prediction|DL|38129980|10.1002/mbo3.1393|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code or data link",<br>"only mentioned in review"<br>]|review|0.5|model_amp_scanner|
|AMP Scanner v2|Antimicrobial Peptide Scanner vr.2|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|29590297|10.1093/bioinformatics/bty179|not_reported_in_available_evidence|http://www.ampscanner.com|provided through the web server (not specified in evidence)|True|[<br>"Not original publication; limited architecture details provided."<br>]|fulltext|0.95|model_ampscanner|
|PepGen 1.0|PepGen 1.0|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide generation|DL|40643674|10.1007/s00284-025-04346-3|not_reported_in_available_evidence|https://bit.ly/2Z281cY|not_reported_in_available_evidence|True|[<br>"No source code repository found; only a shortened URL provided."<br>]|fulltext|0.8|model_amp_scanner_vr_2|
|AmPepGen|AmPepGen|sequence_encoding_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide generation|DL|40643674|10.1007/s00284-025-04346-3|https://github.com/Anorpe/ampepgen-dev|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|model_amp_scanner_vr_2|
|AMP-SEMiner|AMP-SEMiner|sequence_encoding_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|40445833|10.1016/j.celrep.2025.115773|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.9|model_amp_seminer|
|Unnamed AMP predictor from DRAMP 2.0|DRAMP_ML_model|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|31409791|10.1038/s41597-019-0154-y|not_reported_in_available_evidence|not_reported_in_available_evidence|DRAMP database (http://dramp.cpu-bioinfor.org/)|True|[<br>"Model name not provided",<br>"No code or web server link available",<br>"Not yet integrated into DRAMP as stated"<br>]|fulltext|0.5|model_amp_toxicity_prediction_model_hybrid|
|AMP toxicity prediction model (hybrid)|AMP_toxicity_predictor|structure_graph_representation|gnn_models|antimicrobial peptide toxicity prediction|ML|34758751|10.1186/s12859-021-04468-y|https://git.io/JRZaT|not_reported_in_available_evidence|DBAASP database|False|[<br>"Not an AMP prediction model",<br>"Focus on toxicity rather than antimicrobial activity classification"<br>]|fulltext|0.9|model_amp_toxicity_prediction_model_hybrid|
|CalcAMP|CalcAMP|structure_graph_representation|gnn_models|AMP prediction|ML|37107088|10.3390/antibiotics12040725|https://github.com/CDDLeiden/CalcAMP|not_reported_in_available_evidence|https://doi.org/10.5281/zenodo.7588702|True||fulltext|0.95|model_amp_toxicity_prediction_model_hybrid|
|ANN-based AMP prediction model (Torrent et al. 2011)|Torrent-2011-ANN|structure_graph_representation|gnn_models|AMP prediction|ML|21347392|10.1371/journal.pone.0016968|not_reported_in_available_evidence|not_reported_in_available_evidence|CAMP database (http://www.camp.bicnirrh.res.in/) and Uniprot; no direct download link provided|True|[<br>"No code or web server available",<br>"Uses old feature set (8 physicochemical descriptors)"<br>]|fulltext|0.9|model_amp_zgsm|
|Deep learning regression model for antimicrobial peptide design (Witten & Witten 2019)|Witten-2019-CNN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|21347392|10.1101/692681|https://github.com/zswitten/Antimicrobial-Peptides|not_reported_in_available_evidence|GRAMPA database; not directly linked but likely included in the GitHub repository|True|[<br>"Preprint status (no peer-reviewed publication yet)",<br>"Full text not available to confirm details"<br>]|abstract|0.8|model_amp_zgsm|
|AMP-zGSM|AMP-zGSM|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|21347392|10.5220/0014457300004070|https://github.com/DemetParlakSonmez/amp-zGSM|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Conference paper (may not be exhaustive)",<br>"Year listed as 2026 (potential future publication or error)",<br>"Full text unavailable for detailed method verification"<br>]|abstract|0.7|model_amp_zgsm|
|AMP0|AMP0|traditional_physicochemical_statistical_features|machine_learning_models|targeted antimicrobial peptide prediction|ML|32750857|10.1109/TCBB.2020.2999399|not_reported_in_available_evidence|http://ampzero.pythonanywhere.com|not_reported_in_available_evidence|True|[<br>"code not clearly available",<br>"limited training and test data details"<br>]|abstract|0.8|model_amp0|
|sAMPpred-GAT|sAMPpred-GAT|structure_graph_representation|gnn_models|antimicrobial peptide classification|DL|36342186|10.1093/bioinformatics/btac715|https://github.com/HongWuL/sAMPpred-GAT/|http://bliulab.net/sAMPpred-GAT|https://github.com/HongWuL/sAMPpred-GAT/ (likely includes datasets)|True|[<br>"review-level evidence",<br>"needs original paper verification"<br>]|abstract|0.9|model_amp0_webserver|
|PyAMPA|PyAMPA|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML/feature-engineering/web-server|38934543|10.1128/msystems.01358-23|https://github.com/SysBioUAB/PyAMPA|not_reported_in_available_evidence|AMPlify dataset, Liu et al. CPP database, AMPDeep hemolytic database, ToxinPred toxicity database, GRAMPA database (https://github.com/zswitten/Antimicrobial-Peptides)|True||fulltext|0.95|model_ampa|
|AMPA|AMPA|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|web-server|40410382|10.1038/s44320-025-00120-6|not_reported_in_available_evidence|http://tcoffee.crg.cat/apps/ampa|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|model_ampa|
|AntiBP3|AntiBP3|traditional_physicochemical_statistical_features|machine_learning_models|antibacterial peptide classification / antimicrobial peptide classification|ML|38391554|10.3390/antibiotics13020168|https://gitlab.com/raghavalab/antibp3|https://webs.iiitd.edu.in/raghava/antibp3|not_reported_in_available_evidence (training data compiled from public databases, no direct download link provided)|True||fulltext|0.95|model_ampactipred|
|AMPActiPred|AMPActiPred|traditional_physicochemical_statistical_features|machine_learning_models|antibacterial peptide classification and activity prediction|ML|38723168|10.1002/pro.5006|not_reported_in_available_evidence|https://awi.cuhk.edu.cn/∼AMPActiPred/|not_reported_in_available_evidence (elaborate dataset constructed from public sources, no direct download link)|True|[<br>"code not independently available"<br>]|fulltext|0.9|model_ampactipred|
|APEX|APEX|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|DL|38862735|10.1038/s41551-024-01201-x|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"code availability not confirmed from review"<br>]|review|0.6|model_amppredmfa|
|AMPfinder|AMPfinder|sequence_encoding_representation|cnn_dominant_models|AMP discovery / prediction|DL|39540425|10.1093/nar/gkae1019|not_reported_in_available_evidence|https://awi.cuhk.edu.cn/dbAMP/|dbAMP database|True|[<br>"code not independently available"<br>]|fulltext|0.9|model_ampactipred|
|AMPpredictor|AMPpredictor|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML|39540425|10.1093/nar/gkae1019|not_reported_in_available_evidence|https://awi.cuhk.edu.cn/dbAMP/|dbAMP database|True|[<br>"code not independently available"<br>]|fulltext|0.9|model_ampactipred|
|AMPBAN|AMPBAN|protein_language_model_representation|gnn_models|AMP prediction|DL||10.64898/2026.01.20.700468|https://github.com/baiwenhuim/ampban|not_reported_in_available_evidence|https://github.com/baiwenhuim/ampban (dataset in repository)|True||abstract|0.85|model_ampban|
|Generative AMP pipeline (VINCI)|VINCI AMP generator|protein_language_model_representation|rnn_lstm_dominant_models|AMP generation and MIC prediction|DL||10.64898/2026.06.16.732639|not_reported_in_available_evidence|not_reported_in_available_evidence|AMPSphere, DBAASP (links not provided)|True|[<br>"code link not found in abstract; full text needed for repository access"<br>]|abstract|0.7|model_ampban|
|AMPCLGPT|AMPCLGPT|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide classification|DL||10.1101/2025.03.07.642021|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code",<br>"no full text",<br>"preprint"<br>]|abstract|0.7|model_ampclgpt|
|CAmidPred|CAmidPred|protein_language_model_representation|cnn_dominant_models|antimicrobial peptide classification|DL||10.21203/rs.3.rs-7764304/v1|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code available",<br>"no full text",<br>"preprint"<br>]|abstract|0.7|model_ampclgpt|
|PepMCP|PepMCP|structure_graph_representation|cnn_dominant_models|antimicrobial peptide classification|DL||10.64898/2026.02.01.703163|https://github.com/ComputBiophys/PepMCP|not_reported_in_available_evidence|MemAMPdb (described in paper, no explicit link)|True|[<br>"preprint",<br>"no full text",<br>"no web server URL"<br>]|abstract|0.8|model_ampclgpt|
|iMFP-LG|iMFP-LG|protein_language_model_representation|gnn_models|multi-functional peptide prediction including antimicrobial peptide classification|DL|39585308|10.1093/gpbjnl/qzae084|https://github.com/chen-bioinfo/iMFP-LG|https://ngdc.cncb.ac.cn/biocode/tools/BT007494|not_reported_in_available_evidence|True||fulltext|0.95|model_ampd_up|
|Deep learning model for AMP discovery from ruminant gastrointestinal microbiomes|not_provided_in_evidence|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39756573|10.1016/j.jare.2025.01.005|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code or model weights provided",<br>"Dataset not publicly linked",<br>"Full-text cache mismatch (PMCID may be incorrect)"<br>]|abstract|0.7|model_ampep|
|amPEPpy|amPEPpy|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|33135060|10.1093/bioinformatics/btaa917|https://github.com/tlawrence3/amPEPpy|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code"<br>]|abstract|0.8|model_ampeppy|
|panCleave|panCleave|sequence_encoding_representation|machine_learning_models|AMP prediction|ML|37516110|10.1016/j.chom.2023.07.001|https://gitlab.com/machine-biology-group-public/pancleave|not_reported_in_available_evidence|Training and test data (MEROPS substrates) available in the panCleave repository (https://gitlab.com/machine-biology-group-public/pancleave)|True||fulltext|0.9|model_ampeppy|
|Bacteria-specific ML models for E. coli AMP activity|Bacteria-specific ML models for E. coli AMP activity|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|feature-engineering|36912047|10.1021/acs.jcim.2c01551|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No public code or dataset",<br>"Only E. coli activity, not general AMP prediction"<br>]|abstract|0.5|model_ampeppy|
|XGBoost AMP prediction model (Bhangu2025)|XGBoost AMP prediction model (Bhangu2025)|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|40529865|10.1002/smsc.202400579|not_reported_in_available_evidence|not_reported_in_available_evidence|http://cabgrid.res.in:8080/amppred/about.html (and other AMP databases)|True|[<br>"No public code or model weights",<br>"No web server"<br>]|fulltext|0.8|model_ampeppy|
|Multiple DL models reviewed (e.g., AMP-BERT, Deep-AmPEP30, etc.)|Various DL AMP models from review|protein_language_model_representation|cnn_dominant_models|AMP prediction|DL|36290108|10.3390/antibiotics11101451|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|False|[<br>"Review-level evidence",<br>"No specific model details or code"<br>]|review|0.3|model_ampeppy|
|AMPGAN v3|AMPGAN v3|sequence_encoding_representation|cnn_dominant_models|other|DL|42364293|10.1016/j.jmgm.2026.109497|https://github.com/marszzibros/AMPGANv3|not_reported_in_available_evidence|https://github.com/marszzibros/AMPGANv3 (likely contains data)|True||abstract|0.9|model_ampganv3|
|PepAnno|PepAnno|structure_graph_representation|transformer_llm_dominant_models|AMP prediction|DL|42228741|10.1371/journal.pcbi.1014369|not_reported_in_available_evidence|https://bis.zju.edu.cn/pepanno/|not_reported_in_available_evidence|True||abstract|0.8|model_ampgan_v3|
|AMPGP|AMPGP|traditional_physicochemical_statistical_features|cnn_dominant_models|antimicrobial peptide classification|DL|40825014|10.1021/acs.jcim.5c00647|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code or dataset availability reported",<br>"Only abstract available, no full text"<br>]|abstract|0.7|model_ampgp|
|AmpGram|AmpGram|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|ML|32560350|10.3390/ijms21124310|not_reported_in_available_evidence|not_reported_in_available_evidence|Training data not detailed; benchmarked on APD3 and DAMPD datasets|True|[<br>"Original code not in evidence",<br>"Web server not reported"<br>]|fulltext|1.0|model_ampgram|
|Ampir|ampir|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|DL|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original code not available"<br>]|fulltext|0.6|model_ann_based_amp_prediction_model_ref_4|
|Ensemble-AMPPred|Ensemble-AMPPred|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|model_ampir|
|CancerGram|CancerGram|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification||38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|model_ampir|
|PPTPP|PPTPP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification||38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|model_ampir|
|MLBP|MLBP|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification||38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|model_ampir|
|Deep2Pep|Deep2Pep|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|model_ampir|
|Pore-Forming_AMP_SVM|Pore-Forming AMP SVM|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide classification|ML|41391039|10.1002/advs.202516470|https://github.com/ComputBiophys/Pore%E2%80%90Forming_AMP_SVM|not_reported_in_available_evidence|https://github.com/ComputBiophys/Pore%E2%80%90Forming_AMP_SVM (training data included)|True||fulltext|0.95|model_amphgt|
|CG-AMP|CG-AMP|protein_language_model_representation|gnn_models|antimicrobial peptide classification|DL|41286313|10.1038/s41598-025-29666-z|not_reported_in_available_evidence|not_reported_in_available_evidence|AMPlify and DAMP benchmark datasets|True|[<br>"Code not available"<br>]|fulltext|0.85|model_amphgt|
|AmpHGT|AmpHGT|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide classification|DL|40598389|10.1186/s12915-025-02253-4|not_reported_in_available_evidence|not_reported_in_available_evidence|XUAMP, AMPDiscover, NCAA datasets|True|[<br>"Code not available"<br>]|fulltext|0.85|model_amphgt|
|SGAC|SGAC|structure_graph_representation|gnn_models|antimicrobial peptide classification|DL|41662353|10.1093/bib/bbag038|https://github.com/wyxwyx46941930/SGAC|not_reported_in_available_evidence|not_reported_in_available_evidence (paper states 'publicly available AMP and non-AMP datasets')|True||fulltext|0.95|model_amplify|
|TP-LMMSG|TP-LMMSG|protein_language_model_representation|gnn_models|therapeutic peptide prediction (including AMP, antiviral, anticancer)|DL|41978380|10.1093/bib/bbag107|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|False|[<br>"review-level evidence",<br>"needs original paper verification"<br>]|review|0.4|model_amplify|
|PGAT-ABPp|PGAT-ABPp|protein_language_model_representation|gnn_models|antibacterial peptide prediction|DL|41755839|10.1021/jacsau.5c01520|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|False|[<br>"review-level evidence",<br>"needs original paper verification"<br>]|review|0.4|model_amplify|
|Bidirectional LSTM AMP classification model (Wang2021)|Wang2021_LSTM_AMP|sequence_encoding_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|33810011|10.3390/biom11030471|not_reported_in_available_evidence|not_reported_in_available_evidence|CAMP, DBAASP, DRAMP, YADAMP, UniProt (as described in Methods)|True|[<br>"Code not publicly available in a repository, only in Supplementary Materials; unclear if code is accessible."<br>]|fulltext|0.8|model_amplify|
|PrMFTP|PrMFTP|sequence_encoding_representation|cnn_dominant_models|multi-functional therapeutic peptide prediction (including AMP classes like ABP, AFP, AVP, etc.)|DL|36094961|10.1371/journal.pcbi.1010511|not_reported_in_available_evidence|http://bioinfo.ahu.edu.cn/PrMFTP|not_reported_in_available_evidence (constructed from 22 therapeutic peptide datasets; no direct download link provided in evidence)|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|model_amppred|
|DeepAFP|DeepAFP|traditional_physicochemical_statistical_features|cnn_dominant_models|antifungal peptide prediction (AFP identification)|DL|37595093|10.1002/pro.4758|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (DeepAFP-Main dataset, curated, no direct link provided)|True|[<br>"code_repository_not_found",<br>"web_server_not_found",<br>"downloadable_tool_url_missing"<br>]|fulltext|0.85|model_amppred|
|AMPpred|AMPpred|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide prediction|web-server|37914524|10.24272/j.issn.2095-8137.2023.246|not_reported_in_available_evidence|http://cabgrid.res.in:8080/amppred/|not_reported_in_available_evidence|True|[<br>"needs_original_publication"<br>]|fulltext|0.5|model_amppred|
|AMPpred-AAIW|AMPpred-AAIW|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|37120707|10.1142/S0219720023500063|not_reported_in_available_evidence|https://amppred-aaiw.com|DRAMP and other published databases (not reported as link)|True||abstract|0.9|model_amppred_aaiw_web_server|
|MIC prediction ensemble model (BiLSTM-CNN-MBM)|MIC prediction ensemble model|sequence_encoding_representation|cnn_dominant_models|AMP prediction / MIC prediction|DL|39262770|10.48550/arXiv.1810.11363|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.8|model_antifreeze_peptide_discovery|
|AMPpred-EL|AMPpred-EL|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML|35576825|10.1016/j.compbiomed.2022.105577|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, no details"<br>]|fulltext|0.9|model_antifreeze_peptide_discovery|
|AMPpred-MFA|AMPpred-MFA|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|DL||10.1021/acs.jcim.3c01017.s001|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||metadata|0.8|model_amppred_mfa|
|Multifunctional AMP Design Framework (FBGAN-enhanced)|Multifunctional AMP Design Framework|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|Integrated from GRAMPA, APD3, ADAM, CAMPR4, UniProt|True|[<br>"code not available",<br>"no web server"<br>]|abstract|0.7|model_amppredmfa|
|AMPpredMFA|AMPpredMFA|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"lack of original publication details from review"<br>]|review|0.5|model_amppredmfa|
|AMP-META|AMP-META|sequence_encoding_representation|cnn_dominant_models|AMP prediction (strain-specific)|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"lack of original publication details from review"<br>]|review|0.5|model_amppredmfa|
|MBC-attention|MBC-attention|sequence_encoding_representation|cnn_dominant_models|AMP prediction (MIC regression)|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"lack of original publication details from review"<br>]|review|0.5|model_amppredmfa|
|EnDL-HemoLyt|EnDL-HemoLyt|sequence_encoding_representation|cnn_dominant_models|AMP toxicity prediction|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"lack of original publication details from review"<br>]|review|0.5|model_amppredmfa|
|SenseXAMP|SenseXAMP|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"lack of original publication details from review"<br>]|review|0.5|model_amppredmfa|
|AniAMPpred|AniAMPpred|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML/DL|34259329|10.1093/bib/bbab242|not_reported_in_available_evidence|https://aniamppred.anvil.app/|not_reported_in_available_evidence|True|[<br>"no code available",<br>"fulltext provided does not match article (PMC12620532 is a different paper); only abstract evidence used"<br>]|abstract|0.7|model_amps_net|
|Appred|Appred|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|ML|39247292|10.1016/j.heliyon.2024.e36163|not_reported_in_available_evidence|www.soodlab.com/appred|not_reported_in_available_evidence|True|[<br>"no code available",<br>"dataset not publicly linked"<br>]|fulltext|0.9|model_amps_net|
|AMPs-Net|AMPs-Net|structure_graph_representation|gnn_models|antimicrobial peptide classification|DL|35877911|10.3389/fmicb.2021.710199|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"only review evidence",<br>"no code or server found"<br>]|abstract|0.9|model_amps_net|
|LABAMPs|LABAMPs|structure_graph_representation|gnn_models|antimicrobial peptide classification|DL|37521317|10.3389/fbinf.2023.1216362|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"only review evidence",<br>"no code or server found"<br>]|review|0.4|model_amps_net|
|LSTM-based AMP classifier/generator|LSTM AMP classifier (Wang et al. 2021)|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|33810011|10.1016/j.diagmicrobio.2004.02.008|not_reported_in_available_evidence|not_reported_in_available_evidence|not reported (likely from public databases)|True|[<br>"code not available",<br>"no web server reported"<br>]|fulltext|0.8|model_ampscanner|
|AMPSpeciesSpecific|AMPSpeciesSpecific|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|DL|39766503|10.3390/antibiotics13121113|https://github.com/bzlee-bio/AMPSpeciesSpecific|not_reported_in_available_evidence|https://github.com/bzlee-bio/AMPSpeciesSpecific (may contain data)|True||fulltext|0.9|model_ampspeciesspecific|
|PepNet|PepNet|sequence_encoding_representation|cnn_dominant_models|AMP prediction / anti-inflammatory peptide classification|DL|39341947|10.1038/s42003-024-06911-1|https://zenodo.org/records/1322351661, https://zenodo.org/records/1373425862|http://liulab.top/PepNet/server|not_reported_in_available_evidence (described as AMP and AIP test sets from previous studies; likely included in Zenodo records)|True|[<br>"no code, no details"<br>]|fulltext|0.95|model_ampspeciesspecific|
|BPFun|BPFun|protein_language_model_representation|cnn_dominant_models|antimicrobial peptide classification / bioactive peptide function prediction|DL|40691539|10.1186/s12859-025-06190-5|https://github.com/291357657/BPFun|not_reported_in_available_evidence|https://github.com/291357657/BPFun (data included in repository)|True||fulltext|0.95|model_ampspeciesspecific|
|LLAMP|LLAMP|protein_language_model_representation|cnn_dominant_models|AMP prediction (MIC prediction, species-aware)|DL|40676915|10.1093/bib/bbaf343|https://github.com/GIST-CSBL/LLAMP|not_reported_in_available_evidence|https://github.com/GIST-CSBL/LLAMP (data included); DBAASP v3 for MIC data|True||fulltext|0.95|model_ampspeciesspecific|
|CL-ACP|CL-ACP|structure_graph_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|34670488|10.1186/s12859-021-04433-9|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code repository or web server provided"<br>]|fulltext|0.9|model_amptrans_lstm|
|AMPTrans-lstm|AMPTrans-lstm|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|36618982|10.1016/j.csbj.2022.12.029|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Primary purpose is AMP generation, not classification; no standard benchmark testing; no code available"<br>]|fulltext|0.7|model_amptrans_lstm|
|CSAMPPRED|CSAMPPRED|traditional_physicochemical_statistical_features|machine_learning_models|AMPs classification|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original code not available",<br>"Web server link not reported in evidence"<br>]|fulltext|0.7|model_ann_based_amp_prediction_model_ref_4|
|Thomas et al. 2009 AMP prediction model|Thomas et al. 2009 AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original paper not available",<br>"Code not available"<br>]|fulltext|0.4|model_ann_based_amp_prediction_model_ref_4|
|ANN-based AMP prediction model (ref [4])|ANN-based AMP prediction model (ref [4])|structure_graph_representation|gnn_models|AMPs prediction|DL|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No name, no code, no original paper in evidence"<br>]|fulltext|0.3|model_ann_based_amp_prediction_model_ref_4|
|Multiple alignment based AMP predictor (ref [5])|Multiple alignment based AMP predictor (ref [5])|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No name, no code, no original paper"<br>]|fulltext|0.3|model_ann_based_amp_prediction_model_ref_4|
|Two-level fuzzy K-NN model (ref [7])|Two-level fuzzy K-Nearest Neighbor model (ref [7])|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No name, no code, no original paper"<br>]|fulltext|0.3|model_ann_based_amp_prediction_model_ref_4|
|Sequence alignment-SVM-LZ complexity model (ref [8])|Sequence alignment-SVM-LZ complexity model (ref [8])|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No name, no code, no original paper"<br>]|fulltext|0.3|model_ann_based_amp_prediction_model_ref_4|
|Anti-Hepatitis Peptides predictor (ref [9])|Anti-Hepatitis Peptides predictor (ref [9])|traditional_physicochemical_statistical_features|machine_learning_models|specific anti-hepatitis peptide prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Specific to anti-hepatitis, not general AMP",<br>"No code, no original paper"<br>]|fulltext|0.4|model_ann_based_amp_prediction_model_ref_4|
|AmpClass|AmpClass|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|39383429|10.1590/0001-3765202420230756|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code or web server available",<br>"Dataset source not disclosed"<br>]|fulltext|0.8|model_ann_based_amp_prediction_model_ref_4|
|Gabere&Noble AMP predictor|Gabere&Noble AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No model details",<br>"Original code not available"<br>]|fulltext|0.4|model_ann_based_amp_prediction_model_ref_4|
|Wang et al. AMP predictor|Wang et al. AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No model details",<br>"Original code not available"<br>]|fulltext|0.4|model_ann_based_amp_prediction_model_ref_4|
|Witten&Witten AMP predictor|Witten&Witten AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No model details",<br>"Original code not available"<br>]|fulltext|0.4|model_ann_based_amp_prediction_model_ref_4|
|Unnamed CVAE-diffusion AMP generator|Unnamed CVAE-diffusion AMP generator|protein_language_model_representation|transformer_llm_dominant_models|AMP generation and activity prediction|DL|41460918|10.1371/journal.pcbi.1013833|not_reported_in_available_evidence|not_reported_in_available_evidence|UniProt (uniprotkb_reviewed_true_2024_12_17.fasta) for pretraining; GRAMPA for fine-tuning and MIC training|True|[<br>"Code not available",<br>"No public web server or weights"<br>]|fulltext|0.95|model_antibp3|
|Malebary-Khan AMP predictor|Malebary-Khan AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38391554|10.32604/cmc.2021.015041|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code and dataset not available",<br>"No web server or detailed algorithm description"<br>]|abstract|0.5|model_antibp3|
|Anticancer-Peptides-CNN|Anticancer-Peptides-CNN|traditional_physicochemical_statistical_features|cnn_dominant_models|anticancer peptide prediction|deep learning|34880291|10.1038/s41598-021-02703-3|https://github.com/mrzResearchArena/Anticancer-Peptides-CNN|https://anticancer.pythonanywhere.com/|not_reported_in_available_evidence|False|[<br>"task_mismatch"<br>]|repository|0.7|model_anticancer_peptides_cnn|
|APIN|APIN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|31870282|10.1093/bioinformatics/btx679|https://github.com/zhanglabNKU/APIN|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code, no details"<br>]|abstract|0.9|model_apin|
|SeqGAN-BERT-MLP AMP identifier (Cao et al. 2023)|SeqGAN-BERT-MLP AMP identifier|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction|DL|36857616|10.1093/bib/bbad058|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|False|[<br>"No code or model name provided",<br>"Insufficient detail for reproducibility"<br>]|abstract|0.4|model_apin|
|Co-AMPpred GitHub repository|Co-AMPpred GitHub repository|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/onkarS23/CoAMPpred||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|CoAMPpred|CoAMPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/onkarS23/CoAMPpred||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|2020-peptidomics|2020-peptidomics|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/ErikHartman/2020-peptidomics||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AAGP|AAGP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aagpazos/aagpazos.github.io||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.28|github_missing_model_enrichment|
|MetagenomicDC|MetagenomicDC|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/IcarPA-TBlab/MetagenomicDC||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|deep-belief-network|deep-belief-network|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/albertbup/deep-belief-network||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|acp-ope|acp-ope|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/khanhlee/acp-ope||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|2022-iAMP-DL|2022-iAMP-DL|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/mldlproject/2022-iAMP-DL||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|AFP_DL|AFP_DL|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/DongYin521/AFP_DL-QSARES||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|AFP_DL-QSARES|AFP_DL-QSARES|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/DongYin521/AFP_DL-QSARES||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|ANIA_github|ANIA_github|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aniagithub/Nieliniowe||not_reported_in_available_evidence|True||github_search|0.85|github_missing_model_enrichment|
|PC6-protein-encoding-method|PC6-protein-encoding-method|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/LinTzuTang/PC6-protein-encoding-method||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|BAGEL4|BAGEL4|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/ByteDance-Seed/Bagel||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|LinearDisplay|LinearDisplay|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/JCVenterInstitute/LinearDisplay||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|msaconverter|msaconverter|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/linzhi2013/msaconverter||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|LysePred|LysePred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/lincubator/LysePred||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AI4AVP_predictor|AI4AVP_predictor|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/LinTzuTang/AI4AVP_predictor||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|AMP-researchprotein|AMP-researchprotein|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/researchprotein/amp||not_reported_in_available_evidence|True||github_search|0.92|github_missing_model_enrichment|
|learning_sequence_motifs|learning_sequence_motifs|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/p-koo/learning_sequence_motifs||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|AMP-BERT GitHub repository|AMP-BERT GitHub repository|protein_language_model_representation|transformer_llm_dominant_models|||||https://github.com/GIST-CSBL/AMP-BERT||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|LightGBM|LightGBM|traditional_physicochemical_statistical_features|machine_learning_models|||||https://github.com/lightgbm-org/LightGBM||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|shap|shap|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/shap/shap||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|COMPASS database|COMPASS database|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aaronpk/Compass||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AMP-RNNpro web server|AMP-RNNpro web server|sequence_encoding_representation|rnn_lstm_dominant_models|||||https://github.com/Shazzad-Shaon3404/Website_AMPRNNpro||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|deep_AMPpred|deep_AMPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/JunZhao-hash/deep_AMPpred||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|ADAM_web_server|ADAM_web_server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/urban-adam/urban-adam-web||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|ampsphere_web_server|ampsphere_web_server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/BigDataBiology/AMPSphereWebsite||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|MAPLE GitHub repository|MAPLE GitHub repository|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/abdulrahmanbinayub-maker/maple-github-repository||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|Antimicrobial-Peptides|Antimicrobial-Peptides|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zswitten/Antimicrobial-Peptides||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|LMPred_AMP_Prediction|LMPred_AMP_Prediction|protein_language_model_representation|cnn_dominant_models|||||https://github.com/williamdee1/LMPred_AMP_Prediction||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|CDPfold|CDPfold|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zhangch994/CDPfold||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|DDM GitHub|DDM GitHub|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/DDM-Mzp/ddm.github.io||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|UniAMP web server|UniAMP web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Dextro86/Webasto-Ampure-Unite-Home-Assistant-custom-integration||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.38|github_missing_model_enrichment|
|PepProtGraphAnalyzer|PepProtGraphAnalyzer|structure_graph_representation|pipeline_or_ensemble_frameworks|||||https://github.com/cicese-biocom/PepProtGraphAnalyzer||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|esm-AxP-GDL|esm-AxP-GDL|protein_language_model_representation|gnn_models|AMP prediction|DL|not available|not available|https://github.com/cicese-biocom/esm-AxP-GDL|not_reported|not_reported_in_available_evidence|True|[<br>"no code, no details"<br>]|github_search|1.0|github_missing_model_enrichment|
|esm|esm|protein_language_model_representation|gnn_models|||||https://github.com/standard-things/esm||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|E-CLEAP GitHub repository|E-CLEAP GitHub repository|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Wangsicheng52/E-CLEAP||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|AMPScanner vr.2 web server|AMPScanner vr.2 web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/dan-veltri/amp-scanner-v2||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|PepGen 1.0 web server|PepGen 1.0 web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Nate0634034090/nate.283090||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.46|github_missing_model_enrichment|
|CalcAMP GitHub repository|CalcAMP GitHub repository|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/CDDLeiden/CalcAMP||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|Deep-AmPEP30 web server|Deep-AmPEP30 web server|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/Chonwai/Deep_AmPEP30_R||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AMP toxicity prediction code|AMP toxicity prediction code|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/h-khabbaz/amp-toxicity-predictor||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.46|github_missing_model_enrichment|
|AMP0 webserver|AMP0 webserver|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/danielm710/AMP-webserver||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AMPA web server|AMPA web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/miminiyo/ampaweb||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AntiBP3 GitLab|AntiBP3 GitLab|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/AntiBP3||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|AntiBP3 Web Server|AntiBP3 Web Server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/AntiBP3||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|AntiBP3 PyPI|AntiBP3 PyPI|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/AntiBP3||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|dbAMP 3.0 web server|dbAMP 3.0 web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Nate0634034090/bug-free-memory||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.31|github_missing_model_enrichment|
|AMPBenchmark|AMPBenchmark|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/BioGenies/AMPBenchmark||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|StarPep|StarPep|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Grupo-Medicina-Molecular-y-Traslacional/StarPep||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AmpGram R package|AmpGram R package|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/cran/AmpGram||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|PepNet web server|PepNet web server|protein_language_model_representation|transformer_llm_dominant_models|||||https://github.com/VeniQs02/pep.net-web-app||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|Antimicrobial Peptide Scanner vr.2 web server|Antimicrobial Peptide Scanner vr.2 web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/dan-veltri/amp-scanner-v2||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AMPScanner vr.2 web server (alternate)|AMPScanner vr.2 web server (alternate)|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/dan-veltri/amp-scanner-v2||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|ProteoGPT (AMPSorter)|ProteoGPT|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|LABAMPsGCN|LABAMPsGCN|structure_graph_representation|gnn_models|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|AMPidentifier|AMPidentifier|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|SMEP|SMEP|structure_graph_representation|pipeline_or_ensemble_frameworks|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|sAMP-PFPDeep|sAMP-PFPDeep|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|iAMP-CA2L|iAMP-CA2L|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|C_AMPs-predict|C_AMPs-predict|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|UniDL4BioPep|UniDL4BioPep|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|APEX 1.1|APEX 1.1|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|DL|not available|not available|not_reported|not_reported|not_reported|True|[<br>"no code, no details"<br>]|review|0.5||
|ACPred|ACPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer_peptide_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://codes.bio/acpred/|not_reported_in_available_evidence|False|[<br>"original_paper_not_found",<br>"only_usage_reported"<br>]|fulltext|0.3|model_ampfun|
|AMPfun|AMPfun|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP/anticancer/antibacterial prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://fdblab.csie.ncu.edu.tw/AMPfun/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|model_acpred|
|AntiCP|AntiCP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer_peptide_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://crdd.osdd.net/raghava/anticp/|not_reported_in_available_evidence|False|[<br>"webserver_only",<br>"superseded_by_AntiCP2.0"<br>]|fulltext|0.6|model_anticp2_0|
|AntiCP2.0|AntiCP2.0|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer_peptide_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://webs.iiitd.edu.in/raghava/anticp2/|not_reported_in_available_evidence|True|[<br>"webserver_only",<br>"no_source_code",<br>"no_model_weights_provided",<br>"batch_inference_unknown"<br>]|fulltext|0.7|model_anticp2_0|
|HAPPENN|HAPPENN|sequence_encoding_representation|pipeline_or_ensemble_frameworks|hemolysis_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://research.timmons.eu/happenn/|not_reported_in_available_evidence|False|[<br>"webserver_only",<br>"out_of_scope_for_AMP_benchmark"<br>]|fulltext|0.5|model_anticp2_0|
|HemoPred|HemoPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|hemolysis_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://codes.bio/hemopred/|not_reported_in_available_evidence|False|[<br>"webserver_only",<br>"out_of_scope"<br>]|fulltext|0.5|model_anticp2_0|
|ToxinPred|ToxinPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|toxicity_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://crdd.osdd.net/raghava/toxinpred/|not_reported_in_available_evidence|False|[<br>"webserver_only",<br>"out_of_scope"<br>]|fulltext|0.5|model_anticp2_0|
|ToxIBTL|ToxIBTL|sequence_encoding_representation|pipeline_or_ensemble_frameworks|toxicity_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://server.wei-group.net/ToxIBTL/|not_reported_in_available_evidence|False|[<br>"webserver_only",<br>"out_of_scope"<br>]|fulltext|0.5|model_anticp2_0|
|AllerTop|AllerTop|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://www.ddg-pharmfac.net/AllerTOP/|not_reported_in_available_evidence|False|[<br>"webserver_only",<br>"out_of_scope"<br>]|fulltext|0.5|model_anticp2_0|
|AllergenFP|AllergenFP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://ddg-pharmfac.net/AllergenFP/|not_reported_in_available_evidence|False|[<br>"webserver_only",<br>"out_of_scope"<br>]|fulltext|0.5|model_anticp2_0|
|AllerCatPro|AllerCatPro|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://allercatpro.bii.a-star.edu.sg/|not_reported_in_available_evidence|False|[<br>"webserver_only",<br>"out_of_scope"<br>]|fulltext|0.5|model_anticp2_0|
|Deep learning hybrid model (unnamed)|Unknown deep learning hybrid model|multimodal_hybrid_representation|cnn_dominant_models|AMP prediction|deep learning (hybrid)|41731616|10.1186/s40168-025-02326-0|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|False|[<br>"model_name_unknown",<br>"no_details",<br>"review_only"<br>]|fulltext|0.1|model_acpred|
|AxPEP3|AxPEP3|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Webserver-only, no code"<br>]|fulltext|0.7|model_adam|
|RF-AmPEP30|RF-AmPEP30|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No weights reported"<br>]|fulltext|0.8|model_adam|
|CAMPR34|CAMPR34|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Webserver-only, no code"<br>]|fulltext|0.7|model_adam|
|CLASSAMP5|CLASSAMP5|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Webserver-only, no code"<br>]|fulltext|0.7|model_adam|
|DBAASP6|DBAASP6|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Webserver-only, no code"<br>]|fulltext|0.7|model_adam|
|APSvr.2|Antimicrobial Peptide Scanner v.2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|37523405|not_reported_in_available_evidence|not_reported_in_available_evidence|https://aps.unmc.edu/prediction/predict|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|model_adam|
|DBAASPv3.0|DBAASP v3.0|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|AMP prediction|web-server|37523405|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.7|model_adam|
|CAMPR3(RF)|CAMPR3(RF)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.5|model_adam|
|CAMPR3(SVM)|CAMPR3(SVM)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.4|model_adam|
|BAGEL3|BAGEL3|sequence_encoding_representation|pipeline_or_ensemble_frameworks|bacteriocin prediction||28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.5|model_adam|
|BACTIBASE|BACTIBASE|sequence_encoding_representation|pipeline_or_ensemble_frameworks|bacteriocin prediction||28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.4|model_adam|
|AMP prediction server (biosino)|AMP prediction server (biosino)|structure_graph_representation|machine_learning_models|antimicrobial peptide classification|ML/feature-engineering|21533231|10.1371/journal.pone.0018476|not_reported_in_available_evidence|http://amp.biosino.org/|CAMP database (http://www.camp.bicnirrh.res.in/) and UniProt|True|[<br>"Webserver-only, no API for batch processing",<br>"No code"<br>]|fulltext|0.9|model_adam|
|ADMETlab 3|ADMETlab 3|sequence_encoding_representation|pipeline_or_ensemble_frameworks|ADMET property prediction|not_AMP_specific|42276016|10.1016/j.ultsonch.2026.107920|not_reported_in_available_evidence|https://admetlab3.scbdd.com|not_reported_in_available_evidence|False|[<br>"not_AMP_specific"<br>]|fulltext|0.0|model_admetlab_3|
|Multi-label weighted KNN-MLR model|Multi-label WKnn-MLR (Wang2017)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide activity prediction (multi-label classification)|ML|28526820|10.1038/s41598-017-01986-9|not_reported_in_available_evidence|not_reported_in_available_evidence|APD database (May 2016) filtered to 2222 AMPs with 5 activities; APD3 available at https://aps.unmc.edu/AP/|True|[<br>"No code or web server available",<br>"No independent external test set"<br>]|fulltext|0.85|model_amp_de_novo_design_cdgan|
|cdGAN|cdGAN (Tizoc2025)|protein_language_model_representation|cnn_dominant_models|antimicrobial peptide design (generative model with classifier)|deep learning, GAN|41137855|10.1093/bib/bbaf500|https://github.com/aretiz/amp_de_novo_design_cdGAN|not_reported_in_available_evidence|APD3 + UniProt (2600 AMPs, 2600 non-AMPs)|False|[<br>"Generative model, not a direct AMP predictor",<br>"Classifier weights not explicitly provided as standalone predictor"<br>]|fulltext|0.8|model_amp_de_novo_design_cdgan|
|AMP-GSM|AMP-GSM|structure_graph_representation|gnn_models|AMP prediction / antimicrobial peptide classification|ML|41072192|10.3390/app13085106|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no_code_available",<br>"no_dataset_link"<br>]|abstract|0.7|model_amp_gsm|
|ISCAPE|ISCAPE|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / anti-E. coli activity classification|ML|41072192|10.1016/j.jmgm.2025.109188|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no_code_available",<br>"no_dataset_link"<br>]|abstract|0.7|model_amp_gsm|
|AMP MIC predictor (CNN/RNN)|AMP-MIC-predictor-CNN-RNN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|37938588|10.1038/s41467-023-42434-9|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.8|model_amp_mic_predictor_cnn_rnn|
|AxPEP|AxPEP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction||41315055|10.1007/s00248-025-02620-2|https://sourceforge.net/projects/axpep/|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.9|model_amp_scanner_v2|
|AMPGenix|AMPGenix|protein_language_model_representation|transformer_llm_dominant_models|other|DL|40891852|10.1128/spectrum.01504-25|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|False|[<br>"task_type is generation, not prediction"<br>]|fulltext|0.8|model_amp_scanner_v2|
|StackAMP|StackAMP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|29374199|10.1109/tai.2024.3421176|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no_full_text_access",<br>"no_abstract_available"<br>]|metadata|0.3|model_ampep|
|AMPlify_bal|AMPlify_bal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|40891852|10.1128/spectrum.01504-25|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"original_paper_needed"<br>]|fulltext|0.9|model_ampgenix|
|AMPlify_imbal|AMPlify_imbal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|40891852|10.1128/spectrum.01504-25|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"original_paper_needed"<br>]|fulltext|0.9|model_ampgenix|
|PeptideRanker|PeptideRanker|structure_graph_representation|gnn_models|general peptide bioactivity prediction (including antimicrobial)|DL|23056189|10.1371/journal.pone.0045012|not_reported_in_available_evidence|http://bioware.ucd.ie/|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)|True||fulltext|0.9|model_amper|
|HydraAMP|HydraAMP|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide design|DL|23056189|10.1371/journal.pone.0045012|https://github.com/szczurek-lab/hydramp|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.8|model_amper|
|MetaPepticon|MetaPepticon|traditional_physicochemical_statistical_features|machine_learning_models|anticancer peptide prediction from (meta)genomes|ML|23056189|10.1371/journal.pone.0045012|https://github.com/arikanlab/MetaPepticon|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.7|model_amper|
|Venomics artificial intelligence|Venomics artificial intelligence|sequence_encoding_representation|pipeline_or_ensemble_frameworks|unknown (likely venom-related WhatsApp bot)|not_applicable|23056189|10.1371/journal.pone.0045012|https://github.com/vynect/venom|not_reported_in_available_evidence|not_reported_in_available_evidence|False|[<br>"github_search_candidate_requires_manual_verification",<br>"likely_out_of_scope"<br>]|github_search|0.1|model_amper|
|WeightedEnsemble_L3 (Anti_Cp)|WeightedEnsemble_L3|structure_graph_representation|gnn_models|antimicrobial peptide activity classification|ML|38266820|10.1016/j.jare.2024.01.023|https://github.com/xubocheng/Anti_Cp.git|not_reported_in_available_evidence|https://github.com/xubocheng/Anti_Cp.git|True||fulltext|0.9|model_anti_cp|
|PLUM|PLUM|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide generation and classification|DL|42124643|10.64898/2026.02.21.707214|https://github.com/priyamayur/PLUM|not_reported_in_available_evidence|Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository|True||fulltext|0.95|model_antimicrobial|
|APD3|Antimicrobial Peptide Database (APD3)|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|33996914|10.3389/fmolb.2021.669431|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"review_only",<br>"original_paper_needed"<br>]|fulltext|0.8|model_avcpred|
|AVCpred|AVCpred|traditional_physicochemical_statistical_features|machine_learning_models|antiviral peptide prediction|ML|33996914|10.3389/fmolb.2021.669431|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|False|[<br>"out_of_scope_antiviral_only",<br>"review_only"<br>]|fulltext|0.7|model_avcpred|
|ApexGO|ApexGO|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide optimization|DL|42206144|10.1038/s42256-026-01237-5|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (VAE training data not specified, APEX trained on in-house peptides)|True|[<br>"Code and model not publicly available",<br>"APEX predictor weights not available"<br>]|fulltext|0.9|model_apex|
|c_AMPs-prediction|c_AMPs-prediction|protein_language_model_representation|rnn_lstm_dominant_models|AMP prediction|DL|41164228|10.3389/fvets.2025.1689589|https://github.com/mayuefine/c_AMPs-prediction|not_reported_in_available_evidence|https://github.com/mayuefine/c_AMPs-prediction|True|[<br>"original_model_paper_uncertain",<br>"weights_not_reported"<br>]|fulltext|0.8|model_c_amps_prediction|
|AMPlify GitHub|AMPlify GitHub|sequence_encoding_representation|rnn_lstm_dominant_models|||||https://github.com/keonjale/amplifygithubrepo||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|AmPEP web server|AmPEP web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Amal-Thomas/Amal-Thomas-PEP-GP-WebDevProject-Recipe||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.36|github_missing_model_enrichment|
|AMPer web server|AMPer web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/AmirhesamGhahari/Amir_Ghahari_Personal_Website_API_Server||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.46|github_missing_model_enrichment|
|CatBoost AMP predictor|CatBoost AMP predictor|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Ronald106/Surviv.io||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.31|github_missing_model_enrichment|
|Two_Level_Ensemble-classifier-chain|Two_Level_Ensemble-classifier-chain|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/kkzheng/Two_Level_Ensemble-classifier-chain||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|amp_de_novo_design_cdGAN|amp_de_novo_design_cdGAN|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aretiz/amp_de_novo_design_cdGAN||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|MAPLE GitHub|MAPLE GitHub|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Violet-maple/Violet-maple.github.io||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|kneaddata|kneaddata|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/biobakery/kneaddata||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|VirSorter2|VirSorter2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/jiarong/VirSorter2||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|COGclassifier|COGclassifier|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/moshi4/COGclassifier||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|Anti_Cp|Anti_Cp|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/anticp2||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|Anti_Cp.git|Anti_Cp.git|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/AntiO-cps/antio-cps.github.io||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.36|github_missing_model_enrichment|
|PLUM GitHub|PLUM GitHub|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/purpleplum456/purple-plum-GitHub||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|Antimicrobial|Antimicrobial|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zswitten/Antimicrobial-Peptides||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|Urchin|Urchin|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/duckyb/urchin||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|allenCCF|allenCCF|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/cortex-lab/allenCCF||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|phy|phy|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/lo-th/phy||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|iblapps|iblapps|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/int-brain-lab/iblapps||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|Lab|Lab|structure_graph_representation|gnn_models|||||https://github.com/google-deepmind/lab||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|Npx|Npx|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zkat/npx||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|soft-neighbors-supported-clustering|soft-neighbors-supported-clustering|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/DuannYu/soft-neighbors--supported-clustering||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|github_missing_model_enrichment|
|DeepSeaQuence_biofilms|DeepSeaQuence_biofilms|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|||||https://github.com/trongthucnguyen/DeepSeaQuence_biofilms||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|FMT-MetagenomicData|FMT-MetagenomicData|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/pointwei/FMT-MetagenomicData||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|TransDecoder|TransDecoder|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/TransDecoder/TransDecoder||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|macrel2020benchmark|macrel2020benchmark|traditional_physicochemical_statistical_features|machine_learning_models|||||https://github.com/BigDataBiology/macrel2020benchmark||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|nov-fams-pipeline|nov-fams-pipeline|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/AlvaroRodriguezDelRio/nov-fams-pipeline||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|aro|aro|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/attdevsupport/ARO||not_reported_in_available_evidence|True||github_search|1.0|github_missing_model_enrichment|
|StackEnPred|StackEnPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/NK12131/Bankruptcy-Prediction-Using-Financial-KPIs-ML-Pipeline-with-SMOTE-PCA-Stacked-Ensemble||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.5|github_missing_model_enrichment|
|Multi-label WKnn-MLR|Multi-label WKnn-MLR|traditional_physicochemical_statistical_features|machine_learning_models|||||||||[<br>"No code, no web server, cannot be benchmarked"<br>]|very_low|very_low||

## Benchmark Ready Models

|model_name|canonical_name|representation_category|architecture_category|task_type|method_family|source_pmid|source_doi|code_repository_url|web_server_url|dataset_source_or_link|benchmark_candidate|candidate_reason|blocking_issues|evidence_level|confidence|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Co-AMPpred|Co-AMPpred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|34330209|10.1186/s12859-021-04305-2|https://github.com/onkarS23/CoAMPpred|not_reported_in_available_evidence|https://github.com/onkarS23/CoAMPpred (contains training and test data from DEEP-AmPEP30)|True|AMP prediction model with code and dataset available, suitable for benchmark||fulltext|0.9|
|CTCM-Neo & ConformaX-PEP framework|CTCM-Neo & ConformaX-PEP|protein_language_model_representation|gnn_models|antimicrobial peptide classification (antimalarial)|DL|41859462|10.3389/fcimb.2026.1707267|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (likely derived from APD3)|True|Novel deep learning framework for antimalarial peptide prediction with calibrated uncertainty; could be benchmarked for AMP classification|[<br>"no code repository link",<br>"no full text available",<br>"antimalarial-specific may limit general AMP benchmark"<br>]|abstract|0.6|
|A-CaMP|A-CaMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification / anti-cancer peptide prediction|sequence alignment-based / fingerprinting|31870207|10.1080/07391102.2019.1708796|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Described as a tool for anti-cancer and antimicrobial peptide prediction, with reported accuracy 93.4%.||fulltext|0.8|
|PCSPred|PCSPred|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|40781463|10.1109/NEleX59773.2023.10421222|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Direct AMP prediction model using Random Forest for short-chain AMPs, no code or dataset details available from abstract.|[<br>"no_code_available",<br>"no_full_text",<br>"no_dataset_details"<br>]|abstract|0.6|
|iAMPCN|iAMPCN|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|37369638|10.1093/bib/bbad240|https://github.com/joy50706/iAMPCN|not_reported_in_available_evidence|Integrated from multiple databases (APD3, dbAMP, DRAMP, etc.) and UniProt for negatives.|True|Comprehensive AMP and functional activity prediction model, code available, suitable for benchmark.|[<br>"original_model_article_not_this_one",<br>"dataset_not_specified"<br>]|fulltext|0.95|
|SSFGM-Model|SSFGM-Model|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide classification|DL|40462515|10.1186/s12864-020-06978-0|https://github.com/ggcameronnogg/SSFGM-Model|not_reported_in_available_evidence|not_reported_in_available_evidence|True|新型多模态深度学习方法，整合序列、结构、表面特征，性能优于现有方法，代码公开||abstract|0.8|
|ACEP|ACEP|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP recognition|DL|40462515|10.1186/s12864-020-06978-0|https://github.com/Fuhaoyi/ACEP|not_reported_in_available_evidence|APD database (mentioned in fulltext)|True|高性能AMP识别深度学习模型，代码公开，曾被广泛比较||fulltext|0.9|
|MultiPep|MultiPep|sequence_encoding_representation|cnn_dominant_models|multi-label peptide bioactivity classification (potential AMP prediction)|DL|34909478|10.1093/biomethods/bpab021|not_reported_in_available_evidence|not_reported_in_available_evidence|multiple public databases (not specified in abstract)|True|Multi-label classifier for 20 bioactivity classes, which may include antimicrobial peptides; potential AMP prediction model.|[<br>"The abstract does not explicitly list AMP among the 20 bioactivity classes; needs full-text verification."<br>]|abstract|0.5|
|iAMP-2L|iAMP-2L|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Reimplemented in benchmark study, known tool|[<br>"Original code not available",<br>"Web server not reported in evidence"<br>]|fulltext|0.6|
|iAMPred|iAMPred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|综述中提及的AMP预测工具||review|0.5|
|AmPEP|AmPEP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|综述中提及的AMP预测工具|[<br>"github_search_candidate_requires_manual_verification"<br>]|review|0.5|
|AntiBP2|AntiBP2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antibacterial peptide prediction|web-server|37914524|10.24272/j.issn.2095-8137.2023.246|not_reported_in_available_evidence|https://webs.iiitd.edu.in/raghava/antibp2/|not_reported_in_available_evidence|True|Antibacterial peptide prediction web server mentioned in review.|[<br>"needs_original_publication"<br>]|fulltext|0.5|
|CAMPR3|CAMPR3|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|综述中描述的AMP数据库与预测平台，提供多种预测算法|[<br>"possible_mismatch",<br>"github_search_candidate_requires_manual_verification"<br>]|review|0.5|
|ADAM|ADAM|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|http://bioinformatics.cs.ntou.edu.tw/ADAM|not_reported_in_available_evidence|True|Web server for AMP prediction mentioned in review.|[<br>"needs original paper verification"<br>]|review|0.6|
|DBAASP|DBAASP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide activity prediction|web-server|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in a review as a combined database and prediction tool for AMP activity and cytotoxicity.|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|MLAMP|MLAMP|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Reimplemented in benchmark study|[<br>"Original code not available"<br>]|fulltext|0.6|
|CAMP|CAMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|综述中提及的AMP数据库与预测工具|[<br>"original paper not provided; no web server or code link available in this evidence"<br>]|review|0.5|
|ClassAMP|ClassAMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|web-server|37914524|10.24272/j.issn.2095-8137.2023.246|not_reported_in_available_evidence|http://www.bicnirrh.res.in/classamp/|not_reported_in_available_evidence|True|AMP classification web server mentioned in review.|[<br>"needs_original_publication"<br>]|fulltext|0.5|
|AVPpred|AVPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|ML|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|综述中提及的AMP预测工具|[<br>"github_search_candidate_requires_manual_verification"<br>]|review|0.5|
|AMPER|AMPER|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|feature-engineering|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|综述中提及的AMP预测工具|[<br>"original paper not provided; details rely on this reference"<br>]|review|0.5|
|EFC-FCBF|EFC-FCBF|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|AMP prediction|feature-engineering|35305010|10.1093/database/baab012|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|综述中提及的AMP预测工具||review|0.5|
|AMPlify|AMPlify|sequence_encoding_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|35078402|10.1186/s12864-022-08310-4|https://github.com/bcgsc/AMPlify|not_reported_in_available_evidence|Training data from APD, CAMP, etc. (details in paper); test set from Bullfrog genome and other sources.|True|High-performance AMP prediction model with attention, publicly available code, directly comparable to other tools.|[<br>"preprint not yet peer-reviewed, full text verification needed"<br>]|fulltext|0.95|
|E-CLEAP|E-CLEAP|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38722967|10.1371/journal.pone.0300125|https://github.com/Wangsicheng52/E-CLEAP|not_reported_in_available_evidence|https://github.com/Wangsicheng52/E-CLEAP|True|New AMP prediction model with publicly available code and dataset, suitable for benchmark|[<br>"incomplete evidence",<br>"source paper unknown"<br>]|fulltext|0.95|
|UniproLcad|UniproLcad|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/harkic/UniproLcad|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as AMP prediction model with GitHub repository.|[<br>"needs original paper verification"<br>]|review|0.7|
|TriStack|TriStack|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/hjy23/TriStack|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as AMP prediction model with GitHub repository.|[<br>"needs original paper verification"<br>]|review|0.7|
|iAMP-DL|iAMP-DL|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/mldlproject/2022-iAMP-DL|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as AMP prediction model with GitHub repository.|[<br>"needs original paper verification"<br>]|review|0.7|
|amp-gan|amp-gan|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://gitlab.com/vail-uvm/amp-gan|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as AMP prediction/design model with GitLab repository.|[<br>"needs original paper verification"<br>]|review|0.7|
|AVPIden|AVPIden|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|http://awi.cuhk.edu.cn/AVPIden/|not_reported_in_available_evidence|True|Web server for AMP identification mentioned in review.|[<br>"needs original paper verification"<br>]|review|0.6|
|antibp|antibp|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|http://www.imtech.res.in/raghava/antibp/|not_reported_in_available_evidence|True|Web server for AMP prediction mentioned in review.|[<br>"needs original paper verification"<br>]|review|0.6|
|ampsphere|ampsphere|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction / database|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|https://ampsphere.big-data-biology.org/|not_reported_in_available_evidence|True|Web server/database mentioned in review; needs original paper|[<br>"needs original paper verification"<br>]|review|0.6|
|hydramp|hydramp|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|https://hydramp.mimuw.edu.pl|not_reported_in_available_evidence|True|Web server for AMP prediction mentioned in review.|[<br>"Code not reported in this evidence"<br>]|review|0.6|
|AMPDiscover|AF-QSAM AMPDiscover|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML|34081438|10.1021/acs.jcim.1c00251|not_reported_in_available_evidence|https://biocom-ampdiscover.cicese.mx/|not_reported_in_available_evidence|True|Proposes new models and systematically benchmarks against 13 existing AMP prediction tools; provides a web server for access.|[<br>"No code repository available; only web server provided."<br>]|abstract|0.9|
|ESM2-AFPpred|ESM2-AFPpred|protein_language_model_representation|machine_learning_models|AMP prediction / antimicrobial peptide classification|DL|35724626|10.1093/bib/bbac226|https://github.com/DongYin521/AFP_DL|not_reported_in_available_evidence|DRAMP and APD3 databases (no direct download link provided in evidence)|True|Novel antifungal peptide prediction model with public code, suitable for benchmarking AMP classification tasks.||fulltext|0.95|
|ANIA|ANIA|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction|DL|41664908|10.1093/bib/bbag023|https://github.com/SilverGojo4/ANIA.|https://biomics.lab.nycu.edu.tw/ANIA/|DBAASP, dbAMP, DRAMP|True|Novel deep learning model for MIC prediction with web server and code available; compared against ESKAPEE-Pred, AMPActiPred, esAMPMIC|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.95|
|AI4AFP|AI4AFP|protein_language_model_representation|cnn_dominant_models|antimicrobial peptide classification|ML/DL|42146199|10.1021/acsomega.6c00049|not_reported_in_available_evidence|https://axp.iis.sinica.edu.tw/AI4AFP|CAMP, DRAMP, YADAMP, SATPdb, DBAASP (AFPs); UniProtKB/Swiss-Prot (non-AMPs); DBAASP (hemolysis data)|True|New ensemble model for antifungal peptide prediction with integrated hemolysis safety assessment; web server available; compared to existing AFP predictors||fulltext|0.9|
|AI4AMP|AI4AMP|traditional_physicochemical_statistical_features|cnn_rnn_hybrid_models|antimicrobial peptide classification|DL|34783578|10.1128/msystems.00299-21|https://github.com/LinTzuTang/AI4AMP_predictor|http://symbiosis.iis.sinica.edu.tw/PC_6/|not_reported_in_available_evidence|True|Novel AMP predictor with deep learning and PC6 encoding; outperformed existing methods on external test; web server available; code released|[<br>"no code or data link",<br>"only mentioned in review"<br>]|fulltext|0.95|
|Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships|Sparse NN AMP model|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|27870247|10.1002/minf.201600029|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Model directly predicts AMP activity; includes experimental validation.||abstract|0.7|
|SAMP|SAMP|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|39573886|10.1093/bfgp/elae046|https://github.com/wan-mlab/SAMP|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel ensemble ML model for AMP prediction; outperforms existing methods; Python package available||fulltext|0.95|
|DL-QSARES|DL-QSARES|traditional_physicochemical_statistical_features|cnn_dominant_models|antifungal peptide prediction/design|DL|39921483|10.1002/advs.202412488|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel AMP prediction model using deep learning for antifungal peptide screening|[<br>"code not available",<br>"only abstract evidence"<br>]|abstract|0.5|
|AI4AVP|AI4AVP|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|37626205|10.1109/JBHI.2021.3130825|https://github.com/LinTzuTang/AI4AVP_predictor|http://axp.iis.sinica.edu.tw/AI4AVP/|https://github.com/LinTzuTang/AI4AVP_predictor (datasets from APD3, DRAMP, YADAMP, DBAASP, CAMP, AVPdb, UniProt/SwissProt)|True|Deep learning-based antiviral peptide predictor with open-source code and web server; suitable for AMP prediction benchmark|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|
|PepForge|PepForge|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|DL|39705302|10.64898/2026.05.29.728379|https://github.com/wqx1999/PepForge|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel AMP generation and prediction model with publicly available code, can be used for AMP prediction benchmarking|[<br>"github_search_candidate_requires_manual_verification"<br>]|abstract|0.9|
|Al-Omari 2024 AMP prediction model|Al-Omari 2024 AMP prediction model|traditional_physicochemical_statistical_features|cnn_dominant_models|antimicrobial peptide classification|DL|39705302|10.1371/journal.pone.0315477|not_reported_in_available_evidence|not_reported_in_available_evidence|https://dbaasp.org|True|Provides a deep learning-based AMP prediction model with reported accuracy; could be used as benchmark candidate if code becomes available|[<br>"Code not available"<br>]|fulltext|0.8|
|BBATProt|BBATProt|protein_language_model_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|41212592|10.1093/bib/bbaf593|https://github.com/Xukai-YE/BBATProt|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Provides a DL framework with code for AMP prediction, improving accuracy over existing methods; suitable for benchmark||fulltext|0.9|
|AMAP|AMAP|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Reimplemented in benchmark study, simple baseline|[<br>"Original code not available"<br>]|fulltext|0.6|
|AMP|AMP Ensemble Model|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|ML/DL|38972032|10.1007/s12539-024-00640-z|https://github.com/researchprotein/amp|http://amp.denglab.org|https://github.com/researchprotein/amp|True|AMP prediction model with code, web server, and datasets available, good for benchmarking.||abstract|0.8|
|Deep-AmPEP30|Deep-AmPEP30|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP prediction|DL|32464552|10.1016/j.omtn.2020.05.006|not_reported_in_available_evidence|https://cbbio.cis.um.edu.mo/AxPEP|Benchmark dataset of 188 samples (balanced); training set of 1529 positive samples from AMP databases (AmPEP, etc.)|True|Publicly available web server for short AMP prediction, including genome screening; strong experimental validation; suitable for benchmarking.|[<br>"code not available",<br>"no dataset link provided"<br>]|fulltext|0.95|
|EBAMP|EBAMP|sequence_encoding_representation|transformer_llm_dominant_models|antimicrobial peptide design|DL|40906555|10.1016/j.celrep.2025.116215|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel de novo AMP design framework with experimental validation, could be used for comparative analysis if code becomes available.|[<br>"no code or web server available",<br>"method details not fully described"<br>]|abstract|0.5|
|DLFea4AMPGen|DLFea4AMPGen|traditional_physicochemical_statistical_features|cnn_dominant_models|antimicrobial peptide design|DL|41093853|10.1002/adma.202307680|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Innovative design method using DL and SHAP, with experimental validation, potential benchmark if code is released.|[<br>"no code or web server available"<br>]|abstract|0.5|
|AMP-BERT|AMP-BERT|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction|DL|36461699|10.1002/pro.4529|https://github.com/GIST-CSBL/AMP-BERT.|not_reported_in_available_evidence|https://github.com/GIST-CSBL/AMP-BERT.|True|Public code and dataset, strong performance in external test, ideal for benchmark.||fulltext|0.95|
|COMDEL|COMDEL|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39234615|10.1016/j.apsb.2024.05.003|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel DL AMP prediction model with reported test accuracy 94.8%, suitable for benchmarking.||fulltext|0.9|
|C. acnes-targeted AMP generation pipeline (activity classifier)|Dong2024_AMP_activity_classifier|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|38402320|10.1038/s41598-024-55205-3|not_reported_in_available_evidence|not_reported_in_available_evidence|https://dbaasp.org/|True|Predicts antimicrobial activity; could be used in benchmark comparisons, though part of a generation pipeline.|[<br>"No code or web server available",<br>"Focused on C. acnes, not general AMP prediction",<br>"Not intended as a standalone benchmark model"<br>]|fulltext|0.8|
|BERT-based AMP recognition model|Zhang2021_BERT_AMP|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|34037687|10.1093/bib/bbab200|not_reported_in_available_evidence|not_reported_in_available_evidence|Six AMP datasets (not specified in abstract) and a new constructed AMP dataset|True|General AMP recognition model based on BERT, evaluated on multiple datasets, claims superiority over existing methods.|[<br>"No code or web server available",<br>"Fulltext not available; evidence from abstract only",<br>"Dataset details unclear"<br>]|abstract|0.7|
|AmpGPT2|AmpGPT2|protein_language_model_representation|transformer_llm_dominant_models|other|DL|42174216|10.1038/s44259-026-00218-3|https://imigitlab.uni-muenster.de/heiderlab/ampgpt2|not_reported_in_available_evidence|COMPASS database (https://compass.imi.uni-muenster.de)|True|Generative model for AMP sequences; can be used to evaluate novelty and diversity of generated AMPs; paper includes experimental validation.|[<br>"Not a direct AMP activity classifier; requires external classifier for evaluation."<br>]|fulltext|0.95|
|AMP-CapsNet|AMP-CapsNet|structure_graph_representation|gnn_models|AMP prediction|DL|41654884|10.1186/s44342-026-00067-6|not_reported_in_available_evidence|not_reported_in_available_evidence|derived from UniProt and previous study [31]; positive: 1085 AMPs, negative: 1316 non-AMPs|True|Novel AMP prediction classifier with reported accuracy of 97.29% and AUC of 98.91% on test set; uses capsule networks; could be included in benchmark.|[<br>"No code or model weights publicly available",<br>"Dataset not independently accessible"<br>]|fulltext|0.9|
|deepAMP|deepAMP|protein_language_model_representation|transformer_llm_dominant_models|other|DL|41753681|10.3390/microorganisms14020394|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Generative model for AMP design; could be included in benchmark of generative models.|[<br>"Original paper not in this batch; details sparse",<br>"No code availability reported"<br>]|fulltext|0.5|
|AMP-RL|AMP-RL|protein_language_model_representation|transformer_llm_dominant_models|AMP generation and optimization|DL|37992451|10.1016/j.sbi.2023.102733|https://github.com/GIST-CSBL/AMP-RL.|not_reported_in_available_evidence|PeptideAtlas, DBAASP v3 (no direct links provided)|True|New framework for strain-specific AMP generation and optimization; code and evaluation data available; could be used as a benchmark model.||fulltext|0.9|
|PepCVAE|PepCVAE|sequence_encoding_representation|cnn_dominant_models|AMP generation|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as a AMP generation model; potentially useful for benchmark.|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|PrefixProt|PrefixProt|sequence_encoding_representation|cnn_dominant_models|AMP generation / protein design|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as a controllable protein design method; applicable to AMPs.|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|MoFormer|MoFormer|sequence_encoding_representation|transformer_llm_dominant_models|AMP generation / multi-objective optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as a multi-objective optimization model for AMP design; potentially benchmark-relevant.|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|HMAMP|HMAMP|sequence_encoding_representation|cnn_dominant_models|AMP generation / multi-objective optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as a multi-objective optimization model for AMPs; could be a benchmark candidate.|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|AMP-Designer|AMP-Designer|protein_language_model_representation|transformer_llm_dominant_models|AMP generation / optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in review as a GPT-based AMP generation framework; relevant for benchmark.|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|AMP-MIC|AMP-MIC|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|29679519|10.1002/cmdc.201800204|https://github.com/jkwang93/AMP-Designer|not_reported_in_available_evidence|not_reported_in_available_evidence|True|AMP-MIC predicts antimicrobial activity (MIC) for peptides, directly relevant to AMP prediction; code available; uses large-scale data.|[<br>"Paper title/abstract conflict with fulltext; model named AMP-MIC is part of AMP-Designer, not a standalone AMP prediction model; needs verification of original publication."<br>]|fulltext|0.7|
|AP_Sin|AP_Sin|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38416364|10.1038/s41467-018-03746-3|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|新提出的 AMP 预测模型，在论文中与主流工具对比并展现优势，但代码和数据集未公开|[<br>"无代码仓库，训练数据未公开"<br>]|fulltext|0.7|
|AMP-Detector|AMP-Detector|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|DL/ML|39201537|10.3389/fmicb.2018.00323|not_reported_in_available_evidence|not_reported_in_available_evidence|Peptide Atlas (used for discovery)|True|提出了新的 AMP 检测流程，使用蛋白语言模型，性能优于现有方法，但代码和训练数据未公开|[<br>"无代码仓库，训练数据描述不完整"<br>]|fulltext|0.7|
|AMP-RNNpro|AMP-RNNpro|traditional_physicochemical_statistical_features|rnn_lstm_dominant_models|AMP identification|ML/DL|38839785|10.1038/s41598-024-63461-6|not_reported_in_available_evidence|http://13.126.159.30/|not_reported_in_available_evidence (combined dataset from XUAMP, DBAASP, LAMP, DRAMP)|True|AMP identification model with web server, high reported accuracy, suitable for benchmark.|[<br>"No code repository; web server only, may not be suitable for large-scale offline benchmarking."<br>]|fulltext|0.9|
|AMP-Distillation|AMP-Distillation|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction|DL|42155201|10.1016/j.compbiolchem.2026.109129|not_reported_in_available_evidence|not_reported_in_available_evidence|APD3 and DADP databases, CD-HIT deduplication|True|Novel AMP prediction model using knowledge distillation, strong performance metrics reported|[<br>"github_search_candidate_requires_manual_verification"<br>]|abstract|0.75|
|iAMP-SeE|iAMP-SeE|protein_language_model_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|41913931|10.7717/peerj.20978|https://github.com/cqw0715/iAMP-SeE.git|not_reported_in_available_evidence|Dataset 1: DRAMP, dbAMP, CAMPr-4, AMPfun, ADAPTABLE (positive), UniProt (negative); Dataset 2: from deep-AMPpred (Zhao et al. 2024); Zenodo data: https://doi.org/10.5281/zenodo.17398951|True|Open-source code, comprehensive AMP binary and multi-class classification, strong performance||fulltext|0.95|
|STAMP|STAMP|sequence_encoding_representation|machine_learning_models|AMP activity prediction (MIC prediction)|ML/DL|42155201|10.64898/2026.05.28.728246|not_reported_in_available_evidence|not_reported_in_available_evidence|Used three benchmark datasets including two previously published and a new curated dataset from DBAASP|True|Predicts MIC values, cross-species, strong performance (PCC 0.837, R^2 0.70), relevant for AMP discovery|[<br>"No code available in abstract"<br>]|abstract|0.7|
|CF-AMP prediction|CF-AMP prediction|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|42020672|10.1101/2022.11.16.516845|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel ML approach for AMP activity prediction, including combinations. Could be compared with other AMP predictors.|[<br>"No code or data availability",<br>"Preprint, not peer-reviewed",<br>"Only abstract evidence"<br>]|abstract|0.5|
|AMP-DualTransnet|AMP-DualTransnet|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction|DL|42020672|10.1016/j.nexres.2026.101536|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Specific deep learning model for AMP prediction in black pepper, could be benchmarked|[<br>"No abstract or full text",<br>"No code/data",<br>"Journal article with limited info"<br>]|abstract|0.3|
|AMP-FreqNet|AMP-FreqNet|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL||10.1145/3766671.3766835|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Title indicates a new deep learning-based AMP prediction model.|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|
|Collaborative Filtering and Link Prediction model|Unnamed AMP prediction model (Medvedeva et al. 2023)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.1021/acs.jcim.3c00137|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Title indicates a novel AMP activity prediction model using collaborative filtering and link prediction.|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|
|Predictive and Interpretable ML Models|Unnamed AMP prediction models (acsomega 2024)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.1021/acsomega.3c08676.s001|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Title indicates AMP discovery acceleration using predictive and interpretable ML models.|[<br>"No abstract, full text, or code available; only title evidence; possibly a supporting information file"<br>]|metadata|0.3|
|AMP prediction ML model|Unnamed AMP prediction model (Ahmad & Garg 2024)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.54985/peeref.2405p7278831|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Title explicitly states prediction of antimicrobial peptides using machine learning.|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|
|GAC-BiTCNN-AMP|GAC-BiTCNN-AMP|protein_language_model_representation|cnn_dominant_models|AMP prediction|DL|41844874|10.1038/s41598-026-43370-6|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (likely dbAMP 3.0 or similar, no explicit URL)|True|Novel AMP prediction model with deep learning architecture, reported performance on independent test, full text available|[<br>"code not reported"<br>]|fulltext|0.9|
|CVAE-BIO|CVAE-BIO|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML/DL|41849223|10.1093/bib/bbag115|https://github.com/scan2030|not_reported_in_available_evidence|APD3 (http://aps.unmc.edu/)|True|Multi-module pipeline with AMP classifier; experimental validation|[<br>"code availability unclear"<br>]|fulltext|0.85|
|AMPGAN|AMPGAN|sequence_encoding_representation|cnn_dominant_models|AMP generation / prediction|DL|41463765|10.3390/antibiotics14121263|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|GAN-based AMP model mentioned in review; needs original paper|[<br>"needs original paper verification"<br>]|review|0.35|
|Macrel|Macrel|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in a review as a tool for genome/metagenome AMP screening.|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|iAMPpred|iAMPpred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|41463765|10.3390/antibiotics14121263|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|SVM-based AMP predictor mentioned in review; needs original paper|[<br>"needs original paper verification"<br>]|review|0.35|
|AMP-GPT|AMP-GPT|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide generation|DL|40193623|10.1038/s44386-026-00045-6|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Generative model for AMPs; can be used in benchmarking pipelines for de novo AMP design.|[<br>"No code or trained model weights provided",<br>"Training data details missing"<br>]|fulltext|0.85|
|MCL-AMP|MCL-AMP|protein_language_model_representation|cnn_dominant_models|AMP prediction|DL|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel AMP prediction ensemble model with high AUC, but code and data not available|[<br>"code not available",<br>"training data not reported",<br>"no external test details"<br>]|fulltext|0.7|
|MAPLE|MAPLE|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|39792442|10.1021/acs.jcim.4c01913|https://github.com/Harkool/MAPLE|not_reported_in_available_evidence|Benchmark dataset: integrated from dbAMP, DBAASP, APD3, DRAMP, etc. (no single download link); 25,507 AMPs and 72,606 non-AMPs. Independent validation set: 24,582 AMPs, 36,653 non-AMPs.|True|Comprehensive AMP predictor covering 14 functional activities; code available for benchmarking; strong performance on independent test set.||fulltext|0.9|
|PepVAE|PepVAE|sequence_encoding_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|34659152|10.3389/fmicb.2021.725727|not_reported_in_available_evidence|not_reported_in_available_evidence|https://github.com/zswitten/Antimicrobial-Peptides|True|Provides both generative and predictive components for AMP activity; regression models trained on public data and experimentally validated|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|
|LMPred|LMPred|sequence_encoding_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|36699381|10.1101/2020.07.12.199554v3|https://github.com/williamdee1/LMPred_AMP_Prediction|not_reported_in_available_evidence|https://github.com/williamdee1/LMPred_AMP_Prediction|True|Novel input representation using pre-trained language models; code and data publicly available; outperforms previous state-of-the-art||fulltext|0.95|
|AMP prediction SVM-LZ|AMP prediction by SVM-LZ complexity|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML|25802839|10.1093/nar/gkn823|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Method for AMP prediction with reported sensitivity; could be included in benchmarks if implementation becomes available|[<br>"No code or model provided",<br>"Full-text cache mismatch (APD2 database article)"<br>]|abstract|0.6|
|DDM|DDM|protein_language_model_representation|transformer_llm_dominant_models|AMP classification|DL|41692989|10.1093/bioinformatics/btag077|https://github.com/kww567upup/DDM|not_reported_in_available_evidence|https://github.com/kww567upup/DDM (data provided in repository)|True|AMP classification model with code and data available, suitable for inclusion in benchmark comparison.||fulltext|0.95|
|UniAMP|UniAMP|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction|DL|39799358|10.1186/s12859-025-06033-3|not_reported_in_available_evidence|https://amp.starhelix.cn|not_reported_in_available_evidence (dataset constructed from public AMP databases, no direct download link)|True|AMP prediction model with web server, published results, useful for benchmark.|[<br>"No code repository found, only web server; reproducibility may be limited."<br>]|fulltext|0.9|
|AMP Scanner|AMP Scanner|sequence_encoding_representation|cnn_rnn_hybrid_models|AMP prediction|DL|38129980|10.1002/mbo3.1393|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Listed as AMP prediction tool in review, further verification needed|[<br>"no code or data link",<br>"only mentioned in review"<br>]|review|0.5|
|AMP Scanner v2|Antimicrobial Peptide Scanner vr.2|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|29590297|10.1093/bioinformatics/bty179|not_reported_in_available_evidence|http://www.ampscanner.com|provided through the web server (not specified in evidence)|True|Deep learning model for AMP recognition, openly available web server, benchmarked against state-of-the-art|[<br>"Not original publication; limited architecture details provided."<br>]|fulltext|0.95|
|PepGen 1.0|PepGen 1.0|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide generation|DL|40643674|10.1007/s00284-025-04346-3|not_reported_in_available_evidence|https://bit.ly/2Z281cY|not_reported_in_available_evidence|True|AMP sequence generator based on LSTM, used for de novo AMP design and screening.|[<br>"No source code repository found; only a shortened URL provided."<br>]|fulltext|0.8|
|AmPepGen|AmPepGen|sequence_encoding_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide generation|DL|40643674|10.1007/s00284-025-04346-3|https://github.com/Anorpe/ampepgen-dev|not_reported_in_available_evidence|not_reported_in_available_evidence|True|AMP generator using GAN, with code available on GitHub, used for de novo AMP design.|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|
|AMP-SEMiner|AMP-SEMiner|sequence_encoding_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|40445833|10.1016/j.celrep.2025.115773|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel AMP prediction model using deep learning (PLMs) with experimental validation; suitable for benchmarking against other AMP predictors.||fulltext|0.9|
|Unnamed AMP predictor from DRAMP 2.0|DRAMP_ML_model|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|31409791|10.1038/s41597-019-0154-y|not_reported_in_available_evidence|not_reported_in_available_evidence|DRAMP database (http://dramp.cpu-bioinfor.org/)|True|Described as a predictive classifier for AMPs, with reported performance, but lacks name and public code; may be available in future DRAMP updates.|[<br>"Model name not provided",<br>"No code or web server link available",<br>"Not yet integrated into DRAMP as stated"<br>]|fulltext|0.5|
|CalcAMP|CalcAMP|structure_graph_representation|gnn_models|AMP prediction|ML|37107088|10.3390/antibiotics12040725|https://github.com/CDDLeiden/CalcAMP|not_reported_in_available_evidence|https://doi.org/10.5281/zenodo.7588702|True|Publicly available code and dataset; specific models for Gram+ and Gram-; uses experimentally validated non-AMPs as negatives; suitable for benchmarking.||fulltext|0.95|
|ANN-based AMP prediction model (Torrent et al. 2011)|Torrent-2011-ANN|structure_graph_representation|gnn_models|AMP prediction|ML|21347392|10.1371/journal.pone.0016968|not_reported_in_available_evidence|not_reported_in_available_evidence|CAMP database (http://www.camp.bicnirrh.res.in/) and Uniprot; no direct download link provided|True|Early AMP prediction model (2011) widely cited; full-text available; classification accuracy 90%; can serve as historical baseline.|[<br>"No code or web server available",<br>"Uses old feature set (8 physicochemical descriptors)"<br>]|fulltext|0.9|
|Deep learning regression model for antimicrobial peptide design (Witten & Witten 2019)|Witten-2019-CNN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|21347392|10.1101/692681|https://github.com/zswitten/Antimicrobial-Peptides|not_reported_in_available_evidence|GRAMPA database; not directly linked but likely included in the GitHub repository|True|Provides code and dataset; CNN model for AMP activity prediction; outperformed state-of-the-art at classification; includes regression capability.|[<br>"Preprint status (no peer-reviewed publication yet)",<br>"Full text not available to confirm details"<br>]|abstract|0.8|
|AMP-zGSM|AMP-zGSM|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|21347392|10.5220/0014457300004070|https://github.com/DemetParlakSonmez/amp-zGSM|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel feature-ranking framework; high AUCs (0.9737, 0.8846, 0.97); code and datasets publicly available.|[<br>"Conference paper (may not be exhaustive)",<br>"Year listed as 2026 (potential future publication or error)",<br>"Full text unavailable for detailed method verification"<br>]|abstract|0.7|
|AMP0|AMP0|traditional_physicochemical_statistical_features|machine_learning_models|targeted antimicrobial peptide prediction|ML|32750857|10.1109/TCBB.2020.2999399|not_reported_in_available_evidence|http://ampzero.pythonanywhere.com|not_reported_in_available_evidence|True|AMP prediction model with targeted species prediction capability, webserver available|[<br>"code not clearly available",<br>"limited training and test data details"<br>]|abstract|0.8|
|sAMPpred-GAT|sAMPpred-GAT|structure_graph_representation|gnn_models|antimicrobial peptide classification|DL|36342186|10.1093/bioinformatics/btac715|https://github.com/HongWuL/sAMPpred-GAT/|http://bliulab.net/sAMPpred-GAT|https://github.com/HongWuL/sAMPpred-GAT/ (likely includes datasets)|True|State-of-the-art AMP predictor using predicted structure, outperforms existing methods, code and webserver available|[<br>"lack of original publication details from review"<br>]|abstract|0.9|
|PyAMPA|PyAMPA|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML/feature-engineering/web-server|38934543|10.1128/msystems.01358-23|https://github.com/SysBioUAB/PyAMPA|not_reported_in_available_evidence|AMPlify dataset, Liu et al. CPP database, AMPDeep hemolytic database, ToxinPred toxicity database, GRAMPA database (https://github.com/zswitten/Antimicrobial-Peptides)|True|PyAMPA是一个集筛选、验证、性质预测和优化于一体的AMP发现平台，具有高通量蛋白组扫描能力，内部采用随机森林和遗传算法，预测抗菌、溶血、毒性、半衰期等多维性质，并提供了实验验证。适合作为AMP预测benchmark中的综合性工具。||fulltext|0.95|
|AMPA|AMPA|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|web-server|40410382|10.1038/s44320-025-00120-6|not_reported_in_available_evidence|http://tcoffee.crg.cat/apps/ampa|not_reported_in_available_evidence|True|AMPA is a computational tool for predicting antimicrobial regions in protein sequences, directly applicable to AMP prediction benchmarks.|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|
|AntiBP3|AntiBP3|traditional_physicochemical_statistical_features|machine_learning_models|antibacterial peptide classification / antimicrobial peptide classification|ML|38391554|10.3390/antibiotics13020168|https://gitlab.com/raghavalab/antibp3|https://webs.iiitd.edu.in/raghava/antibp3|not_reported_in_available_evidence (training data compiled from public databases, no direct download link provided)|True|Provides a new method for predicting antibacterial peptides against three bacterial groups, with web server, standalone package, and comparison with existing tools.||fulltext|0.95|
|AMPActiPred|AMPActiPred|traditional_physicochemical_statistical_features|machine_learning_models|antibacterial peptide classification and activity prediction|ML|38723168|10.1002/pro.5006|not_reported_in_available_evidence|https://awi.cuhk.edu.cn/∼AMPActiPred/|not_reported_in_available_evidence (elaborate dataset constructed from public sources, no direct download link)|True|Proposes a three-stage framework for ABP identification and activity prediction, with web server, and claims state-of-the-art performance.|[<br>"code not independently available"<br>]|fulltext|0.9|
|APEX|APEX|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|DL|38862735|10.1038/s41551-024-01201-x|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Deep learning model for antibiotic discovery from extinct organisms; potential benchmark candidate|[<br>"code availability not confirmed from review"<br>]|review|0.6|
|AMPfinder|AMPfinder|sequence_encoding_representation|cnn_dominant_models|AMP discovery / prediction|DL|39540425|10.1093/nar/gkae1019|not_reported_in_available_evidence|https://awi.cuhk.edu.cn/dbAMP/|dbAMP database|True|集成在 dbAMP 3.0 中的 AMP 发现工具，可用于从基因组/宏基因组数据中挖掘 AMP。|[<br>"code not independently available"<br>]|fulltext|0.9|
|AMPpredictor|AMPpredictor|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML|39540425|10.1093/nar/gkae1019|not_reported_in_available_evidence|https://awi.cuhk.edu.cn/dbAMP/|dbAMP database|True|集成在 dbAMP 3.0 中的机器学习工具，用于预测 AMP 功能活性。|[<br>"code not independently available"<br>]|fulltext|0.9|
|AMPBAN|AMPBAN|protein_language_model_representation|gnn_models|AMP prediction|DL||10.64898/2026.01.20.700468|https://github.com/baiwenhuim/ampban|not_reported_in_available_evidence|https://github.com/baiwenhuim/ampban (dataset in repository)|True|Novel deep learning AMP predictor using multimodal fusion; outperforms nine state-of-the-art models; code and datasets publicly available.||abstract|0.85|
|Generative AMP pipeline (VINCI)|VINCI AMP generator|protein_language_model_representation|rnn_lstm_dominant_models|AMP generation and MIC prediction|DL||10.64898/2026.06.16.732639|not_reported_in_available_evidence|not_reported_in_available_evidence|AMPSphere, DBAASP (links not provided)|True|Transfer learning-based AMP generator with MIC prediction; open-source announced; evaluation planned via AMP Challenge.|[<br>"code link not found in abstract; full text needed for repository access"<br>]|abstract|0.7|
|AMPCLGPT|AMPCLGPT|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide classification|DL||10.1101/2025.03.07.642021|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel GPT-based AMP generator and MIC predictor, suitable for benchmarking|[<br>"no code",<br>"no full text",<br>"preprint"<br>]|abstract|0.7|
|CAmidPred|CAmidPred|protein_language_model_representation|cnn_dominant_models|antimicrobial peptide classification|DL||10.21203/rs.3.rs-7764304/v1|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel AMP prediction model targeting amidated AMPs, uses ESM2|[<br>"no code available",<br>"no full text",<br>"preprint"<br>]|abstract|0.7|
|PepMCP|PepMCP|structure_graph_representation|cnn_dominant_models|antimicrobial peptide classification|DL||10.64898/2026.02.01.703163|https://github.com/ComputBiophys/PepMCP|not_reported_in_available_evidence|MemAMPdb (described in paper, no explicit link)|True|Novel graph-based predictor with code and database, suitable for benchmarking|[<br>"preprint",<br>"no full text",<br>"no web server URL"<br>]|abstract|0.8|
|iMFP-LG|iMFP-LG|protein_language_model_representation|gnn_models|multi-functional peptide prediction including antimicrobial peptide classification|DL|39585308|10.1093/gpbjnl/qzae084|https://github.com/chen-bioinfo/iMFP-LG|https://ngdc.cncb.ac.cn/biocode/tools/BT007494|not_reported_in_available_evidence|True|Directly predicts antimicrobial peptide function as part of multi-functional peptide identification; outperforms state-of-the-art methods on public benchmarks; code available||fulltext|0.95|
|Deep learning model for AMP discovery from ruminant gastrointestinal microbiomes|not_provided_in_evidence|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39756573|10.1016/j.jare.2025.01.005|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel deep learning model for AMP prediction with experimental validation; could be included in future benchmarks if code/data become available.|[<br>"No code or model weights provided",<br>"Dataset not publicly linked",<br>"Full-text cache mismatch (PMCID may be incorrect)"<br>]|abstract|0.7|
|amPEPpy|amPEPpy|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|33135060|10.1093/bioinformatics/btaa917|https://github.com/tlawrence3/amPEPpy|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Open-source AMP prediction tool using random forest; could be included in benchmark comparison.||abstract|0.8|
|panCleave|panCleave|sequence_encoding_representation|machine_learning_models|AMP prediction|ML|37516110|10.1016/j.chom.2023.07.001|https://gitlab.com/machine-biology-group-public/pancleave|not_reported_in_available_evidence|Training and test data (MEROPS substrates) available in the panCleave repository (https://gitlab.com/machine-biology-group-public/pancleave)|True|Machine learning pipeline for mining encrypted antimicrobial peptides from proteomes; panCleave random forest predicts cleavage sites, and the resulting fragments are screened for antimicrobial activity, enabling proteome-wide AMP prospection.||fulltext|0.9|
|Bacteria-specific ML models for E. coli AMP activity|Bacteria-specific ML models for E. coli AMP activity|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|feature-engineering|36912047|10.1021/acs.jcim.2c01551|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Proposes a supervised ML pipeline for E. coli-specific AMP prediction; methodology and features are described but no code or data are provided.|[<br>"No public code or dataset",<br>"Only E. coli activity, not general AMP prediction"<br>]|abstract|0.5|
|XGBoost AMP prediction model (Bhangu2025)|XGBoost AMP prediction model (Bhangu2025)|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|40529865|10.1002/smsc.202400579|not_reported_in_available_evidence|not_reported_in_available_evidence|http://cabgrid.res.in:8080/amppred/about.html (and other AMP databases)|True|Full-text available; model is well-described, trained on standard AMP data, and experimentally validated. Suitable for benchmark if code or weights become available.|[<br>"No public code or model weights",<br>"No web server"<br>]|fulltext|0.8|
|AMPGAN v3|AMPGAN v3|sequence_encoding_representation|cnn_dominant_models|other|DL|42364293|10.1016/j.jmgm.2026.109497|https://github.com/marszzibros/AMPGANv3|not_reported_in_available_evidence|https://github.com/marszzibros/AMPGANv3 (likely contains data)|True|Novel generative AMP model with code, in vitro validation, and reported superiority over prior models, suitable for AMP prediction benchmarks.||abstract|0.9|
|PepAnno|PepAnno|structure_graph_representation|transformer_llm_dominant_models|AMP prediction|DL|42228741|10.1371/journal.pcbi.1014369|not_reported_in_available_evidence|https://bis.zju.edu.cn/pepanno/|not_reported_in_available_evidence|True|Comprehensive DL web server for AMP prediction among other bioactivities, competitive performance, publicly available.||abstract|0.8|
|AMPGP|AMPGP|traditional_physicochemical_statistical_features|cnn_dominant_models|antimicrobial peptide classification|DL|40825014|10.1021/acs.jcim.5c00647|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|High accuracy (98.46%) on independent test set; novel deep learning combining generation and prediction.|[<br>"No code or dataset availability reported",<br>"Only abstract available, no full text"<br>]|abstract|0.7|
|AmpGram|AmpGram|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|ML|32560350|10.3390/ijms21124310|not_reported_in_available_evidence|not_reported_in_available_evidence|Training data not detailed; benchmarked on APD3 and DAMPD datasets|True|Published AMP prediction tool with web-server and R package, outperforms top-ranking classifiers; designed for long AMPs and proteomic screening.|[<br>"Original code not in evidence",<br>"Web server not reported"<br>]|fulltext|1.0|
|Ampir|ampir|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|DL|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Reimplemented in benchmark study, deep learning method|[<br>"Original code not available"<br>]|fulltext|0.6|
|Ensemble-AMPPred|Ensemble-AMPPred|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in a review as an ensemble-based AMP predictor.|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|CancerGram|CancerGram|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification||38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in a review as a tool for distinguishing anticancer and antibacterial peptides; relevant for AMP prediction.|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|PPTPP|PPTPP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification||38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in a review as a multi-function peptide predictor including antibacterial activity.|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|MLBP|MLBP|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification||38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in a review as a multi-function peptide predictor including antimicrobial activity.|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|Deep2Pep|Deep2Pep|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Mentioned in a review as a deep learning tool for discovering antimicrobial, antihypertensive, and antihyperglycemic bioactivities.|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|Pore-Forming_AMP_SVM|Pore-Forming AMP SVM|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide classification|ML|41391039|10.1002/advs.202516470|https://github.com/ComputBiophys/Pore%E2%80%90Forming_AMP_SVM|not_reported_in_available_evidence|https://github.com/ComputBiophys/Pore%E2%80%90Forming_AMP_SVM (training data included)|True|Publicly available code and data, SVM classifier for AMP prediction with novel mechanism-driven screening pipeline.||fulltext|0.95|
|CG-AMP|CG-AMP|protein_language_model_representation|gnn_models|antimicrobial peptide classification|DL|41286313|10.1038/s41598-025-29666-z|not_reported_in_available_evidence|not_reported_in_available_evidence|AMPlify and DAMP benchmark datasets|True|State-of-the-art DL model for AMP prediction with strong benchmark results, but code not provided.|[<br>"Code not available"<br>]|fulltext|0.85|
|AmpHGT|AmpHGT|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide classification|DL|40598389|10.1186/s12915-025-02253-4|not_reported_in_available_evidence|not_reported_in_available_evidence|XUAMP, AMPDiscover, NCAA datasets|True|Advanced model handling non-canonical amino acids, competitive on benchmarks, but code not provided.|[<br>"Code not available"<br>]|fulltext|0.85|
|SGAC|SGAC|structure_graph_representation|gnn_models|antimicrobial peptide classification|DL|41662353|10.1093/bib/bbag038|https://github.com/wyxwyx46941930/SGAC|not_reported_in_available_evidence|not_reported_in_available_evidence (paper states 'publicly available AMP and non-AMP datasets')|True|Novel GNN-based AMP classifier with publicly available code; proposes a structure-aware approach and addresses class imbalance.||fulltext|0.95|
|Bidirectional LSTM AMP classification model (Wang2021)|Wang2021_LSTM_AMP|sequence_encoding_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|33810011|10.3390/biom11030471|not_reported_in_available_evidence|not_reported_in_available_evidence|CAMP, DBAASP, DRAMP, YADAMP, UniProt (as described in Methods)|True|AMP classification model using bidirectional LSTM, described in fulltext, could be evaluated if code is available.|[<br>"Code not publicly available in a repository, only in Supplementary Materials; unclear if code is accessible."<br>]|fulltext|0.8|
|PrMFTP|PrMFTP|sequence_encoding_representation|cnn_dominant_models|multi-functional therapeutic peptide prediction (including AMP classes like ABP, AFP, AVP, etc.)|DL|36094961|10.1371/journal.pcbi.1010511|not_reported_in_available_evidence|http://bioinfo.ahu.edu.cn/PrMFTP|not_reported_in_available_evidence (constructed from 22 therapeutic peptide datasets; no direct download link provided in evidence)|True|Predicts multiple therapeutic peptide functions including AMP categories (antibacterial, antifungal, antiviral, etc.), provides web server, and can be used for AMP prediction tasks.|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|
|DeepAFP|DeepAFP|traditional_physicochemical_statistical_features|cnn_dominant_models|antifungal peptide prediction (AFP identification)|DL|37595093|10.1002/pro.4758|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (DeepAFP-Main dataset, curated, no direct link provided)|True|Specific deep learning model for AFP identification, state-of-the-art performance, downloadable tool mentioned but no URL found.|[<br>"code_repository_not_found",<br>"web_server_not_found",<br>"downloadable_tool_url_missing"<br>]|fulltext|0.85|
|AMPpred|AMPpred|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide prediction|web-server|37914524|10.24272/j.issn.2095-8137.2023.246|not_reported_in_available_evidence|http://cabgrid.res.in:8080/amppred/|not_reported_in_available_evidence|True|AMP prediction web server mentioned in review.|[<br>"needs_original_publication"<br>]|fulltext|0.5|
|AMPpred-AAIW|AMPpred-AAIW|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|37120707|10.1142/S0219720023500063|not_reported_in_available_evidence|https://amppred-aaiw.com|DRAMP and other published databases (not reported as link)|True|Proposed AMP recognition models using AAIW encoding, achieved over 93% accuracy and 0.87 MCC on independent test sets, web server available.||abstract|0.9|
|MIC prediction ensemble model (BiLSTM-CNN-MBM)|MIC prediction ensemble model|sequence_encoding_representation|cnn_dominant_models|AMP prediction / MIC prediction|DL|39262770|10.48550/arXiv.1810.11363|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Novel regression model for AMP minimum inhibitory concentration prediction; directly relevant to AMP benchmark.||fulltext|0.8|
|AMPpred-EL|AMPpred-EL|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / antimicrobial peptide classification|ML|35576825|10.1016/j.compbiomed.2022.105577|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Explicit AMP prediction model with reported benchmark performance, suitable for inclusion.||fulltext|0.9|
|AMPpred-MFA|AMPpred-MFA|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|DL||10.1021/acs.jcim.3c01017.s001|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|明确的 AMP 预测模型，采用 stacking 架构和多头注意力，适合纳入基准测试。||metadata|0.8|
|Multifunctional AMP Design Framework (FBGAN-enhanced)|Multifunctional AMP Design Framework|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|Integrated from GRAMPA, APD3, ADAM, CAMPR4, UniProt|True|Novel deep learning framework for multifunctional AMP generation and prediction; could be used for benchmarking if code becomes available|[<br>"code not available",<br>"no web server"<br>]|abstract|0.7|
|AMPpredMFA|AMPpredMFA|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Published AMP prediction model; possible benchmark candidate|[<br>"lack of original publication details from review"<br>]|review|0.5|
|AMP-META|AMP-META|sequence_encoding_representation|cnn_dominant_models|AMP prediction (strain-specific)|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Strain-specific AMP predictor; potential benchmark|[<br>"lack of original publication details from review"<br>]|review|0.5|
|MBC-attention|MBC-attention|sequence_encoding_representation|cnn_dominant_models|AMP prediction (MIC regression)|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|CNN-based MIC predictor for E. coli; could be benchmarked|[<br>"lack of original publication details from review"<br>]|review|0.5|
|EnDL-HemoLyt|EnDL-HemoLyt|sequence_encoding_representation|cnn_dominant_models|AMP toxicity prediction|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Toxicity predictor for AMPs; could be benchmarked|[<br>"lack of original publication details from review"<br>]|review|0.5|
|SenseXAMP|SenseXAMP|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction|DL|40806517|10.3390/ijms26157387|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|AMP predictor using fused embeddings; could be benchmarked|[<br>"lack of original publication details from review"<br>]|review|0.5|
|AniAMPpred|AniAMPpred|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML/DL|34259329|10.1093/bib/bbab242|not_reported_in_available_evidence|https://aniamppred.anvil.app/|not_reported_in_available_evidence|True|AMP prediction model with SVM and deep features, online server available; abstract describes high accuracy and non-biased classification.|[<br>"no code available",<br>"fulltext provided does not match article (PMC12620532 is a different paper); only abstract evidence used"<br>]|abstract|0.7|
|Appred|Appred|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|ML|39247292|10.1016/j.heliyon.2024.e36163|not_reported_in_available_evidence|www.soodlab.com/appred|not_reported_in_available_evidence|True|Predicts antiprotozoal peptides (AMP subclass) with high accuracy, provides web server, suitable for benchmarking.|[<br>"no code available",<br>"dataset not publicly linked"<br>]|fulltext|0.9|
|AMPs-Net|AMPs-Net|structure_graph_representation|gnn_models|antimicrobial peptide classification|DL|35877911|10.3389/fmicb.2021.710199|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Deep learning model explicitly designed for AMP prediction; outperforms prior method by 8.8% in average precision; handles both antibacterial and antiviral activity.|[<br>"only review evidence",<br>"no code or server found"<br>]|abstract|0.9|
|LABAMPs|LABAMPs|structure_graph_representation|gnn_models|antimicrobial peptide classification|DL|37521317|10.3389/fbinf.2023.1216362|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Predicts lactic acid bacteria AMPs, outperformed other ML algorithms according to review.|[<br>"only review evidence",<br>"no code or server found"<br>]|review|0.4|
|LSTM-based AMP classifier/generator|LSTM AMP classifier (Wang et al. 2021)|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|33810011|10.1016/j.diagmicrobio.2004.02.008|not_reported_in_available_evidence|not_reported_in_available_evidence|not reported (likely from public databases)|True|Deep learning (LSTM) model for AMP classification, potentially useful for benchmark|[<br>"code not available",<br>"no web server reported"<br>]|fulltext|0.8|
|AMPSpeciesSpecific|AMPSpeciesSpecific|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|DL|39766503|10.3390/antibiotics13121113|https://github.com/bzlee-bio/AMPSpeciesSpecific|not_reported_in_available_evidence|https://github.com/bzlee-bio/AMPSpeciesSpecific (may contain data)|True|AMP prediction model with public code, used in a discovery pipeline with experimental validation||fulltext|0.9|
|PepNet|PepNet|sequence_encoding_representation|cnn_dominant_models|AMP prediction / anti-inflammatory peptide classification|DL|39341947|10.1038/s42003-024-06911-1|https://zenodo.org/records/1322351661, https://zenodo.org/records/1373425862|http://liulab.top/PepNet/server|not_reported_in_available_evidence (described as AMP and AIP test sets from previous studies; likely included in Zenodo records)|True|Novel deep learning model for AMP and AIP prediction with web server and code, outperforming existing methods||fulltext|0.95|
|BPFun|BPFun|protein_language_model_representation|cnn_dominant_models|antimicrobial peptide classification / bioactive peptide function prediction|DL|40691539|10.1186/s12859-025-06190-5|https://github.com/291357657/BPFun|not_reported_in_available_evidence|https://github.com/291357657/BPFun (data included in repository)|True|Multi-label deep learning model for bioactive peptide prediction including AMP, with code and data available||fulltext|0.95|
|LLAMP|LLAMP|protein_language_model_representation|cnn_dominant_models|AMP prediction (MIC prediction, species-aware)|DL|40676915|10.1093/bib/bbaf343|https://github.com/GIST-CSBL/LLAMP|not_reported_in_available_evidence|https://github.com/GIST-CSBL/LLAMP (data included); DBAASP v3 for MIC data|True|Species-aware AMP activity prediction model based on language model, with code and data, predicts MIC values||fulltext|0.95|
|CL-ACP|CL-ACP|structure_graph_representation|cnn_dominant_models|AMP prediction / antimicrobial peptide classification|DL|34670488|10.1186/s12859-021-04433-9|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Explicit AMP/ACP prediction model; evaluated on standard datasets with published metrics; suitable for benchmark comparison|[<br>"No code repository or web server provided"<br>]|fulltext|0.9|
|AMPTrans-lstm|AMPTrans-lstm|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|36618982|10.1016/j.csbj.2022.12.029|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Contains AMP activity prediction component (QSAR classifiers) and generates novel AMPs; could be used for benchmarking AMP prediction pipelines|[<br>"Primary purpose is AMP generation, not classification; no standard benchmark testing; no code available"<br>]|fulltext|0.7|
|CSAMPPRED|CSAMPPRED|traditional_physicochemical_statistical_features|machine_learning_models|AMPs classification|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Well-known AMP prediction model, reimplemented in benchmark study|[<br>"Original code not available",<br>"Web server link not reported in evidence"<br>]|fulltext|0.7|
|Thomas et al. 2009 AMP prediction model|Thomas et al. 2009 AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Early influential model, referenced in review|[<br>"Original paper not available",<br>"Code not available"<br>]|fulltext|0.4|
|ANN-based AMP prediction model (ref [4])|ANN-based AMP prediction model (ref [4])|structure_graph_representation|gnn_models|AMPs prediction|DL|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Referenced ANN model|[<br>"No name, no code, no original paper in evidence"<br>]|fulltext|0.3|
|Multiple alignment based AMP predictor (ref [5])|Multiple alignment based AMP predictor (ref [5])|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Referenced model with good performance|[<br>"No name, no code, no original paper"<br>]|fulltext|0.3|
|Two-level fuzzy K-NN model (ref [7])|Two-level fuzzy K-Nearest Neighbor model (ref [7])|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Referenced model|[<br>"No name, no code, no original paper"<br>]|fulltext|0.3|
|Sequence alignment-SVM-LZ complexity model (ref [8])|Sequence alignment-SVM-LZ complexity model (ref [8])|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|High sensitivity reported (95.28% in jackknife)|[<br>"No name, no code, no original paper"<br>]|fulltext|0.3|
|Anti-Hepatitis Peptides predictor (ref [9])|Anti-Hepatitis Peptides predictor (ref [9])|traditional_physicochemical_statistical_features|machine_learning_models|specific anti-hepatitis peptide prediction|ML|29379261|10.6026/97320630013415|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Referenced model with 94% accuracy (10-fold CV)|[<br>"Specific to anti-hepatitis, not general AMP",<br>"No code, no original paper"<br>]|fulltext|0.4|
|AmpClass|AmpClass|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|39383429|10.1590/0001-3765202420230756|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|New AMP predictor reported to outperform classical models|[<br>"No code or web server available",<br>"Dataset source not disclosed"<br>]|fulltext|0.8|
|Gabere&Noble AMP predictor|Gabere&Noble AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Reimplemented in benchmark study|[<br>"No model details",<br>"Original code not available"<br>]|fulltext|0.4|
|Wang et al. AMP predictor|Wang et al. AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Reimplemented in benchmark study|[<br>"No model details",<br>"Original code not available"<br>]|fulltext|0.4|
|Witten&Witten AMP predictor|Witten&Witten AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Reimplemented in benchmark study|[<br>"No model details",<br>"Original code not available"<br>]|fulltext|0.4|
|Unnamed CVAE-diffusion AMP generator|Unnamed CVAE-diffusion AMP generator|protein_language_model_representation|transformer_llm_dominant_models|AMP generation and activity prediction|DL|41460918|10.1371/journal.pcbi.1013833|not_reported_in_available_evidence|not_reported_in_available_evidence|UniProt (uniprotkb_reviewed_true_2024_12_17.fasta) for pretraining; GRAMPA for fine-tuning and MIC training|True|Novel generative framework with MIC predictor; can be benchmarked against other AMP design/prediction tools.|[<br>"Code not available",<br>"No public web server or weights"<br>]|fulltext|0.95|
|Malebary-Khan AMP predictor|Malebary-Khan AMP predictor|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38391554|10.32604/cmc.2021.015041|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Simple ML-based AMP classifier with reported accuracy of 95.43%; could serve as baseline.|[<br>"Code and dataset not available",<br>"No web server or detailed algorithm description"<br>]|abstract|0.5|
|APIN|APIN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|31870282|10.1093/bioinformatics/btx679|https://github.com/zhanglabNKU/APIN|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Deep learning model with available code, outperformed state-of-the-art on multiple AMP datasets including APD3 benchmark||abstract|0.9|
|Co-AMPpred GitHub repository|Co-AMPpred GitHub repository|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/onkarS23/CoAMPpred||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|CoAMPpred|CoAMPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/onkarS23/CoAMPpred||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|2020-peptidomics|2020-peptidomics|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/ErikHartman/2020-peptidomics||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AAGP|AAGP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aagpazos/aagpazos.github.io||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.28|
|ACP-DL|ACP-DL|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/haichengyi/ACP-DL||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Anticancer-Peptides-CNN|Anticancer-Peptides-CNN|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/RafsanjaniHub/Anticancer-Peptides-CNN||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|MetagenomicDC|MetagenomicDC|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/IcarPA-TBlab/MetagenomicDC||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|deep-belief-network|deep-belief-network|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/albertbup/deep-belief-network||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|acp-ope|acp-ope|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/khanhlee/acp-ope||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|2022-iAMP-DL|2022-iAMP-DL|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/mldlproject/2022-iAMP-DL||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|AFP_DL|AFP_DL|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/DongYin521/AFP_DL-QSARES||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|AFP_DL-QSARES|AFP_DL-QSARES|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/DongYin521/AFP_DL-QSARES||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|ANIA_github|ANIA_github|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aniagithub/Nieliniowe||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|0.85|
|PC6-protein-encoding-method|PC6-protein-encoding-method|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/LinTzuTang/PC6-protein-encoding-method||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|BAGEL4|BAGEL4|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/ByteDance-Seed/Bagel||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|LinearDisplay|LinearDisplay|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/JCVenterInstitute/LinearDisplay||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|msaconverter|msaconverter|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/linzhi2013/msaconverter||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|LysePred|LysePred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/lincubator/LysePred||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AI4AVP_predictor|AI4AVP_predictor|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/LinTzuTang/AI4AVP_predictor||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|AMP-researchprotein|AMP-researchprotein|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/researchprotein/amp||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|0.92|
|learning_sequence_motifs|learning_sequence_motifs|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/p-koo/learning_sequence_motifs||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|AMP-BERT GitHub repository|AMP-BERT GitHub repository|protein_language_model_representation|transformer_llm_dominant_models|||||https://github.com/GIST-CSBL/AMP-BERT||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|LightGBM|LightGBM|traditional_physicochemical_statistical_features|machine_learning_models|||||https://github.com/lightgbm-org/LightGBM||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|shap|shap|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/shap/shap||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|COMPASS database|COMPASS database|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aaronpk/Compass||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AMP-RNNpro web server|AMP-RNNpro web server|sequence_encoding_representation|rnn_lstm_dominant_models|||||https://github.com/Shazzad-Shaon3404/Website_AMPRNNpro||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|deep_AMPpred|deep_AMPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/JunZhao-hash/deep_AMPpred||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|ADAM_web_server|ADAM_web_server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/urban-adam/urban-adam-web||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|ampsphere_web_server|ampsphere_web_server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/BigDataBiology/AMPSphereWebsite||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|MAPLE GitHub repository|MAPLE GitHub repository|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/abdulrahmanbinayub-maker/maple-github-repository||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Antimicrobial-Peptides|Antimicrobial-Peptides|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zswitten/Antimicrobial-Peptides||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|LMPred_AMP_Prediction|LMPred_AMP_Prediction|protein_language_model_representation|cnn_dominant_models|||||https://github.com/williamdee1/LMPred_AMP_Prediction||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|CDPfold|CDPfold|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zhangch994/CDPfold||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|DDM GitHub|DDM GitHub|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/DDM-Mzp/ddm.github.io||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|UniAMP web server|UniAMP web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Dextro86/Webasto-Ampure-Unite-Home-Assistant-custom-integration||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.38|
|PepProtGraphAnalyzer|PepProtGraphAnalyzer|structure_graph_representation|pipeline_or_ensemble_frameworks|||||https://github.com/cicese-biocom/PepProtGraphAnalyzer||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|esm-AxP-GDL|esm-AxP-GDL|protein_language_model_representation|gnn_models|||||https://github.com/cicese-biocom/esm-AxP-GDL||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|esm|esm|protein_language_model_representation|gnn_models|||||https://github.com/standard-things/esm||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|E-CLEAP GitHub repository|E-CLEAP GitHub repository|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Wangsicheng52/E-CLEAP||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|AMPScanner vr.2 web server|AMPScanner vr.2 web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/dan-veltri/amp-scanner-v2||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|PepGen 1.0 web server|PepGen 1.0 web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Nate0634034090/nate.283090||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.46|
|CalcAMP GitHub repository|CalcAMP GitHub repository|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/CDDLeiden/CalcAMP||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Deep-AmPEP30 web server|Deep-AmPEP30 web server|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/Chonwai/Deep_AmPEP30_R||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AMP toxicity prediction code|AMP toxicity prediction code|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/h-khabbaz/amp-toxicity-predictor||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.46|
|AMP0 webserver|AMP0 webserver|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/danielm710/AMP-webserver||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AMPA web server|AMPA web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/miminiyo/ampaweb||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AntiBP3 GitLab|AntiBP3 GitLab|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/AntiBP3||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|AntiBP3 Web Server|AntiBP3 Web Server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/AntiBP3||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|AntiBP3 PyPI|AntiBP3 PyPI|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/AntiBP3||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|dbAMP 3.0 web server|dbAMP 3.0 web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Nate0634034090/bug-free-memory||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.31|
|AMPBenchmark|AMPBenchmark|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/BioGenies/AMPBenchmark||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|StarPep|StarPep|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Grupo-Medicina-Molecular-y-Traslacional/StarPep||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AmpGram R package|AmpGram R package|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/cran/AmpGram||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|TP-LMMSG|TP-LMMSG|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/NanjunChen37/TP_LMMSG||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|PGAT-ABPp|PGAT-ABPp|structure_graph_representation|gnn_models|||||https://github.com/moonseter/PGAT-ABPp||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|PepNet web server|PepNet web server|protein_language_model_representation|transformer_llm_dominant_models|||||https://github.com/VeniQs02/pep.net-web-app||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|Antimicrobial Peptide Scanner vr.2 web server|Antimicrobial Peptide Scanner vr.2 web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/dan-veltri/amp-scanner-v2||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AMPScanner vr.2 web server (alternate)|AMPScanner vr.2 web server (alternate)|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/dan-veltri/amp-scanner-v2||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|ACPred|ACPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer peptide prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://codes.bio/acpred/|not_reported_in_available_evidence|True|Webserver actively used for anticancer peptide classification; could represent anticancer AMP classifiers.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AMPfun|AMPfun|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP/anticancer/antibacterial prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://fdblab.csie.ncu.edu.tw/AMPfun/|not_reported_in_available_evidence|True|Used for anticancer and antibacterial prediction in benchmark study.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AntiCP|AntiCP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer peptide prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://crdd.osdd.net/raghava/anticp/|not_reported_in_available_evidence|True|Anticancer peptide prediction webserver used in benchmark.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AntiCP2.0|AntiCP2.0|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer_peptide_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://webs.iiitd.edu.in/raghava/anticp2/|not_reported_in_available_evidence|True|Widely used webserver for anticancer peptide prediction; web interface available, but programmatic access uncertain.|[<br>"webserver_only",<br>"no_source_code",<br>"no_model_weights_provided",<br>"batch_inference_unknown"<br>]|fulltext|0.7|
|HAPPENN|HAPPENN|sequence_encoding_representation|pipeline_or_ensemble_frameworks|hemolysis prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://research.timmons.eu/happenn/|not_reported_in_available_evidence|True|Hemolysis prediction tool used in benchmark.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|HemoPred|HemoPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|hemolysis prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://codes.bio/hemopred/|not_reported_in_available_evidence|True|Hemolysis prediction tool used in benchmark.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|ToxinPred|ToxinPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|toxicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://crdd.osdd.net/raghava/toxinpred/|not_reported_in_available_evidence|True|Cytotoxicity prediction tool used in benchmark.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|ToxIBTL|ToxIBTL|sequence_encoding_representation|pipeline_or_ensemble_frameworks|toxicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://server.wei-group.net/ToxIBTL/|not_reported_in_available_evidence|True|Cytotoxicity prediction tool used in benchmark.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AllerTop|AllerTop|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://www.ddg-pharmfac.net/AllerTOP/|not_reported_in_available_evidence|True|Allergenicity prediction tool used in benchmark.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AllergenFP|AllergenFP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://ddg-pharmfac.net/AllergenFP/|not_reported_in_available_evidence|True|Allergenicity prediction tool used in benchmark.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AllerCatPro|AllerCatPro|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://allercatpro.bii.a-star.edu.sg/|not_reported_in_available_evidence|True|Allergenicity prediction tool used in benchmark.|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AxPEP3|AxPEP3|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Used as AMP prediction tool in a published study; could be included in a benchmark of AMP predictors.||fulltext|0.7|
|RF-AmPEP30|RF-AmPEP30|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|RF-based AMP predictor; part of AmPEP30 suite; useful for ensemble benchmarking.||fulltext|0.8|
|CAMPR34|CAMPR34|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Multi-algorithm AMP prediction webserver; candidate for benchmark.||fulltext|0.7|
|CLASSAMP5|CLASSAMP5|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|AMP prediction tool using SVM and RF; benchmark candidate.||fulltext|0.7|
|DBAASP6|DBAASP6|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|AMP prediction tool based on physicochemical properties; benchmark candidate.||fulltext|0.7|
|APSvr.2|Antimicrobial Peptide Scanner v.2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|37523405|not_reported_in_available_evidence|not_reported_in_available_evidence|https://aps.unmc.edu/prediction/predict|not_reported_in_available_evidence|True|Webserver-based AMP scanner; URL found in extracted links; widely used.|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|
|DBAASPv3.0|DBAASP v3.0|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|AMP prediction|web-server|37523405|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|AMP prediction tool from DBAASP database; benchmark candidate.||fulltext|0.7|
|CAMPR3(RF)|CAMPR3(RF)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Identified as the best-performing general AMP prediction tool in the benchmark study.|[<br>"needs original paper verification"<br>]|review|0.5|
|CAMPR3(SVM)|CAMPR3(SVM)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Evaluated as one of the general AMP prediction tools in the benchmark study.|[<br>"needs original paper verification"<br>]|review|0.4|
|BAGEL3|BAGEL3|sequence_encoding_representation|pipeline_or_ensemble_frameworks|bacteriocin prediction||28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|One of two bacteriocin prediction tools evaluated; outperformed BACTIBASE on the larger benchmark.|[<br>"needs original paper verification"<br>]|review|0.5|
|BACTIBASE|BACTIBASE|sequence_encoding_representation|pipeline_or_ensemble_frameworks|bacteriocin prediction||28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Bacteriocin prediction tool; outperformed by BAGEL3 on larger benchmark.|[<br>"needs original paper verification"<br>]|review|0.4|
|AMP prediction server (biosino)|AMP prediction server (biosino)|structure_graph_representation|machine_learning_models|antimicrobial peptide classification|ML/feature-engineering|21533231|10.1371/journal.pone.0018476|not_reported_in_available_evidence|http://amp.biosino.org/|CAMP database (http://www.camp.bicnirrh.res.in/) and UniProt|True|Novel AMP prediction method with a publicly available web server, evaluated via jackknife test on a benchmark dataset.||fulltext|0.9|
|Multi-label weighted KNN-MLR model|Multi-label WKnn-MLR (Wang2017)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide activity prediction (multi-label classification)|ML|28526820|10.1038/s41598-017-01986-9|not_reported_in_available_evidence|not_reported_in_available_evidence|APD database (May 2016) filtered to 2222 AMPs with 5 activities; APD3 available at https://aps.unmc.edu/AP/|True|Novel multi-label method for AMP activity prediction with clear methodology and publicly available dataset (APD), suitable for inclusion in benchmark comparison.|[<br>"No code or web server available",<br>"No independent external test set"<br>]|fulltext|0.85|
|AMP-GSM|AMP-GSM|structure_graph_representation|gnn_models|AMP prediction / antimicrobial peptide classification|ML|41072192|10.3390/app13085106|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Provides performance comparison (AUC 99% Gram-negative, 98% Gram-positive) and uses feature grouping; could be benchmarked if code/data become available.|[<br>"no_code_available",<br>"no_dataset_link"<br>]|abstract|0.7|
|ISCAPE|ISCAPE|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / anti-E. coli activity classification|ML|41072192|10.1016/j.jmgm.2025.109188|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Reports AUROC 91.83% and MCC 71.86%, outperforms AntiMPmod; model is interpretable. Could be benchmarked if code/data available.|[<br>"no_code_available",<br>"no_dataset_link"<br>]|abstract|0.7|
|AMP MIC predictor (CNN/RNN)|AMP-MIC-predictor-CNN-RNN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|37938588|10.1038/s41467-023-42434-9|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Predictive deep learning models (CNN/RNN) trained on AMP sequence-MIC data can predict antimicrobial activity, making them suitable for AMP prediction benchmarks.||fulltext|0.8|
|AxPEP|AxPEP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction||41315055|10.1007/s00248-025-02620-2|https://sourceforge.net/projects/axpep/|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Publicly available AMP prediction tool; source code repository linked.||fulltext|0.9|
|StackAMP|StackAMP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|29374199|10.1109/tai.2024.3421176|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|AMP prediction model from 2024, to be evaluated for inclusion|[<br>"no_full_text_access",<br>"no_abstract_available"<br>]|metadata|0.3|
|AMPlify_bal|AMPlify_bal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|40891852|10.1128/spectrum.01504-25|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Existing AMP prediction model used for validation in the paper (Fig 1i).|[<br>"original_paper_needed"<br>]|fulltext|0.9|
|AMPlify_imbal|AMPlify_imbal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|40891852|10.1128/spectrum.01504-25|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|Existing AMP prediction model used for validation in the paper (Fig 1i).|[<br>"original_paper_needed"<br>]|fulltext|0.9|
|PeptideRanker|PeptideRanker|structure_graph_representation|gnn_models|general peptide bioactivity prediction (including antimicrobial)|DL|23056189|10.1371/journal.pone.0045012|not_reported_in_available_evidence|http://bioware.ucd.ie/|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)|True|Compared directly to AMP-specific predictors (CAMP, AntiBP2) and can serve as a general baseline for AMP prediction benchmarks.||fulltext|0.9|
|HydraAMP|HydraAMP|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide design|DL|23056189|10.1371/journal.pone.0045012|https://github.com/szczurek-lab/hydramp|not_reported_in_available_evidence|not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.8|
|MetaPepticon|MetaPepticon|traditional_physicochemical_statistical_features|machine_learning_models|anticancer peptide prediction from (meta)genomes|ML|23056189|10.1371/journal.pone.0045012|https://github.com/arikanlab/MetaPepticon|not_reported_in_available_evidence|not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.7|
|WeightedEnsemble_L3 (Anti_Cp)|WeightedEnsemble_L3|structure_graph_representation|gnn_models|antimicrobial peptide activity classification|ML|38266820|10.1016/j.jare.2024.01.023|https://github.com/xubocheng/Anti_Cp.git|not_reported_in_available_evidence|https://github.com/xubocheng/Anti_Cp.git|True|ML model for predicting AMP activity levels with available code and data; suitable for benchmarking feature-based AMP classifiers.||fulltext|0.9|
|PLUM|PLUM|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide generation and classification|DL|42124643|10.64898/2026.02.21.707214|https://github.com/priyamayur/PLUM|not_reported_in_available_evidence|Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository|True|Provides a trained AMP classifier (AUROC 0.988) and MIC predictor (R² 0.875) that can be used to classify peptides as AMP or non-AMP and estimate potency, making it a candidate for benchmarking AMP prediction tasks.||fulltext|0.95|
|APD3|Antimicrobial Peptide Database (APD3)|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|33996914|10.3389/fmolb.2021.669431|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|抗菌肽数据库和预测工具，广泛引用，具备基准测试潜力|[<br>"review_only",<br>"original_paper_needed"<br>]|fulltext|0.8|
|ApexGO|ApexGO|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide optimization|DL|42206144|10.1038/s42256-026-01237-5|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (VAE training data not specified, APEX trained on in-house peptides)|True|Generative AI model for optimizing antimicrobial peptides; high experimental success rate in enhancing activity|[<br>"Code and model not publicly available",<br>"APEX predictor weights not available"<br>]|fulltext|0.9|
|c_AMPs-prediction|c_AMPs-prediction|protein_language_model_representation|rnn_lstm_dominant_models|AMP prediction|DL|41164228|10.3389/fvets.2025.1689589|https://github.com/mayuefine/c_AMPs-prediction|not_reported_in_available_evidence|https://github.com/mayuefine/c_AMPs-prediction|True|AMP prediction model with available code; used in a published AMP discovery study|[<br>"original_model_paper_uncertain",<br>"weights_not_reported"<br>]|fulltext|0.8|
|Venomics artificial intelligence|Venomics artificial intelligence|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/vynect/venom||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.92|
|AMPlify GitHub|AMPlify GitHub|sequence_encoding_representation|rnn_lstm_dominant_models|||||https://github.com/keonjale/amplifygithubrepo||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|AmPEP web server|AmPEP web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Amal-Thomas/Amal-Thomas-PEP-GP-WebDevProject-Recipe||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.36|
|AMPer web server|AMPer web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/AmirhesamGhahari/Amir_Ghahari_Personal_Website_API_Server||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.46|
|CatBoost AMP predictor|CatBoost AMP predictor|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Ronald106/Surviv.io||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.31|
|Two_Level_Ensemble-classifier-chain|Two_Level_Ensemble-classifier-chain|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/kkzheng/Two_Level_Ensemble-classifier-chain||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|amp_de_novo_design_cdGAN|amp_de_novo_design_cdGAN|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aretiz/amp_de_novo_design_cdGAN||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|MAPLE GitHub|MAPLE GitHub|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Violet-maple/Violet-maple.github.io||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|kneaddata|kneaddata|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/biobakery/kneaddata||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|VirSorter2|VirSorter2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/jiarong/VirSorter2||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|COGclassifier|COGclassifier|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/moshi4/COGclassifier||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Anti_Cp|Anti_Cp|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/anticp2||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Anti_Cp.git|Anti_Cp.git|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/AntiO-cps/antio-cps.github.io||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.36|
|PLUM GitHub|PLUM GitHub|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/purpleplum456/purple-plum-GitHub||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Antimicrobial|Antimicrobial|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zswitten/Antimicrobial-Peptides||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Urchin|Urchin|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/duckyb/urchin||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|allenCCF|allenCCF|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/cortex-lab/allenCCF||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|phy|phy|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/lo-th/phy||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|iblapps|iblapps|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/int-brain-lab/iblapps||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Lab|Lab|structure_graph_representation|gnn_models|||||https://github.com/google-deepmind/lab||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|Npx|Npx|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zkat/npx||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|soft-neighbors-supported-clustering|soft-neighbors-supported-clustering|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/DuannYu/soft-neighbors--supported-clustering||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|1.0|
|DeepSeaQuence_biofilms|DeepSeaQuence_biofilms|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|||||https://github.com/trongthucnguyen/DeepSeaQuence_biofilms||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|FMT-MetagenomicData|FMT-MetagenomicData|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/pointwei/FMT-MetagenomicData||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|TransDecoder|TransDecoder|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/TransDecoder/TransDecoder||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|ADMETlab 3|ADMETlab 3|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/kucukkal/admetlab3.0||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|macrel2020benchmark|macrel2020benchmark|traditional_physicochemical_statistical_features|machine_learning_models|||||https://github.com/BigDataBiology/macrel2020benchmark||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|nov-fams-pipeline|nov-fams-pipeline|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/AlvaroRodriguezDelRio/nov-fams-pipeline||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|aro|aro|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/attdevsupport/ARO||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.||github_search|1.0|
|StackEnPred|StackEnPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/NK12131/Bankruptcy-Prediction-Using-Financial-KPIs-ML-Pipeline-with-SMOTE-PCA-Stacked-Ensemble||not_reported_in_available_evidence|True|GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.5|

## Selected Models

|model_name|canonical_name|representation_category|architecture_category|task_type|method_family|source_pmid|source_doi|code_repository_url|web_server_url|dataset_source_or_link|benchmark_candidate|blocking_issues|evidence_level|confidence|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Co-AMPpred|Co-AMPpred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|34330209|10.1186/s12859-021-04305-2|https://github.com/onkarS23/CoAMPpred|not_reported|https://github.com/onkarS23/CoAMPpred (DEEP-AmPEP30 derived)|True|[<br>"pre-trained weights not confirmed; may require training from scratch"<br>]|fulltext|0.9|
|ACEP|ACEP|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP recognition|DL|40462515|10.1186/s12864-020-06978-0|https://github.com/Fuhaoyi/ACEP|not_reported|APD database|True|[<br>"pre-trained weights not confirmed"<br>]|fulltext|0.9|
|SSFGM-Model|SSFGM-Model|protein_language_model_representation|transformer_llm_dominant_models|AMP classification|DL|40462515|10.1186/s12864-020-06978-0|https://github.com/ggcameronnogg/SSFGM-Model|not_reported|not_reported|True|[<br>"evidence only from abstract; full text mismatch suspected",<br>"pre-trained weights not confirmed",<br>"dataset not reported"<br>]|abstract|0.8|
|ESM2-AFPpred|ESM2-AFPpred|protein_language_model_representation|machine_learning_models|antifungal peptide prediction|DL|35724626|10.1093/bib/bbac226|https://github.com/DongYin521/AFP_DL|not_reported|DRAMP and APD3 (anti-Candida peptides)|False|[<br>"specific to antifungal peptides; removed from main AMP benchmark",<br>"no pre-trained weights"<br>]|fulltext|0.95|
|ANIA|ANIA|sequence_encoding_representation|transformer_llm_dominant_models|MIC prediction (regression)|DL|41664908|10.1093/bib/bbag023|https://github.com/SilverGojo4/ANIA|https://biomics.lab.nycu.edu.tw/ANIA/|DBAASP, dbAMP, DRAMP|False|[<br>"regression task (MIC), not binary AMP classification"<br>]|fulltext|0.95|
|AI4AFP|AI4AFP|protein_language_model_representation|cnn_dominant_models|antifungal peptide classification|ML/DL ensemble|42146199|10.1021/acsomega.6c00049|not_reported|https://axp.iis.sinica.edu.tw/AI4AFP|CAMP, DRAMP, YADAMP, SATPdb, DBAASP (AFPs); UniProtKB/Swiss-Prot (non-AMPs)|False|[<br>"specific to antifungal peptides; removed from main AMP benchmark",<br>"no code repository"<br>]|fulltext|0.9|
|AMPDiscover|AMPDiscover|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|34081438|not_reported|not_reported|https://biocom-ampdiscover.cicese.mx/|not_reported|False|[<br>"no code repository, only webserver",<br>"non-reproducible locally"<br>]|abstract|0.7|
|AMPlify|AMPlify|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/bcgsc/AMPlify|not_reported|not_reported|True|[<br>"pre-trained weights not confirmed"<br>]|search_result|0.8|
|iAMPCN|iAMPCN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39330266|10.1093/bib/bbad240|https://github.com/joy50706/iAMPCN|not_reported|not_reported|True|[<br>"original_model_article_not_this_one",<br>"dataset_not_specified",<br>"no pre-trained weights"<br>]|fulltext|0.95|
|MultiPep|MultiPep|sequence_encoding_representation|cnn_dominant_models|multi-label peptide bioactivity classification|DL|34909478|10.1093/biomethods/bpab021|not_reported|not_reported|multiple public databases|True|[<br>"AMP class not confirmed in abstract",<br>"no code",<br>"no weights"<br>]|abstract|0.5|
|PCSPred|PCSPred|traditional_physicochemical_statistical_features|machine_learning_models|short-chain AMP classification|ML|40781463|10.1109/NEleX59773.2023.10421222|not_reported|not_reported|not_reported|True|[<br>"no_code_available",<br>"no_full_text",<br>"no_dataset_details",<br>"no weights"<br>]|abstract|0.6|
|A-CaMP|A-CaMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification / anti-cancer peptide prediction|sequence alignment-based / fingerprinting|31870207|10.1080/07391102.2019.1708796|not_reported|not_reported|not_reported|False|[<br>"no code, no dataset",<br>"task boundary unclear (also predicts anticancer peptides)"<br>]|fulltext|0.8|
|iAMP-2L|iAMP-2L|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.1093/bib/bbac343|not_reported|not_reported|not_reported|False|[<br>"no code or detailed dataset from review",<br>"only described in review"<br>]|review|0.6|
|CAMPR3|CAMPR3|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|35305010|10.1093/database/baab012|not_reported|not_reported|not_reported|False|[<br>"no code, webserver only",<br>"non-reproducible locally"<br>]|review|0.5|
|ADAM|ADAM|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|35305010|10.1007/s12602-024-10402-4|not_reported|http://bioinformatics.cs.ntou.edu.tw/ADAM|not_reported|False|[<br>"no code, webserver only",<br>"non-reproducible locally"<br>]|review|0.6|
|MLAMP|MLAMP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.1093/bib/bbac343|not_reported|not_reported|not_reported|False|[<br>"no code or dataset from review",<br>"only described in review"<br>]|review|0.6|
|CAMP|CAMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|35305010|10.1093/database/baab012|not_reported|not_reported|not_reported|False|[<br>"no code, webserver only",<br>"only described in review"<br>]|review|0.5|
|ClassAMP|ClassAMP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|unknown|35305010|10.24272/j.issn.2095-8137.2023.246|not_reported|not_reported|not_reported|False|[<br>"no code, review only"<br>]|review|0.5|
|AVPpred|AVPpred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|unknown|35305010|10.1093/database/baab012|not_reported|not_reported|not_reported|False|[<br>"no code, review only"<br>]|review|0.5|
|AntiBP2|AntiBP2|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.24272/j.issn.2095-8137.2023.246|not_reported|not_reported|not_reported|False|[<br>"no code, review only"<br>]|review|0.5|
|iAMPred|iAMPred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.1093/database/baab012|not_reported|not_reported|not_reported|False|[<br>"no code, review only"<br>]|review|0.5|
|AmPEP|AmPEP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|35305010|10.1093/database/baab012|not_reported|not_reported|not_reported|False|[<br>"no code, review only"<br>]|review|0.5|
|AMPER|AMPER|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|unknown|35305010|10.1093/database/baab012|not_reported|not_reported|not_reported|False|[<br>"no code, review only"<br>]|review|0.5|
|EFC-FCBF|EFC-FCBF|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|AMP prediction|feature-engineering|35305010|10.1093/database/baab012|not_reported|not_reported|not_reported|False|[<br>"no code, review only"<br>]|review|0.5|
|E-CLEAP|E-CLEAP|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1371/journal.pone.0300125|https://github.com/Wangsicheng52/E-CLEAP|not_reported|not_reported|True|[<br>"no published paper details",<br>"no pre-trained weights"<br>]|review|0.95|
|UniproLcad|UniproLcad|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/harkic/UniproLcad|not_reported|not_reported|True|[<br>"no published paper details",<br>"no pre-trained weights"<br>]|review|0.7|
|TriStack|TriStack|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/hjy23/TriStack|not_reported|not_reported|True|[<br>"no published paper details",<br>"no pre-trained weights"<br>]|review|0.7|
|iAMP-DL|iAMP-DL|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39557756|10.1007/s12602-024-10402-4|https://github.com/mldlproject/2022-iAMP-DL|not_reported|not_reported|True|[<br>"no published paper details",<br>"no pre-trained weights"<br>]|review|0.7|
|amp-gan|amp-gan|sequence_encoding_representation|cnn_dominant_models|AMP prediction/design|DL|39557756|10.1007/s12602-024-10402-4|https://gitlab.com/vail-uvm/amp-gan|not_reported|not_reported|False|[<br>"generative model, not classification"<br>]|review|0.7|
|AVPIden|AVPIden|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP identification|web-server|39557756|10.1007/s12602-024-10402-4|not_reported|http://awi.cuhk.edu.cn/AVPIden/|not_reported|False|[<br>"no code, webserver only",<br>"non-reproducible locally"<br>]|review|0.6|
|antibp|antibp|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported|http://www.imtech.res.in/raghava/antibp/|not_reported|False|[<br>"no code, webserver only",<br>"non-reproducible locally"<br>]|review|0.6|
|hydramp|hydramp|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|39557756|10.1007/s12602-024-10402-4|not_reported|https://hydramp.mimuw.edu.pl|not_reported|False|[<br>"no code, webserver only",<br>"non-reproducible locally"<br>]|review|0.6|
|CTCM-Neo & ConformaX-PEP|CTCM-Neo & ConformaX-PEP|traditional_physicochemical_statistical_features|cnn_dominant_models|antimalarial peptide classification|DL|41859462|10.3389/fcimb.2026.1707267|not_reported|not_reported|not_reported|False|[<br>"antimalarial-specific, no code, no full text"<br>]|abstract|0.6|
|ACP-DL|ACP-DL|traditional_physicochemical_statistical_features|cnn_dominant_models|anticancer peptide prediction|deep learning|34880291||https://github.com/haichengyi/ACP-DL|https://anticancer.pythonanywhere.com/|not_reported|False|[<br>"targets anticancer peptides, not antimicrobial peptides"<br>]|repository||
|DBAASP|DBAASP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide activity prediction|web-server|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|ampsphere|ampsphere|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction / database|web-server|39557756|10.1007/s12602-024-10402-4|not_reported_in_available_evidence|https://ampsphere.big-data-biology.org/|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.6|
|AI4AMP|AI4AMP|traditional_physicochemical_statistical_features|cnn_rnn_hybrid_models|antimicrobial peptide classification|DL|34783578|10.1128/msystems.00299-21|https://github.com/LinTzuTang/AI4AMP_predictor|http://symbiosis.iis.sinica.edu.tw/PC_6/|not_reported_in_available_evidence|True|[<br>"no code or data link",<br>"only mentioned in review"<br>]|fulltext|0.95|
|Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships|Sparse NN AMP model|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|27870247|10.1002/minf.201600029|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||abstract|0.7|
|SAMP|SAMP|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|39573886|10.1093/bfgp/elae046|https://github.com/wan-mlab/SAMP|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.95|
|DL-QSARES|DL-QSARES|traditional_physicochemical_statistical_features|cnn_dominant_models|antifungal peptide prediction/design|DL|39921483|10.1002/advs.202412488|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"code not available",<br>"only abstract evidence"<br>]|abstract|0.5|
|AI4AVP|AI4AVP|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL|37626205|10.1109/JBHI.2021.3130825|https://github.com/LinTzuTang/AI4AVP_predictor|http://axp.iis.sinica.edu.tw/AI4AVP/|https://github.com/LinTzuTang/AI4AVP_predictor (datasets from APD3, DRAMP, YADAMP, DBAASP, CAMP, AVPdb, UniProt/SwissProt)|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|
|PepForge|PepForge|sequence_encoding_representation|pipeline_or_ensemble_frameworks|antimicrobial peptide classification|DL|39705302|10.64898/2026.05.29.728379|https://github.com/wqx1999/PepForge|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|abstract|0.9|
|Al-Omari 2024 AMP prediction model|Al-Omari 2024 AMP prediction model|traditional_physicochemical_statistical_features|cnn_dominant_models|antimicrobial peptide classification|DL|39705302|10.1371/journal.pone.0315477|not_reported_in_available_evidence|not_reported_in_available_evidence|https://dbaasp.org|True|[<br>"Code not available"<br>]|fulltext|0.8|
|BBATProt|BBATProt|protein_language_model_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|41212592|10.1093/bib/bbaf593|https://github.com/Xukai-YE/BBATProt|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.9|
|AMAP|AMAP|traditional_physicochemical_statistical_features|machine_learning_models|AMPs prediction|ML|35988923|10.1093/bib/bbac343|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original code not available"<br>]|fulltext|0.6|
|AMP|AMP Ensemble Model|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|ML/DL|38972032|10.1007/s12539-024-00640-z|https://github.com/researchprotein/amp|http://amp.denglab.org|https://github.com/researchprotein/amp|True||abstract|0.8|
|Deep-AmPEP30|Deep-AmPEP30|traditional_physicochemical_statistical_features|cnn_dominant_models|AMP prediction|DL|32464552|10.1016/j.omtn.2020.05.006|not_reported_in_available_evidence|https://cbbio.cis.um.edu.mo/AxPEP|Benchmark dataset of 188 samples (balanced); training set of 1529 positive samples from AMP databases (AmPEP, etc.)|True|[<br>"code not available",<br>"no dataset link provided"<br>]|fulltext|0.95|
|EBAMP|EBAMP|sequence_encoding_representation|transformer_llm_dominant_models|antimicrobial peptide design|DL|40906555|10.1016/j.celrep.2025.116215|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code or web server available",<br>"method details not fully described"<br>]|abstract|0.5|
|DLFea4AMPGen|DLFea4AMPGen|traditional_physicochemical_statistical_features|cnn_dominant_models|antimicrobial peptide design|DL|41093853|10.1002/adma.202307680|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no code or web server available"<br>]|abstract|0.5|
|AMP-BERT|AMP-BERT|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction|DL|36461699|10.1002/pro.4529|https://github.com/GIST-CSBL/AMP-BERT.|not_reported_in_available_evidence|https://github.com/GIST-CSBL/AMP-BERT.|True||fulltext|0.95|
|COMDEL|COMDEL|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|39234615|10.1016/j.apsb.2024.05.003|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.9|
|C. acnes-targeted AMP generation pipeline (activity classifier)|Dong2024_AMP_activity_classifier|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|38402320|10.1038/s41598-024-55205-3|not_reported_in_available_evidence|not_reported_in_available_evidence|https://dbaasp.org/|True|[<br>"No code or web server available",<br>"Focused on C. acnes, not general AMP prediction",<br>"Not intended as a standalone benchmark model"<br>]|fulltext|0.8|
|BERT-based AMP recognition model|Zhang2021_BERT_AMP|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|34037687|10.1093/bib/bbab200|not_reported_in_available_evidence|not_reported_in_available_evidence|Six AMP datasets (not specified in abstract) and a new constructed AMP dataset|True|[<br>"No code or web server available",<br>"Fulltext not available; evidence from abstract only",<br>"Dataset details unclear"<br>]|abstract|0.7|
|AmpGPT2|AmpGPT2|protein_language_model_representation|transformer_llm_dominant_models|other|DL|42174216|10.1038/s44259-026-00218-3|https://imigitlab.uni-muenster.de/heiderlab/ampgpt2|not_reported_in_available_evidence|COMPASS database (https://compass.imi.uni-muenster.de)|True|[<br>"Not a direct AMP activity classifier; requires external classifier for evaluation."<br>]|fulltext|0.95|
|AMP-CapsNet|AMP-CapsNet|structure_graph_representation|gnn_models|AMP prediction|DL|41654884|10.1186/s44342-026-00067-6|not_reported_in_available_evidence|not_reported_in_available_evidence|derived from UniProt and previous study [31]; positive: 1085 AMPs, negative: 1316 non-AMPs|True|[<br>"No code or model weights publicly available",<br>"Dataset not independently accessible"<br>]|fulltext|0.9|
|deepAMP|deepAMP|protein_language_model_representation|transformer_llm_dominant_models|other|DL|41753681|10.3390/microorganisms14020394|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Original paper not in this batch; details sparse",<br>"No code availability reported"<br>]|fulltext|0.5|
|AMP-RL|AMP-RL|protein_language_model_representation|transformer_llm_dominant_models|AMP generation and optimization|DL|37992451|10.1016/j.sbi.2023.102733|https://github.com/GIST-CSBL/AMP-RL.|not_reported_in_available_evidence|PeptideAtlas, DBAASP v3 (no direct links provided)|True||fulltext|0.9|
|PepCVAE|PepCVAE|sequence_encoding_representation|cnn_dominant_models|AMP generation|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|PrefixProt|PrefixProt|sequence_encoding_representation|cnn_dominant_models|AMP generation / protein design|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|MoFormer|MoFormer|sequence_encoding_representation|transformer_llm_dominant_models|AMP generation / multi-objective optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|HMAMP|HMAMP|sequence_encoding_representation|cnn_dominant_models|AMP generation / multi-objective optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|AMP-Designer|AMP-Designer|protein_language_model_representation|transformer_llm_dominant_models|AMP generation / optimization|DL|37992451|10.1016/j.sbi.2023.102733|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Code not reported in this evidence"<br>]|review|0.5|
|AMP-MIC|AMP-MIC|protein_language_model_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide classification|DL|29679519|10.1002/cmdc.201800204|https://github.com/jkwang93/AMP-Designer|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Paper title/abstract conflict with fulltext; model named AMP-MIC is part of AMP-Designer, not a standalone AMP prediction model; needs verification of original publication."<br>]|fulltext|0.7|
|AP_Sin|AP_Sin|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38416364|10.1038/s41467-018-03746-3|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"无代码仓库，训练数据未公开"<br>]|fulltext|0.7|
|AMP-Detector|AMP-Detector|sequence_encoding_representation|machine_learning_models|antimicrobial peptide classification|DL/ML|39201537|10.3389/fmicb.2018.00323|not_reported_in_available_evidence|not_reported_in_available_evidence|Peptide Atlas (used for discovery)|True|[<br>"无代码仓库，训练数据描述不完整"<br>]|fulltext|0.7|
|AMP-RNNpro|AMP-RNNpro|traditional_physicochemical_statistical_features|rnn_lstm_dominant_models|AMP identification|ML/DL|38839785|10.1038/s41598-024-63461-6|not_reported_in_available_evidence|http://13.126.159.30/|not_reported_in_available_evidence (combined dataset from XUAMP, DBAASP, LAMP, DRAMP)|True|[<br>"No code repository; web server only, may not be suitable for large-scale offline benchmarking."<br>]|fulltext|0.9|
|AMP-Distillation|AMP-Distillation|sequence_encoding_representation|rnn_lstm_dominant_models|AMP prediction|DL|42155201|10.1016/j.compbiolchem.2026.109129|not_reported_in_available_evidence|not_reported_in_available_evidence|APD3 and DADP databases, CD-HIT deduplication|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|abstract|0.75|
|iAMP-SeE|iAMP-SeE|protein_language_model_representation|rnn_lstm_dominant_models|antimicrobial peptide classification|DL|41913931|10.7717/peerj.20978|https://github.com/cqw0715/iAMP-SeE.git|not_reported_in_available_evidence|Dataset 1: DRAMP, dbAMP, CAMPr-4, AMPfun, ADAPTABLE (positive), UniProt (negative); Dataset 2: from deep-AMPpred (Zhao et al. 2024); Zenodo data: https://doi.org/10.5281/zenodo.17398951|True||fulltext|0.95|
|STAMP|STAMP|sequence_encoding_representation|machine_learning_models|AMP activity prediction (MIC prediction)|ML/DL|42155201|10.64898/2026.05.28.728246|not_reported_in_available_evidence|not_reported_in_available_evidence|Used three benchmark datasets including two previously published and a new curated dataset from DBAASP|True|[<br>"No code available in abstract"<br>]|abstract|0.7|
|CF-AMP prediction|CF-AMP prediction|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|42020672|10.1101/2022.11.16.516845|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code or data availability",<br>"Preprint, not peer-reviewed",<br>"Only abstract evidence"<br>]|abstract|0.5|
|AMP-DualTransnet|AMP-DualTransnet|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction|DL|42020672|10.1016/j.nexres.2026.101536|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract or full text",<br>"No code/data",<br>"Journal article with limited info"<br>]|abstract|0.3|
|AMP-FreqNet|AMP-FreqNet|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide classification|DL||10.1145/3766671.3766835|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|
|Collaborative Filtering and Link Prediction model|Unnamed AMP prediction model (Medvedeva et al. 2023)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.1021/acs.jcim.3c00137|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|
|Predictive and Interpretable ML Models|Unnamed AMP prediction models (acsomega 2024)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.1021/acsomega.3c08676.s001|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract, full text, or code available; only title evidence; possibly a supporting information file"<br>]|metadata|0.3|
|AMP prediction ML model|Unnamed AMP prediction model (Ahmad & Garg 2024)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML||10.54985/peeref.2405p7278831|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No abstract, full text, or code available; only title evidence"<br>]|metadata|0.4|
|GAC-BiTCNN-AMP|GAC-BiTCNN-AMP|protein_language_model_representation|cnn_dominant_models|AMP prediction|DL|41844874|10.1038/s41598-026-43370-6|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (likely dbAMP 3.0 or similar, no explicit URL)|True|[<br>"code not reported"<br>]|fulltext|0.9|
|CVAE-BIO|CVAE-BIO|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML/DL|41849223|10.1093/bib/bbag115|https://github.com/scan2030|not_reported_in_available_evidence|APD3 (http://aps.unmc.edu/)|True|[<br>"code availability unclear"<br>]|fulltext|0.85|
|AMPGAN|AMPGAN|sequence_encoding_representation|cnn_dominant_models|AMP generation / prediction|DL|41463765|10.3390/antibiotics14121263|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.35|
|Macrel|Macrel|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|38877295|10.1002/2211-5463.13847|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"Review mentions tool, but no detailed performance data in this source"<br>]|abstract|0.5|
|iAMPpred|iAMPpred|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|41463765|10.3390/antibiotics14121263|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.35|
|AMP-GPT|AMP-GPT|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide generation|DL|40193623|10.1038/s44386-026-00045-6|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"No code or trained model weights provided",<br>"Training data details missing"<br>]|fulltext|0.85|
|ACPred|ACPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer peptide prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://codes.bio/acpred/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AMPfun|AMPfun|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP/anticancer/antibacterial prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://fdblab.csie.ncu.edu.tw/AMPfun/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AntiCP|AntiCP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer peptide prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://crdd.osdd.net/raghava/anticp/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AntiCP2.0|AntiCP2.0|sequence_encoding_representation|pipeline_or_ensemble_frameworks|anticancer_peptide_prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://webs.iiitd.edu.in/raghava/anticp2/|not_reported_in_available_evidence|True|[<br>"webserver_only",<br>"no_source_code",<br>"no_model_weights_provided",<br>"batch_inference_unknown"<br>]|fulltext|0.7|
|HAPPENN|HAPPENN|sequence_encoding_representation|pipeline_or_ensemble_frameworks|hemolysis prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://research.timmons.eu/happenn/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|HemoPred|HemoPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|hemolysis prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://codes.bio/hemopred/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|ToxinPred|ToxinPred|sequence_encoding_representation|pipeline_or_ensemble_frameworks|toxicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|http://crdd.osdd.net/raghava/toxinpred/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|ToxIBTL|ToxIBTL|sequence_encoding_representation|pipeline_or_ensemble_frameworks|toxicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://server.wei-group.net/ToxIBTL/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AllerTop|AllerTop|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://www.ddg-pharmfac.net/AllerTOP/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AllergenFP|AllergenFP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://ddg-pharmfac.net/AllergenFP/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AllerCatPro|AllerCatPro|sequence_encoding_representation|pipeline_or_ensemble_frameworks|allergenicity prediction|not_reported_in_available_evidence|41155367|10.3390/ijms262010077|not_reported_in_available_evidence|https://allercatpro.bii.a-star.edu.sg/|not_reported_in_available_evidence|True|[<br>"original_paper_not_found",<br>"no_dataset_or_code_reported"<br>]|fulltext|0.3|
|AxPEP3|AxPEP3|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.7|
|RF-AmPEP30|RF-AmPEP30|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.8|
|CAMPR34|CAMPR34|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.7|
|CLASSAMP5|CLASSAMP5|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.7|
|DBAASP6|DBAASP6|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|web-server|34867843|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.7|
|APSvr.2|Antimicrobial Peptide Scanner v.2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction|web-server|37523405|not_reported_in_available_evidence|not_reported_in_available_evidence|https://aps.unmc.edu/prediction/predict|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|fulltext|0.9|
|DBAASPv3.0|DBAASP v3.0|traditional_physicochemical_statistical_features|pipeline_or_ensemble_frameworks|AMP prediction|web-server|37523405|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.7|
|CAMPR3(RF)|CAMPR3(RF)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.5|
|CAMPR3(SVM)|CAMPR3(SVM)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide classification|ML|28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.4|
|BAGEL3|BAGEL3|sequence_encoding_representation|pipeline_or_ensemble_frameworks|bacteriocin prediction||28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.5|
|BACTIBASE|BACTIBASE|sequence_encoding_representation|pipeline_or_ensemble_frameworks|bacteriocin prediction||28203715|10.1093/bioinformatics/btx081|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"needs original paper verification"<br>]|review|0.4|
|AMP prediction server (biosino)|AMP prediction server (biosino)|structure_graph_representation|machine_learning_models|antimicrobial peptide classification|ML/feature-engineering|21533231|10.1371/journal.pone.0018476|not_reported_in_available_evidence|http://amp.biosino.org/|CAMP database (http://www.camp.bicnirrh.res.in/) and UniProt|True||fulltext|0.9|
|Multi-label weighted KNN-MLR model|Multi-label WKnn-MLR (Wang2017)|traditional_physicochemical_statistical_features|machine_learning_models|antimicrobial peptide activity prediction (multi-label classification)|ML|28526820|10.1038/s41598-017-01986-9|not_reported_in_available_evidence|not_reported_in_available_evidence|APD database (May 2016) filtered to 2222 AMPs with 5 activities; APD3 available at https://aps.unmc.edu/AP/|True|[<br>"No code or web server available",<br>"No independent external test set"<br>]|fulltext|0.85|
|AMP-GSM|AMP-GSM|structure_graph_representation|gnn_models|AMP prediction / antimicrobial peptide classification|ML|41072192|10.3390/app13085106|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no_code_available",<br>"no_dataset_link"<br>]|abstract|0.7|
|ISCAPE|ISCAPE|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction / anti-E. coli activity classification|ML|41072192|10.1016/j.jmgm.2025.109188|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no_code_available",<br>"no_dataset_link"<br>]|abstract|0.7|
|AMP MIC predictor (CNN/RNN)|AMP-MIC-predictor-CNN-RNN|sequence_encoding_representation|cnn_dominant_models|AMP prediction|DL|37938588|10.1038/s41467-023-42434-9|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.8|
|AxPEP|AxPEP|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction||41315055|10.1007/s00248-025-02620-2|https://sourceforge.net/projects/axpep/|not_reported_in_available_evidence|not_reported_in_available_evidence|True||fulltext|0.9|
|AMP Scanner v2|AMP Scanner V2|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|[<br>"41315055",<br>"40891852"<br>]|[<br>"10.1007/s00248-025-02620-2",<br>"10.1128/spectrum.01504-25"<br>]|https://github.com/dan-veltri/amp-scanner-v2|https://www.dveltri.com/ascan/v2/ascan.html|not_reported_in_available_evidence|True|[<br>"original_paper_needed"<br>]|fulltext|0.9|
|StackAMP|StackAMP|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|29374199|10.1109/tai.2024.3421176|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"no_full_text_access",<br>"no_abstract_available"<br>]|metadata|0.3|
|AMPlify_bal|AMPlify_bal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|40891852|10.1128/spectrum.01504-25|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"original_paper_needed"<br>]|fulltext|0.9|
|AMPlify_imbal|AMPlify_imbal|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|40891852|10.1128/spectrum.01504-25|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"original_paper_needed"<br>]|fulltext|0.9|
|PeptideRanker|PeptideRanker|structure_graph_representation|gnn_models|general peptide bioactivity prediction (including antimicrobial)|DL|23056189|10.1371/journal.pone.0045012|not_reported_in_available_evidence|http://bioware.ucd.ie/|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)|True||fulltext|0.9|
|HydraAMP|HydraAMP|sequence_encoding_representation|cnn_dominant_models|antimicrobial peptide design|DL|23056189|10.1371/journal.pone.0045012|https://github.com/szczurek-lab/hydramp|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.8|
|MetaPepticon|MetaPepticon|traditional_physicochemical_statistical_features|machine_learning_models|anticancer peptide prediction from (meta)genomes|ML|23056189|10.1371/journal.pone.0045012|https://github.com/arikanlab/MetaPepticon|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.7|
|WeightedEnsemble_L3 (Anti_Cp)|WeightedEnsemble_L3|structure_graph_representation|gnn_models|antimicrobial peptide activity classification|ML|38266820|10.1016/j.jare.2024.01.023|https://github.com/xubocheng/Anti_Cp.git|not_reported_in_available_evidence|https://github.com/xubocheng/Anti_Cp.git|True||fulltext|0.9|
|PLUM|PLUM|protein_language_model_representation|transformer_llm_dominant_models|antimicrobial peptide generation and classification|DL|42124643|10.64898/2026.02.21.707214|https://github.com/priyamayur/PLUM|not_reported_in_available_evidence|Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository|True||fulltext|0.95|
|APD3|Antimicrobial Peptide Database (APD3)|traditional_physicochemical_statistical_features|machine_learning_models|AMP prediction|ML|33996914|10.3389/fmolb.2021.669431|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|[<br>"review_only",<br>"original_paper_needed"<br>]|fulltext|0.8|
|APEX|APEX|sequence_encoding_representation|pipeline_or_ensemble_frameworks|AMP prediction (MIC prediction)|DL|39764027|10.1101/2024.12.17.628923|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (training data not described, in-house peptides mentioned)|True|[<br>"Code and trained model not publicly available",<br>"Training data not publicly available"<br>]|fulltext|0.9|
|ApexGO|ApexGO|sequence_encoding_representation|transformer_llm_dominant_models|AMP prediction / antimicrobial peptide optimization|DL|42206144|10.1038/s42256-026-01237-5|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence (VAE training data not specified, APEX trained on in-house peptides)|True|[<br>"Code and model not publicly available",<br>"APEX predictor weights not available"<br>]|fulltext|0.9|
|c_AMPs-prediction|c_AMPs-prediction|protein_language_model_representation|rnn_lstm_dominant_models|AMP prediction|DL|41164228|10.3389/fvets.2025.1689589|https://github.com/mayuefine/c_AMPs-prediction|not_reported_in_available_evidence|https://github.com/mayuefine/c_AMPs-prediction|True|[<br>"original_model_paper_uncertain",<br>"weights_not_reported"<br>]|fulltext|0.8|
|MAPLE|MAPLE|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/subframe7536/maple-font||not_reported_in_available_evidence|True||github_search|1.0|
|Deep-AmPEP30 web server|Deep-AmPEP30 web server|sequence_encoding_representation|cnn_dominant_models|||||https://github.com/Chonwai/Deep_AmPEP30_R||not_reported_in_available_evidence|True||github_search|1.0|
|Venomics artificial intelligence|Venomics artificial intelligence|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/vynect/venom||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.92|
|AMPlify GitHub|AMPlify GitHub|sequence_encoding_representation|rnn_lstm_dominant_models|||||https://github.com/keonjale/amplifygithubrepo||not_reported_in_available_evidence|True||github_search|1.0|
|AmPEP web server|AmPEP web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Amal-Thomas/Amal-Thomas-PEP-GP-WebDevProject-Recipe||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.36|
|AMPer web server|AMPer web server|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/AmirhesamGhahari/Amir_Ghahari_Personal_Website_API_Server||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.46|
|CatBoost AMP predictor|CatBoost AMP predictor|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Ronald106/Surviv.io||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.31|
|Two_Level_Ensemble-classifier-chain|Two_Level_Ensemble-classifier-chain|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/kkzheng/Two_Level_Ensemble-classifier-chain||not_reported_in_available_evidence|True||github_search|1.0|
|amp_de_novo_design_cdGAN|amp_de_novo_design_cdGAN|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/aretiz/amp_de_novo_design_cdGAN||not_reported_in_available_evidence|True||github_search|1.0|
|MAPLE GitHub|MAPLE GitHub|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/Violet-maple/Violet-maple.github.io||not_reported_in_available_evidence|True||github_search|1.0|
|kneaddata|kneaddata|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/biobakery/kneaddata||not_reported_in_available_evidence|True||github_search|1.0|
|VirSorter2|VirSorter2|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/jiarong/VirSorter2||not_reported_in_available_evidence|True||github_search|1.0|
|COGclassifier|COGclassifier|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/moshi4/COGclassifier||not_reported_in_available_evidence|True||github_search|1.0|
|Anti_Cp|Anti_Cp|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/raghavagps/anticp2||not_reported_in_available_evidence|True||github_search|1.0|
|Anti_Cp.git|Anti_Cp.git|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/AntiO-cps/antio-cps.github.io||not_reported_in_available_evidence|True|[<br>"github_search_candidate_requires_manual_verification"<br>]|github_search|0.36|
|PLUM GitHub|PLUM GitHub|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/purpleplum456/purple-plum-GitHub||not_reported_in_available_evidence|True||github_search|1.0|
|Antimicrobial|Antimicrobial|sequence_encoding_representation|pipeline_or_ensemble_frameworks|||||https://github.com/zswitten/Antimicrobial-Peptides||not_reported_in_available_evidence|True||github_search|1.0|

## Repositories

|name|url|repository_type|matched_model_name|source_pmid|source_doi|evidence_level|
|---|---|---|---|---|---|---|
|Co-AMPpred GitHub repository|https://github.com/onkarS23/CoAMPpred|code|Co-AMPpred|34330209|10.1186/s12859-021-04305-2|fulltext|
|CoAMPpred|https://github.com/onkarS23/CoAMPpred|code|Co-AMPpred|34330209|10.1186/s12859-021-04305-2|fulltext|
|2020-peptidomics|https://github.com/ErikHartman/2020-peptidomics|code|not_reported|33613550|10.3389/fimmu.2020.620707|fulltext|
|AAGP.|https://github.com/saptawtf/AAGP.|code_or_web|AAGP (anti-aging, excluded)|40781463|10.1038/s41598-025-12759-0|fulltext|
|iAMPCN|https://github.com/joy50706/iAMPCN|code_or_web|iAMPCN|39330266|10.3390/md22090385|fulltext|
|SSFGM-Model|https://github.com/ggcameronnogg/SSFGM-Model|code|SSFGM-Model|40462515|10.1186/s12864-020-06978-0|abstract|
|ACEP|https://github.com/Fuhaoyi/ACEP|code|ACEP|40462515|10.1186/s12864-020-06978-0|fulltext|
|ACP-DL|https://github.com/haichengyi/ACP-DL|code_or_web|ACP-DL|34880291|10.1038/s41598-021-02703-3|regex_fulltext_or_metadata|
|Anticancer-Peptides-CNN|https://github.com/mrzResearchArena/Anticancer-Peptides-CNN|code_or_web||34880291|10.1038/s41598-021-02703-3|regex_fulltext_or_metadata|
|MetagenomicDC|https://github.com/IcarPA-TBlab/MetagenomicDC|code_or_web||30066629|10.1186/s12859-018-2182-6|regex_fulltext_or_metadata|
|deep-belief-network.|https://github.com/albertbup/deep-belief-network.|code_or_web||30066629|10.1186/s12859-018-2182-6|regex_fulltext_or_metadata|
|acp-ope|https://github.com/khanhlee/acp-ope|code_or_web|acp-ope (anticancer, excluded)|36642410|10.1093/bib/bbac630|regex_fulltext_or_metadata|
|E-CLEAP|https://github.com/Wangsicheng52/E-CLEAP|code_or_web|E-CLEAP|39557756|10.1007/s12602-024-10402-4|regex_fulltext_or_metadata|
|AMPlify|https://github.com/bcgsc/AMPlify|code_or_web|AMPlify|39557756|10.1007/s12602-024-10402-4|regex_fulltext_or_metadata|
|UniproLcad|https://github.com/harkic/UniproLcad|code_or_web|UniproLcad|39557756|10.1007/s12602-024-10402-4|regex_fulltext_or_metadata|
|TriStack|https://github.com/hjy23/TriStack|code_or_web|TriStack|39557756|10.1007/s12602-024-10402-4|regex_fulltext_or_metadata|
|2022-iAMP-DL|https://github.com/mldlproject/2022-iAMP-DL|code_or_web|iAMP-DL|39557756|10.1007/s12602-024-10402-4|regex_fulltext_or_metadata|
|AMPDiscover|https://biocom-ampdiscover.cicese.mx/|webserver|AMPDiscover|34081438|10.1021/acs.jcim.1c00251|abstract|
|AFP_DL|https://github.com/DongYin521/AFP_DL|code|ESM2-AFPpred|35724626|10.1093/bib/bbac226|fulltext|
|AFP_DL-QSARES|https://github.com/DongYin521/AFP_DL‐QSARES|code|ESM2-AFPpred|35724626|10.1093/bib/bbac226|fulltext|
|ANIA_github|https://github.com/SilverGojo4/ANIA.|code|ANIA|41664908|10.1093/bib/bbag023|fulltext|
|ANIA_webserver|https://biomics.lab.nycu.edu.tw/ANIA/|webserver|ANIA|41664908|10.1093/bib/bbag023|fulltext|
|AI4AFP_webserver|https://axp.iis.sinica.edu.tw/AI4AFP|webserver|AI4AFP|42146199|10.1021/acsomega.6c00049|fulltext|
|ANIA.|https://github.com/SilverGojo4/ANIA.|code|ANIA|41664908|10.1093/bib/bbag023|fulltext|
|ANIA._github_duplicate|https://github.com/SilverGojo4/ANIA.|code||41664908|10.1093/bib/bbag023|fulltext|
|AI4AMP_predictor|https://github.com/LinTzuTang/AI4AMP_predictor|code|AI4AMP|34783578|10.1128/msystems.00299-21|fulltext|
|PC6-protein-encoding-method|https://github.com/LinTzuTang/PC6-protein-encoding-method|code|AI4AMP|34783578|10.1128/msystems.00299-21|fulltext|
|SAMP|https://github.com/wan-mlab/SAMP|code|SAMP|39573886|10.1093/bfgp/elae046|fulltext|
|BAGEL4|https://github.com/annejong/BAGEL4|code|BAGEL4|41148698|10.3390/antibiotics14101004|fulltext|
|LinearDisplay|https://github.com/JCVenterInstitute/LinearDisplay|code|not_reported_in_available_evidence|41148698|10.3390/antibiotics14101004|fulltext|
|msaconverter|https://github.com/linzhi2013/msaconverter|code|not_reported_in_available_evidence|41148698|10.3390/antibiotics14101004|fulltext|
|LysePred.|https://github.com/lincubator/LysePred.|code|LysePred|42338220|10.1021/acssynbio.6c00173|fulltext|
|AI4AVP_predictor|https://github.com/LinTzuTang/AI4AVP_predictor|code|AI4AVP|37626205|10.1109/JBHI.2021.3130825|chunk_summary|
|amp_gan|https://github.com/lsbnb/amp_gan|code|AI4AVP|37626205|10.1109/JBHI.2021.3130825|chunk_summary|
|AI4AVP_web_server|http://axp.iis.sinica.edu.tw/AI4AVP/|webserver|AI4AVP|37626205|10.1109/JBHI.2021.3130825|chunk_summary|
|PepForge|https://github.com/wqx1999/PepForge|code|PepForge|39705302|10.64898/2026.05.29.728379|abstract|
|BBATProt|https://github.com/Xukai-YE/BBATProt|code|BBATProt|41212592|10.1093/bib/bbaf593|fulltext|
|AMAP webserver|http://faculty.pieas.edu.pk/fayyaz/software.html#AMAP|webserver|AMAP|30831306|10.1016/j.compbiomed.2019.02.018|abstract|
|AMP-researchprotein|https://github.com/researchprotein/amp|code|AMP|38972032|10.1007/s12539-024-00640-z|abstract|
|AxPEP web server|https://cbbio.cis.um.edu.mo/AxPEP|webserver|Deep-AmPEP30|32464552|10.1109/INDCON.2011.6139332|abstract|
|learning_sequence_motifs.|https://github.com/p-koo/learning_sequence_motifs.|code|AMP|38972032|10.1093/nar/gkab1080|abstract|
|amp|https://github.com/researchprotein/amp|code|AMP|38972032|10.1093/nar/gkab1080|abstract|
|AMP-BERT GitHub repository|https://github.com/GIST-CSBL/AMP-BERT.|code|AMP-BERT|36461699|10.1002/pro.4529|fulltext|
|treexplainer-study|https://github.com/suinleelab/treexplainer-study|code||36290108|10.1038/s42256-019-0138-9|fulltext|
|LightGBM|https://github.com/Microsoft/LightGBM|code||36290108|10.1038/s42256-019-0138-9|fulltext|
|shap|https://github.com/slundberg/shap|code||36290108|10.1038/s42256-019-0138-9|fulltext|
|AmpGPT2 code repository|https://imigitlab.uni-muenster.de/heiderlab/ampgpt2|code|AmpGPT2|42174216|10.1038/s44259-026-00218-3|fulltext|
|COMPASS database|https://compass.imi.uni-muenster.de|webserver|AmpGPT2|42174216|10.1038/s44259-026-00218-3|fulltext|
|AMP-RL|https://github.com/GIST-CSBL/AMP-RL.|code|AMP-RL|37992451|10.1016/j.sbi.2023.102733|fulltext|
|AMP-Designer|https://github.com/jkwang93/AMP-Designer|code|AMP-MIC|29679519|10.1002/cmdc.201800204|fulltext|
|AMP-RNNpro web server|http://13.126.159.30/|webserver|AMP-RNNpro|38839785|10.1016/j.csbj.2022.07.043|abstract|
|iAMP-SeE|https://github.com/cqw0715/iAMP-SeE.git|code|iAMP-SeE|41913931|10.7717/peerj.20978|fulltext|
|deep_AMPpred.|https://github.com/JunZhao-hash/deep_AMPpred.|code||41913931|10.7717/peerj.20978|fulltext|
|scan2030 (potential CVAE-BIO code)|https://github.com/scan2030|code|CVAE-BIO|41849223|10.1093/bib/bbag115|fulltext|
|iAMP-DL|https://github.com/mldlproject/2022-iAMP-DL|code|iAMP-DL|39557756|10.1007/s12602-024-10402-4|review|
|AVPIden_web_server|http://awi.cuhk.edu.cn/AVPIden/|webserver|AVPIden|39557756|10.1007/s12602-024-10402-4|review|
|ADAM_web_server|http://bioinformatics.cs.ntou.edu.tw/ADAM|webserver|ADAM|39557756|10.1007/s12602-024-10402-4|review|
|antibp_web_server|http://www.imtech.res.in/raghava/antibp/|webserver|AntiBP|39557756|10.1007/s12602-024-10402-4|review|
|ampsphere_web_server|https://ampsphere.big-data-biology.org/|webserver|ampsphere|39557756|10.1007/s12602-024-10402-4|review|
|MAPLE GitHub repository|https://github.com/Harkool/MAPLE|code|MAPLE|39792442|10.1021/acs.jcim.4c01913|fulltext|
|LMPred GitHub repository|https://github.com/williamdee1/LMPred_AMP_Prediction|code|LMPred|36699381|10.1101/2020.07.12.199554v3|fulltext|
|GRAMPA dataset repository|https://github.com/zswitten/Antimicrobial-Peptides|dataset|PepVAE|34659152|10.3389/fmicb.2021.725727|fulltext|
|Antimicrobial-Peptides.|https://github.com/zswitten/Antimicrobial-Peptides.|code_or_web|PyAMPA|34659152|10.3389/fmicb.2021.725727|mixed|
|LMPred_AMP_Prediction.|https://github.com/williamdee1/LMPred_AMP_Prediction.|code_or_web||36699381|10.1101/2020.07.12.199554v3|mixed|
|CDPfold.|https://github.com/zhangch994/CDPfold.|code_or_web||36699381|10.1101/2020.07.12.199554v3|mixed|
|DDM GitHub|https://github.com/kww567upup/DDM|code|DDM|41692989|10.1093/bioinformatics/btag077|fulltext|
|UniAMP web server|https://amp.starhelix.cn|webserver|UniAMP|39799358|10.1186/s12859-025-06033-3|fulltext|
|PepProtGraphAnalyzer|https://github.com/cicese-biocom/PepProtGraphAnalyzer|code|not_reported_in_available_evidence|41594075|10.3390/antibiotics15010039|fulltext|
|esm-AxP-GDL|https://github.com/cicese-biocom/esm-AxP-GDL|code|not_reported_in_available_evidence|41594075|10.3390/antibiotics15010039|fulltext|
|esm|https://github.com/facebookresearch/esm|code|not_reported_in_available_evidence|41594075|10.3390/antibiotics15010039|fulltext|
|E-CLEAP GitHub repository|https://github.com/Wangsicheng52/E-CLEAP|code|E-CLEAP|38722967|10.1371/journal.pone.0300125|fulltext|
|AMPScanner vr.2 web server|https://www.dveltri.com/ascan/v2/ascan.html|webserver|AMPScanner vr.2|37851665|10.1371/journal.pone.0292947|fulltext|
|PepGen 1.0 web server|https://bit.ly/2Z281cY|webserver|PepGen 1.0|40643674|10.1007/s00284-025-04346-3|fulltext|
|AmPepGen GitHub repository|https://github.com/Anorpe/ampepgen-dev|code|AmPepGen|40643674|10.1007/s00284-025-04346-3|fulltext|
|CalcAMP GitHub repository|https://github.com/CDDLeiden/CalcAMP|code|CalcAMP|37107088|10.3390/antibiotics12040725|fulltext|
|Deep-AmPEP30 web server|https://cbbio.cis.um.edu.mo/AxPEP|webserver|Deep-AmPEP30|32464552|10.1016/j.omtn.2020.05.006|fulltext|
|AMP toxicity prediction code|https://git.io/JRZaT|code|AMP toxicity prediction model (hybrid)|34758751|10.1186/s12859-021-04468-y|fulltext|
|DRAMP database website|http://dramp.cpu-bioinfor.org/|webserver|Unnamed AMP predictor from DRAMP 2.0|31409791|10.1038/s41597-019-0154-y|fulltext|
|CalcAMP.|https://github.com/CDDLeiden/CalcAMP.|code|CalcAMP|37107088|10.3390/antibiotics12040725|fulltext|
|Antimicrobial-Peptides (Witten & Witten)|https://github.com/zswitten/Antimicrobial-Peptides|code|Witten-2019-CNN|21347392|10.1101/692681|abstract|
|amp-zGSM|https://github.com/DemetParlakSonmez/amp-zGSM|code|AMP-zGSM|21347392|10.5220/0014457300004070|abstract|
|sAMPpred-GAT GitHub|https://github.com/HongWuL/sAMPpred-GAT/|code|sAMPpred-GAT|36342186|10.1093/bioinformatics/btac715|abstract|
|AMP0 webserver|http://ampzero.pythonanywhere.com|webserver|AMP0|32750857|10.1109/TCBB.2020.2999399|abstract|
|sAMPpred-GAT (regex match)|https://github.com/HongWuL/sAMPpred-GAT|code_or_web||36342186|10.1093/bioinformatics/btac715|metadata|
|SysBioUAB/PyAMPA|https://github.com/SysBioUAB/PyAMPA|code|PyAMPA|38934543|10.1128/msystems.01358-23|fulltext|
|AMPA web server|http://tcoffee.crg.cat/apps/ampa|webserver|AMPA|40410382|10.1038/s44320-025-00120-6|fulltext|
|AntiBP3 GitLab|https://gitlab.com/raghavalab/antibp3|code|AntiBP3|38391554|10.3390/antibiotics13020168|fulltext|
|AntiBP3 Web Server|https://webs.iiitd.edu.in/raghava/antibp3|webserver|AntiBP3|38391554|10.3390/antibiotics13020168|fulltext|
|AntiBP3 PyPI|https://pypi.org/project/antibp3/|code|AntiBP3|38391554|10.3390/antibiotics13020168|fulltext|
|AMPActiPred Web Server|https://awi.cuhk.edu.cn/∼AMPActiPred/|webserver|AMPActiPred|38723168|10.1002/pro.5006|fulltext|
|dbAMP 3.0 web server|https://awi.cuhk.edu.cn/dbAMP/|webserver|AMPfinder, AMPpredictor, AMPActiPred|39540425|10.1093/nar/gkae1019|fulltext|
|battleamp-snakemake|https://github.com/szczurek-lab/battleamp-snakemake|code|BATTLE-AMP framework||10.64898/2026.06.19.733349|abstract|
|AMPBAN|https://github.com/baiwenhuim/ampban|code|AMPBAN||10.64898/2026.01.20.700468|abstract|
|AMPBenchmark|http://BioGenies.info/AMPBenchmark|webserver|AMPBenchmark|38416364|10.1101/2022.05.30.493946|abstract|
|PepMCP|https://github.com/ComputBiophys/PepMCP|code|PepMCP||10.64898/2026.02.01.703163|abstract|
|iMFP-LG GitHub|https://github.com/chen-bioinfo/iMFP-LG|code|iMFP-LG|39585308|10.1093/gpbjnl/qzae084|fulltext|
|iMFP-LG BioCode Tool|https://ngdc.cncb.ac.cn/biocode/tools/BT007494|webserver|iMFP-LG|39585308|10.1093/gpbjnl/qzae084|fulltext|
|CAPTP GitHub|https://github.com/jiaoshihu/CAPTP.|code|CAPTP (peptide toxicity prediction)|38696758|10.1093/bioinformatics/btae297|regex_fulltext_or_metadata|
|AMPd-Up GitHub|https://github.com/bcgsc/AMPd-Up.|code|AMPd-Up (de novo AMP design)|38988311|10.1002/pro.5088|regex_fulltext_or_metadata|
|amPEPpy|https://github.com/tlawrence3/amPEPpy|code|amPEPpy|33135060|10.1093/bioinformatics/btaa917|abstract|
|panCleave|https://gitlab.com/machine-biology-group-public/pancleave|code|panCleave|37516110|10.1016/j.chom.2023.07.001|fulltext|
|peptides_molecular_fingerprints_classification|https://github.com/scikit-fingerprints/peptides_molecular_fingerprints_classification|code|not applicable (benchmark code)|33774670|10.1093/bib/bbab083|fulltext|
|StarPep|http://mobiosd-hub.com/starpep/|webserver||33093586|10.1038/s41598-020-75029-1|abstract|
|scan2030 GitHub (potential CVAE-BIO code)|https://github.com/scan2030|code|CVAE-BIO|41849223|10.1093/bib/bbag115|fulltext|
|AMPGANv3|https://github.com/marszzibros/AMPGANv3|code|AMPGAN v3|42364293|10.1016/j.jmgm.2026.109497|abstract|
|PepAnno|https://bis.zju.edu.cn/pepanno/|webserver|PepAnno|42228741|10.1371/journal.pcbi.1014369|abstract|
|SAMP GitHub repository|https://github.com/wan-mlab/SAMP|code|SAMP|39573886|10.1101/gr.254557.119|abstract|
|AmpGram R package|not_reported_in_available_evidence|code|AmpGram|32560350|10.3390/ijms21124310|fulltext|
|AmpGram web server|not_reported_in_available_evidence|webserver|AmpGram|32560350|10.3390/ijms21124310|fulltext|
|SHARP|https://github.com/shibiaowan/SHARP|code|SAMP|38712184|10.1128/aac.02340-16|regex_fulltext_or_metadata|
|AmpGram R package on CRAN|not_reported_in_available_evidence|code|AmpGram|32560350|10.1074/jbc.M111.303602|abstract|
|Pore-Forming_AMP_SVM|https://github.com/ComputBiophys/Pore%E2%80%90Forming_AMP_SVM|code/model|Pore-Forming AMP SVM|41391039|10.1002/advs.202516470|fulltext|
|Pore|https://github.com/ComputBiophys/Pore|code_or_web||41391039|10.1002/advs.202516470|regex_fulltext_or_metadata|
|MAPLE|https://github.com/Harkool/MAPLE|code|MAPLE|39927895|10.1021/acs.jcim.5c00006|fulltext|
|iFeature|https://github.com/Superzchen/iFeature|code_or_web||30867681|10.1186/s13040-019-0196-x|regex_fulltext_or_metadata|
|SGAC|https://github.com/wyxwyx46941930/SGAC|code|SGAC|41662353|10.1093/bib/bbag038|fulltext|
|keras-multi-head|https://github.com/CyberZHG/keras-multi-head|code_or_web||35078402|10.1186/s12864-022-08310-4|regex_fulltext_or_metadata|
|keras_attention.|https://github.com/lzfelix/keras_attention.|code_or_web||35078402|10.1186/s12864-022-08310-4|regex_fulltext_or_metadata|
|PrMFTP web server|http://bioinfo.ahu.edu.cn/PrMFTP|webserver|PrMFTP|36094961|10.1371/journal.pcbi.1010511|fulltext|
|AMPpred-AAIW web server|https://amppred-aaiw.com|webserver|AMPpred-AAIW|37120707|10.1142/S0219720023500063|abstract|
|Antifreeze-Peptide-Discovery.|https://github.com/imamabi/Antifreeze-Peptide-Discovery.|code_or_web||35576825|10.1016/j.compbiomed.2022.105577|regex_fulltext_or_metadata|
|AniAMPpred webserver|https://aniamppred.anvil.app/|webserver|AniAMPpred|34259329|10.1093/bib/bbab242|abstract|
|Appred webserver|www.soodlab.com/appred|webserver|Appred|39247292|10.1016/j.heliyon.2024.e36163|fulltext|
|Antimicrobial Peptide Scanner vr.2|http://www.ampscanner.com|webserver|AMPScanner v2|29590297|10.1093/bioinformatics/bty179|fulltext|
|AMPSpeciesSpecific GitHub|https://github.com/bzlee-bio/AMPSpeciesSpecific|code|AMPSpeciesSpecific|39766503|10.3390/antibiotics13121113|fulltext|
|PepNet Zenodo record 1|https://zenodo.org/records/1322351661|code|PepNet|39341947|10.1038/s42003-024-06911-1|fulltext|
|PepNet Zenodo record 2|https://zenodo.org/records/1373425862|code|PepNet|39341947|10.1038/s42003-024-06911-1|fulltext|
|PepNet web server|http://liulab.top/PepNet/server|webserver|PepNet|39341947|10.1038/s42003-024-06911-1|fulltext|
|BPFun GitHub|https://github.com/291357657/BPFun|code|BPFun|40691539|10.1186/s12859-025-06190-5|fulltext|
|LLAMP GitHub|https://github.com/GIST-CSBL/LLAMP|code|LLAMP|40676915|10.1093/bib/bbaf343|fulltext|
|Antimicrobial-Peptides GitHub|https://github.com/zswitten/Antimicrobial-Peptides|code|LLAMP|40676915|10.1093/bib/bbaf343|fulltext|
|BioGenies/NegativeDatasets|https://github.com/BioGenies/NegativeDatasets|dataset|N/A|35988923|10.1093/bib/bbac343|fulltext|
|BioGenies/NegativeDatasetsArchitectures|https://github.com/BioGenies/NegativeDatasetsArchitectures|code|N/A|35988923|10.1093/bib/bbac343|fulltext|
|AntiBP3 GitHub|https://github.com/raghavagps/AntiBP3|code|AntiBP3|38391554|10.5281/zenodo.19911030|repository|
|Antimicrobial Peptide Scanner vr.2 web server|http://www.ampscanner.com|webserver|AMPScanner v2|29590297|10.1093/bioinformatics/bty179|fulltext|
|AMPScanner vr.2 web server (alternate)|https://www.dveltri.com/ascan/v2/ascan.html|webserver|AMPScanner vr.2|37851665|10.1371/journal.pone.0292947|fulltext|
|PyAMPA.|https://github.com/SysBioUAB/PyAMPA.|code_or_web||38934543|10.1128/msystems.01358-23|regex_fulltext_or_metadata|
|APIN GitHub|https://github.com/zhanglabNKU/APIN|code|APIN|31870282|10.1093/bioinformatics/btx679|abstract|
|lsgkm|https://github.com/Dongwon-Lee/lsgkm|code|APIN|31870282|10.1093/bioinformatics/btx679|uncertain|
|Basset|https://github.com/davek44/Basset|code|APIN|31870282|10.1093/bioinformatics/btx679|uncertain|
|Deopen.|https://github.com/kimmo1019/Deopen.|code|APIN|31870282|10.1093/bioinformatics/btx679|uncertain|
|APIN.|https://github.com/zhanglabNKU/APIN.|code|APIN|31870282|10.1093/bioinformatics/btx679|uncertain|
|Co-AMPpred GitHub|https://github.com/onkarS23/CoAMPpred|code|Co-AMPpred|||fulltext|
|ACEP GitHub|https://github.com/Fuhaoyi/ACEP|code|ACEP|||fulltext|
|SSFGM-Model GitHub|https://github.com/ggcameronnogg/SSFGM-Model|code|SSFGM-Model|||abstract|
|AMPlify GitHub|https://github.com/bcgsc/AMPlify|code|AMPlify|40100125|10.1093/nar/gki524|review|
|iAMPCN GitHub|https://github.com/joy50706/iAMPCN|code|iAMPCN|||fulltext|
|ESM2-AFPpred GitHub|https://github.com/DongYin521/AFP_DL|code|ESM2-AFPpred||||
|AMPDiscover Webserver|https://biocom-ampdiscover.cicese.mx/|webserver|AMPDiscover|34081438|10.1021/acs.jcim.1c00251|abstract|
|E-CLEAP GitHub|https://github.com/Wangsicheng52/E-CLEAP|code|E-CLEAP||||
|UniproLcad GitHub|https://github.com/harkic/UniproLcad|code|UniproLcad||||
|TriStack GitHub|https://github.com/hjy23/TriStack|code|TriStack||||
|iAMP-DL GitHub|https://github.com/mldlproject/2022-iAMP-DL|code|iAMP-DL||||
|amp-gan GitLab|https://gitlab.com/vail-uvm/amp-gan|code|amp-gan||||
|AVPIden Webserver|http://awi.cuhk.edu.cn/AVPIden/|webserver|AVPIden|39557756|10.1007/s12602-024-10402-4|review|
|antibp Webserver|http://www.imtech.res.in/raghava/antibp/|webserver|antibp|39557756|10.1007/s12602-024-10402-4|review|
|hydramp Webserver|https://hydramp.mimuw.edu.pl|webserver|hydramp|39557756|10.1007/s12602-024-10402-4|review|
|ADAM Webserver|http://bioinformatics.cs.ntou.edu.tw/ADAM|webserver|ADAM|39557756|10.1007/s12602-024-10402-4|review|
|ACP-DL GitHub|https://github.com/haichengyi/ACP-DL|code|ACP-DL||||
|2020-peptidomics GitHub|https://github.com/ErikHartman/2020-peptidomics|code|not_reported||||
|ANIA Web Server|https://biomics.lab.nycu.edu.tw/ANIA/|webserver|ANIA||||
|AI4AFP Web Server|https://axp.iis.sinica.edu.tw/AI4AFP|webserver|AI4AFP||||
|iAamir3924/A-CaMP|https://github.com/iAamir3924/A-CaMP|code_candidate|A-CaMP|||github_search|
|Mikaellesmana/ADAM2|https://github.com/Mikaellesmana/ADAM2|code_candidate|ADAM|||github_search|
|szczurek-lab/hydramp|https://github.com/szczurek-lab/hydramp|code_candidate|hydramp|||github_search|
|ali-ghulam/AMP-CapsNet|https://github.com/ali-ghulam/AMP-CapsNet|code_candidate|AMP-CapsNet|||github_search|
|AhmedDonkol/DeepAMPNet|https://github.com/AhmedDonkol/DeepAMPNet|code_candidate|deepAMP|||github_search|
|Iseeu233/deepAMPNet|https://github.com/Iseeu233/deepAMPNet|code_candidate|deepAMP|||github_search|
|akv84/DeepAMP|https://github.com/akv84/DeepAMP|code_candidate|deepAMP|||github_search|
|scan2030/cvae_bio_amp_discovery|https://github.com/scan2030/cvae_bio_amp_discovery|code_candidate|CVAE-BIO|||github_search|
|JackKuo666/AMP-SEMiner-Portal|https://github.com/JackKuo666/AMP-SEMiner-Portal|code_candidate|AMP-SEMiner|||github_search|
|zjlab-BioGene/AMP-SEMiner|https://github.com/zjlab-BioGene/AMP-SEMiner|code_candidate|AMP-SEMiner|||github_search|
|onkarS23/CoAMPpred|https://github.com/onkarS23/CoAMPpred|code_candidate|Co-AMPpred|||github_search|
|ErikHartman/2020-peptidomics|https://github.com/ErikHartman/2020-peptidomics|code_candidate|2020-peptidomics|||github_search|
|forthespada/CampusShame|https://github.com/forthespada/CampusShame|code_candidate|A-CaMP|||github_search|
|ipfs/camp|https://github.com/ipfs/camp|code_candidate|A-CaMP|||github_search|
|twopin/CAMP|https://github.com/twopin/CAMP|code_candidate|A-CaMP|||github_search|
|AngryBytesTech/pcsprediction|https://github.com/AngryBytesTech/pcsprediction|code_candidate|PCSPred|||github_search|
|Davidsondextor/PCS-Prediction|https://github.com/Davidsondextor/PCS-Prediction|code_candidate|PCSPred|||github_search|
|nompaixg06/PcsPredict|https://github.com/nompaixg06/PcsPredict|code_candidate|PCSPred|||github_search|
|zhiqan/AMPCNN|https://github.com/zhiqan/AMPCNN|code_candidate|iAMPCN|||github_search|
|joy50706/iAMPCN|https://github.com/joy50706/iAMPCN|code_candidate|iAMPCN|||github_search|
|ruhluku/ampCNC|https://github.com/ruhluku/ampCNC|code_candidate|iAMPCN|||github_search|
|aagpazos/aagpazos.github.io|https://github.com/aagpazos/aagpazos.github.io|code_candidate|AAGP|||github_search|
|AAgps547/aagps.github.io|https://github.com/AAgps547/aagps.github.io|code_candidate|AAGP|||github_search|
|AAgprogrammer/AAgprogrammer.github.io|https://github.com/AAgprogrammer/AAgprogrammer.github.io|code_candidate|AAGP|||github_search|
|thomas0809/SSFGM|https://github.com/thomas0809/SSFGM|code_candidate|SSFGM-Model|||github_search|
|ggcameronnogg/SSFGM-Model|https://github.com/ggcameronnogg/SSFGM-Model|code_candidate|SSFGM-Model|||github_search|
|agusnieto77/ACEP|https://github.com/agusnieto77/ACEP|code_candidate|ACEP|||github_search|
|Fuhaoyi/ACEP|https://github.com/Fuhaoyi/ACEP|code_candidate|ACEP|||github_search|
|NEO722315/ACEP|https://github.com/NEO722315/ACEP|code_candidate|ACEP|||github_search|
|haichengyi/ACP-DL|https://github.com/haichengyi/ACP-DL|code_candidate|ACP-DL|||github_search|
|YouHongfeng101/ACP-DL|https://github.com/YouHongfeng101/ACP-DL|code_candidate|ACP-DL|||github_search|
|EdvardNA-999/ACP-DL|https://github.com/EdvardNA-999/ACP-DL|code_candidate|ACP-DL|||github_search|
|RafsanjaniHub/Anticancer-Peptides-CNN|https://github.com/RafsanjaniHub/Anticancer-Peptides-CNN|code_candidate|Anticancer-Peptides-CNN|||github_search|
|IcarPA-TBlab/MetagenomicDC|https://github.com/IcarPA-TBlab/MetagenomicDC|code_candidate|MetagenomicDC|||github_search|
|albertbup/deep-belief-network|https://github.com/albertbup/deep-belief-network|code_candidate|deep-belief-network|||github_search|
|mehulrastogi/Deep-Belief-Network-pytorch|https://github.com/mehulrastogi/Deep-Belief-Network-pytorch|code_candidate|deep-belief-network|||github_search|
|AmanPriyanshu/Deep-Belief-Networks-in-PyTorch|https://github.com/AmanPriyanshu/Deep-Belief-Networks-in-PyTorch|code_candidate|deep-belief-network|||github_search|
|scheelelab/MultiPep|https://github.com/scheelelab/MultiPep|code_candidate|MultiPep|||github_search|
|srivathsanb14/MultiPeptide|https://github.com/srivathsanb14/MultiPeptide|code_candidate|MultiPep|||github_search|
|wollok/multipepita|https://github.com/wollok/multipepita|code_candidate|MultiPep|||github_search|
|khanhlee/acp-ope|https://github.com/khanhlee/acp-ope|code_candidate|acp-ope|||github_search|
|OperatorACP/OperatorACP|https://github.com/OperatorACP/OperatorACP|code_candidate|acp-ope|||github_search|
|ranxianglei/opencode-acp|https://github.com/ranxianglei/opencode-acp|code_candidate|acp-ope|||github_search|
|amphp/amp|https://github.com/amphp/amp|code_candidate|iAMP-2L|||github_search|
|jmacdonald/amp|https://github.com/jmacdonald/amp|code_candidate|iAMP-2L|||github_search|
|CubeCoders/AMP|https://github.com/CubeCoders/AMP|code_candidate|iAMP-2L|||github_search|
|sayalaruano/AMPredST|https://github.com/sayalaruano/AMPredST|code_candidate|iAMPred|||github_search|
|ruihan-dong/AMPredictor|https://github.com/ruihan-dong/AMPredictor|code_candidate|iAMPred|||github_search|
|zhaoqi106/AMPred-MFG|https://github.com/zhaoqi106/AMPred-MFG|code_candidate|iAMPred|||github_search|
|tlawrence3/amPEPpy|https://github.com/tlawrence3/amPEPpy|code_candidate|AmPEP|||github_search|
|Chonwai/Ampep_Python|https://github.com/Chonwai/Ampep_Python|code_candidate|AmPEP|||github_search|
|ShirleyWISiu/AmPEP|https://github.com/ShirleyWISiu/AmPEP|code_candidate|AmPEP|||github_search|
|raghavagps/AntiBP3|https://github.com/raghavagps/AntiBP3|code_candidate|AntiBP2|||github_search|
|raghavagps/AntiBP2|https://github.com/raghavagps/AntiBP2|code_candidate|AntiBP2|||github_search|
|raghavagps/AntiBP|https://github.com/raghavagps/AntiBP|code_candidate|AntiBP2|||github_search|
|Campr-Project-Management/campr|https://github.com/Campr-Project-Management/campr|code_candidate|CAMPR3|||github_search|
|blladnar/HappyCamprFramework|https://github.com/blladnar/HappyCamprFramework|code_candidate|CAMPR3|||github_search|
|tgulacsi/camproxy|https://github.com/tgulacsi/camproxy|code_candidate|CAMPR3|||github_search|
|bigdatagenomics/adam|https://github.com/bigdatagenomics/adam|code_candidate|ADAM|||github_search|
|Malinskiy/adam|https://github.com/Malinskiy/adam|code_candidate|ADAM|||github_search|
|gbionics/adam|https://github.com/gbionics/adam|code_candidate|ADAM|||github_search|
|mlampros/mlampros.github.io|https://github.com/mlampros/mlampros.github.io|code_candidate|MLAMP|||github_search|
|mirosval/mLamp|https://github.com/mirosval/mLamp|code_candidate|MLAMP|||github_search|
|rkmcloud99/mlamp|https://github.com/rkmcloud99/mlamp|code_candidate|MLAMP|||github_search|
|chikitang/A|https://github.com/chikitang/A|code_candidate|ClassAMP|||github_search|
|doni21122005/classamp|https://github.com/doni21122005/classamp|code_candidate|ClassAMP|||github_search|
|FIllxe/ClassAmplifier|https://github.com/FIllxe/ClassAmplifier|code_candidate|ClassAMP|||github_search|
|zyweizm/AVPpred-BWR|https://github.com/zyweizm/AVPpred-BWR|code_candidate|AVPpred|||github_search|
|jpetazzo/ampernetacle|https://github.com/jpetazzo/ampernetacle|code_candidate|AMPER|||github_search|
|JetBrains/amper|https://github.com/JetBrains/amper|code_candidate|AMPER|||github_search|
|BLeeEZ/amperfy|https://github.com/BLeeEZ/amperfy|code_candidate|AMPER|||github_search|
|aws-amplify/amplify-js|https://github.com/aws-amplify/amplify-js|code_candidate|AMPlify|||github_search|
|aws-amplify/amplify-cli|https://github.com/aws-amplify/amplify-cli|code_candidate|AMPlify|||github_search|
|ageitgey/amplify|https://github.com/ageitgey/amplify|code_candidate|AMPlify|||github_search|
|Wangsicheng52/E-CLEAP|https://github.com/Wangsicheng52/E-CLEAP|code_candidate|E-CLEAP|||github_search|
|CleapedByEnes/CleapedByEnes|https://github.com/CleapedByEnes/CleapedByEnes|code_candidate|E-CLEAP|||github_search|
|BoetaV/ecLeap24|https://github.com/BoetaV/ecLeap24|code_candidate|E-CLEAP|||github_search|
|harkic/UniproLcad|https://github.com/harkic/UniproLcad|code_candidate|UniproLcad|||github_search|
|hjy23/TriStack|https://github.com/hjy23/TriStack|code_candidate|TriStack|||github_search|
|tristacksolutions/tristack-solution-|https://github.com/tristacksolutions/tristack-solution-|code_candidate|TriStack|||github_search|
|Arthisuresh210/TriStack|https://github.com/Arthisuresh210/TriStack|code_candidate|TriStack|||github_search|
|LucaCerina/ampdLib|https://github.com/LucaCerina/ampdLib|code_candidate|iAMP-DL|||github_search|
|mldlproject/2022-iAMP-DL|https://github.com/mldlproject/2022-iAMP-DL|code_candidate|iAMP-DL|||github_search|
|merissamm24-hue/AMPDLife|https://github.com/merissamm24-hue/AMPDLife|code_candidate|iAMP-DL|||github_search|
|lsbnb/amp_gan|https://github.com/lsbnb/amp_gan|code_candidate|amp-gan|||github_search|
|marszzibros/AMPGANv3|https://github.com/marszzibros/AMPGANv3|code_candidate|amp-gan|||github_search|
|zswitten/Antimicrobial-Peptides|https://github.com/zswitten/Antimicrobial-Peptides|code_candidate|amp-gan|||github_search|
|BiOmicsLab/AVPIden|https://github.com/BiOmicsLab/AVPIden|code_candidate|AVPIden|||github_search|
|siranhe888/AVP-Identification|https://github.com/siranhe888/AVP-Identification|code_candidate|AVPIden|||github_search|
|SinghVishakha/Deep-AVPiden|https://github.com/SinghVishakha/Deep-AVPiden|code_candidate|AVPIden|||github_search|
|BigDataBiology/SantosJunior_Torres_2024_AMPSphere_v1|https://github.com/BigDataBiology/SantosJunior_Torres_2024_AMPSphere_v1|code_candidate|ampsphere|||github_search|
|BigDataBiology/AMPSphereWebsite|https://github.com/BigDataBiology/AMPSphereWebsite|code_candidate|ampsphere|||github_search|
|BigDataBiology/AMPSphereFrontendv1|https://github.com/BigDataBiology/AMPSphereFrontendv1|code_candidate|ampsphere|||github_search|
|conda-forge/hydrampp-feedstock|https://github.com/conda-forge/hydrampp-feedstock|code_candidate|hydramp|||github_search|
|marmarmarmar/pytorch-hydramp|https://github.com/marmarmarmar/pytorch-hydramp|code_candidate|hydramp|||github_search|
|quangnhbk/antimicrobial-peptides|https://github.com/quangnhbk/antimicrobial-peptides|code_candidate|2022-iAMP-DL|||github_search|
|shreyabansal-sb/-WCCI-26-DL-NL--03-Bias-Amplification-in-Toxicity-and-Sentiment-Classifiers|https://github.com/shreyabansal-sb/-WCCI-26-DL-NL--03-Bias-Amplification-in-Toxicity-and-Sentiment-Classifiers|code_candidate|2022-iAMP-DL|||github_search|
|Null-Phnix/amp-discovery|https://github.com/Null-Phnix/amp-discovery|code_candidate|AMPDiscover|||github_search|
|ajuni-sohota/amp-discovery-dl|https://github.com/ajuni-sohota/amp-discovery-dl|code_candidate|AMPDiscover|||github_search|
|iamvarshag/amp_discovery_pipeline|https://github.com/iamvarshag/amp_discovery_pipeline|code_candidate|AMPDiscover|||github_search|
|DongYin521/AFP_DL-QSARES|https://github.com/DongYin521/AFP_DL-QSARES|code_candidate|AFP_DL|||github_search|
|sinosoftjhao/dlisafp|https://github.com/sinosoftjhao/dlisafp|code_candidate|AFP_DL|||github_search|
|afpa-mx2017/afpa-bay|https://github.com/afpa-mx2017/afpa-bay|code_candidate|AFP_DL|||github_search|
|AliAlgur/Ania|https://github.com/AliAlgur/Ania|code_candidate|ANIA|||github_search|
|Tetous/Ania|https://github.com/Tetous/Ania|code_candidate|ANIA|||github_search|
|Animxer18/aniarch-frontend|https://github.com/Animxer18/aniarch-frontend|code_candidate|ANIA|||github_search|
|wccheng1210/AI4AFP|https://github.com/wccheng1210/AI4AFP|code_candidate|AI4AFP|||github_search|
|lsbnb/AI4AFP|https://github.com/lsbnb/AI4AFP|code_candidate|AI4AFP|||github_search|
|LinTzuTang/AI4AFP_predictor|https://github.com/LinTzuTang/AI4AFP_predictor|code_candidate|AI4AFP|||github_search|
|aniagithub/Nieliniowe|https://github.com/aniagithub/Nieliniowe|code_candidate|ANIA_github|||github_search|
|aniagithub/Sterowanie_ZUM|https://github.com/aniagithub/Sterowanie_ZUM|code_candidate|ANIA_github|||github_search|
|aniagithub/Roboty-mobilne|https://github.com/aniagithub/Roboty-mobilne|code_candidate|ANIA_github|||github_search|
|ben-vargas/ai-amp-cli|https://github.com/ben-vargas/ai-amp-cli|code_candidate|AI4AMP|||github_search|
|Mohammedvaraliya/AI-Amplify-Hackathon|https://github.com/Mohammedvaraliya/AI-Amplify-Hackathon|code_candidate|AI4AMP|||github_search|
|LinTzuTang/AI4AMP_predictor|https://github.com/LinTzuTang/AI4AMP_predictor|code_candidate|AI4AMP|||github_search|
|mohamedhassanmus/SAMP|https://github.com/mohamedhassanmus/SAMP|code_candidate|SAMP|||github_search|
|HongWuL/sAMPpred-GAT|https://github.com/HongWuL/sAMPpred-GAT|code_candidate|SAMP|||github_search|
|GRGServer/SAMP|https://github.com/GRGServer/SAMP|code_candidate|SAMP|||github_search|
|LinTzuTang/PC6-protein-encoding-method|https://github.com/LinTzuTang/PC6-protein-encoding-method|code_candidate|PC6-protein-encoding-method|||github_search|
|ByteDance-Seed/Bagel|https://github.com/ByteDance-Seed/Bagel|code_candidate|BAGEL4|||github_search|
|yagiz/Bagel|https://github.com/yagiz/Bagel|code_candidate|BAGEL4|||github_search|
|EnhancedJax/Bagels|https://github.com/EnhancedJax/Bagels|code_candidate|BAGEL4|||github_search|
|JCVenterInstitute/LinearDisplay|https://github.com/JCVenterInstitute/LinearDisplay|code_candidate|LinearDisplay|||github_search|
|ravianandfbg/Linear-queue-to-insert-delete-display-using-array|https://github.com/ravianandfbg/Linear-queue-to-insert-delete-display-using-array|code_candidate|LinearDisplay|||github_search|
|rrtt2323/DisplayGammaUIInLinearSpace|https://github.com/rrtt2323/DisplayGammaUIInLinearSpace|code_candidate|LinearDisplay|||github_search|
|linzhi2013/msaconverter|https://github.com/linzhi2013/msaconverter|code_candidate|msaconverter|||github_search|
|DysnomianC/msaConverter|https://github.com/DysnomianC/msaConverter|code_candidate|msaconverter|||github_search|
|lincubator/LysePred|https://github.com/lincubator/LysePred|code_candidate|LysePred|||github_search|
|LinTzuTang/AI4AVP_predictor|https://github.com/LinTzuTang/AI4AVP_predictor|code_candidate|AI4AVP|||github_search|
|vwfang/AI4AVP_predictor_improved|https://github.com/vwfang/AI4AVP_predictor_improved|code_candidate|AI4AVP|||github_search|
|jinf2/AVP_AIAgent|https://github.com/jinf2/AVP_AIAgent|code_candidate|AI4AVP|||github_search|
|wqx1999/PepForge|https://github.com/wqx1999/PepForge|code_candidate|PepForge|||github_search|
|SanghunWoo-23/Pepforge|https://github.com/SanghunWoo-23/Pepforge|code_candidate|PepForge|||github_search|
|HakimTaoufik/PeptideForge|https://github.com/HakimTaoufik/PeptideForge|code_candidate|PepForge|||github_search|
|Xukai-YE/BBATProt|https://github.com/Xukai-YE/BBATProt|code_candidate|BBATProt|||github_search|
|SindenDev/amap|https://github.com/SindenDev/amap|code_candidate|AMAP|||github_search|
|wuwenrufeng/amap|https://github.com/wuwenrufeng/amap|code_candidate|AMAP|||github_search|
|qq2241025/amap|https://github.com/qq2241025/amap|code_candidate|AMAP|||github_search|
|Chonwai/Deep_AmPEP30_R|https://github.com/Chonwai/Deep_AmPEP30_R|code_candidate|Deep-AmPEP30|||github_search|
|ebampoagyemang/ebampoagyemang|https://github.com/ebampoagyemang/ebampoagyemang|code_candidate|EBAMP|||github_search|
|dendem1980/eBamplus|https://github.com/dendem1980/eBamplus|code_candidate|EBAMP|||github_search|
|hgao12345/DLFea4AMPGen|https://github.com/hgao12345/DLFea4AMPGen|code_candidate|DLFea4AMPGen|||github_search|
|researchprotein/amp|https://github.com/researchprotein/amp|code_candidate|AMP-researchprotein|||github_search|
|p-koo/learning_sequence_motifs|https://github.com/p-koo/learning_sequence_motifs|code_candidate|learning_sequence_motifs|||github_search|
|jertubiana/ProteinMotifRBM|https://github.com/jertubiana/ProteinMotifRBM|code_candidate|learning_sequence_motifs|||github_search|
|gpattarone/Deep-Learning-DNA|https://github.com/gpattarone/Deep-Learning-DNA|code_candidate|learning_sequence_motifs|||github_search|
|GIST-CSBL/AMP-BERT|https://github.com/GIST-CSBL/AMP-BERT|code_candidate|AMP-BERT|||github_search|
|fcf2/amp-bert|https://github.com/fcf2/amp-bert|code_candidate|AMP-BERT|||github_search|
|ArthurWallaceIFB/AMP_BERT_IA|https://github.com/ArthurWallaceIFB/AMP_BERT_IA|code_candidate|AMP-BERT|||github_search|
|stephenlofgren/ComDelete|https://github.com/stephenlofgren/ComDelete|code_candidate|COMDEL|||github_search|
|comdelex/comdelex.github.io|https://github.com/comdelex/comdelex.github.io|code_candidate|COMDEL|||github_search|
|axelbros/comdel|https://github.com/axelbros/comdel|code_candidate|COMDEL|||github_search|
|TheNuber/AMP-BERT-BIOCHEM|https://github.com/TheNuber/AMP-BERT-BIOCHEM|code_candidate|AMP-BERT GitHub repository|||github_search|
|lightgbm-org/LightGBM|https://github.com/lightgbm-org/LightGBM|code_candidate|LightGBM|||github_search|
|apachecn/lightgbm-doc-zh|https://github.com/apachecn/lightgbm-doc-zh|code_candidate|LightGBM|||github_search|
|StatMixedML/LightGBMLSS|https://github.com/StatMixedML/LightGBMLSS|code_candidate|LightGBM|||github_search|
|shap/shap|https://github.com/shap/shap|code_candidate|shap|||github_search|
|Sahana1412/Machine-Learning-Based-Classification-of-Antimicrobial-Peptides-and-Toxicity-Prediction|https://github.com/Sahana1412/Machine-Learning-Based-Classification-of-Antimicrobial-Peptides-and-Toxicity-Prediction|code_candidate|shap|||github_search|
|MAIF/shapash|https://github.com/MAIF/shapash|code_candidate|shap|||github_search|
|MartAlae-AAGP/AAGP|https://github.com/MartAlae-AAGP/AAGP|code_candidate|AAGP|||github_search|
|Featheredpluto6/AAGP|https://github.com/Featheredpluto6/AAGP|code_candidate|AAGP|||github_search|
|Mdtvs/AAGP|https://github.com/Mdtvs/AAGP|code_candidate|AAGP|||github_search|
|LYRHeidi/BroadAMP-GPT|https://github.com/LYRHeidi/BroadAMP-GPT|code_candidate|AmpGPT2|||github_search|
|Dar-kSun/AMP-GPT|https://github.com/Dar-kSun/AMP-GPT|code_candidate|AmpGPT2|||github_search|
|LYRHeidi/BroadAMP-GPTno|https://github.com/LYRHeidi/BroadAMP-GPTno|code_candidate|AmpGPT2|||github_search|
|panapina/pina|https://github.com/panapina/pina|code_candidate|AMP-CapsNet|||github_search|
|amirpandi/Deep_AMP|https://github.com/amirpandi/Deep_AMP|code_candidate|deepAMP|||github_search|
|jimmyrate/deepAMP|https://github.com/jimmyrate/deepAMP|code_candidate|deepAMP|||github_search|
|aaronpk/Compass|https://github.com/aaronpk/Compass|code_candidate|COMPASS database|||github_search|
|sogou-biztech/compass|https://github.com/sogou-biztech/compass|code_candidate|COMPASS database|||github_search|
|arunkumar9t2/compass|https://github.com/arunkumar9t2/compass|code_candidate|COMPASS database|||github_search|
|Gudegi/IsaacLab_AMP_rl-games|https://github.com/Gudegi/IsaacLab_AMP_rl-games|code_candidate|AMP-RL|||github_search|
|GIST-CSBL/AMP-RL|https://github.com/GIST-CSBL/AMP-RL|code_candidate|AMP-RL|||github_search|
|wsxajd/AMP|https://github.com/wsxajd/AMP|code_candidate|AMP-RL|||github_search|
|chen-bioinfo/PrefixProt|https://github.com/chen-bioinfo/PrefixProt|code_candidate|PrefixProt|||github_search|
|PaulFarry/ProtobufLengthPrefix|https://github.com/PaulFarry/ProtobufLengthPrefix|code_candidate|PrefixProt|||github_search|
|2x10/move-proton-prefix|https://github.com/2x10/move-proton-prefix|code_candidate|PrefixProt|||github_search|
|zcao0420/MOFormer|https://github.com/zcao0420/MOFormer|code_candidate|MoFormer|||github_search|
|OmicsML/scMoFormer|https://github.com/OmicsML/scMoFormer|code_candidate|MoFormer|||github_search|
|wl-wl/MOFormer|https://github.com/wl-wl/MOFormer|code_candidate|MoFormer|||github_search|
|wl-wl/HMAMP-main|https://github.com/wl-wl/HMAMP-main|code_candidate|HMAMP|||github_search|
|hjttu/DRL_HMAMP|https://github.com/hjttu/DRL_HMAMP|code_candidate|HMAMP|||github_search|
|jkwang93/AMP-Designer|https://github.com/jkwang93/AMP-Designer|code_candidate|AMP-Designer|||github_search|
|olegkapitonov/tubeAmp-Designer|https://github.com/olegkapitonov/tubeAmp-Designer|code_candidate|AMP-Designer|||github_search|
|AntonS-bio/EnviroAmpDesigner|https://github.com/AntonS-bio/EnviroAmpDesigner|code_candidate|AMP-Designer|||github_search|
|61-Keys/AMP-MIC-Predictor|https://github.com/61-Keys/AMP-MIC-Predictor|code_candidate|AMP-MIC|||github_search|
|chungcr/esAMPMIC|https://github.com/chungcr/esAMPMIC|code_candidate|AMP-MIC|||github_search|
|ankushk5/Amp_microblogging|https://github.com/ankushk5/Amp_microblogging|code_candidate|AMP-MIC|||github_search|
|microsoft/APSINet|https://github.com/microsoft/APSINet|code_candidate|AP_Sin|||github_search|
|apsinghdev/apsinghdev|https://github.com/apsinghdev/apsinghdev|code_candidate|AP_Sin|||github_search|
|sgogula5588/Apsin|https://github.com/sgogula5588/Apsin|code_candidate|AP_Sin|||github_search|
|bunny9411/AMPD|https://github.com/bunny9411/AMPD|code_candidate|AMP-Detector|||github_search|
|vpobleteacustica/amphibian-vae-latent-detector|https://github.com/vpobleteacustica/amphibian-vae-latent-detector|code_candidate|AMP-Detector|||github_search|
|david-svitov/AmphibianDetector|https://github.com/david-svitov/AmphibianDetector|code_candidate|AMP-Detector|||github_search|
|Shazzad-Shaon3404/Website_AMPRNNpro|https://github.com/Shazzad-Shaon3404/Website_AMPRNNpro|code_candidate|AMP-RNNpro|||github_search|
|FahimSultan-cyb/AMP-RNNPro|https://github.com/FahimSultan-cyb/AMP-RNNPro|code_candidate|AMP-RNNpro|||github_search|
|cloudera/CML_AMP_Knowledge_Distillation_With_Private_Data|https://github.com/cloudera/CML_AMP_Knowledge_Distillation_With_Private_Data|code_candidate|AMP-Distillation|||github_search|
|andreast6/SDS_Distillation_AMP|https://github.com/andreast6/SDS_Distillation_AMP|code_candidate|AMP-Distillation|||github_search|
|kp27302/Typeagent--AMP|https://github.com/kp27302/Typeagent--AMP|code_candidate|AMP-Distillation|||github_search|
|YougLin-dev/amp-server|https://github.com/YougLin-dev/amp-server|code_candidate|iAMP-SeE|||github_search|
|USCPOSH/AMPSE|https://github.com/USCPOSH/AMPSE|code_candidate|iAMP-SeE|||github_search|
|stampit-org/stampit|https://github.com/stampit-org/stampit|code_candidate|STAMP|||github_search|
|torodb/stampede|https://github.com/torodb/stampede|code_candidate|STAMP|||github_search|
|xxss0903/drawstamputils|https://github.com/xxss0903/drawstamputils|code_candidate|STAMP|||github_search|
|JunZhao-hash/deep_AMPpred|https://github.com/JunZhao-hash/deep_AMPpred|code_candidate|deep_AMPpred|||github_search|
|1fuyuhe/deep-AMPpred|https://github.com/1fuyuhe/deep-AMPpred|code_candidate|deep_AMPpred|||github_search|
|mfyz/cf-amp-test|https://github.com/mfyz/cf-amp-test|code_candidate|CF-AMP prediction|||github_search|
|mariantalla/cf-amphora-release|https://github.com/mariantalla/cf-amphora-release|code_candidate|CF-AMP prediction|||github_search|
|dankerizer/ampinstant-cf|https://github.com/dankerizer/ampinstant-cf|code_candidate|CF-AMP prediction|||github_search|
|xintail/Hierarchical-amplitude-frequency-prediction-network|https://github.com/xintail/Hierarchical-amplitude-frequency-prediction-network|code_candidate|AMP-FreqNet|||github_search|
|cloudera/CML_AMP_MLFlow_Tracking|https://github.com/cloudera/CML_AMP_MLFlow_Tracking|code_candidate|AMP prediction ML model|||github_search|
|sayalaruano/PredAMP-ML|https://github.com/sayalaruano/PredAMP-ML|code_candidate|AMP prediction ML model|||github_search|
|flystar233/AMPml|https://github.com/flystar233/AMPml|code_candidate|AMP prediction ML model|||github_search|
|Farman335/GAC-BiTCNN-AMP|https://github.com/Farman335/GAC-BiTCNN-AMP|code_candidate|GAC-BiTCNN-AMP|||github_search|
|BigDataBiology/macrel|https://github.com/BigDataBiology/macrel|code_candidate|Macrel|||github_search|
|koenbok/MacReloader|https://github.com/koenbok/MacReloader|code_candidate|Macrel|||github_search|
|drbarq/macrelay|https://github.com/drbarq/macrelay|code_candidate|Macrel|||github_search|
|Jiangle525/AMPpred-MFA|https://github.com/Jiangle525/AMPpred-MFA|code_candidate|iAMPpred|||github_search|
|urban-adam/urban-adam-web|https://github.com/urban-adam/urban-adam-web|code_candidate|ADAM_web_server|||github_search|
|AdamFerguson06/adam_website|https://github.com/AdamFerguson06/adam_website|code_candidate|ADAM_web_server|||github_search|
|minnieb35/adamwebsiteversion2|https://github.com/minnieb35/adamwebsiteversion2|code_candidate|ADAM_web_server|||github_search|
|mumuyang666/AMPGPT|https://github.com/mumuyang666/AMPGPT|code_candidate|AMP-GPT|||github_search|
|Foreast/McLamp|https://github.com/Foreast/McLamp|code_candidate|MCL-AMP|||github_search|
|Mclamp815/mclamp815.github.io|https://github.com/Mclamp815/mclamp815.github.io|code_candidate|MCL-AMP|||github_search|
|mclamp/fitpog|https://github.com/mclamp/fitpog|code_candidate|MCL-AMP|||github_search|
|subframe7536/maple-font|https://github.com/subframe7536/maple-font|code_candidate|MAPLE|||github_search|
|MapleTechLabs/maple|https://github.com/MapleTechLabs/maple|code_candidate|MAPLE|||github_search|
|YtFlow/Maple|https://github.com/YtFlow/Maple|code_candidate|MAPLE|||github_search|
|abdulrahmanbinayub-maker/maple-github-repository|https://github.com/abdulrahmanbinayub-maker/maple-github-repository|code_candidate|MAPLE GitHub repository|||github_search|
|olga-r/Interpretable-VAE-for-Antimicrobial-Peptide-Design|https://github.com/olga-r/Interpretable-VAE-for-Antimicrobial-Peptide-Design|code_candidate|PepVAE|||github_search|
|shuan4638/PeptideTempVAE|https://github.com/shuan4638/PeptideTempVAE|code_candidate|PepVAE|||github_search|
|AFneedWater/peptide_vae|https://github.com/AFneedWater/peptide_vae|code_candidate|PepVAE|||github_search|
|williamdee1/LMPred_AMP_Prediction|https://github.com/williamdee1/LMPred_AMP_Prediction|code_candidate|LMPred|||github_search|
|grushaprasad/psycholing-lm-predict|https://github.com/grushaprasad/psycholing-lm-predict|code_candidate|LMPred|||github_search|
|1Mike-e/LMPrediction|https://github.com/1Mike-e/LMPrediction|code_candidate|LMPred|||github_search|
|dan-veltri/amp-scanner-v2|https://github.com/dan-veltri/amp-scanner-v2|code_candidate|Antimicrobial-Peptides|||github_search|
|zhangch994/CDPfold|https://github.com/zhangch994/CDPfold|code_candidate|CDPfold|||github_search|
|torchDDM/DDM|https://github.com/torchDDM/DDM|code_candidate|DDM|||github_search|
|MCG-NJU/DDM|https://github.com/MCG-NJU/DDM|code_candidate|DDM|||github_search|
|NeXAIS/DDM|https://github.com/NeXAIS/DDM|code_candidate|DDM|||github_search|
|quietbamboo/UniAMP|https://github.com/quietbamboo/UniAMP|code_candidate|UniAMP|||github_search|
|awslabs/aws-amplify-unicorntrivia-workshop|https://github.com/awslabs/aws-amplify-unicorntrivia-workshop|code_candidate|UniAMP|||github_search|
|tkusal/Projeto-Ampliar-UniCesumar-01-2024|https://github.com/tkusal/Projeto-Ampliar-UniCesumar-01-2024|code_candidate|UniAMP|||github_search|
|DDM-Mzp/ddm.github.io|https://github.com/DDM-Mzp/ddm.github.io|code_candidate|DDM GitHub|||github_search|
|Khero001/ddm.github.io|https://github.com/Khero001/ddm.github.io|code_candidate|DDM GitHub|||github_search|
|c3server-github-testuser/1768448659058-thKqkVrY-koDDmkotT4bTXYpc-GitHubTestBase|https://github.com/c3server-github-testuser/1768448659058-thKqkVrY-koDDmkotT4bTXYpc-GitHubTestBase|code_candidate|DDM GitHub|||github_search|
|Dextro86/Webasto-Ampure-Unite-Home-Assistant-custom-integration|https://github.com/Dextro86/Webasto-Ampure-Unite-Home-Assistant-custom-integration|code_candidate|UniAMP web server|||github_search|
|csuyeon190/amplifyReactUnityWebGL|https://github.com/csuyeon190/amplifyReactUnityWebGL|code_candidate|UniAMP web server|||github_search|
|dholesh/amplify-unicorn-website|https://github.com/dholesh/amplify-unicorn-website|code_candidate|UniAMP web server|||github_search|
|cicese-biocom/PepProtGraphAnalyzer|https://github.com/cicese-biocom/PepProtGraphAnalyzer|code_candidate|PepProtGraphAnalyzer|||github_search|
|cicese-biocom/esm-AxP-GDL|https://github.com/cicese-biocom/esm-AxP-GDL|code_candidate|esm-AxP-GDL|||github_search|
|cicese-biocom/esm-AxP-GDL_v2|https://github.com/cicese-biocom/esm-AxP-GDL_v2|code_candidate|esm-AxP-GDL|||github_search|
|standard-things/esm|https://github.com/standard-things/esm|code_candidate|esm|||github_search|
|facebookresearch/esm|https://github.com/facebookresearch/esm|code_candidate|esm|||github_search|
|Biohub/esm|https://github.com/Biohub/esm|code_candidate|esm|||github_search|
|czeslaw-milosz/ampscannerv2|https://github.com/czeslaw-milosz/ampscannerv2|code_candidate|AMP Scanner|||github_search|
|FusRaDa/AMPScanner_NF|https://github.com/FusRaDa/AMPScanner_NF|code_candidate|AMP Scanner|||github_search|
|uclahs-cds/package-moPepGen|https://github.com/uclahs-cds/package-moPepGen|code_candidate|PepGen 1.0|||github_search|
|KalyanPalepu/PepGen|https://github.com/KalyanPalepu/PepGen|code_candidate|PepGen 1.0|||github_search|
|bigbio/pepgenome|https://github.com/bigbio/pepgenome|code_candidate|PepGen 1.0|||github_search|
|Anorpe/ampepgen-dev|https://github.com/Anorpe/ampepgen-dev|code_candidate|AmPepGen|||github_search|
|computational-genomics-lab/AMP_peptide_generation|https://github.com/computational-genomics-lab/AMP_peptide_generation|code_candidate|AmPepGen|||github_search|
|martskow/Generative-models-of-amyloid-peptides|https://github.com/martskow/Generative-models-of-amyloid-peptides|code_candidate|AmPepGen|||github_search|
|Nate0634034090/nate.283090|https://github.com/Nate0634034090/nate.283090|code_candidate|PepGen 1.0 web server|||github_search|
|websimapi/websim-brupr-pepsi-man-night-generator|https://github.com/websimapi/websim-brupr-pepsi-man-night-generator|code_candidate|PepGen 1.0 web server|||github_search|
|KeKo6988/Karma|https://github.com/KeKo6988/Karma|code_candidate|PepGen 1.0 web server|||github_search|
|CDDLeiden/CalcAMP|https://github.com/CDDLeiden/CalcAMP|code_candidate|CalcAMP|||github_search|
|anyahirota/calcamper|https://github.com/anyahirota/calcamper|code_candidate|CalcAMP|||github_search|
|mariiko-dev/calcamper|https://github.com/mariiko-dev/calcamper|code_candidate|CalcAMP|||github_search|
|Colin-CompChem/CalcAMP|https://github.com/Colin-CompChem/CalcAMP|code_candidate|CalcAMP GitHub repository|||github_search|
|cleissonheggdorne/calcamp|https://github.com/cleissonheggdorne/calcamp|code_candidate|CalcAMP GitHub repository|||github_search|
|h-khabbaz/amp-toxicity-predictor|https://github.com/h-khabbaz/amp-toxicity-predictor|code_candidate|AMP toxicity prediction code|||github_search|
|danielm710/AMP-webserver|https://github.com/danielm710/AMP-webserver|code_candidate|AMP0 webserver|||github_search|
|agrove15/amp_webserver|https://github.com/agrove15/amp_webserver|code_candidate|AMP0 webserver|||github_search|
|pcipov/Docker-AMP-webserver|https://github.com/pcipov/Docker-AMP-webserver|code_candidate|AMP0 webserver|||github_search|
|AidaSousa/ampa|https://github.com/AidaSousa/ampa|code_candidate|AMPA|||github_search|
|h-khabbaz/ampact|https://github.com/h-khabbaz/ampact|code_candidate|AMPA|||github_search|
|iTeam-S/Ampalibe|https://github.com/iTeam-S/Ampalibe|code_candidate|AMPA|||github_search|
|miminiyo/ampaweb|https://github.com/miminiyo/ampaweb|code_candidate|AMPA web server|||github_search|
|abrilrdzt/AMPAweb|https://github.com/abrilrdzt/AMPAweb|code_candidate|AMPA web server|||github_search|
|jmanuelascacibar/ampa-web|https://github.com/jmanuelascacibar/ampa-web|code_candidate|AMPA web server|||github_search|
|lantianyao/AMPActiPred|https://github.com/lantianyao/AMPActiPred|code_candidate|AMPActiPred|||github_search|
|abdullah-abunada/amps-activity-prediction-model|https://github.com/abdullah-abunada/amps-activity-prediction-model|code_candidate|AMPActiPred|||github_search|
|wangpuai/multilabel-AMP-activity-prediction|https://github.com/wangpuai/multilabel-AMP-activity-prediction|code_candidate|AMPActiPred|||github_search|
|NVIDIA/apex|https://github.com/NVIDIA/apex|code_candidate|APEX|||github_search|
|oracle/apex|https://github.com/oracle/apex|code_candidate|APEX|||github_search|
|pensarai/apex|https://github.com/pensarai/apex|code_candidate|APEX|||github_search|
|abcair/AMPFinder|https://github.com/abcair/AMPFinder|code_candidate|AMPfinder|||github_search|
|jhjhong/m-Meta-AMPfinder|https://github.com/jhjhong/m-Meta-AMPfinder|code_candidate|AMPfinder|||github_search|
|bioaicuhksz/AMPfinder|https://github.com/bioaicuhksz/AMPfinder|code_candidate|AMPfinder|||github_search|
|ANTIBUNGARIZKIAH/project3-bpx-anti|https://github.com/ANTIBUNGARIZKIAH/project3-bpx-anti|code_candidate|AntiBP3 GitLab|||github_search|
|Nate0634034090/bug-free-memory|https://github.com/Nate0634034090/bug-free-memory|code_candidate|dbAMP 3.0 web server|||github_search|
|udinparla/aa.py|https://github.com/udinparla/aa.py|code_candidate|dbAMP 3.0 web server|||github_search|
|BioGenies/AMPBenchmark|https://github.com/BioGenies/AMPBenchmark|code_candidate|AMPBenchmark|||github_search|
|yutarochan/AMP_Benchmark|https://github.com/yutarochan/AMP_Benchmark|code_candidate|AMPBenchmark|||github_search|
|rpcme/amp-benchmark-mcu|https://github.com/rpcme/amp-benchmark-mcu|code_candidate|AMPBenchmark|||github_search|
|GHodg1/AmideYieldPredictor|https://github.com/GHodg1/AmideYieldPredictor|code_candidate|CAmidPred|||github_search|
|maxreed/amide_hydrogen_shift_predictor_structural_v1|https://github.com/maxreed/amide_hydrogen_shift_predictor_structural_v1|code_candidate|CAmidPred|||github_search|
|ZYChen33/ML-assisted-amidase-catalytic-enantioselectivity-prediction-and-rational-design|https://github.com/ZYChen33/ML-assisted-amidase-catalytic-enantioselectivity-prediction-and-rational-design|code_candidate|CAmidPred|||github_search|
|Grupo-Medicina-Molecular-y-Traslacional/StarPep|https://github.com/Grupo-Medicina-Molecular-y-Traslacional/StarPep|code_candidate|StarPep|||github_search|
|GnanaloshiniA27/star-pep-project|https://github.com/GnanaloshiniA27/star-pep-project|code_candidate|StarPep|||github_search|
|peppo0star/peppo0star|https://github.com/peppo0star/peppo0star|code_candidate|StarPep|||github_search|
|chinmayaNK22/PepAnnotate|https://github.com/chinmayaNK22/PepAnnotate|code_candidate|PepAnno|||github_search|
|pepkit/pep_annotationhub|https://github.com/pepkit/pep_annotationhub|code_candidate|PepAnno|||github_search|
|edammer/MQ1pepAnnotate|https://github.com/edammer/MQ1pepAnnotate|code_candidate|PepAnno|||github_search|
|michbur/AmpGram|https://github.com/michbur/AmpGram|code_candidate|AmpGram|||github_search|
|cran/AmpGram|https://github.com/cran/AmpGram|code_candidate|AmpGram|||github_search|
|michbur/AmpGramModel|https://github.com/michbur/AmpGramModel|code_candidate|AmpGram|||github_search|
|Legana/ampir|https://github.com/Legana/ampir|code_candidate|Ampir|||github_search|
|Aveglia/vAMPirus|https://github.com/Aveglia/vAMPirus|code_candidate|Ampir|||github_search|
|quotra12/AmpirV|https://github.com/quotra12/AmpirV|code_candidate|Ampir|||github_search|
|Amth274/Ensemble-protein-embedding-framework-for-AMP-prediction|https://github.com/Amth274/Ensemble-protein-embedding-framework-for-AMP-prediction|code_candidate|Ensemble-AMPPred|||github_search|
|BioGenies/CancerGram|https://github.com/BioGenies/CancerGram|code_candidate|CancerGram|||github_search|
|BioGenies/CancerGramModel|https://github.com/BioGenies/CancerGramModel|code_candidate|CancerGram|||github_search|
|BioGenies/CancerGram-analysis|https://github.com/BioGenies/CancerGram-analysis|code_candidate|CancerGram|||github_search|
|YPZ858/PPTPP|https://github.com/YPZ858/PPTPP|code_candidate|PPTPP|||github_search|
|pld-linux/pptpproxy|https://github.com/pld-linux/pptpproxy|code_candidate|PPTPP|||github_search|
|TheSilentWolf1886/pptpp|https://github.com/TheSilentWolf1886/pptpp|code_candidate|PPTPP|||github_search|
|tangwending/MLBP|https://github.com/tangwending/MLBP|code_candidate|MLBP|||github_search|
|Irbaaz786/mlbp|https://github.com/Irbaaz786/mlbp|code_candidate|MLBP|||github_search|
|chrisjackson4256/MLBPitchPredictor|https://github.com/chrisjackson4256/MLBPitchPredictor|code_candidate|MLBP|||github_search|
|saikrishna-1996/deep_pepper_chess|https://github.com/saikrishna-1996/deep_pepper_chess|code_candidate|Deep2Pep|||github_search|
|IBPA/DeepPep|https://github.com/IBPA/DeepPep|code_candidate|Deep2Pep|||github_search|
|fteufel/DeepPeptide|https://github.com/fteufel/DeepPeptide|code_candidate|Deep2Pep|||github_search|
|ghli16/CG-AMP|https://github.com/ghli16/CG-AMP|code_candidate|CG-AMP|||github_search|
|klyLab/CGAMP|https://github.com/klyLab/CGAMP|code_candidate|CG-AMP|||github_search|
|timiabayomi/cascode-cs-cg-amplifier|https://github.com/timiabayomi/cascode-cs-cg-amplifier|code_candidate|CG-AMP|||github_search|
|AledHe/AmpHGT|https://github.com/AledHe/AmpHGT|code_candidate|AmpHGT|||github_search|
|AledHe/AmpHGT_db|https://github.com/AledHe/AmpHGT_db|code_candidate|AmpHGT|||github_search|
|NanjunChen37/TP_LMMSG|https://github.com/NanjunChen37/TP_LMMSG|code_candidate|TP-LMMSG|||github_search|
|moonseter/PGAT-ABPp|https://github.com/moonseter/PGAT-ABPp|code_candidate|PGAT-ABPp|||github_search|
|xialab-ahu/PrMFTP|https://github.com/xialab-ahu/PrMFTP|code_candidate|PrMFTP|||github_search|
|nahid18/PrMFTP-wf|https://github.com/nahid18/PrMFTP-wf|code_candidate|PrMFTP|||github_search|
|saeedalsarhi/Mini-File-Transfer-Program-MFTP|https://github.com/saeedalsarhi/Mini-File-Transfer-Program-MFTP|code_candidate|PrMFTP|||github_search|
|lantianyao/DeepAFP|https://github.com/lantianyao/DeepAFP|code_candidate|DeepAFP|||github_search|
|wangyuze18/DeepSeek-W4AFP8-AWQ|https://github.com/wangyuze18/DeepSeek-W4AFP8-AWQ|code_candidate|DeepAFP|||github_search|
|jerry1984Y/AFP-Deep|https://github.com/jerry1984Y/AFP-Deep|code_candidate|DeepAFP|||github_search|
|ThammakornS/amppred-aaiw|https://github.com/ThammakornS/amppred-aaiw|code_candidate|AMPpred-AAIW|||github_search|
|agrawalpiyush-srm/AMP_MetaAnalysis|https://github.com/agrawalpiyush-srm/AMP_MetaAnalysis|code_candidate|AMP-META|||github_search|
|ampproject/meta|https://github.com/ampproject/meta|code_candidate|AMP-META|||github_search|
|jieluyan/MBC-Attention|https://github.com/jieluyan/MBC-Attention|code_candidate|MBC-attention|||github_search|
|szczurek-lab/BattleAMP-mbc-attention|https://github.com/szczurek-lab/BattleAMP-mbc-attention|code_candidate|MBC-attention|||github_search|
|William-Zhanng/SenseXAMP|https://github.com/William-Zhanng/SenseXAMP|code_candidate|SenseXAMP|||github_search|
|szczurek-lab/BattleAMP-senseXAMP|https://github.com/szczurek-lab/BattleAMP-senseXAMP|code_candidate|SenseXAMP|||github_search|
|Chaste/ApPredict|https://github.com/Chaste/ApPredict|code_candidate|Appred|||github_search|
|MKSBarbosa/AppRedes|https://github.com/MKSBarbosa/AppRedes|code_candidate|Appred|||github_search|
|ioskrish/StreetStyleStore-AppRedesign|https://github.com/ioskrish/StreetStyleStore-AppRedesign|code_candidate|Appred|||github_search|
|BCV-Uniandes/AMPs-Net|https://github.com/BCV-Uniandes/AMPs-Net|code_candidate|AMPs-Net|||github_search|
|vohidjon123/google|https://github.com/vohidjon123/google|code_candidate|AMPs-Net|||github_search|
|noviamandaps/Novi-Amanda-PS-OOP-NET|https://github.com/noviamandaps/Novi-Amanda-PS-OOP-NET|code_candidate|AMPs-Net|||github_search|
|chainreaction/LabAmp|https://github.com/chainreaction/LabAmp|code_candidate|LABAMPs|||github_search|
|lkytal/PepNet|https://github.com/lkytal/PepNet|code_candidate|PepNet|||github_search|
|openvax/pepnet|https://github.com/openvax/pepnet|code_candidate|PepNet|||github_search|
|hjy23/PepNet|https://github.com/hjy23/PepNet|code_candidate|PepNet|||github_search|
|VeniQs02/pep.net-web-app|https://github.com/VeniQs02/pep.net-web-app|code_candidate|PepNet web server|||github_search|
|landy22granatt/Kumpulan-Script-Termux|https://github.com/landy22granatt/Kumpulan-Script-Termux|code_candidate|PepNet web server|||github_search|
|stdlib-js/lapack-base-clacpy|https://github.com/stdlib-js/lapack-base-clacpy|code_candidate|CL-ACP|||github_search|
|Hassanjaved4157/CLACP|https://github.com/Hassanjaved4157/CLACP|code_candidate|CL-ACP|||github_search|
|Dwip055/Clacpro|https://github.com/Dwip055/Clacpro|code_candidate|CL-ACP|||github_search|
|AspirinCode/AMPTrans-lstm|https://github.com/AspirinCode/AMPTrans-lstm|code_candidate|AMPTrans-lstm|||github_search|
|shunsunsun/AMPTrans-lstm|https://github.com/shunsunsun/AMPTrans-lstm|code_candidate|AMPTrans-lstm|||github_search|
|AnikaSharma17/Explainable-AMP-Transformer|https://github.com/AnikaSharma17/Explainable-AMP-Transformer|code_candidate|AMPTrans-lstm|||github_search|
|kren-ai-lab/amp_class_ml|https://github.com/kren-ai-lab/amp_class_ml|code_candidate|AmpClass|||github_search|
|AlanRavelo/AmpClassD---TPA3118---Texas-Instruments|https://github.com/AlanRavelo/AmpClassD---TPA3118---Texas-Instruments|code_candidate|AmpClass|||github_search|
|AFP_DL GitHub|https://github.com/DongYin521/AFP_DL|code||||fulltext|
|Ampir GitHub|https://github.com/legana/Ampir|code||||repository|
|AMP Scanner v2 GitHub|https://github.com/amp-scanner/AMP-Scanner-v2|code||||repository|
|Macrel GitHub|https://github.com/BigDataBiology/macrel|code||||repository|
|amPEPpy GitHub|https://github.com/tlawrence3/amPEPpy|code||||repository|
|ampsphere portal|https://ampsphere.big-data-biology.org/|webserver|ampsphere|39557756|10.1007/s12602-024-10402-4|review|
|https://github.com/onkarS23/CoAMPpred|||||||
|https://github.com/joy50706/iAMPCN|||||||
|https://github.com/Fuhaoyi/ACEP|||||||
|https://github.com/ggcameronnogg/SSFGM-Model|||||||
|https://github.com/DongYin521/AFP_DL|||||||
|https://github.com/bcgsc/AMPlify|||||||
|https://github.com/Wangsicheng52/E-CLEAP|||||||
|https://github.com/harkic/UniproLcad|||||||
|https://github.com/hjy23/TriStack|||||||
|https://github.com/mldlproject/2022-iAMP-DL|||||||
|https://gitlab.com/vail-uvm/amp-gan|||||||
|https://github.com/SilverGojo4/ANIA|||||||
|https://github.com/haichengyi/ACP-DL|||||||
|AMPfun|http://fdblab.csie.ncu.edu.tw/AMPfun/|webserver|AMPfun|41155367|10.3390/ijms262010077|fulltext|
|AntiCP|http://crdd.osdd.net/raghava/anticp/|webserver|AntiCP|41155367|10.3390/ijms262010077|fulltext|
|AntiCP2.0|https://webs.iiitd.edu.in/raghava/anticp2/|webserver|AntiCP2.0|41155367|10.3390/ijms262010077|fulltext|
|ACPred|http://codes.bio/acpred/|webserver|ACPred|41155367|10.3390/ijms262010077|fulltext|
|iAMPpred|http://cabgrid.res.in:8080/amppred/index/|webserver|iAMPpred|41155367|10.3390/ijms262010077|fulltext|
|Macrel|https://www.big-data-biology.org/software/macrel/|code|Macrel|41155367|10.3390/ijms262010077|fulltext|
|HAPPENN|https://research.timmons.eu/happenn/|webserver|HAPPENN|41155367|10.3390/ijms262010077|fulltext|
|HemoPred|http://codes.bio/hemopred/|webserver|HemoPred|41155367|10.3390/ijms262010077|fulltext|
|ToxinPred|http://crdd.osdd.net/raghava/toxinpred/|webserver|ToxinPred|41155367|10.3390/ijms262010077|fulltext|
|ToxIBTL|https://server.wei-group.net/ToxIBTL/|webserver|ToxIBTL|41155367|10.3390/ijms262010077|fulltext|
|AllerTop|https://www.ddg-pharmfac.net/AllerTOP/|webserver|AllerTop|41155367|10.3390/ijms262010077|fulltext|
|AllergenFP|https://ddg-pharmfac.net/AllergenFP/|webserver|AllergenFP|41155367|10.3390/ijms262010077|fulltext|
|AllerCatPro|https://allercatpro.bii.a-star.edu.sg/|webserver|AllerCatPro|41155367|10.3390/ijms262010077|fulltext|
|Deep learning hybrid model (unnamed)|not_reported_in_available_evidence|code|Deep learning hybrid model (unnamed)|41731616|10.1186/s40168-025-02326-0|fulltext|
|Antimicrobial Peptide Scanner (APSvr.2) webserver|https://aps.unmc.edu/prediction/predict|webserver|APSvr.2|37523405|10.1371/journal.ppat.1011508|fulltext|
|AMP prediction server (biosino)|http://amp.biosino.org/|webserver|AMP prediction server (biosino)|21533231|10.1371/journal.pone.0018476|fulltext|
|PeptideRanker|http://distilldeep.ucd.ie/PeptideRanker/|webserver|not_AMP_specific|42276016|10.1016/j.ultsonch.2026.107920|fulltext|
|ADMETlab 3|https://admetlab3.scbdd.com|webserver|not_AMP_specific|42276016|10.1016/j.ultsonch.2026.107920|fulltext|
|Urchin|https://github.com/VirtualBrainLab/Urchin|code_or_web||40233747|10.1016/j.neuron.2025.03.020|regex_fulltext_or_metadata|
|allenCCF|https://github.com/cortex-lab/allenCCF|code_or_web||40233747|10.1016/j.neuron.2025.03.020|regex_fulltext_or_metadata|
|phy|https://github.com/cortex-lab/phy|code_or_web||40233747|10.1016/j.neuron.2025.03.020|regex_fulltext_or_metadata|
|iblapps|https://github.com/int-brain-lab/iblapps|code_or_web||40233747|10.1016/j.neuron.2025.03.020|regex_fulltext_or_metadata|
|Lab|https://github.com/tortugar/Lab|code_or_web||40233747|10.1016/j.neuron.2025.03.020|regex_fulltext_or_metadata|
|Npx.|https://github.com/tortugar/Npx.|code_or_web||40233747|10.1016/j.neuron.2025.03.020|regex_fulltext_or_metadata|
|amp_de_novo_design_cdGAN|https://github.com/aretiz/amp_de_novo_design_cdGAN|code|cdGAN|41137855|10.1093/bib/bbaf500|fulltext|
|axpep|https://sourceforge.net/projects/axpep/|code|AxPEP|41315055|10.1007/s00248-025-02620-2|fulltext|
|AMP Scanner V2 webserver|https://www.dveltri.com/ascan/v2/ascan.html|webserver|AMP Scanner V2|41315055|10.1007/s00248-025-02620-2|fulltext|
|AMP Scanner V2 code|https://github.com/dan-veltri/amp-scanner-v2|code|AMP Scanner V2|41315055|10.1007/s00248-025-02620-2|github_search|
|hydramp|https://github.com/szczurek-lab/hydramp|code|hydramp|41315055|10.1007/s00248-025-02620-2|github_search|
|AntiBP2|https://github.com/raghavagps/AntiBP2|code|AntiBP2|41315055|10.1007/s00248-025-02620-2|github_search|
|PrefixProt|https://github.com/chen-bioinfo/PrefixProt|code|PrefixProt|41315055|10.1007/s00248-025-02620-2|github_search|
|Deep-AmPEP30|https://github.com/Chonwai/Deep_AmPEP30_R|code|Deep-AmPEP30|41315055|10.1007/s00248-025-02620-2|github_search|
|CAMPR3|https://github.com/Campr-Project-Management/campr|code|CAMPR3|41315055|10.1007/s00248-025-02620-2|github_search|
|kneaddata|https://github.com/biobakery/kneaddata|tool||41315055|10.1007/s00248-025-02620-2|fulltext|
|VirSorter2|https://github.com/jiarong/VirSorter2|tool||41315055|10.1007/s00248-025-02620-2|fulltext|
|COGclassifier|https://github.com/moshi4/COGclassifier|tool||41315055|10.1007/s00248-025-02620-2|fulltext|
|AmPEP web server|http://cbbio.cis.umac.mo/software/AmPEP/|webserver|AmPEP|29374199|10.1038/s41598-018-19752-w|fulltext|
|hydramp GitHub repo|https://github.com/szczurek-lab/hydramp|github|hydramp|29374199|10.1038/s41598-018-19752-w|github_search|
|Macrel GitHub repo|https://github.com/BigDataBiology/macrel|github|Macrel|29374199|10.1038/s41598-018-19752-w|github_search|
|AmPEP GitHub repo|https://github.com/ShirleyWISiu/AmPEP|github|AmPEP|29374199|10.1038/s41598-018-19752-w|github_search|
|AMP Scanner v2 GitHub repo|https://github.com/dan-veltri/amp-scanner-v2|github|AMP Scanner v2|29374199|10.1038/s41598-018-19752-w|github_search|
|AntiBP2 GitHub repo|https://github.com/raghavagps/AntiBP2|github|AntiBP2|29374199|10.1038/s41598-018-19752-w|github_search|
|Deep-AmPEP30 GitHub repo|https://github.com/Chonwai/Deep_AmPEP30_R|github|Deep-AmPEP30|29374199|10.1038/s41598-018-19752-w|github_search|
|PrefixProt GitHub repo|https://github.com/chen-bioinfo/PrefixProt|github|PrefixProt|29374199|10.1038/s41598-018-19752-w|github_search|
|PeptideRanker web server|http://bioware.ucd.ie/|webserver|PeptideRanker|23056189|10.1371/journal.pone.0045012|fulltext|
|AMPer web server|http://marray.cmdr.ubc.ca/cgi-bin/amp.pl|webserver|AMPer|23056189|10.1371/journal.pone.0045012|fulltext|
|HydrAMP feedstock|https://github.com/conda-forge/hydrampp-feedstock|code|HydrAMP|github_enrichment|10.3390/ijms22062857|github_search|
|PyTorch HydrAMP|https://github.com/marmarmarmar/pytorch-hydramp|code|HydrAMP|github_enrichment|10.3390/ijms22062857|github_search|
|AmPEP|https://github.com/ShirleyWISiu/AmPEP|code|AmPEP|github_enrichment|10.3390/ijms22062857|github_search|
|Ampep_Python|https://github.com/Chonwai/Ampep_Python|code|AmPEP|github_enrichment|10.3390/ijms22062857|github_search|
|AMP Scanner v2|https://github.com/dan-veltri/amp-scanner-v2|code|AMP Scanner v2|github_enrichment|10.3390/ijms22062857|github_search|
|Deep_AmPEP30_R|https://github.com/Chonwai/Deep_AmPEP30_R|code|Deep-AmPEP30|github_enrichment|10.3390/ijms22062857|github_search|
|MetaPepticon|https://github.com/arikanlab/MetaPepticon|code|MetaPepticon|github_enrichment|10.3390/ijms22062857|github_search|
|AVPpred-BWR|https://github.com/zyweizm/AVPpred-BWR|code|AVPpred|github_enrichment|10.3390/ijms22062857|github_search|
|Anti_Cp|https://github.com/xubocheng/Anti_Cp.git|code|WeightedEnsemble_L3|38266820|10.1016/j.jare.2024.01.023|fulltext|
|PLUM GitHub|https://github.com/priyamayur/PLUM|code|PLUM|42124643|10.64898/2026.02.21.707214|fulltext|
|Antimicrobial (regex match)|https://github.com/zswitten/Antimicrobial|code|PLUM|42124643|10.64898/2026.02.21.707214|uncertain|
|nov-fams-pipeline|https://github.com/AlvaroRodriguezDelRio/nov-fams-pipeline|code||38109938|10.1038/s41586-023-06955-z|fulltext|
|aro|https://github.com/arpcard/aro|code||38109938|10.1038/s41586-023-06955-z|fulltext|
|c_AMPs-prediction|https://github.com/mayuefine/c_AMPs-prediction|model|c_AMPs-prediction|41164228|10.3389/fvets.2025.1689589|fulltext|
|FMT-MetagenomicData|https://github.com/pointwei/FMT-MetagenomicData|code||41164228|10.3389/fvets.2025.1689589|fulltext|
|DeepSeaQuence_biofilms|https://github.com/trongthucnguyen/DeepSeaQuence_biofilms|code||42104260|10.1186/s12866-026-05098-1|fulltext|
|vynect/venom|https://github.com/vynect/venom|code_candidate|Venomics artificial intelligence|||github_search|
|sugeth/xxx|https://github.com/sugeth/xxx|code_candidate|Venomics artificial intelligence|||github_search|
|keonjale/amplifygithubrepo|https://github.com/keonjale/amplifygithubrepo|code_candidate|AMPlify GitHub|||github_search|
|arikanlab/MetaPepticon|https://github.com/arikanlab/MetaPepticon|code_candidate|MetaPepticon|||github_search|
|full-stack-serverless/full-stack-amplify|https://github.com/full-stack-serverless/full-stack-amplify|code_candidate|StackAMP|||github_search|
|AmpolStack/AmpolStack|https://github.com/AmpolStack/AmpolStack|code_candidate|StackAMP|||github_search|
|dabit3/expo-amplify-full-stack-cloud-workshop|https://github.com/dabit3/expo-amplify-full-stack-cloud-workshop|code_candidate|StackAMP|||github_search|
|Amal-Thomas/Amal-Thomas-PEP-GP-WebDevProject-Recipe|https://github.com/Amal-Thomas/Amal-Thomas-PEP-GP-WebDevProject-Recipe|code_candidate|AmPEP web server|||github_search|
|harshal0004/Ecommerce|https://github.com/harshal0004/Ecommerce|code_candidate|AmPEP web server|||github_search|
|Iskingcomet/Shamar-Roberts|https://github.com/Iskingcomet/Shamar-Roberts|code_candidate|AmPEP web server|||github_search|
|AmirhesamGhahari/Amir_Ghahari_Personal_Website_API_Server|https://github.com/AmirhesamGhahari/Amir_Ghahari_Personal_Website_API_Server|code_candidate|AMPer web server|||github_search|
|wasim15185/Amazon-Clone-PERN-stack|https://github.com/wasim15185/Amazon-Clone-PERN-stack|code_candidate|AMPer web server|||github_search|
|questionmark1122/cnn10|https://github.com/questionmark1122/cnn10|code_candidate|AMPer web server|||github_search|
|Ronald106/Surviv.io|https://github.com/Ronald106/Surviv.io|code_candidate|CatBoost AMP predictor|||github_search|
|kkzheng/Two_Level_Ensemble-classifier-chain|https://github.com/kkzheng/Two_Level_Ensemble-classifier-chain|code_candidate|Two_Level_Ensemble-classifier-chain|||github_search|
|aretiz/amp_de_novo_design_cdGAN|https://github.com/aretiz/amp_de_novo_design_cdGAN|code_candidate|amp_de_novo_design_cdGAN|||github_search|
|thienhaiblue/mbed_ampm_gsm_uip_lwip|https://github.com/thienhaiblue/mbed_ampm_gsm_uip_lwip|code_candidate|AMP-GSM|||github_search|
|nabtodaemon/GSM-AMP|https://github.com/nabtodaemon/GSM-AMP|code_candidate|AMP-GSM|||github_search|
|DemetParlakSonmez/amp-zGSM|https://github.com/DemetParlakSonmez/amp-zGSM|code_candidate|AMP-GSM|||github_search|
|ImaginaryLandscape/iscape-jobboard|https://github.com/ImaginaryLandscape/iscape-jobboard|code_candidate|ISCAPE|||github_search|
|lorochka85/iscape-djangonews|https://github.com/lorochka85/iscape-djangonews|code_candidate|ISCAPE|||github_search|
|Bindo56/Iscape-Out|https://github.com/Bindo56/Iscape-Out|code_candidate|ISCAPE|||github_search|
|Violet-maple/Violet-maple.github.io|https://github.com/Violet-maple/Violet-maple.github.io|code_candidate|MAPLE GitHub|||github_search|
|ForstMaple/ForstMaple.github.io|https://github.com/ForstMaple/ForstMaple.github.io|code_candidate|MAPLE GitHub|||github_search|
|FlyMaple/FlyMaple.github.io|https://github.com/FlyMaple/FlyMaple.github.io|code_candidate|MAPLE GitHub|||github_search|
|axepttv/Axpep|https://github.com/axepttv/Axpep|code_candidate|AxPEP|||github_search|
|Chonwai/AxPEP_Backend|https://github.com/Chonwai/AxPEP_Backend|code_candidate|AxPEP|||github_search|
|Chonwai/AxPEP_Ecotoxicology_Core|https://github.com/Chonwai/AxPEP_Ecotoxicology_Core|code_candidate|AxPEP|||github_search|
|biobakery/kneaddata|https://github.com/biobakery/kneaddata|code_candidate|kneaddata|||github_search|
|bmorganpalmer/kneaddata|https://github.com/bmorganpalmer/kneaddata|code_candidate|kneaddata|||github_search|
|EagleGenomics-cookbooks/KneadData|https://github.com/EagleGenomics-cookbooks/KneadData|code_candidate|kneaddata|||github_search|
|jiarong/VirSorter2|https://github.com/jiarong/VirSorter2|code_candidate|VirSorter2|||github_search|
|simroux/VirSorter|https://github.com/simroux/VirSorter|code_candidate|VirSorter2|||github_search|
|simroux/VirSorter2_to_Anvio|https://github.com/simroux/VirSorter2_to_Anvio|code_candidate|VirSorter2|||github_search|
|moshi4/COGclassifier|https://github.com/moshi4/COGclassifier|code_candidate|COGclassifier|||github_search|
|pworden/COGclassifier|https://github.com/pworden/COGclassifier|code_candidate|COGclassifier|||github_search|
|raghavagps/anticp2|https://github.com/raghavagps/anticp2|code_candidate|Anti_Cp|||github_search|
|MartinHessler/antiCPy|https://github.com/MartinHessler/antiCPy|code_candidate|Anti_Cp|||github_search|
|xubocheng/Anti_Cp|https://github.com/xubocheng/Anti_Cp|code_candidate|Anti_Cp|||github_search|
|AntiO-cps/antio-cps.github.io|https://github.com/AntiO-cps/antio-cps.github.io|code_candidate|Anti_Cp.git|||github_search|
|rime/plum|https://github.com/rime/plum|code_candidate|PLUM|||github_search|
|beartype/plum|https://github.com/beartype/plum|code_candidate|PLUM|||github_search|
|yegor256/plum|https://github.com/yegor256/plum|code_candidate|PLUM|||github_search|
|purpleplum456/purple-plum-GitHub|https://github.com/purpleplum456/purple-plum-GitHub|code_candidate|PLUM GitHub|||github_search|
|eeapbh/plum-github-web|https://github.com/eeapbh/plum-github-web|code_candidate|PLUM GitHub|||github_search|
|PlumGithub/atlas-app|https://github.com/PlumGithub/atlas-app|code_candidate|PLUM GitHub|||github_search|
|BirolLab/AMPlify|https://github.com/BirolLab/AMPlify|code_candidate|Antimicrobial|||github_search|
|duckyb/urchin|https://github.com/duckyb/urchin|code_candidate|Urchin|||github_search|
|GPeye/urchin-peripheral-animation|https://github.com/GPeye/urchin-peripheral-animation|code_candidate|Urchin|||github_search|
|matheusgomes28/urchin|https://github.com/matheusgomes28/urchin|code_candidate|Urchin|||github_search|
|cortex-lab/allenCCF|https://github.com/cortex-lab/allenCCF|code_candidate|allenCCF|||github_search|
|thewtex/allen-ccf-itk-vtk-zarr|https://github.com/thewtex/allen-ccf-itk-vtk-zarr|code_candidate|allenCCF|||github_search|
|bjmiao/allenCCF|https://github.com/bjmiao/allenCCF|code_candidate|allenCCF|||github_search|
|lo-th/phy|https://github.com/lo-th/phy|code_candidate|phy|||github_search|
|cortex-lab/phy|https://github.com/cortex-lab/phy|code_candidate|phy|||github_search|
|NVIDIA/physicsnemo|https://github.com/NVIDIA/physicsnemo|code_candidate|phy|||github_search|
|int-brain-lab/iblapps|https://github.com/int-brain-lab/iblapps|code_candidate|iblapps|||github_search|
|google-deepmind/lab|https://github.com/google-deepmind/lab|code_candidate|Lab|||github_search|
|zaquestion/lab|https://github.com/zaquestion/lab|code_candidate|Lab|||github_search|
|wywu/LAB|https://github.com/wywu/LAB|code_candidate|Lab|||github_search|
|zkat/npx|https://github.com/zkat/npx|code_candidate|Npx|||github_search|
|npm/npx|https://github.com/npm/npx|code_candidate|Npx|||github_search|
|sigma-py/npx|https://github.com/sigma-py/npx|code_candidate|Npx|||github_search|
|DuannYu/soft-neighbors--supported-clustering|https://github.com/DuannYu/soft-neighbors--supported-clustering|code_candidate|soft-neighbors-supported-clustering|||github_search|
|sayantann11/all-classification-templetes-for-ML|https://github.com/sayantann11/all-classification-templetes-for-ML|code_candidate|soft-neighbors-supported-clustering|||github_search|
|apex/apex-go|https://github.com/apex/apex-go|code_candidate|ApexGO|||github_search|
|shunjikonishi/apex-google-api|https://github.com/shunjikonishi/apex-google-api|code_candidate|ApexGO|||github_search|
|Yimeng-Zeng/APEXGo|https://github.com/Yimeng-Zeng/APEXGo|code_candidate|ApexGO|||github_search|
|mayuefine/c_AMPs-prediction|https://github.com/mayuefine/c_AMPs-prediction|code_candidate|c_AMPs-prediction|||github_search|
|ChenSizhe13893461199/Fast-AMPs-Discovery-Projects|https://github.com/ChenSizhe13893461199/Fast-AMPs-Discovery-Projects|code_candidate|c_AMPs-prediction|||github_search|
|trongthucnguyen/DeepSeaQuence_biofilms|https://github.com/trongthucnguyen/DeepSeaQuence_biofilms|code_candidate|DeepSeaQuence_biofilms|||github_search|
|pointwei/FMT-MetagenomicData|https://github.com/pointwei/FMT-MetagenomicData|code_candidate|FMT-MetagenomicData|||github_search|
|noodles/ampfun|https://github.com/noodles/ampfun|code_candidate|AMPfun|||github_search|
|TearsWaiting/ACPred-LAF|https://github.com/TearsWaiting/ACPred-LAF|code_candidate|ACPred|||github_search|
|tamshun/Large-Scale_ACPrediction|https://github.com/tamshun/Large-Scale_ACPrediction|code_candidate|ACPred|||github_search|
|leleshidawang/CNBT-ACPred|https://github.com/leleshidawang/CNBT-ACPred|code_candidate|ACPred|||github_search|
|tejaskale19/happenn|https://github.com/tejaskale19/happenn|code_candidate|HAPPENN|||github_search|
|khalidmadih/Happenn|https://github.com/khalidmadih/Happenn|code_candidate|HAPPENN|||github_search|
|HAPPENnewbie/HAPPENnewbie|https://github.com/HAPPENnewbie/HAPPENnewbie|code_candidate|HAPPENN|||github_search|
|Ranggaalan/HemoPredict-Streamlit-App-Using-ABC-Optimized-XGBoost-for-Hemodialysis-Complication-Prediction|https://github.com/Ranggaalan/HemoPredict-Streamlit-App-Using-ABC-Optimized-XGBoost-for-Hemodialysis-Complication-Prediction|code_candidate|HemoPred|||github_search|
|maahi89/-HemoPredict-Harnessing-Data-for-Blood-Cancer-Prognosis|https://github.com/maahi89/-HemoPredict-Harnessing-Data-for-Blood-Cancer-Prognosis|code_candidate|HemoPred|||github_search|
|chaninn/HemoPred|https://github.com/chaninn/HemoPred|code_candidate|HemoPred|||github_search|
|raghavagps/toxinpred3|https://github.com/raghavagps/toxinpred3|code_candidate|ToxinPred|||github_search|
|raghavagps/toxinpred2|https://github.com/raghavagps/toxinpred2|code_candidate|ToxinPred|||github_search|
|zxl124/Ocean-toxin-prediction|https://github.com/zxl124/Ocean-toxin-prediction|code_candidate|ToxinPred|||github_search|
|WLYLab/ToxIBTL|https://github.com/WLYLab/ToxIBTL|code_candidate|ToxIBTL|||github_search|
|dennyjames/allertop|https://github.com/dennyjames/allertop|code_candidate|AllerTop|||github_search|
|GhostTroops/TOP|https://github.com/GhostTroops/TOP|code_candidate|AllerTop|||github_search|
|sobia-naaz/vaccine-design-bioinformatics|https://github.com/sobia-naaz/vaccine-design-bioinformatics|code_candidate|AllerTop|||github_search|
|zszszszsz/.config|https://github.com/zszszszsz/.config|code_candidate|AllerCatPro|||github_search|
|ttzt/catalog_of_requirements_for_ai_products|https://github.com/ttzt/catalog_of_requirements_for_ai_products|code_candidate|AllerCatPro|||github_search|
|SuperRogerio/css|https://github.com/SuperRogerio/css|code_candidate|AllerCatPro|||github_search|
|TransDecoder/TransDecoder|https://github.com/TransDecoder/TransDecoder|code_candidate|TransDecoder|||github_search|
|sghignone/TransDecoder|https://github.com/sghignone/TransDecoder|code_candidate|TransDecoder|||github_search|
|TransDecoder/PyTransDecoder|https://github.com/TransDecoder/PyTransDecoder|code_candidate|TransDecoder|||github_search|
|kucukkal/admetlab3.0|https://github.com/kucukkal/admetlab3.0|code_candidate|ADMETlab 3|||github_search|
|royalananth/pregnancy-drug-card|https://github.com/royalananth/pregnancy-drug-card|code_candidate|ADMETlab 3|||github_search|
|naiff001212-lang/MTUyMTAzMjcyNzAxNTcyMzA3OQ.GLLLaX.pePW2uwpxpJvxncI85eCVLRhuh-0W9pvGfivbw|https://github.com/naiff001212-lang/MTUyMTAzMjcyNzAxNTcyMzA3OQ.GLLLaX.pePW2uwpxpJvxncI85eCVLRhuh-0W9pvGfivbw|code_candidate|AxPEP3|||github_search|
|melomcr/dbaasp_api_helper_libraries|https://github.com/melomcr/dbaasp_api_helper_libraries|code_candidate|DBAASP6|||github_search|
|Lee-ChinCheng/dbaasp-crawler|https://github.com/Lee-ChinCheng/dbaasp-crawler|code_candidate|DBAASP6|||github_search|
|SalasNorman/dbaasp|https://github.com/SalasNorman/dbaasp|code_candidate|DBAASP6|||github_search|
|Srinjay-GIT/Surface_Roughness_Prediction_|https://github.com/Srinjay-GIT/Surface_Roughness_Prediction_|code_candidate|ADAM (prediction tool)|||github_search|
|ManjirGrg/House-Price-Prediction|https://github.com/ManjirGrg/House-Price-Prediction|code_candidate|ADAM (prediction tool)|||github_search|
|dillard889/apsvrx|https://github.com/dillard889/apsvrx|code_candidate|APSvr.2|||github_search|
|apsvr/FFT|https://github.com/apsvr/FFT|code_candidate|APSvr.2|||github_search|
|saeedghoorchian/An-epsilon-SVR-Approach-for-Model-Identification|https://github.com/saeedghoorchian/An-epsilon-SVR-Approach-for-Model-Identification|code_candidate|APSvr.2|||github_search|
|dinalbenj/AmplifyBallotBox|https://github.com/dinalbenj/AmplifyBallotBox|code_candidate|AMPlify_bal|||github_search|
|BigDataBiology/macrel2020benchmark|https://github.com/BigDataBiology/macrel2020benchmark|code_candidate|macrel2020benchmark|||github_search|
|AlvaroRodriguezDelRio/nov-fams-pipeline|https://github.com/AlvaroRodriguezDelRio/nov-fams-pipeline|code_candidate|nov-fams-pipeline|||github_search|
|attdevsupport/ARO|https://github.com/attdevsupport/ARO|code_candidate|aro|||github_search|
|arpcard/aro|https://github.com/arpcard/aro|code_candidate|aro|||github_search|
|InterruptedLobster/ARO|https://github.com/InterruptedLobster/ARO|code_candidate|aro|||github_search|
|NK12131/Bankruptcy-Prediction-Using-Financial-KPIs-ML-Pipeline-with-SMOTE-PCA-Stacked-Ensemble|https://github.com/NK12131/Bankruptcy-Prediction-Using-Financial-KPIs-ML-Pipeline-with-SMOTE-PCA-Stacked-Ensemble|code_candidate|StackEnPred|||github_search|
|Pranov1984/Prediction-of-cement-compressive-strength-using-stacked-ensemble-modelling|https://github.com/Pranov1984/Prediction-of-cement-compressive-strength-using-stacked-ensemble-modelling|code_candidate|StackEnPred|||github_search|
|LBMercado/stacked-generalization-ensemble-learning-for-air-pollutant-concentration-prediction|https://github.com/LBMercado/stacked-generalization-ensemble-learning-for-air-pollutant-concentration-prediction|code_candidate|StackEnPred|||github_search|
|Co-AMPpred|https://github.com/onkarS23/CoAMPpred|code|Co-AMPpred||||

## GitHub Missing-Link Enrichment Evidence

|model_name|matched_model_name|name|url|description|stars|language|match_score|confidence_label|needs_manual_verification|evidence_level|query|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Co-AMPpred|Co-AMPpred|onkarS23/CoAMPpred|https://github.com/onkarS23/CoAMPpred||3|Python|1.0|high_confidence_repo|False|github_search|coamppred in:name|
|CTCM-Neo & ConformaX-PEP framework|CTCM-Neo & ConformaX-PEP framework||||||0.0|no_candidate|True|github_search_no_hit||
|Co-AMPpred GitHub repository|Co-AMPpred GitHub repository|onkarS23/CoAMPpred|https://github.com/onkarS23/CoAMPpred||3|Python|1.0|high_confidence_repo|False|github_search|coamppred in:name|
|CoAMPpred|CoAMPpred|onkarS23/CoAMPpred|https://github.com/onkarS23/CoAMPpred||3|Python|1.0|high_confidence_repo|False|github_search|coamppred in:name|
|2020-peptidomics|2020-peptidomics|ErikHartman/2020-peptidomics|https://github.com/ErikHartman/2020-peptidomics|This git repository contains the code for a research project at Lunds University.|0|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|2020-peptidomics|
|A-CaMP|A-CaMP|forthespada/CampusShame|https://github.com/forthespada/CampusShame|互联网仍有记忆！那些曾经在校招过程中毁过口头offer、意向书、三方的公司！纵然人微言轻，也想尽绵薄之力！|3370|JavaScript|1.0|high_confidence_repo|False|github_search|"camp"|
|PCSPred|PCSPred|AngryBytesTech/pcsprediction|https://github.com/AngryBytesTech/pcsprediction|Pancakeswap Prediction NodeJS module|1|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|pcspred in:name|
|iAMPCN|iAMPCN|zhiqan/AMPCNN|https://github.com/zhiqan/AMPCNN|A Fault Diagnosis Method of Rotor System Based on Parallel Convolutional Neural Network Architecture with Attention Mechanism|38|Python|1.0|high_confidence_repo|False|github_search|ampcn in:name|
|AAGP|AAGP|MartAlae-AAGP/AAGP|https://github.com/MartAlae-AAGP/AAGP|Adjacency-Adaptive Gaussian Process|0|Python|1.0|high_confidence_repo|False|github_search|aagp in:name|
|SSFGM-Model|SSFGM-Model|thomas0809/SSFGM|https://github.com/thomas0809/SSFGM|Semi-Supervised Factor Graph Model|2|Python|1.0|high_confidence_repo|False|github_search|ssfgm in:name|
|ACEP|ACEP|agusnieto77/ACEP|https://github.com/agusnieto77/ACEP|Análisis Computacional de Eventos de Protesta (ACEP). Computer-Aided Protest Event Analysis (CAPEA)|11|R|1.0|high_confidence_repo|False|github_search|acep in:name|
|ACP-DL|ACP-DL|haichengyi/ACP-DL|https://github.com/haichengyi/ACP-DL|A deep learning model to predict anticancer peptides.|25|Python|1.0|high_confidence_repo|False|github_search|acp dl|
|Anticancer-Peptides-CNN|Anticancer-Peptides-CNN|RafsanjaniHub/Anticancer-Peptides-CNN|https://github.com/RafsanjaniHub/Anticancer-Peptides-CNN|Anticancer Peptide Identification employing Multi-headed Deep-CNN|13|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|anticancer-peptides-cnn in:name|
|MetagenomicDC|MetagenomicDC|IcarPA-TBlab/MetagenomicDC|https://github.com/IcarPA-TBlab/MetagenomicDC||19|Python|1.0|high_confidence_repo|False|github_search|metagenomic-dc in:name|
|deep-belief-network|deep-belief-network|albertbup/deep-belief-network|https://github.com/albertbup/deep-belief-network|A Python implementation of Deep Belief Networks built upon NumPy and TensorFlow with scikit-learn compatibility|510|Python|1.0|high_confidence_repo|False|github_search|deep-belief-network in:name|
|MultiPep|MultiPep|scheelelab/MultiPep|https://github.com/scheelelab/MultiPep|MultiPep stand-alone program and network parameters|5|Python|1.0|high_confidence_repo|False|github_search|multi-pep in:name|
|acp-ope|acp-ope|khanhlee/acp-ope|https://github.com/khanhlee/acp-ope|Config files for my GitHub profile.|4|Python|1.0|high_confidence_repo|False|github_search|acp ope|
|iAMP-2L|iAMP-2L|amphp/amp|https://github.com/amphp/amp|A non-blocking concurrency framework for PHP applications. 🐘|4426|PHP|1.0|high_confidence_repo|False|github_search|amp in:name|
|iAMPred|iAMPred|sayalaruano/AMPredST|https://github.com/sayalaruano/AMPredST|Streamlit web application to deploy a machine learning binary classifier to predict the activity of antimicrobial peptides|10|Python|1.0|high_confidence_repo|False|github_search|ampred in:name|
|AmPEP|AmPEP|tlawrence3/amPEPpy|https://github.com/tlawrence3/amPEPpy|Sequence-based Identification of Antimicrobial Peptides using Distribution Patterns of Amino Acid Properties|30|Python|1.0|high_confidence_repo|False|github_search|am-pep in:name|
|AntiBP2|AntiBP2|raghavagps/AntiBP3|https://github.com/raghavagps/AntiBP3|An improved method for predicting of antibacterial peptides using machine learning yechniques|4|Python|1.0|high_confidence_repo|False|github_search|antibp in:name|
|CAMPR3|CAMPR3|Campr-Project-Management/campr|https://github.com/Campr-Project-Management/campr|Campr Workspace App|39|PHP|1.0|high_confidence_repo|False|github_search|campr in:name|
|ADAM|ADAM|bigdatagenomics/adam|https://github.com/bigdatagenomics/adam|ADAM is a genomics analysis platform with specialized file formats built using Apache Avro, Apache Spark, and Apache Parquet. Apache 2 licensed.|1053|Scala|1.0|high_confidence_repo|False|github_search|adam in:name|
|MLAMP|MLAMP|mlampros/mlampros.github.io|https://github.com/mlampros/mlampros.github.io|My personal blog|4|CSS|1.0|high_confidence_repo|False|github_search|mlamp in:name|
|ClassAMP|ClassAMP|chikitang/A|https://github.com/chikitang/A|!DOCTYPE html> <html lang="en" data-color-mode="auto" data-light-theme="light" data-dark-theme="dark" data-a11y-animated-images="system">   <head>     <meta charset="utf-8">   <link rel="dns-prefetch" href="https://github.githubassets.com">   <link rel="dns-prefetch" href="https://avatars.githubusercontent.com">   <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">   <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">   <link rel="preconnect" href="https://github.githubassets.com" crossorigin>   <link rel="preconnect" href="https://avatars.githubusercontent.com">      <link crossorigin="anonymous" media="all" integrity="sha512-ksfTgQOOnE+FFXf+yNfVjKSlEckJAdufFIYGK7ZjRhWcZgzAGcmZqqArTgMLpu90FwthqcCX4ldDgKXbmVMeuQ==" rel="stylesheet" href="https://github.githubassets.com/assets/light-92c7d381038e.css" /><link crossorigin="anonymous" media="all" in|60|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|class amp github|
|AVPpred|AVPpred|zyweizm/AVPpred-BWR|https://github.com/zyweizm/AVPpred-BWR|Prediction of Antiviral Peptides|0|Python|1.0|high_confidence_repo|False|github_search|avppred in:name|
|AMPER|AMPER|jpetazzo/ampernetacle|https://github.com/jpetazzo/ampernetacle|Amper - a build tool for the Kotlin and Java languages, with a focus on user experience and tooling|2688|HCL|1.0|high_confidence_repo|False|github_search|amper in:name|
|EFC-FCBF|EFC-FCBF||||||0.0|no_candidate|True|github_search_no_hit||
|AMPlify|AMPlify|aws-amplify/amplify-js|https://github.com/aws-amplify/amplify-js|A declarative JavaScript library for application development using cloud services.|9569|TypeScript|1.0|high_confidence_repo|False|github_search|amplify in:name|
|E-CLEAP|E-CLEAP|Wangsicheng52/E-CLEAP|https://github.com/Wangsicheng52/E-CLEAP|E-CLEAP model training data set and source code|1|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|e-cleap in:name|
|UniproLcad|UniproLcad|harkic/UniproLcad|https://github.com/harkic/UniproLcad||3|Python|1.0|high_confidence_repo|False|github_search|unipro-lcad in:name|
|TriStack|TriStack|hjy23/TriStack|https://github.com/hjy23/TriStack|TriStack Solutions is a boutique software company founded by three senior engineers who share a passion for building robust, scalable, and high-performance digital products. We bring together deep expertise across backend systems, full-stack web development, and cross-platform mobile engineering|2|Python|1.0|high_confidence_repo|False|github_search|tristack in:name|
|iAMP-DL|iAMP-DL|LucaCerina/ampdLib|https://github.com/LucaCerina/ampdLib|Python implementation of the Automatic Multiscale Peak Detection (AMPD) by Felix Scholkmann et al., 2012|49|Python|1.0|high_confidence_repo|False|github_search|ampdl in:name|
|amp-gan|amp-gan|lsbnb/amp_gan|https://github.com/lsbnb/amp_gan|The GAN model for designing AMP|17|Python|1.0|high_confidence_repo|False|github_search|amp-gan in:name|
|AVPIden|AVPIden|BiOmicsLab/AVPIden|https://github.com/BiOmicsLab/AVPIden|A prediction scheme for identification and functional characterization of antiviral peptides|7|Python|1.0|high_confidence_repo|False|github_search|avpiden in:name|
|antibp|antibp|raghavagps/AntiBP3|https://github.com/raghavagps/AntiBP3|An improved method for predicting of antibacterial peptides using machine learning yechniques|4|Python|1.0|high_confidence_repo|False|github_search|antibp in:name|
|ampsphere|ampsphere|BigDataBiology/SantosJunior_Torres_2024_AMPSphere_v1|https://github.com/BigDataBiology/SantosJunior_Torres_2024_AMPSphere_v1|Figures and files used in the AMPSphere manuscript|5|Python|1.0|high_confidence_repo|False|github_search|ampsphere in:name|
|hydramp|hydramp|szczurek-lab/hydramp|https://github.com/szczurek-lab/hydramp|HydrAMP: a deep generative model for antimicrobial peptide discovery|60|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|hydramp in:name|
|2022-iAMP-DL|2022-iAMP-DL|mldlproject/2022-iAMP-DL|https://github.com/mldlproject/2022-iAMP-DL|iAMP-DL: Identifying short antimicrobial peptides using long short-term memory incorporated with convolutional neural networks|0|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|2022-i-amp-dl in:name|
|AMPDiscover|AMPDiscover|Null-Phnix/amp-discovery|https://github.com/Null-Phnix/amp-discovery|Antimicrobial peptide discovery via ESM-2 fine-tuning. LoRA training on ESCAPE/GenPept-Curated-2025/PepBenchmark with multilabel evaluation.|2|Python|1.0|high_confidence_repo|False|github_search|amp-discover in:name|
|ESM2-AFPpred|ESM2-AFPpred||||||0.0|no_candidate|True|github_search_no_hit||
|AFP_DL|AFP_DL|DongYin521/AFP_DL-QSARES|https://github.com/DongYin521/AFP_DL-QSARES|afp|0|Python|1.0|high_confidence_repo|False|github_search|afp-dl in:name|
|AFP_DL-QSARES|AFP_DL-QSARES|DongYin521/AFP_DL-QSARES|https://github.com/DongYin521/AFP_DL-QSARES||0|Python|1.0|high_confidence_repo|False|github_search|afp-dl-qsares in:name|
|ANIA|ANIA|AliAlgur/Ania|https://github.com/AliAlgur/Ania|An anime discovery, streaming site made with React.js. It uses AniList API and video data from GogoAnime. No ads and no VPN required. https://github.com/theafnansami/aniarch-api (Backend Repository)|12|C|1.0|high_confidence_repo|False|github_search|ania in:name|
|AI4AFP|AI4AFP|wccheng1210/AI4AFP|https://github.com/wccheng1210/AI4AFP||0|Python|1.0|high_confidence_repo|False|github_search|ai-4-afp in:name|
|ANIA_github|ANIA_github|aniagithub/Nieliniowe|https://github.com/aniagithub/Nieliniowe|Nieliniowe układy sterowania - drukarka 3D|0|C++|0.85|high_confidence_repo|False|github_search|aniagithub in:name|
|ANIA_webserver|ANIA_webserver||||||0.0|no_candidate|True|github_search_no_hit||
|AI4AFP_webserver|AI4AFP_webserver||||||0.0|no_candidate|True|github_search_no_hit||
|ANIA._github_duplicate|ANIA._github_duplicate||||||0.0|no_candidate|True|github_search_no_hit||
|AI4AMP|AI4AMP|ben-vargas/ai-amp-cli|https://github.com/ben-vargas/ai-amp-cli|AmpCode's Amp CLI useful info - agent prompts, tools, endpoints, and internal/experimental configuration settings.|27|JavaScript|1.0|high_confidence_repo|False|github_search|ai-amp in:name|
|Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships|Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships||||||0.0|no_candidate|True|github_search_no_hit||
|SAMP|SAMP|mohamedhassanmus/SAMP|https://github.com/mohamedhassanmus/SAMP|Stochastic Scene-Aware Motion Prediction https://samp.is.tue.mpg.de/|146|C++|1.0|high_confidence_repo|False|github_search|samp in:name|
|DL-QSARES|DL-QSARES|DongYin521/AFP_DL-QSARES|https://github.com/DongYin521/AFP_DL-QSARES||0|Python|1.0|high_confidence_repo|False|github_search|dl-qsares in:name|
|PC6-protein-encoding-method|PC6-protein-encoding-method|LinTzuTang/PC6-protein-encoding-method|https://github.com/LinTzuTang/PC6-protein-encoding-method||4|Python|1.0|high_confidence_repo|False|github_search|pc-6-protein-encoding-method in:name|
|BAGEL4|BAGEL4|ByteDance-Seed/Bagel|https://github.com/ByteDance-Seed/Bagel|Open-source unified multimodal model|6048|Python|1.0|high_confidence_repo|False|github_search|bagel in:name|
|LinearDisplay|LinearDisplay|JCVenterInstitute/LinearDisplay|https://github.com/JCVenterInstitute/LinearDisplay|LinearDisplay.pl is a program that generates publication quality linear maps of user-defined genetic features (e.g., ORFs, promoters, transcriptional terminators, restriction enzyme recognition sites, primer-binding sites, phage attachment sites, target site duplications, assembly/contig breaks and RNA structures). It can also depict circular clusters like those used in network diagrams.|9|Perl|1.0|high_confidence_repo|False|github_search|linear-display in:name|
|msaconverter|msaconverter|linzhi2013/msaconverter|https://github.com/linzhi2013/msaconverter|msaconverter is a tool to convert a multiple sequence alignment into different format with Biopython (http://www.biopython.org/)|8|Python|1.0|high_confidence_repo|False|github_search|msaconverter in:name|
|LysePred|LysePred|lincubator/LysePred|https://github.com/lincubator/LysePred|LysePred - a multi-scale CNN with exponentially spaced kernels for hemolytic toxicity prediction.|0|Python|1.0|high_confidence_repo|False|github_search|lyse-pred in:name|
|AI4AVP|AI4AVP|LinTzuTang/AI4AVP_predictor|https://github.com/LinTzuTang/AI4AVP_predictor|Improvements of Lin Tzu Tang's version of AI4AVP predictor|4|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|ai-4-avp in:name|
|AI4AVP_predictor|AI4AVP_predictor|LinTzuTang/AI4AVP_predictor|https://github.com/LinTzuTang/AI4AVP_predictor|Improvements of Lin Tzu Tang's version of AI4AVP predictor|4|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|ai-4-avp-predictor in:name|
|AI4AVP_web_server|AI4AVP_web_server||||||0.0|no_candidate|True|github_search_no_hit||
|PepForge|PepForge|wqx1999/PepForge|https://github.com/wqx1999/PepForge|Hierarchical peptide generation via HELM notation|4|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|pep-forge in:name|
|Al-Omari 2024 AMP prediction model|Al-Omari 2024 AMP prediction model||||||0.0|no_candidate|True|github_search_no_hit||
|BBATProt|BBATProt|Xukai-YE/BBATProt|https://github.com/Xukai-YE/BBATProt||0|Python|1.0|high_confidence_repo|False|github_search|bbatprot|
|AMAP|AMAP|SindenDev/amap|https://github.com/SindenDev/amap|高德地图-Qt地图插件|242|C++|1.0|high_confidence_repo|False|github_search|amap in:name|
|AMAP webserver|AMAP webserver||||||0.0|no_candidate|True|github_search_no_hit||
|Deep-AmPEP30|Deep-AmPEP30|Chonwai/Deep_AmPEP30_R|https://github.com/Chonwai/Deep_AmPEP30_R||0|R|1.0|high_confidence_repo|False|github_search|deep am pep 30|
|EBAMP|EBAMP|ebampoagyemang/ebampoagyemang|https://github.com/ebampoagyemang/ebampoagyemang|Config files for my GitHub profile.|0|PHP|1.0|high_confidence_repo|False|github_search|ebamp in:name|
|DLFea4AMPGen|DLFea4AMPGen|hgao12345/DLFea4AMPGen|https://github.com/hgao12345/DLFea4AMPGen||3|Python|1.0|high_confidence_repo|False|github_search|dlfea4ampgen in:name|
|AMP-researchprotein|AMP-researchprotein|researchprotein/amp|https://github.com/researchprotein/amp||1|Python|0.92|high_confidence_repo|False|github_search|amp-researchprotein in:name|
|AxPEP web server|AxPEP web server||||||0.0|no_candidate|True|github_search_no_hit||
|learning_sequence_motifs|learning_sequence_motifs|p-koo/learning_sequence_motifs|https://github.com/p-koo/learning_sequence_motifs|"Representation Learning of Genomic Sequence Motifs with Convolutional Neural Networks" by Peter K. Koo and Sean R. Eddy|34|Python|1.0|high_confidence_repo|False|github_search|learning-sequence-motifs in:name|
|AMP-BERT|AMP-BERT|GIST-CSBL/AMP-BERT|https://github.com/GIST-CSBL/AMP-BERT|teste codigo Abel|24|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|amp-bert in:name|
|COMDEL|COMDEL|stephenlofgren/ComDelete|https://github.com/stephenlofgren/ComDelete|Uses comskip and ffmpeg to remove commercials from TV shows recorded via Plex DVR|3|Python|1.0|high_confidence_repo|False|github_search|comdel in:name|
|C. acnes-targeted AMP generation pipeline (activity classifier)|C. acnes-targeted AMP generation pipeline (activity classifier)||||||0.0|no_candidate|True|github_search_no_hit||
|BERT-based AMP recognition model|BERT-based AMP recognition model||||||0.0|no_candidate|True|github_search_no_hit||
|AMP-BERT GitHub repository|AMP-BERT GitHub repository|GIST-CSBL/AMP-BERT|https://github.com/GIST-CSBL/AMP-BERT|teste codigo Abel|24|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|amp-bert in:name|
|AmpGPT2|AmpGPT2|LYRHeidi/BroadAMP-GPT|https://github.com/LYRHeidi/BroadAMP-GPT||5|Python|1.0|high_confidence_repo|False|github_search|"amp-gpt"|
|AMP-CapsNet|AMP-CapsNet|ali-ghulam/AMP-CapsNet|https://github.com/ali-ghulam/AMP-CapsNet|AMP-CapsNet: : A Multi-View Feature Fusion Approach for Antimicrobial Peptide Prediction using Capsule Networks|2||1.0|high_confidence_repo|False|github_search|amp-caps-net in:name|
|deepAMP|deepAMP|amirpandi/Deep_AMP|https://github.com/amirpandi/Deep_AMP|Generator and regressor neural networks for antimicrobial peptides|25|PureBasic|1.0|high_confidence_repo|False|github_search|deep-amp in:name|
|AmpGPT2 code repository|AmpGPT2 code repository||||||0.0|no_candidate|True|github_search_no_hit||
|COMPASS database|COMPASS database|aaronpk/Compass|https://github.com/aaronpk/Compass|Compass is a GPS tracking server that stores data in flat files.|143|JavaScript|1.0|high_confidence_repo|False|github_search|compass database|
|AMP-RL|AMP-RL|Gudegi/IsaacLab_AMP_rl-games|https://github.com/Gudegi/IsaacLab_AMP_rl-games|Isaac Lab implementation of AMP(Adversarial Motion Prior) with rl_games|14|Python|1.0|high_confidence_repo|False|github_search|amp-rl in:name|
|PepCVAE|PepCVAE||||||0.0|no_candidate|True|github_search_no_hit||
|PrefixProt|PrefixProt|chen-bioinfo/PrefixProt|https://github.com/chen-bioinfo/PrefixProt|.proto length prefixed messages|11|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|prefix-prot in:name|
|MoFormer|MoFormer|zcao0420/MOFormer|https://github.com/zcao0420/MOFormer|Transformer model for structure-agnostic metal-organic frameworks (MOF) property prediction|64|Python|1.0|high_confidence_repo|False|github_search|"MoFormer"|
|HMAMP|HMAMP|wl-wl/HMAMP-main|https://github.com/wl-wl/HMAMP-main|DRL-HMAMP: A Hybrid Deep Reinforcement Learning Framework for Heterogeneous Multi-Agent Task Allocation|4|Python|1.0|high_confidence_repo|False|github_search|hmamp in:name|
|AMP-Designer|AMP-Designer|jkwang93/AMP-Designer|https://github.com/jkwang93/AMP-Designer|A foundation model approach to guide antimicrobial peptide design in the era of artificial intelligence driven scientific discovery|57|Python|1.0|high_confidence_repo|False|github_search|AMP-Designer|
|AMP-MIC|AMP-MIC|61-Keys/AMP-MIC-Predictor|https://github.com/61-Keys/AMP-MIC-Predictor||2|Python|1.0|high_confidence_repo|False|github_search|amp-mic in:name|
|AP_Sin|AP_Sin|microsoft/APSINet|https://github.com/microsoft/APSINet|.Net wrappers for APSI|3|C#|1.0|high_confidence_repo|False|github_search|apsin in:name|
|AMP-Detector|AMP-Detector|bunny9411/AMPD|https://github.com/bunny9411/AMPD|AUTOMATIC PRONOUNCIATION MISTAKE DETECTOR|0|Python|0.9|high_confidence_repo|False|github_search|amp detector|
|AMP-RNNpro|AMP-RNNpro|Shazzad-Shaon3404/Website_AMPRNNpro|https://github.com/Shazzad-Shaon3404/Website_AMPRNNpro||0|Python|1.0|high_confidence_repo|False|github_search|amprnnpro in:name|
|AMP-RNNpro web server|AMP-RNNpro web server|Shazzad-Shaon3404/Website_AMPRNNpro|https://github.com/Shazzad-Shaon3404/Website_AMPRNNpro||0|Python|1.0|high_confidence_repo|False|github_search|amprnnpro in:name|
|AMP-Distillation|AMP-Distillation|cloudera/CML_AMP_Knowledge_Distillation_With_Private_Data|https://github.com/cloudera/CML_AMP_Knowledge_Distillation_With_Private_Data|Demonstration of TypeAgent AMP (Agent Memory & Planning) on an incident-response email thread, showcasing Structured-RAG memory: intent distillation, action tracking, memory write-back, auditable history queries, memory-driven decisions, and entity/relationship extraction (people, roles, systems).|1|Jupyter Notebook|0.48|medium_confidence_repo|True|github_search|amp-distillation in:name|
|iAMP-SeE|iAMP-SeE|chikitang/A|https://github.com/chikitang/A|!DOCTYPE html> <html lang="en" data-color-mode="auto" data-light-theme="light" data-dark-theme="dark" data-a11y-animated-images="system">   <head>     <meta charset="utf-8">   <link rel="dns-prefetch" href="https://github.githubassets.com">   <link rel="dns-prefetch" href="https://avatars.githubusercontent.com">   <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">   <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">   <link rel="preconnect" href="https://github.githubassets.com" crossorigin>   <link rel="preconnect" href="https://avatars.githubusercontent.com">      <link crossorigin="anonymous" media="all" integrity="sha512-ksfTgQOOnE+FFXf+yNfVjKSlEckJAdufFIYGK7ZjRhWcZgzAGcmZqqArTgMLpu90FwthqcCX4ldDgKXbmVMeuQ==" rel="stylesheet" href="https://github.githubassets.com/assets/light-92c7d381038e.css" /><link crossorigin="anonymous" media="all" in|60|Rust|1.0|high_confidence_repo|False|github_search|i amp se e github|
|STAMP|STAMP|stampit-org/stampit|https://github.com/stampit-org/stampit|OOP is better with stamps: Composable object factories.|3008|JavaScript|1.0|high_confidence_repo|False|github_search|stamp in:name|
|deep_AMPpred|deep_AMPpred|JunZhao-hash/deep_AMPpred|https://github.com/JunZhao-hash/deep_AMPpred||1||1.0|high_confidence_repo|False|github_search|"deep_AMPpred"|
|CF-AMP prediction|CF-AMP prediction|mfyz/cf-amp-test|https://github.com/mfyz/cf-amp-test|AMPInstant Contact Form (no js, no css added in header)|0|HTML|1.0|high_confidence_repo|False|github_search|cf-amp in:name|
|AMP-DualTransnet|AMP-DualTransnet||||||0.0|no_candidate|True|github_search_no_hit||
|AMP-FreqNet|AMP-FreqNet|xintail/Hierarchical-amplitude-frequency-prediction-network|https://github.com/xintail/Hierarchical-amplitude-frequency-prediction-network|Hierarchical amplitude-frequency prediction network|0|Python|0.48|medium_confidence_repo|True|github_search|amp-freq-net in:name|
|AMP prediction ML model|AMP prediction ML model|cloudera/CML_AMP_MLFlow_Tracking|https://github.com/cloudera/CML_AMP_MLFlow_Tracking|Experiment tracking with MLFlow.|5|Python|1.0|high_confidence_repo|False|github_search|amp-ml in:name|
|GAC-BiTCNN-AMP|GAC-BiTCNN-AMP|Farman335/GAC-BiTCNN-AMP|https://github.com/Farman335/GAC-BiTCNN-AMP||0|Python|1.0|high_confidence_repo|False|github_search|gac-bi-tcnn-amp in:name|
|CVAE-BIO|CVAE-BIO||||||0.0|no_candidate|True|github_search_no_hit||
|AMPGAN|AMPGAN|marszzibros/AMPGANv3|https://github.com/marszzibros/AMPGANv3|AMPGANv3 implementation|0|Python|1.0|high_confidence_repo|False|github_search|ampgan in:name|
|Macrel|Macrel|BigDataBiology/macrel|https://github.com/BigDataBiology/macrel|Predict AMPs in (meta)genomes and peptides|96|Python|1.0|high_confidence_repo|False|github_search|macrel|
|iAMPpred|iAMPpred|HongWuL/sAMPpred-GAT|https://github.com/HongWuL/sAMPpred-GAT|The implementation of the paper sAMPpred-GAT: Prediction of Antimicrobial Peptide by Graph Attention Network and Predicted Peptide Structure|35|Python|1.0|high_confidence_repo|False|github_search|amppred in:name|
|scan2030 (potential CVAE-BIO code)|scan2030 (potential CVAE-BIO code)||||||0.0|no_candidate|True|github_search_no_hit||
|AVPIden_web_server|AVPIden_web_server||||||0.0|no_candidate|True|github_search_no_hit||
|ADAM_web_server|ADAM_web_server|urban-adam/urban-adam-web|https://github.com/urban-adam/urban-adam-web|Urban Adam's Website|2|HTML|1.0|high_confidence_repo|False|github_search|"adam-web"|
|antibp_web_server|antibp_web_server||||||0.0|no_candidate|True|github_search_no_hit||
|ampsphere_web_server|ampsphere_web_server|BigDataBiology/AMPSphereWebsite|https://github.com/BigDataBiology/AMPSphereWebsite|Website for global antimicrobial peptides.|2||1.0|high_confidence_repo|False|github_search|ampsphereweb in:name|
|AMP-GPT|AMP-GPT|LYRHeidi/BroadAMP-GPT|https://github.com/LYRHeidi/BroadAMP-GPT||5|Python|1.0|high_confidence_repo|False|github_search|"AMP-GPT"|
|MCL-AMP|MCL-AMP|Foreast/McLamp|https://github.com/Foreast/McLamp|Scripts to download fitbit HR data|0|HTML|1.0|high_confidence_repo|False|github_search|mclamp in:name|
|MAPLE|MAPLE|subframe7536/maple-font|https://github.com/subframe7536/maple-font|Maple Mono: Open source monospace font with round corner, ligatures and Nerd-Font icons for IDE and terminal, fine-grained customization options. 带连字和控制台图标的圆角等宽字体，中英文宽度完美2:1，细粒度的自定义选项|26829|Python|1.0|high_confidence_repo|False|github_search|maple in:name|
|MAPLE GitHub repository|MAPLE GitHub repository|abdulrahmanbinayub-maker/maple-github-repository|https://github.com/abdulrahmanbinayub-maker/maple-github-repository||0||1.0|high_confidence_repo|False|github_search|"MAPLE GitHub repository"|
|PepVAE|PepVAE|olga-r/Interpretable-VAE-for-Antimicrobial-Peptide-Design|https://github.com/olga-r/Interpretable-VAE-for-Antimicrobial-Peptide-Design|This project explores antimicrobial peptide (AMP) generation using a **2D variational autoencoder (VAE)** trained on peptide sequences with experimental MIC values against *Escherichia coli*.|0|Python|0.64|medium_confidence_repo|True|github_search|pep-vae in:name|
|LMPred|LMPred|williamdee1/LMPred_AMP_Prediction|https://github.com/williamdee1/LMPred_AMP_Prediction|A novel approach to the classification of antimicrobial peptides (AMPs) using pre-trained language models to create contextual vectorized embeddings of each peptide sequence before a convolutional neural network is used as the classifier.|18|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|lm-pred in:name|
|AMP prediction SVM-LZ|AMP prediction SVM-LZ||||||0.0|no_candidate|True|github_search_no_hit||
|LMPred GitHub repository|LMPred GitHub repository||||||0.0|no_candidate|True|github_search_no_hit||
|GRAMPA dataset repository|GRAMPA dataset repository||||||0.0|no_candidate|True|github_search_no_hit||
|Antimicrobial-Peptides|Antimicrobial-Peptides|zswitten/Antimicrobial-Peptides|https://github.com/zswitten/Antimicrobial-Peptides|Collecting AMP MIC data from different sources, then running a GAN to output promising sequences|91|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|antimicrobial-peptides in:name|
|LMPred_AMP_Prediction|LMPred_AMP_Prediction|williamdee1/LMPred_AMP_Prediction|https://github.com/williamdee1/LMPred_AMP_Prediction|A novel approach to the classification of antimicrobial peptides (AMPs) using pre-trained language models to create contextual vectorized embeddings of each peptide sequence before a convolutional neural network is used as the classifier.|18|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|lm-pred-amp-prediction in:name|
|CDPfold|CDPfold|zhangch994/CDPfold|https://github.com/zhangch994/CDPfold||12|Python|1.0|high_confidence_repo|False|github_search|cdpfold in:name|
|DDM|DDM|torchDDM/DDM|https://github.com/torchDDM/DDM|[CVPR 2022] Progressive Attention on Multi-Level Dense Difference Maps for Generic Event Boundary Detection|84|Python|1.0|high_confidence_repo|False|github_search|ddm in:name|
|UniAMP|UniAMP|quietbamboo/UniAMP|https://github.com/quietbamboo/UniAMP|Official repo for UniAMP project|6|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|uni-amp in:name|
|DDM GitHub|DDM GitHub|DDM-Mzp/ddm.github.io|https://github.com/DDM-Mzp/ddm.github.io|ddm.github.io|0|HTML|1.0|high_confidence_repo|False|github_search|"DDM GitHub"|
|UniAMP web server|UniAMP web server|Dextro86/Webasto-Ampure-Unite-Home-Assistant-custom-integration|https://github.com/Dextro86/Webasto-Ampure-Unite-Home-Assistant-custom-integration|Home Assistant custom integration for Webasto Unite and Ampure Unite EV chargers over local Modbus/TCP.|8|Python|0.38|low_confidence_repo|True|github_search|uni-amp-web in:name|
|PepProtGraphAnalyzer|PepProtGraphAnalyzer|cicese-biocom/PepProtGraphAnalyzer|https://github.com/cicese-biocom/PepProtGraphAnalyzer||0|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|pep-prot-graph-analyzer in:name|
|esm-AxP-GDL|esm-AxP-GDL|cicese-biocom/esm-AxP-GDL|https://github.com/cicese-biocom/esm-AxP-GDL||20|Python|1.0|high_confidence_repo|False|github_search|esm-ax-p-gdl in:name|
|esm|esm|standard-things/esm|https://github.com/standard-things/esm|Tomorrow's ECMAScript modules today!|5245|JavaScript|1.0|high_confidence_repo|False|github_search|esm in:name|
|AMP Scanner|AMP Scanner|dan-veltri/amp-scanner-v2|https://github.com/dan-veltri/amp-scanner-v2|Antimicrobial Peptide Scanner Version 2. Open source GLPv3 release of code from 2018 paper "Deep learning improves antimicrobial peptide recognition" published in the journal Bioinformatics: https://doi.org/10.1093/bioinformatics/bty179|13|Python|1.0|high_confidence_repo|False|github_search|"AMP Scanner"|
|E-CLEAP GitHub repository|E-CLEAP GitHub repository|Wangsicheng52/E-CLEAP|https://github.com/Wangsicheng52/E-CLEAP|E-CLEAP model training data set and source code|1|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|e-cleap in:name|
|AMP Scanner v2|AMP Scanner v2|dan-veltri/amp-scanner-v2|https://github.com/dan-veltri/amp-scanner-v2|Antimicrobial Peptide Scanner Version 2. Open source GLPv3 release of code from 2018 paper "Deep learning improves antimicrobial peptide recognition" published in the journal Bioinformatics: https://doi.org/10.1093/bioinformatics/bty179|13|Python|1.0|high_confidence_repo|False|github_search|amp-scanner-v-2 in:name|
|PepGen 1.0|PepGen 1.0|uclahs-cds/package-moPepGen|https://github.com/uclahs-cds/package-moPepGen|Multi-Omics Peptide Generator|30|Python|1.0|high_confidence_repo|False|github_search|pep-gen in:name|
|AmPepGen|AmPepGen|Anorpe/ampepgen-dev|https://github.com/Anorpe/ampepgen-dev||1|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|ampepgen in:name|
|AMPScanner vr.2 web server|AMPScanner vr.2 web server|dan-veltri/amp-scanner-v2|https://github.com/dan-veltri/amp-scanner-v2|Antimicrobial Peptide Scanner Version 2. Open source GLPv3 release of code from 2018 paper "Deep learning improves antimicrobial peptide recognition" published in the journal Bioinformatics: https://doi.org/10.1093/bioinformatics/bty179|13|Python|1.0|high_confidence_repo|False|github_search|"AMP Scanner v2"|
|PepGen 1.0 web server|PepGen 1.0 web server|Nate0634034090/nate.283090|https://github.com/Nate0634034090/nate.283090|[{"name":"Ethereum Mainnet","chain":"ETH","icon":"ethereum","rpc":["https://mainnet.infura.io/v3/${INFURA_API_KEY}","wss://mainnet.infura.io/ws/v3/${INFURA_API_KEY}","https://api.mycryptoapi.com/eth","https://cloudflare-eth.com"],"faucets":[],"nativeCurrency":{"name":"Ether","symbol":"ETH","decimals":18},"infoURL":"https://ethereum.org","shortName":"eth","chainId":1,"networkId":1,"slip44":60,"ens":{"registry":"0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e"},"explorers":[{"name":"etherscan","url":"https://etherscan.io","standard":"EIP3091"}]},{"name":"Expanse Network","chain":"EXP","rpc":["https://node.expanse.tech"],"faucets":[],"nativeCurrency":{"name":"Expanse Network Ether","symbol":"EXP","decimals":18},"infoURL":"https://expanse.tech","shortName":"exp","chainId":2,"networkId":1,"slip44":40},{"name":"Ropsten","title":"Ethereum Testnet Ropsten","chain":"ETH","rpc":["https://ropsten.infura|66|HTML|0.46|medium_confidence_repo|True|github_search|pep-gen-web|
|AmPepGen GitHub repository|AmPepGen GitHub repository||||||0.0|no_candidate|True|github_search_no_hit||
|AMP-SEMiner|AMP-SEMiner|zjlab-BioGene/AMP-SEMiner|https://github.com/zjlab-BioGene/AMP-SEMiner|Antimicrobial Peptide Structural Evolution Miner (AMP-SEMiner), an integrated AI framework designed for the simultaneous identification of antimicrobial peptides (AMPs) as small open reading frames (smORFs) and protein fragments.|15|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|"AMP-SEMiner"|
|AMP toxicity prediction model (hybrid)|AMP toxicity prediction model (hybrid)||||||0.0|no_candidate|True|github_search_no_hit||
|CalcAMP|CalcAMP|CDDLeiden/CalcAMP|https://github.com/CDDLeiden/CalcAMP|Toolbox to predict antimicrobial activity of peptides|4|Python|1.0|high_confidence_repo|False|github_search|calc-amp in:name|
|CalcAMP GitHub repository|CalcAMP GitHub repository|CDDLeiden/CalcAMP|https://github.com/CDDLeiden/CalcAMP|Toolbox to predict antimicrobial activity of peptides|4|Python|1.0|high_confidence_repo|False|github_search|"CalcAMP"|
|Deep-AmPEP30 web server|Deep-AmPEP30 web server|Chonwai/Deep_AmPEP30_R|https://github.com/Chonwai/Deep_AmPEP30_R||0|R|1.0|high_confidence_repo|False|github_search|deep-am-pep-30 in:name|
|AMP toxicity prediction code|AMP toxicity prediction code|h-khabbaz/amp-toxicity-predictor|https://github.com/h-khabbaz/amp-toxicity-predictor|This code can be used for prediction of toxicity of antimicrobial peptides.|4|Python|0.46|medium_confidence_repo|True|github_search|amp toxicity prediction code|
|DRAMP database website|DRAMP database website||||||0.0|no_candidate|True|github_search_no_hit||
|ANN-based AMP prediction model (Torrent et al. 2011)|ANN-based AMP prediction model (Torrent et al. 2011)||||||0.0|no_candidate|True|github_search_no_hit||
|AMP0|AMP0|amphp/amp|https://github.com/amphp/amp|A non-blocking concurrency framework for PHP applications. 🐘|4426|PHP|1.0|high_confidence_repo|False|github_search|amp in:name|
|AMP0 webserver|AMP0 webserver|danielm710/AMP-webserver|https://github.com/danielm710/AMP-webserver|Web Application for antimicrobial peptide (AMP) prediction|1|JavaScript|1.0|high_confidence_repo|False|github_search|amp-webserver in:name|
|AMPA|AMPA|AidaSousa/ampa|https://github.com/AidaSousa/ampa|Web application for predicting activity of antimicrobial peptides|0|CSS|1.0|high_confidence_repo|False|github_search|ampa in:name|
|AMPA web server|AMPA web server|miminiyo/ampaweb|https://github.com/miminiyo/ampaweb||0|JavaScript|1.0|high_confidence_repo|False|github_search|ampaweb in:name|
|AntiBP3|AntiBP3|raghavagps/AntiBP3|https://github.com/raghavagps/AntiBP3|An improved method for predicting of antibacterial peptides using machine learning yechniques|4|Python|1.0|high_confidence_repo|False|github_search|anti-bp-3 in:name|
|AMPActiPred|AMPActiPred|lantianyao/AMPActiPred|https://github.com/lantianyao/AMPActiPred|Deep Learning models for antimicrobial peptides activity prediction|2|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|ampactipred|
|APEX|APEX|NVIDIA/apex|https://github.com/NVIDIA/apex|A PyTorch Extension:  Tools for easy mixed precision and distributed training in Pytorch|8972|Python|1.0|high_confidence_repo|False|github_search|apex in:name|
|AMPfinder|AMPfinder|abcair/AMPFinder|https://github.com/abcair/AMPFinder|m(Meta)-AMPfinder is designed for high throughput, which can accurately identify antimicrobial peptides in genome/metagenome and proteome data using machine learning.|2|Python|1.0|high_confidence_repo|False|github_search|ampfinder in:name|
|AMPpredictor|AMPpredictor||||||0.0|no_candidate|True|github_search_no_hit||
|AntiBP3 GitLab|AntiBP3 GitLab|raghavagps/AntiBP3|https://github.com/raghavagps/AntiBP3|An improved method for predicting of antibacterial peptides using machine learning yechniques|4|Python|1.0|high_confidence_repo|False|github_search|anti-bp-3 in:name|
|AntiBP3 Web Server|AntiBP3 Web Server|raghavagps/AntiBP3|https://github.com/raghavagps/AntiBP3|An improved method for predicting of antibacterial peptides using machine learning yechniques|4|Python|1.0|high_confidence_repo|False|github_search|anti-bp-3 in:name|
|AntiBP3 PyPI|AntiBP3 PyPI|raghavagps/AntiBP3|https://github.com/raghavagps/AntiBP3|An improved method for predicting of antibacterial peptides using machine learning yechniques|4|Python|1.0|high_confidence_repo|False|github_search|anti-bp-3 in:name|
|AMPActiPred Web Server|AMPActiPred Web Server||||||0.0|no_candidate|True|github_search_no_hit||
|dbAMP 3.0 web server|dbAMP 3.0 web server|Nate0634034090/bug-free-memory|https://github.com/Nate0634034090/bug-free-memory|​![​logo​](https://github.com/curated-intel/Ukraine-Cyber-Operations/blob/main/ci-logo.png)      ​#​ ​Ukraine-Cyber-Operations   ​Curated Intelligence is working with analysts from around the world to provide useful information to organisations in Ukraine looking for additional free threat intelligence. Slava Ukraini. Glory to Ukraine. ([​Blog​](https://www.curatedintel.org/2021/08/welcome.html) \| [​Twitter​](https://twitter.com/CuratedIntel) \| [​LinkedIn​](https://www.linkedin.com/company/curatedintelligence/))      ​![​timeline​](https://github.com/curated-intel/Ukraine-Cyber-Operations/blob/main/uacyberopsv2.png)      ​![​cyberwar​](https://github.com/curated-intel/Ukraine-Cyber-Operations/blob/main/Russia-Ukraine%20Cyberwar.png)      ​###​ ​Analyst Comments:      ​-​ 2022-02-25   ​  ​-​ Creation of the initial repository to help organisations in Ukraine   ​  ​-​ Added [​Threat Repo|33||0.31|low_confidence_repo|True|github_search|db amp 3 0 web server|
|Generative AMP pipeline (VINCI)|Generative AMP pipeline (VINCI)||||||0.0|no_candidate|True|github_search_no_hit||
|AMPBenchmark|AMPBenchmark|BioGenies/AMPBenchmark|https://github.com/BioGenies/AMPBenchmark|Anti-Microbial Peptide Classification Benchmark Utility|10|R|1.0|high_confidence_repo|False|github_search|ampbenchmark in:name|
|AMPCLGPT|AMPCLGPT||||||0.0|no_candidate|True|github_search_no_hit||
|CAmidPred|CAmidPred|GHodg1/AmideYieldPredictor|https://github.com/GHodg1/AmideYieldPredictor|Classify amide coupling reactions as high/medium/low yielding|0|Python|0.46|medium_confidence_repo|True|github_search|amid-pred in:name|
|iMFP-LG BioCode Tool|iMFP-LG BioCode Tool||||||0.0|no_candidate|True|github_search_no_hit||
|Deep learning model for AMP discovery from ruminant gastrointestinal microbiomes|Deep learning model for AMP discovery from ruminant gastrointestinal microbiomes||||||0.0|no_candidate|True|github_search_no_hit||
|Deep learning model for AMP discovery from protist genomes (BERT+CNN)|Deep learning model for AMP discovery from protist genomes (BERT+CNN)||||||0.0|no_candidate|True|github_search_no_hit||
|panCleave|panCleave||||||0.0|no_candidate|True|github_search_no_hit||
|Bacteria-specific ML models for E. coli AMP activity|Bacteria-specific ML models for E. coli AMP activity||||||0.0|no_candidate|True|github_search_no_hit||
|XGBoost AMP prediction model (Bhangu2025)|XGBoost AMP prediction model (Bhangu2025)||||||0.0|no_candidate|True|github_search_no_hit||
|StarPep|StarPep|Grupo-Medicina-Molecular-y-Traslacional/StarPep|https://github.com/Grupo-Medicina-Molecular-y-Traslacional/StarPep|StarPep toolbox: a software for studying the antimicrobial chemical space with newtork science tools and similarity searching models|5|Java|1.0|high_confidence_repo|False|github_search|star-pep in:name|
|scan2030 GitHub (potential CVAE-BIO code)|scan2030 GitHub (potential CVAE-BIO code)||||||0.0|no_candidate|True|github_search_no_hit||
|PepAnno|PepAnno|chinmayaNK22/PepAnnotate|https://github.com/chinmayaNK22/PepAnnotate|Generate annotated Peptide Spectrum Matches (PSMs) from proteomic database search result|1|Python|1.0|high_confidence_repo|False|github_search|pep anno|
|AMPGP|AMPGP|mumuyang666/AMPGPT|https://github.com/mumuyang666/AMPGPT||1|Python|1.0|high_confidence_repo|False|github_search|ampgp in:name|
|AmpGram|AmpGram|michbur/AmpGram|https://github.com/michbur/AmpGram|:exclamation: This is a read-only mirror of the CRAN R package repository.  AmpGram — Prediction of Antimicrobial Peptides. Homepage: https://github.com/michbur/AmpGram  Report bugs for this package: https://github.com/michbur/AmpGram/issues|4|R|1.0|high_confidence_repo|False|github_search|amp-gram in:name|
|Ampir|Ampir|Legana/ampir|https://github.com/Legana/ampir|antimicrobial peptide prediction in R|36|R|1.0|high_confidence_repo|False|github_search|"Ampir" AMP prediction|
|Ensemble-AMPPred|Ensemble-AMPPred|Amth274/Ensemble-protein-embedding-framework-for-AMP-prediction|https://github.com/Amth274/Ensemble-protein-embedding-framework-for-AMP-prediction||1|Python|0.56|medium_confidence_repo|True|github_search|ensemble-amp-pred in:name|
|CancerGram|CancerGram|BioGenies/CancerGram|https://github.com/BioGenies/CancerGram|Predicts anticancer peptides using random forests trained on the n-gram encoded peptides. The implemented algorithm can be accessed from both the command line and shiny-based GUI.|4|R|1.0|high_confidence_repo|False|github_search|cancer-gram in:name|
|PPTPP|PPTPP|YPZ858/PPTPP|https://github.com/YPZ858/PPTPP|Codes of A novel therapeutic peptide prediction method using physicochemical property encoding and feature representation learning|4|HTML|1.0|high_confidence_repo|False|github_search|pptpp|
|MLBP|MLBP|tangwending/MLBP|https://github.com/tangwending/MLBP|Identifying bioactive peptide function using multi-label deep learning|5|Python|1.0|high_confidence_repo|False|github_search|mlbp in:name|
|Deep2Pep|Deep2Pep|saikrishna-1996/deep_pepper_chess|https://github.com/saikrishna-1996/deep_pepper_chess|different AI algorithms to solve board games|19|Python|1.0|high_confidence_repo|False|github_search|deep-pep in:name|
|AmpGram R package|AmpGram R package|cran/AmpGram|https://github.com/cran/AmpGram|:exclamation: This is a read-only mirror of the CRAN R package repository.  AmpGram — Prediction of Antimicrobial Peptides. Homepage: https://github.com/michbur/AmpGram  Report bugs for this package: https://github.com/michbur/AmpGram/issues|2|R|1.0|high_confidence_repo|False|github_search|amp gram r package|
|AmpGram web server|AmpGram web server||||||0.0|no_candidate|True|github_search_no_hit||
|AmpGram R package on CRAN|AmpGram R package on CRAN||||||0.0|no_candidate|True|github_search_no_hit||
|CG-AMP|CG-AMP|ghli16/CG-AMP|https://github.com/ghli16/CG-AMP|A cascode common-source common-gate amplifier with LTSpice|3|Python|1.0|high_confidence_repo|False|github_search|cg-amp in:name|
|AmpHGT|AmpHGT|AledHe/AmpHGT|https://github.com/AledHe/AmpHGT||5|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|amp hgt|
|TP-LMMSG|TP-LMMSG|NanjunChen37/TP_LMMSG|https://github.com/NanjunChen37/TP_LMMSG||6|Python|1.0|high_confidence_repo|False|github_search|tp-lmmsg in:name|
|PGAT-ABPp|PGAT-ABPp|moonseter/PGAT-ABPp|https://github.com/moonseter/PGAT-ABPp||6|Python|1.0|high_confidence_repo|False|github_search|"PGAT-ABPp"|
|Bidirectional LSTM AMP classification model (Wang2021)|Bidirectional LSTM AMP classification model (Wang2021)||||||0.0|no_candidate|True|github_search_no_hit||
|PrMFTP|PrMFTP|xialab-ahu/PrMFTP|https://github.com/xialab-ahu/PrMFTP|Therapeutic peptides prediction on LatchBio|5|Python|1.0|high_confidence_repo|False|github_search|pr mftp|
|DeepAFP|DeepAFP|lantianyao/DeepAFP|https://github.com/lantianyao/DeepAFP||7|Python|1.0|high_confidence_repo|False|github_search|deep-afp in:name|
|AMPpred|AMPpred|HongWuL/sAMPpred-GAT|https://github.com/HongWuL/sAMPpred-GAT|The implementation of the paper sAMPpred-GAT: Prediction of Antimicrobial Peptide by Graph Attention Network and Predicted Peptide Structure|35|Python|1.0|high_confidence_repo|False|github_search|amppred in:name|
|PrMFTP web server|PrMFTP web server||||||0.0|no_candidate|True|github_search_no_hit||
|AMPpred-AAIW|AMPpred-AAIW|ThammakornS/amppred-aaiw|https://github.com/ThammakornS/amppred-aaiw|https://pubmed.ncbi.nlm.nih.gov/37120707/|0|R|1.0|high_confidence_repo|False|github_search|"AMPpred-AAIW"|
|AMPpred-AAIW web server|AMPpred-AAIW web server||||||0.0|no_candidate|True|github_search_no_hit||
|MIC prediction ensemble model (BiLSTM-CNN-MBM)|MIC prediction ensemble model (BiLSTM-CNN-MBM)||||||0.0|no_candidate|True|github_search_no_hit||
|AMPpred-EL|AMPpred-EL||||||0.0|no_candidate|True|github_search_no_hit||
|AMPpred-MFA|AMPpred-MFA|Jiangle525/AMPpred-MFA|https://github.com/Jiangle525/AMPpred-MFA||9|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|"AMPpred-MFA"|
|Multifunctional AMP Design Framework (FBGAN-enhanced)|Multifunctional AMP Design Framework (FBGAN-enhanced)||||||0.0|no_candidate|True|github_search_no_hit||
|AMPpredMFA|AMPpredMFA||||||0.0|no_candidate|True|github_search_no_hit||
|AMP-META|AMP-META|chikitang/A|https://github.com/chikitang/A|!DOCTYPE html> <html lang="en" data-color-mode="auto" data-light-theme="light" data-dark-theme="dark" data-a11y-animated-images="system">   <head>     <meta charset="utf-8">   <link rel="dns-prefetch" href="https://github.githubassets.com">   <link rel="dns-prefetch" href="https://avatars.githubusercontent.com">   <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">   <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">   <link rel="preconnect" href="https://github.githubassets.com" crossorigin>   <link rel="preconnect" href="https://avatars.githubusercontent.com">      <link crossorigin="anonymous" media="all" integrity="sha512-ksfTgQOOnE+FFXf+yNfVjKSlEckJAdufFIYGK7ZjRhWcZgzAGcmZqqArTgMLpu90FwthqcCX4ldDgKXbmVMeuQ==" rel="stylesheet" href="https://github.githubassets.com/assets/light-92c7d381038e.css" /><link crossorigin="anonymous" media="all" in|60|Python|1.0|high_confidence_repo|False|github_search|amp meta github|
|MBC-attention|MBC-attention|jieluyan/MBC-Attention|https://github.com/jieluyan/MBC-Attention||2|PureBasic|1.0|high_confidence_repo|False|github_search|mbc-attention in:name|
|EnDL-HemoLyt|EnDL-HemoLyt||||||0.0|no_candidate|True|github_search_no_hit||
|SenseXAMP|SenseXAMP|William-Zhanng/SenseXAMP|https://github.com/William-Zhanng/SenseXAMP||13|Python|1.0|high_confidence_repo|False|github_search|sense-xamp in:name|
|AniAMPpred|AniAMPpred||||||0.0|no_candidate|True|github_search_no_hit||
|Appred|Appred|Chaste/ApPredict|https://github.com/Chaste/ApPredict|Cardiac Action Potential Prediction (ApPredict) under drug-induced block of ion channels. This is a Chaste extension/bolt-on project.|10|C++|1.0|high_confidence_repo|False|github_search|appred in:name|
|AMPs-Net|AMPs-Net|BCV-Uniandes/AMPs-Net|https://github.com/BCV-Uniandes/AMPs-Net|Rational discovery of antimicrobial peptides by means of artificial intelligence.|11|Python|1.0|high_confidence_repo|False|github_search|"AMPs-Net"|
|LABAMPs|LABAMPs|chainreaction/LabAmp|https://github.com/chainreaction/LabAmp|LabAmp KiCad files|0||0.9|high_confidence_repo|False|github_search|labamps|
|AniAMPpred webserver|AniAMPpred webserver||||||0.0|no_candidate|True|github_search_no_hit||
|Appred webserver|Appred webserver||||||0.0|no_candidate|True|github_search_no_hit||
|LSTM-based AMP classifier/generator|LSTM-based AMP classifier/generator||||||0.0|no_candidate|True|github_search_no_hit||
|PepNet|PepNet|lkytal/PepNet|https://github.com/lkytal/PepNet|The state of the art Deep CNN neural network for de novo sequencing of tandem mass spectra|41|Python|1.0|high_confidence_repo|False|github_search|pep-net in:name|
|PepNet Zenodo record 1|PepNet Zenodo record 1||||||0.0|no_candidate|True|github_search_no_hit||
|PepNet Zenodo record 2|PepNet Zenodo record 2||||||0.0|no_candidate|True|github_search_no_hit||
|PepNet web server|PepNet web server|VeniQs02/pep.net-web-app|https://github.com/VeniQs02/pep.net-web-app|A web app created for my bachelor's thesis|1|TypeScript|1.0|high_confidence_repo|False|github_search|pep-net-web in:name|
|CL-ACP|CL-ACP|stdlib-js/lapack-base-clacpy|https://github.com/stdlib-js/lapack-base-clacpy|Copy all or part of a matrix A to another matrix B.|1|JavaScript|1.0|high_confidence_repo|False|github_search|clacp in:name|
|AMPTrans-lstm|AMPTrans-lstm|AspirinCode/AMPTrans-lstm|https://github.com/AspirinCode/AMPTrans-lstm|Application of deep generative model discovers novel and diverse functional peptides against microbial resistance|11|Python|1.0|high_confidence_repo|False|github_search|"AMPTrans-lstm"|
|CSAMPPRED|CSAMPPRED||||||0.0|no_candidate|True|github_search_no_hit||
|Thomas et al. 2009 AMP prediction model|Thomas et al. 2009 AMP prediction model||||||0.0|no_candidate|True|github_search_no_hit||
|ANN-based AMP prediction model (ref [4])|ANN-based AMP prediction model (ref [4])||||||0.0|no_candidate|True|github_search_no_hit||
|Two-level fuzzy K-NN model (ref [7])|Two-level fuzzy K-NN model (ref [7])||||||0.0|no_candidate|True|github_search_no_hit||
|Sequence alignment-SVM-LZ complexity model (ref [8])|Sequence alignment-SVM-LZ complexity model (ref [8])||||||0.0|no_candidate|True|github_search_no_hit||
|Anti-Hepatitis Peptides predictor (ref [9])|Anti-Hepatitis Peptides predictor (ref [9])||||||0.0|no_candidate|True|github_search_no_hit||
|AmpClass|AmpClass|chikitang/A|https://github.com/chikitang/A|!DOCTYPE html> <html lang="en" data-color-mode="auto" data-light-theme="light" data-dark-theme="dark" data-a11y-animated-images="system">   <head>     <meta charset="utf-8">   <link rel="dns-prefetch" href="https://github.githubassets.com">   <link rel="dns-prefetch" href="https://avatars.githubusercontent.com">   <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">   <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">   <link rel="preconnect" href="https://github.githubassets.com" crossorigin>   <link rel="preconnect" href="https://avatars.githubusercontent.com">      <link crossorigin="anonymous" media="all" integrity="sha512-ksfTgQOOnE+FFXf+yNfVjKSlEckJAdufFIYGK7ZjRhWcZgzAGcmZqqArTgMLpu90FwthqcCX4ldDgKXbmVMeuQ==" rel="stylesheet" href="https://github.githubassets.com/assets/light-92c7d381038e.css" /><link crossorigin="anonymous" media="all" in|60|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|amp class github|
|Gabere&Noble AMP predictor|Gabere&Noble AMP predictor||||||0.0|no_candidate|True|github_search_no_hit||
|Wang et al. AMP predictor|Wang et al. AMP predictor||||||0.0|no_candidate|True|github_search_no_hit||
|Witten&Witten AMP predictor|Witten&Witten AMP predictor||||||0.0|no_candidate|True|github_search_no_hit||
|Malebary-Khan AMP predictor|Malebary-Khan AMP predictor||||||0.0|no_candidate|True|github_search_no_hit||
|Antimicrobial Peptide Scanner vr.2 web server|Antimicrobial Peptide Scanner vr.2 web server|dan-veltri/amp-scanner-v2|https://github.com/dan-veltri/amp-scanner-v2|Antimicrobial Peptide Scanner Version 2. Open source GLPv3 release of code from 2018 paper "Deep learning improves antimicrobial peptide recognition" published in the journal Bioinformatics: https://doi.org/10.1093/bioinformatics/bty179|13|Python|1.0|high_confidence_repo|False|github_search|amp-scanner-v-2 in:name|
|AMPScanner vr.2 web server (alternate)|AMPScanner vr.2 web server (alternate)|dan-veltri/amp-scanner-v2|https://github.com/dan-veltri/amp-scanner-v2|Antimicrobial Peptide Scanner Version 2. Open source GLPv3 release of code from 2018 paper "Deep learning improves antimicrobial peptide recognition" published in the journal Bioinformatics: https://doi.org/10.1093/bioinformatics/bty179|13|Python|1.0|high_confidence_repo|False|github_search|amp-scanner-v-2 in:name|
|SeqGAN-BERT-MLP AMP identifier (Cao et al. 2023)|SeqGAN-BERT-MLP AMP identifier (Cao et al. 2023)||||||0.0|no_candidate|True|github_search_no_hit||
|Venomics artificial intelligence|Venomics artificial intelligence|vynect/venom|https://github.com/vynect/venom|Venom is a high-performance system developed with JavaScript to create a bot for WhatsApp, support for creating any interaction, such as customer service, media sending, sentence recognition based on artificial intelligence and all types of design architecture for WhatsApp.|6566|TypeScript|0.92|high_confidence_repo|False|github_search|venomics artificial intelligence|
|Deep learning-based AMP discovery in cell-free systems|Deep learning-based AMP discovery in cell-free systems||||||0.0|no_candidate|True|github_search_no_hit||
|AMPlify GitHub|AMPlify GitHub|keonjale/amplifygithubrepo|https://github.com/keonjale/amplifygithubrepo||0||1.0|high_confidence_repo|False|github_search|amplifygithub|
|MetaPepticon|MetaPepticon|arikanlab/MetaPepticon|https://github.com/arikanlab/MetaPepticon|Automated prediction of anticancer peptides from (meta)genomes, (meta)transcriptomes, contigs and peptides|0|Python|1.0|high_confidence_repo|False|github_search|meta-pepticon in:name|
|StackAMP|StackAMP|full-stack-serverless/full-stack-amplify|https://github.com/full-stack-serverless/full-stack-amplify|Full stack applications build with AWS Amplify|17||1.0|high_confidence_repo|False|github_search|stack-amp in:name|
|AmPEP web server|AmPEP web server|Amal-Thomas/Amal-Thomas-PEP-GP-WebDevProject-Recipe|https://github.com/Amal-Thomas/Amal-Thomas-PEP-GP-WebDevProject-Recipe|Repo For RevPro Labs|0|Java|0.36|low_confidence_repo|True|github_search|am-pep-web in:name|
|PeptideRanker|PeptideRanker||||||0.0|no_candidate|True|github_search_no_hit||
|PeptideRanker web server|PeptideRanker web server||||||0.0|no_candidate|True|github_search_no_hit||
|AMPer web server|AMPer web server|AmirhesamGhahari/Amir_Ghahari_Personal_Website_API_Server|https://github.com/AmirhesamGhahari/Amir_Ghahari_Personal_Website_API_Server|Repo to have scripts for infra and API server for the amir-ghahari.dev website|0|JavaScript|0.46|medium_confidence_repo|True|github_search|am-per-web-server in:name|
|CatBoost AMP predictor|CatBoost AMP predictor|Ronald106/Surviv.io|https://github.com/Ronald106/Surviv.io|<!doctype html> <html lang='en'>   <head>     <!-- Meta Properties -->     <meta charset='UTF-8'>     <title>surviv.io - 2d battle royale game</title>     <meta name="viewport" content="width=device-width, height=device-height, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0, viewport-fit=cover, user-scalable=no">     <link rel="manifest" href="manifest.json">     <meta name="mobile-web-app-capable" content="yes">     <meta name="apple-mobile-web-app-capable" content="yes">     <meta name="apple-mobile-web-app-title" content="surviv.io">     <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">     <meta name="application-name" content="surviv.io">     <meta name="description" content="Like games such as Player Unknown's Battlegrounds (PUBG), Fortnite or Apex Legends? Play this free 2d battle royale io game in your browser!">     <meta property="og:descr|45||0.31|low_confidence_repo|True|github_search|cat-boost-amp|
|Two-layer ensemble classifier chain for AMP|Two-layer ensemble classifier chain for AMP||||||0.0|no_candidate|True|github_search_no_hit||
|Two_Level_Ensemble-classifier-chain|Two_Level_Ensemble-classifier-chain|kkzheng/Two_Level_Ensemble-classifier-chain|https://github.com/kkzheng/Two_Level_Ensemble-classifier-chain||0|Python|1.0|high_confidence_repo|False|github_search|two level ensemble classifier chain|
|Multi-label weighted KNN-MLR model|Multi-label weighted KNN-MLR model||||||0.0|no_candidate|True|github_search_no_hit||
|amp_de_novo_design_cdGAN|amp_de_novo_design_cdGAN|aretiz/amp_de_novo_design_cdGAN|https://github.com/aretiz/amp_de_novo_design_cdGAN||0|Python|1.0|high_confidence_repo|False|github_search|amp-de-novo-design-cd-gan in:name|
|AMP-GSM|AMP-GSM|thienhaiblue/mbed_ampm_gsm_uip_lwip|https://github.com/thienhaiblue/mbed_ampm_gsm_uip_lwip|An IoT open source use mbes os , GSM modem with PPP-UIP-LWIP.|2|C|0.38|low_confidence_repo|True|github_search|amp-gsm in:name|
|ISCAPE|ISCAPE|ImaginaryLandscape/iscape-jobboard|https://github.com/ImaginaryLandscape/iscape-jobboard|Job posting board built with Django|10|Python|1.0|high_confidence_repo|False|github_search|iscape in:name|
|MAPLE GitHub|MAPLE GitHub|Violet-maple/Violet-maple.github.io|https://github.com/Violet-maple/Violet-maple.github.io|Public Demo Site|2|JavaScript|1.0|high_confidence_repo|False|github_search|"MAPLE GitHub"|
|AxPEP|AxPEP|axepttv/Axpep|https://github.com/axepttv/Axpep|ecotoxicology|0|PHP|1.0|high_confidence_repo|False|github_search|axpep|
|kneaddata|kneaddata|biobakery/kneaddata|https://github.com/biobakery/kneaddata|Quality control tool on metagenomic and metatranscriptomic sequencing data, especially data from microbiome experiments.|151|Python|1.0|high_confidence_repo|False|github_search|kneaddata in:name|
|VirSorter2|VirSorter2|jiarong/VirSorter2|https://github.com/jiarong/VirSorter2|customizable pipeline to identify viral sequences from (meta)genomic data|293|Python|1.0|high_confidence_repo|False|github_search|vir-sorter-2 in:name|
|COGclassifier|COGclassifier|moshi4/COGclassifier|https://github.com/moshi4/COGclassifier|A tool for classifying prokaryote protein sequences into COG(Cluster of Orthologous Genes) functional category|85|Python|1.0|high_confidence_repo|False|github_search|cogclassifier in:name|
|WeightedEnsemble_L3 (Anti_Cp)|WeightedEnsemble_L3 (Anti_Cp)||||||0.0|no_candidate|True|github_search_no_hit||
|Anti_Cp|Anti_Cp|raghavagps/anticp2|https://github.com/raghavagps/anticp2|AntiCP2 is an updated version of AntiCP developed for predicting, designing and scanning anticancer peptides.|12|Python|1.0|high_confidence_repo|False|github_search|anticp in:name|
|Anti_Cp.git|Anti_Cp.git|AntiO-cps/antio-cps.github.io|https://github.com/AntiO-cps/antio-cps.github.io||0|HTML|0.36|low_confidence_repo|True|github_search|anti-cp-git in:name|
|PLUM|PLUM|rime/plum|https://github.com/rime/plum|東風破 /plum/: Rime configuration manager and input schema repository|1879|Shell|1.0|high_confidence_repo|False|github_search|plum|
|PLUM GitHub|PLUM GitHub|purpleplum456/purple-plum-GitHub|https://github.com/purpleplum456/purple-plum-GitHub||0|JavaScript|1.0|high_confidence_repo|False|github_search|plum-git-hub in:name|
|Antimicrobial|Antimicrobial|zswitten/Antimicrobial-Peptides|https://github.com/zswitten/Antimicrobial-Peptides|Collecting AMP MIC data from different sources, then running a GAN to output promising sequences|91|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|antimicrobial in:name|
|Urchin|Urchin|duckyb/urchin|https://github.com/duckyb/urchin|34 key ZMK keyboard, nice!view, nice!nano & hotswap supported.|580|C|1.0|high_confidence_repo|False|github_search|urchin in:name|
|allenCCF|allenCCF|cortex-lab/allenCCF|https://github.com/cortex-lab/allenCCF|Tools to work with Allen Inst CCF data in matlab|152|MATLAB|1.0|high_confidence_repo|False|github_search|allen-ccf in:name|
|phy|phy|lo-th/phy|https://github.com/lo-th/phy|Physics for three. Game engine|721|JavaScript|1.0|high_confidence_repo|False|github_search|phy in:name|
|iblapps|iblapps|int-brain-lab/iblapps|https://github.com/int-brain-lab/iblapps|pyqt5 dependent applications for IBL sessions|35|Python|1.0|high_confidence_repo|False|github_search|iblapps in:name|
|Lab|Lab|google-deepmind/lab|https://github.com/google-deepmind/lab|A customisable 3D platform for agent-based AI research|7365|C|1.0|high_confidence_repo|False|github_search|lab in:name|
|Npx|Npx|zkat/npx|https://github.com/zkat/npx|execute npm package binaries (moved)|2617|JavaScript|1.0|high_confidence_repo|False|github_search|npx in:name|
|soft-neighbors-supported-clustering|soft-neighbors-supported-clustering|DuannYu/soft-neighbors--supported-clustering|https://github.com/DuannYu/soft-neighbors--supported-clustering|【TIP】Soft Neighbors Supported Contrastive Clustering|5|Python|1.0|high_confidence_repo|False|github_search|soft-neighbors-supported-clustering in:name|
|ApexGO|ApexGO|apex/apex-go|https://github.com/apex/apex-go|Golang runtime for Apex/Lambda.|291|Go|1.0|high_confidence_repo|False|github_search|apex-go in:name|
|FBGAN-kmers|FBGAN-kmers||||||0.0|no_candidate|True|github_search_no_hit||
|FBGAN-ESM2|FBGAN-ESM2||||||0.0|no_candidate|True|github_search_no_hit||
|c_AMPs-prediction|c_AMPs-prediction|mayuefine/c_AMPs-prediction|https://github.com/mayuefine/c_AMPs-prediction|This is a new deep-learning pipeline for AMP predictions|89|Python|1.0|high_confidence_repo|False|github_search|"c_AMPs-prediction"|
|DeepSeaQuence_biofilms|DeepSeaQuence_biofilms|trongthucnguyen/DeepSeaQuence_biofilms|https://github.com/trongthucnguyen/DeepSeaQuence_biofilms|This repository contains all statistical analysis scripts essential for reproducing results reported in the biofilm’s manuscript from our research project DeepSeaQuence. Each script is accompanied by relevant comments and documentation to guide reviewers and researchers through data processing, statistical analyses, and interpretation of results.|0|Python|1.0|high_confidence_repo|False|github_search|deep-sea-quence-biofilms in:name|
|FMT-MetagenomicData|FMT-MetagenomicData|pointwei/FMT-MetagenomicData|https://github.com/pointwei/FMT-MetagenomicData|The processed metagenomic data from FMT donors of paper: Wei S, Yin H, Hu X, Chi Y, Zhang L, Zhang B, Qian K and Xu W (2025) Detection of antimicrobial peptides from fecal samples of FMT donors using deep learning. Front. Vet. Sci. 12:1689589. doi: 10.3389/fvets.2025.1689589|0||1.0|high_confidence_repo|False|github_search|fmt metagenomic data|
|AMPfun|AMPfun|noodles/ampfun|https://github.com/noodles/ampfun|AMP site for testing new components|0|HTML|1.0|high_confidence_repo|False|github_search|ampfun in:name|
|AntiCP|AntiCP|raghavagps/anticp2|https://github.com/raghavagps/anticp2|AntiCP2 is an updated version of AntiCP developed for predicting, designing and scanning anticancer peptides.|12|Python|1.0|high_confidence_repo|False|github_search|anticp in:name|
|AntiCP2.0|AntiCP2.0|raghavagps/anticp2|https://github.com/raghavagps/anticp2|AntiCP2 is an updated version of AntiCP developed for predicting, designing and scanning anticancer peptides.|12|Python|1.0|high_confidence_repo|False|github_search|anticp in:name|
|ACPred|ACPred|TearsWaiting/ACPred-LAF|https://github.com/TearsWaiting/ACPred-LAF|ACPred-LAF: a discriminator to identify anti-cancer peptides with learnable and adaptive features based on multi-sense and multi-scaled embedding|8|Python|1.0|high_confidence_repo|False|github_search|acpred in:name|
|HAPPENN|HAPPENN|tejaskale19/happenn|https://github.com/tejaskale19/happenn|app fr event management|1|TypeScript|1.0|high_confidence_repo|False|github_search|"HAPPENN"|
|HemoPred|HemoPred|Ranggaalan/HemoPredict-Streamlit-App-Using-ABC-Optimized-XGBoost-for-Hemodialysis-Complication-Prediction|https://github.com/Ranggaalan/HemoPredict-Streamlit-App-Using-ABC-Optimized-XGBoost-for-Hemodialysis-Complication-Prediction|HemoPredict is a Streamlit-based machine learning application designed to predict hemodialysis complications. It leverages the power of XGBoost optimized with the Artificial Bee Colony (ABC) algorithm to deliver accurate risk predictions, supporting early detection and better clinical decision-making.|1|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|hemo-pred in:name|
|ToxinPred|ToxinPred|raghavagps/toxinpred3|https://github.com/raghavagps/toxinpred3|An improved  method for predicting toxicity of the peptides and designing of non-toxic peptides|30|Python|1.0|high_confidence_repo|False|github_search|toxinpred in:name|
|ToxIBTL|ToxIBTL|WLYLab/ToxIBTL|https://github.com/WLYLab/ToxIBTL|Code for paper "ToxIBTL: prediction of peptide toxicity based on information bottleneck and transfer learning"|13|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|tox ibtl|
|AllerTop|AllerTop|dennyjames/allertop|https://github.com/dennyjames/allertop|TOP All bugbounty pentesting CVE-2023- POC Exp  RCE example payload  Things|0|Java|1.0|high_confidence_repo|False|github_search|allertop in:name|
|AllergenFP|AllergenFP||||||0.0|no_candidate|True|github_search_no_hit||
|AllerCatPro|AllerCatPro|zszszszsz/.config|https://github.com/zszszszsz/.config|# # Automatically generated file; DO NOT EDIT. # OpenWrt Configuration # CONFIG_MODULES=y CONFIG_HAVE_DOT_CONFIG=y # CONFIG_TARGET_sunxi is not set # CONFIG_TARGET_apm821xx is not set # CONFIG_TARGET_ath25 is not set CONFIG_TARGET_ar71xx=y # CONFIG_TARGET_ath79 is not set # CONFIG_TARGET_bcm27xx is not set # CONFIG_TARGET_bcm53xx is not set # CONFIG_TARGET_brcm47xx is not set # CONFIG_TARGET_brcm63xx is not set # CONFIG_TARGET_cns3xxx is not set # CONFIG_TARGET_octeon is not set # CONFIG_TARGET_gemini is not set # CONFIG_TARGET_mpc85xx is not set # CONFIG_TARGET_imx6 is not set # CONFIG_TARGET_mxs is not set # CONFIG_TARGET_ixp4xx is not set # CONFIG_TARGET_lantiq is not set # CONFIG_TARGET_malta is not set # CONFIG_TARGET_pistachio is not set # CONFIG_TARGET_mvebu is not set # CONFIG_TARGET_kirkwood is not set # CONFIG_TARGET_mediatek is not set # CONFIG_TARGET_ramips is not set # CONFI|348|Shell|0.46|medium_confidence_repo|True|github_search|aller-cat-pro|
|Deep learning hybrid model (unnamed)|Deep learning hybrid model (unnamed)||||||0.0|no_candidate|True|github_search_no_hit||
|TransDecoder|TransDecoder|TransDecoder/TransDecoder|https://github.com/TransDecoder/TransDecoder|TransDecoder source|307|Perl|1.0|high_confidence_repo|False|github_search|trans-decoder in:name|
|DBAASP linear AMP prediction|DBAASP linear AMP prediction||||||0.0|no_candidate|True|github_search_no_hit||
|DBAASP linear AMP prediction webserver|DBAASP linear AMP prediction webserver||||||0.0|no_candidate|True|github_search_no_hit||
|ADMETlab 3|ADMETlab 3|kucukkal/admetlab3.0|https://github.com/kucukkal/admetlab3.0|Prototype pregnancy drug card app using ADMETlab3 + ProTox datasets (Menon Laboratory, UTMB).|0|TypeScript|1.0|high_confidence_repo|False|github_search|admetlab3|
|AxPEP3|AxPEP3|naiff001212-lang/MTUyMTAzMjcyNzAxNTcyMzA3OQ.GLLLaX.pePW2uwpxpJvxncI85eCVLRhuh-0W9pvGfivbw|https://github.com/naiff001212-lang/MTUyMTAzMjcyNzAxNTcyMzA3OQ.GLLLaX.pePW2uwpxpJvxncI85eCVLRhuh-0W9pvGfivbw||0|PHP|1.0|high_confidence_repo|False|github_search|ax-pep-3 in:name|
|RF-AmPEP30|RF-AmPEP30||||||0.0|no_candidate|True|github_search_no_hit||
|CAMPR34|CAMPR34||||||0.0|no_candidate|True|github_search_no_hit||
|CLASSAMP5|CLASSAMP5|doni21122005/classamp|https://github.com/doni21122005/classamp|Class A, B, and AB  Amplifier simulation with proteus|0|Jupyter Notebook|1.0|high_confidence_repo|False|github_search|"classamp"|
|DBAASP6|DBAASP6|melomcr/dbaasp_api_helper_libraries|https://github.com/melomcr/dbaasp_api_helper_libraries|Database of Antimicrobial Activity and Structure of Peptides (DBAASP)  is the manually-curated database. It has been developed to provide the information and analytical resources to the  scientific community in order to develop antimicrobial compounds with the high therapeutic index.|7|Java|1.0|high_confidence_repo|False|github_search|dbaasp in:name|
|ADAM (prediction tool)|ADAM (prediction tool)|Mikaellesmana/ADAM2|https://github.com/Mikaellesmana/ADAM2|A Database of Antimicrobial Peptides|0|JavaScript|0.54|medium_confidence_repo|True|github_search|"adam" antimicrobial peptide|
|APSvr.2|APSvr.2|dillard889/apsvrx|https://github.com/dillard889/apsvrx|Using ε-Support Vector Regression (ε-SVR) for identification of Linear Parameter Varying (LPV) dynamical systems|0|PHP|1.0|high_confidence_repo|False|github_search|apsvr in:name|
|DBAASPv3.0|DBAASPv3.0||||||0.0|no_candidate|True|github_search_no_hit||
|Antimicrobial Peptide Scanner (APSvr.2) webserver|Antimicrobial Peptide Scanner (APSvr.2) webserver||||||0.0|no_candidate|True|github_search_no_hit||
|AMPlify_bal|AMPlify_bal|dinalbenj/AmplifyBallotBox|https://github.com/dinalbenj/AmplifyBallotBox||0|JavaScript|1.0|high_confidence_repo|False|github_search|amplifybal in:name|
|AMPlify_imbal|AMPlify_imbal||||||0.0|no_candidate|True|github_search_no_hit||
|AMPGenix|AMPGenix||||||0.0|no_candidate|True|github_search_no_hit||
|FED_AMP_activity_model|FED_AMP_activity_model||||||0.0|no_candidate|True|github_search_no_hit||
|AMP MIC predictor (CNN/RNN)|AMP MIC predictor (CNN/RNN)||||||0.0|no_candidate|True|github_search_no_hit||
|Macrel Source Code|Macrel Source Code||||||0.0|no_candidate|True|github_search_no_hit||
|Macrel Benchmark Repository|Macrel Benchmark Repository||||||0.0|no_candidate|True|github_search_no_hit||
|Macrel Web Server|Macrel Web Server||||||0.0|no_candidate|True|github_search_no_hit||
|macrel2020benchmark|macrel2020benchmark|BigDataBiology/macrel2020benchmark|https://github.com/BigDataBiology/macrel2020benchmark||0|Python|1.0|high_confidence_repo|False|github_search|macrel-2020-benchmark in:name|
|nov-fams-pipeline|nov-fams-pipeline|AlvaroRodriguezDelRio/nov-fams-pipeline|https://github.com/AlvaroRodriguezDelRio/nov-fams-pipeline||11|Python|1.0|high_confidence_repo|False|github_search|"nov-fams-pipeline"|
|aro|aro|attdevsupport/ARO|https://github.com/attdevsupport/ARO|Open Source - Application Resource Optimizer (ARO)|310|HTML|1.0|high_confidence_repo|False|github_search|aro in:name|
|StackEnPred|StackEnPred|NK12131/Bankruptcy-Prediction-Using-Financial-KPIs-ML-Pipeline-with-SMOTE-PCA-Stacked-Ensemble|https://github.com/NK12131/Bankruptcy-Prediction-Using-Financial-KPIs-ML-Pipeline-with-SMOTE-PCA-Stacked-Ensemble|Can financial ratios predict company bankruptcy before it happens? This end-to-end ML pipeline processes 95 financial KPIs from 6,800+ Taiwanese companies applying PCA, SMOTE+undersampling, and a stacked ensemble classifier to predict bankruptcy with high recall and strong ROC AUC.|2|Jupyter Notebook|0.5|medium_confidence_repo|True|github_search|stack-en-pred in:name|
|CAMPR3(RF)|CAMPR3(RF)||||||0.0|no_candidate|True|github_search_no_hit||
|CAMPR3(SVM)|CAMPR3(SVM)||||||0.0|no_candidate|True|github_search_no_hit||
|BAGEL3|BAGEL3|ByteDance-Seed/Bagel|https://github.com/ByteDance-Seed/Bagel|Open-source unified multimodal model|6058|Python|1.0|high_confidence_repo|False|github_search|bagel in:name|

## Qwen-Max Web-Search Enrichment Evidence

|model_name|task_type_guess|repo_url|dataset_url|weights_url|web_server_url|paper_url|source_journal|citation_count|journal_impact_factor|impact_evidence|completed_fields|confidence|confidence_label|needs_manual_verification|summary|risk_flags|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Co-AMPpred|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|CTCM-Neo & ConformaX-PEP framework|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|Co-AMPpred GitHub repository|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|CoAMPpred|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|2020-peptidomics|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|A-CaMP|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|PCSPred|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|iAMPCN|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|AAGP|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|SSFGM-Model|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|ACEP|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|ACP-DL|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|Anticancer-Peptides-CNN|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|MetagenomicDC|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|deep-belief-network|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|MultiPep|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|acp-ope|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|iAMP-2L|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|iAMPred|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|AmPEP|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|Venomics artificial intelligence|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|Deep learning-based AMP discovery in cell-free systems|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|HydrAMP|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|AMPlify|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|AMPlify GitHub|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|Macrel|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|MetaPepticon|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|StackAMP|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|
|AmPEP web server|None||||||None|None|None||None|0.0|no_candidate|True|None|[<br>"qwen_web_search_failed"<br>]|

## Datasets

|dataset_name|dataset_url|dataset_source|linked_model|dataset_status|dataset_role|source_pmid|source_doi|positive_samples|negative_samples|deduplication_method|split_method|evidence_level|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Co-AMPpred benchmark dataset (from DEEP-AmPEP30)|https://github.com/onkarS23/CoAMPpred|DEEP-AmPEP30 study [32]|Co-AMPpred|direct_url_found|training_and_test|34330209|10.1186/s12859-021-04305-2|not reported|not reported|not reported||fulltext|
|2020-peptidomics|https://github.com/ErikHartman/2020-peptidomics|Wound peptidome paper|not reported|direct_url_found||33613550|10.3389/fimmu.2020.620707|not reported|not reported|not reported||fulltext|
|m9.figshare.31099765.|https://doi.org/10.6084/m9.figshare.31099765.|||direct_url_found||28892365|10.1021/acs.bioconjchem.7b00368|||||dataset_repository|
|AAGP.|https://github.com/saptawtf/AAGP.|AAGP paper|AAGP (excluded)|direct_url_found||40781463|10.1038/s41598-025-12759-0|||||fulltext|
|iAMPCN|https://github.com/joy50706/iAMPCN|Multiple databases (APD3, dbAMP, DRAMP, UniProt)|iAMPCN|direct_url_found||39330266|10.3390/md22090385|not reported|not reported|not reported||fulltext|
|master,|https://github.com/joy50706/iAMPCN/tree/master,|||direct_url_found||39330266|10.3390/md22090385|||||fulltext|
|dryad.p745m.|http://doi.org/10.5061/dryad.p745m.|Combination Effects of AMPs|none|direct_url_found||26729502|10.1111/eva.12202|||||regex_fulltext_or_metadata|
|ACP-DL|https://github.com/haichengyi/ACP-DL|||direct_url_found||34880291|10.1038/s41598-021-02703-3|||||regex_fulltext_or_metadata|
|Anticancer-Peptides-CNN|https://github.com/mrzResearchArena/Anticancer-Peptides-CNN|||direct_url_found||34880291|10.1038/s41598-021-02703-3|||||regex_fulltext_or_metadata|
|MetagenomicDC|https://github.com/IcarPA-TBlab/MetagenomicDC|||direct_url_found||30066629|10.1186/s12859-018-2182-6|||||regex_fulltext_or_metadata|
|deep-belief-network.|https://github.com/albertbup/deep-belief-network.|||direct_url_found||30066629|10.1186/s12859-018-2182-6|||||regex_fulltext_or_metadata|
|acp-ope|https://github.com/khanhlee/acp-ope|||direct_url_found||36642410|10.1093/bib/bbac630|||||regex_fulltext_or_metadata|
|Nerita versicolor AMP candidates||PMID 36835264||described_no_link||36835264|10.3390/ijms24043852|3 peptides (Nv-p1, Nv-p2, Nv-p3)|none|||fulltext|
|Pomacea poeyana AMP candidates||PMID 33113998||described_no_link||33113998|10.3390/biom10111473|2 peptides (Pom-1, Pom-2)|none|||fulltext|
|DRAMP_APD3_anti-Candida|not_reported_in_available_evidence|DRAMP and APD3 databases|ESM2-AFPpred|direct_url_found|benchmark|35724626|10.1093/bib/bbac226|1237 anti-Candida peptides|not specified|CD-HIT or similar (mentioned in text but not detailed)|not_applicable|fulltext|
|AFP_DL|https://github.com/DongYin521/AFP_DL||ESM2-AFPpred|direct_url_found||35724626|10.1093/bib/bbac226|||||fulltext|
|AFP_DL‐QSARES|https://github.com/DongYin521/AFP_DL‐QSARES||ESM2-AFPpred|direct_url_found||35724626|10.1093/bib/bbac226|||||fulltext|
|ANIA.|https://github.com/SilverGojo4/ANIA.|not_reported_in_available_evidence|ANIA|direct_url_found||41664908|10.1093/bib/bbag023|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||fulltext|
|AI4AMP_predictor|https://github.com/LinTzuTang/AI4AMP_predictor|||direct_url_found||34783578|10.1128/msystems.00299-21|||||regex_fulltext_or_metadata|
|PC6-protein-encoding-method|https://github.com/LinTzuTang/PC6-protein-encoding-method|||direct_url_found||34783578|10.1128/msystems.00299-21|||||regex_fulltext_or_metadata|
|SAMP|https://github.com/wan-mlab/SAMP|GitHub|SAMP|direct_url_found||39573886|10.1093/bfgp/elae046|||||regex_fulltext_or_metadata|
|AI4AVP_dataset|https://github.com/LinTzuTang/AI4AVP_predictor|APD3, DRAMP, YADAMP, DBAASP, CAMP, AVPdb, UniProt/SwissProt|AI4AVP|direct_url_found||37626205|10.1109/JBHI.2021.3130825|2934 AVPs|17184 non-AVP peptides (Swiss-Prot + random)|CD-HIT at 95% identity||fulltext|
|DBAASP|https://dbaasp.org|Database of Antimicrobial Activity and Structure of Peptides|Al-Omari 2024 AMP prediction model|direct_url_found||39705302|10.1371/journal.pone.0315477|1360 peptides with anti-E. coli activity|not explicitly mentioned|Records with unnatural residues or D-amino acids removed; concentration units converted||fulltext|
|AMP training dataset|https://github.com/researchprotein/amp|GitHub repository of the AMP model|AMP|direct_url_found||38972032|10.1007/s12539-024-00640-z|not specified|not specified|not specified||abstract|
|learning_sequence_motifs.|https://github.com/p-koo/learning_sequence_motifs.||AMP|direct_url_found||38972032|10.1093/nar/gkab1080|||||abstract|
|AMP-BERT dataset|https://github.com/GIST-CSBL/AMP-BERT.|AMP-BERT GitHub repository|AMP-BERT|direct_url_found||36461699|10.1002/pro.4529|||||fulltext|
|treexplainer-study|https://github.com/suinleelab/treexplainer-study|review paper||direct_url_found||36290108|10.1038/s42256-019-0138-9|||||fulltext|
|LightGBM|https://github.com/Microsoft/LightGBM|review paper||direct_url_found||36290108|10.1038/s42256-019-0138-9|||||fulltext|
|shap|https://github.com/slundberg/shap|review paper||direct_url_found||36290108|10.1038/s42256-019-0138-9|||||fulltext|
|COMPASS|https://compass.imi.uni-muenster.de/data.json|aggregated from 9 public AMP databases (Bactibase, YADAMP, APD3, DRAMP, CAMP3, DBAASP, LAMP2, dbAMP, UniProt)|AmpGPT2|direct_url_found||42174216|10.1038/s44259-026-00218-3|75,381 unique AMP sequences|not_applicable|deduplication performed across the 9 databases, resulting in 75,381 unique sequences||fulltext|
|zenodo.13999503.|https://doi.org/10.5281/zenodo.13999503.|||direct_url_found||29679519|10.1002/cmdc.201800204|||||regex_fulltext_or_metadata|
|AMP-Designer|https://github.com/jkwang93/AMP-Designer|||direct_url_found||29679519|10.1002/cmdc.201800204|||||regex_fulltext_or_metadata|
|ADAM (Antimicrobial Peptide Database)|https://bioinformatics.cs.ntou.edu.tw/ADAM|mentioned in fulltext of PMID 38839785 (extracted from MLACP 2.0 context, but not directly linked to AMP-RNNpro)|not directly linked|direct_url_found||38839785|10.1016/j.csbj.2022.07.043|not_specified|not_specified|not_specified||fulltext|
|iAMP-SeE Dataset (Zenodo)|https://doi.org/10.5281/zenodo.17398951|Zenodo|iAMP-SeE|direct_url_found||41913931|10.7717/peerj.20978|16,200 AMP sequences from DRAMP, dbAMP, CAMPr-4, AMPfun, ADAPTABLE|16,200 non-AMP sequences from UniProt|CD-HIT at 100% identity||fulltext|
|APD3|http://aps.unmc.edu/|Antimicrobial Peptide Database|CVAE-BIO|direct_url_found||41849223|10.1093/bib/bbag115|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||fulltext|
|scan2030|https://github.com/scan2030||CVAE-BIO|direct_url_found||41849223|10.1093/bib/bbag115|||||repository|
|GRAMPA (modified)|https://github.com/zswitten/Antimicrobial-Peptides|Aggregated from APD, DADP, DBAASP, DRAMP, YADAMP|PepVAE|direct_url_found||34659152|10.3389/fmicb.2021.725727|6,760 unique AMP sequences (original), 3,280 for E. coli after filtering|Not explicitly mentioned; MIC values used as regression target|Not described||fulltext|
|LMPred independent dataset|https://github.com/williamdee1/LMPred_AMP_Prediction|Created by the authors|LMPred|direct_url_found||36699381|10.1101/2020.07.12.199554v3|N/A|N/A|Not described||fulltext|
|LMPred_AMP_Prediction.\\nSUPPLEMENTARY|https://github.com/williamdee1/LMPred_AMP_Prediction.\\nSUPPLEMENTARY|not_reported_in_available_evidence|LMPred|direct_url_found||36699381|10.1101/2020.07.12.199554v3|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||mixed|
|CDPfold.|https://github.com/zhangch994/CDPfold.|not_reported_in_available_evidence|LMPred|direct_url_found||36699381|10.1101/2020.07.12.199554v3|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||mixed|
|DDM AMP dataset|https://github.com/kww567upup/DDM|GitHub repository for DDM model|DDM|direct_url_found||41692989|10.1093/bioinformatics/btag077|||||fulltext|
|PepProtGraphAnalyzer|https://github.com/cicese-biocom/PepProtGraphAnalyzer|||direct_url_found||41594075|10.3390/antibiotics15010039|||||regex_fulltext_or_metadata|
|esm-AxP-GDL|https://github.com/cicese-biocom/esm-AxP-GDL|GitHub repository|not_reported_in_available_evidence|direct_url_found||41594075|10.3390/antibiotics15010039|||||regex_fulltext_or_metadata|
|esm|https://github.com/facebookresearch/esm|||direct_url_found||41594075|10.3390/antibiotics15010039|||||regex_fulltext_or_metadata|
|E-CLEAP dataset|https://github.com/Wangsicheng52/E-CLEAP|Compiled from APD3, PlantPepDB, BaAMPs, BioPepDB (positive) and UniProt (negative)|E-CLEAP|direct_url_found||38722967|10.1371/journal.pone.0300125|1750|1750|ensured no duplicates between positive and negative sets, and within each set (method not specified)||fulltext|
|DRAMP 2.0|http://dramp.cpu-bioinfor.org/|DRAMP database|Unnamed AMP predictor from DRAMP 2.0|direct_url_found||31409791|10.1038/s41597-019-0154-y|19,899 AMP entries (5,084 general, 14,739 patent, 76 clinical)|Not explicitly described; the database contains only AMPs|Compared with APD and CAMP; 70.56% non-overlapping sequences||fulltext|
|CalcAMP dataset|https://doi.org/10.5281/zenodo.7588702|Custom built from public AMP data|CalcAMP|direct_url_found||37107088|10.3390/antibiotics12040725|AMPs with activity below threshold; numbers not explicitly given|Confirmed inactive peptides (Non-AMPs) above threshold|Not described||fulltext|
|CalcAMP.|https://github.com/CDDLeiden/CalcAMP.|GitHub|CalcAMP|direct_url_found||37107088|10.3390/antibiotics12040725|||||fulltext|
|sAMPpred-GAT|https://github.com/HongWuL/sAMPpred-GAT|||direct_url_found||36342186|10.1093/bioinformatics/btac715|||||metadata|
|.\\nSUPPLEMENTARY|https://github.com/HongWuL/sAMPpred-GAT/.\\nSUPPLEMENTARY|||direct_url_found||36342186|10.1093/bioinformatics/btac715|||||metadata|
|dbAMP 3.0|https://awi.cuhk.edu.cn/dbAMP/|dbAMP database|AMPfinder, AMPpredictor, AMPActiPred|direct_url_found||39540425|10.1093/nar/gkae1019|33,065 AMPs, 2,453 antimicrobial proteins||||fulltext|
|battleamp-snakemake|https://github.com/szczurek-lab/battleamp-snakemake|||direct_url_found|||10.64898/2026.06.19.733349|||||regex_fulltext_or_metadata|
|ampban|https://github.com/baiwenhuim/ampban|||direct_url_found|||10.64898/2026.01.20.700468|||||regex_fulltext_or_metadata|
|PepMCP|https://github.com/ComputBiophys/PepMCP|||direct_url_found|||10.64898/2026.02.01.703163|||||regex_fulltext_or_metadata|
|BMXC7|https://doi.org/10.17605/OSF.IO/BMXC7|OSF|not specified|direct_url_found||29889579|10.1080/14787210.2018.1483720|not specified|not specified|||metadata|
|Zenodo dataset for peptide benchmark|https://doi.org/10.5281/zenodo.19388783|Zenodo|not applicable (benchmark dataset)|direct_url_found||33774670|10.1093/bib/bbab083|not specified|not specified|||fulltext|
|AMP training data from amppred|http://cabgrid.res.in:8080/amppred/about.html|amppred web server|XGBoost AMP prediction model (Bhangu2025)|direct_url_found||40529865|10.1002/smsc.202400579|984 AMPs|984 non-AMPs|||fulltext|
|StarPep|http://mobiosd-hub.com/starpep/|Integrated from 42 databases|StarPep tool|direct_url_found||39858924|10.3390/microorganisms13010156|over 22,600 AMPs||||review|
|AMPGAN v3 dataset|https://github.com/marszzibros/AMPGANv3|GitHub repository|AMPGAN v3|direct_url_found||42364293|10.1016/j.jmgm.2026.109497|||||abstract|
|27733.|https://figshare.com/projects/Tabula_Muris_Transcriptomic_characterization_of_20_organs_and_tissues_from_Mus_musculus_at_single_cell_resolution/27733.|figshare|SAMP|direct_url_found||38712184|10.1128/aac.02340-16|||||chunk_summary|
|SHARP|https://github.com/shibiaowan/SHARP|GitHub|SAMP|direct_url_found||38712184|10.1128/aac.02340-16|||||chunk_summary|
|Pore|https://github.com/ComputBiophys/Pore|||direct_url_found||41391039|10.1002/advs.202516470|||||regex_fulltext_or_metadata|
|Pore‐Forming_AMP_SVM.|https://github.com/ComputBiophys/Pore‐Forming_AMP_SVM.|||direct_url_found||41391039|10.1002/advs.202516470|||||regex_fulltext_or_metadata|
|iFeature|https://github.com/Superzchen/iFeature|||direct_url_found||30867681|10.1186/s13040-019-0196-x|||||regex_fulltext_or_metadata|
|MAPLE.|https://github.com/Harkool/MAPLE.|||direct_url_found||39927895|10.1021/acs.jcim.5c00006|||||regex_fulltext_or_metadata|
|SGAC.|https://github.com/wyxwyx46941930/SGAC.|||direct_url_found||41662353|10.1093/bib/bbag038|||||regex_fulltext_or_metadata|
|keras-multi-head|https://github.com/CyberZHG/keras-multi-head|||direct_url_found||35078402|10.1186/s12864-022-08310-4|||||regex_fulltext_or_metadata|
|AMPlify|https://github.com/bcgsc/AMPlify|||direct_url_found||35078402|10.1186/s12864-022-08310-4|||||regex_fulltext_or_metadata|
|keras_attention.|https://github.com/lzfelix/keras_attention.|||direct_url_found||35078402|10.1186/s12864-022-08310-4|||||regex_fulltext_or_metadata|
|APD (Antimicrobial Peptide Database)|https://aps.unmc.edu/AP/|curated from literature|not specific|direct_url_found|training_and_validation|37914524|10.24272/j.issn.2095-8137.2023.246|more than 3,500 AMPs cataloged|N/A|not described|leave-one-out cross-validation|fulltext|
|DBAASP|http://www.biomedicine.org.ge/dbaasp/|database|not specific|direct_url_found||37914524|10.24272/j.issn.2095-8137.2023.246||N/A|||fulltext|
|LAMP|http://biotechlab.fudan.edu.cn/database/lamp|database|not specific|direct_url_found||37914524|10.24272/j.issn.2095-8137.2023.246||N/A|||fulltext|
|Antifreeze-Peptide-Discovery.|https://github.com/imamabi/Antifreeze-Peptide-Discovery.|||direct_url_found||35576825|10.1016/j.compbiomed.2022.105577|||||regex_fulltext_or_metadata|
|SendongZhao.|https://github.com/SendongZhao.|||direct_url_found||36227057|10.1093/bioinformatics/btac675|||||regex_fulltext_or_metadata|
|AMPSpeciesSpecific dataset|https://github.com/bzlee-bio/AMPSpeciesSpecific|likely included in GitHub repository|AMPSpeciesSpecific|direct_url_found||39766503|10.3390/antibiotics13121113|||||fulltext|
|PepNet Zenodo 1322351661|https://zenodo.org/records/1322351661|Zenodo repository (likely code and data)|PepNet|direct_url_found||39341947|10.1038/s42003-024-06911-1|||||fulltext|
|PepNet Zenodo 1373425862|https://zenodo.org/records/1373425862|Zenodo repository (likely code and data)|PepNet|direct_url_found||39341947|10.1038/s42003-024-06911-1|||||fulltext|
|BPFun dataset|https://github.com/291357657/BPFun|GitHub repository; includes AMP, ACP, ADP, AHP, AIP, AAP, AOP peptides|BPFun|direct_url_found||40691539|10.1186/s12859-025-06190-5|2409 AMPs, etc.|not explicitly reported|CD-HIT at 0.9 threshold||fulltext|
|LLAMP dataset|https://github.com/GIST-CSBL/LLAMP|DBAASP v3, processed; included in GitHub|LLAMP|direct_url_found||40676915|10.1093/bib/bbaf343|~1.7 million peptide-MIC pairs|not applicable (regression)|||fulltext|
|grampa.csv|https://github.com/zswitten/Antimicrobial-Peptides/blob/master/data/grampa.csv|file in Antimicrobial-Peptides repository|LLAMP|direct_url_found||40676915|10.1093/bib/bbaf343|||||fulltext|
|peptides_molecular_fingerprints_classification|https://github.com/scikit-fingerprints/peptides_molecular_fingerprints_classification|||direct_url_found||34037687|10.1093/bib/bbab200|||||regex_fulltext_or_metadata|
|AntiBP3 dataset|https://doi.org/10.5281/zenodo.19911030|Curated from APD3, AntiBP2, dbAMP 2.0, CAMPR3, DRAMP, ABP-Finder|AntiBP3|direct_url_found||38391554|10.3390/antibiotics13020168|GP: 930; GN: 1455; GV: 8985|GP: 1860; GN: 2910; GV: 17970|non-redundant ABPs; non-ABPs randomly selected from Swiss-Prot/UniProt excluding ABPs||repository|
|zenodo.5347031|https://doi.org/10.5281/zenodo.5347031|||direct_url_found||40410382|10.5281/zenodo.5347031|||||regex_fulltext_or_metadata|
|models?filter=beit|https://huggingface.co/models?filter=beit|||direct_url_found||40410382|10.5281/zenodo.5347031|||||regex_fulltext_or_metadata|
|models?filter=layoutlmv2|https://huggingface.co/models?filter=layoutlmv2|||direct_url_found||40410382|10.5281/zenodo.5347031|||||regex_fulltext_or_metadata|
|5347031|https://zenodo.org/record/5347031|||direct_url_found||40410382|10.5281/zenodo.5347031|||||regex_fulltext_or_metadata|
|AI4AFP_AFP_dataset|not_reported|CAMP, DRAMP, YADAMP, SATPdb, DBAASP; UniProtKB/Swiss-Prot|AI4AFP|source_database_named|training|35724626|10.1093/bib/bbac226|3011 AFPs|3011 (half random, half UniProt non-AMPs)|CD-HIT at 0.95 identity||fulltext|
|||||||||1237|not specified||||
|||||||||3011|3011||||
|||||||||8283/7582/5621 (per bacteria)|N/A||||
|||||||||7|1||||
|Co-AMPpred dataset (DEEP-AmPEP30 derived)|||||||||||||
|AI4AFP dataset|||||||||||||
|ANIA training set|||||||||||||
|Nerita versicolor & Pomacea poeyana AMP candidates|||||||||||||
|Collagen-derived AMP set|||||||||||||
|zenodo.19462601|https://doi.org/10.5281/zenodo.19462601|DOI||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|Urchin|https://github.com/VirtualBrainLab/Urchin|GitHub||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|allenCCF|https://github.com/cortex-lab/allenCCF|GitHub||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|phy|https://github.com/cortex-lab/phy|GitHub||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|iblapps|https://github.com/int-brain-lab/iblapps|GitHub||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|atlaselectrophysiology|https://github.com/int-brain-lab/iblapps/tree/master/atlaselectrophysiology|GitHub||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|Lab|https://github.com/tortugar/Lab|GitHub||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|PySleep|https://github.com/tortugar/Lab/tree/master/PySleep|GitHub||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|Npx.|https://github.com/tortugar/Npx.|GitHub||direct_url_found||40233747|10.1016/j.neuron.2025.03.020|||||repository|
|APD3 + UniProt balanced dataset|https://github.com/aretiz/amp_de_novo_design_cdGAN|APD3 for AMPs, reviewed UniProt for non-AMPs, clustered with MMseqs2 at 50% identity|cdGAN|direct_url_found|training|41137855|10.1093/bib/bbaf500|2600 AMPs (10-50 residues)|2600 non-AMPs (10-50 residues)|MMseqs2 clustering at 50% sequence identity for non-AMPs; average pairwise similarity 0.2|not explicitly described (likely used for GAN training)|fulltext|
|PRJNA600247|https://www.ncbi.nlm.nih.gov/bioproject/PRJNA600247|NCBI BioProject|Macrel, AxPEP, AMP Scanner V2|source_database_named|external_test|41315055|10.1007/s00248-025-02620-2|||||fulltext|
|PRJNA646512|https://www.ncbi.nlm.nih.gov/bioproject/PRJNA646512|NCBI BioProject|Macrel, AxPEP, AMP Scanner V2|source_database_named|external_test|41315055|10.1007/s00248-025-02620-2|||||fulltext|
|DBAASP-derived AMP activity dataset (MRSA focus)|https://github.com/xubocheng/Anti_Cp.git|DBAASP database|WeightedEnsemble_L3|direct_url_found|training|38266820|10.1016/j.jare.2024.01.023|high activity (MIC <= 32 μg/ml)|low activity (32 < MIC <= 128 μg/ml) and no activity (MIC > 128 μg/ml)|not_reported_in_available_evidence|80% train, 20% test|fulltext|
|dataset_for_|https://github.com/zswitten/Antimicrobial|||direct_url_found||42124643|10.64898/2026.02.21.707214|||||chunk_summary|
|ConoServer|https://www.conoserver.org/|ConoServer|APEX|direct_url_found|external_test|39764027|10.1101/2024.12.17.628923|Conopeptides (not all AMPs)|not applicable||not applicable|fulltext|
|ArachnoServer|https://arachnoserver.qfab.org/mainMenu.html|ArachnoServer|APEX|direct_url_found|external_test|39764027|10.1101/2024.12.17.628923|Spider proteins|not applicable||not applicable|fulltext|
|ISOB|https://www.snakebd.com/|ISOB (Indigenous Snake Proteins)|APEX|direct_url_found|external_test|39764027|10.1101/2024.12.17.628923|Snake proteins|not applicable||not applicable|fulltext|
|VenomZone|https://venomzone.expasy.org/|VenomZone (UniProtKB)|APEX|direct_url_found|external_test|39764027|10.1101/2024.12.17.628923|Venom proteins from six taxa|not applicable||not applicable|fulltext|
|FESNov antimicrobial peptide families|https://novelfams.cgmlab.org|Nature 2023 paper (doi:10.1038/s41586-023-06955-z)||direct_url_found|benchmark|38109938|10.1038/s41586-023-06955-z|240 FESNov gene families with antimicrobial signatures||not_reported_in_available_evidence|not_reported_in_available_evidence|fulltext|
|nov-fams-pipeline|https://github.com/AlvaroRodriguezDelRio/nov-fams-pipeline.|GitHub repository linked to Nature 2023 paper||direct_url_found||38109938|10.1038/s41586-023-06955-z|||not_reported_in_available_evidence|not_reported_in_available_evidence|repository|
|aro|https://github.com/arpcard/aro|GitHub repository linked to Nature 2023 paper||direct_url_found||38109938|10.1038/s41586-023-06955-z|||not_reported_in_available_evidence|not_reported_in_available_evidence|repository|
|FMT donor fecal AMP candidates|https://github.com/pointwei/FMT-MetagenomicData|Fecal metagenomes from 120 FMT donors|c_AMPs-prediction|direct_url_found|experimental_validation|41164228|10.3389/fvets.2025.1689589|2,820,488 potential AMPs predicted||redundancy removed by Perl script, details not specified||fulltext|
|AMOR biofilm AMPs dataset|https://github.com/trongthucnguyen/DeepSeaQuence_biofilms|Arctic deep-sea hydrothermal vent biofilm metagenomes||direct_url_found||42104260|10.1186/s12866-026-05098-1|961 predicted AMPs (873 unique)||CD-HIT||fulltext|

## Model-Dataset Links

|model_name|dataset_name|dataset_role|dataset_source|dataset_url|dataset_status|source_pmid|source_doi|positive_samples|negative_samples|deduplication_method|split_method|needs_followup|evidence_level|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Co-AMPpred|https://github.com/onkarS23/CoAMPpred (contains training and test data from DEEP-AmPEP30)|training_or_benchmark_unspecified|https://github.com/onkarS23/CoAMPpred (contains training and test data from DEEP-AmPEP30)|https://github.com/onkarS23/CoAMPpred (contains training and test data from DEEP-AmPEP30)|direct_url_found|34330209|10.1186/s12859-021-04305-2|||||False|fulltext|
|CTCM-Neo & ConformaX-PEP framework|not_reported_in_available_evidence (likely derived from APD3)|training_or_benchmark_unspecified|not_reported_in_available_evidence (likely derived from APD3)||described_no_link|41859462|10.3389/fcimb.2026.1707267|||||True|abstract|
|A-CaMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|31870207|10.1080/07391102.2019.1708796|||||True|fulltext|
|PCSPred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40781463|10.1109/NEleX59773.2023.10421222|||||True|abstract|
|iAMPCN|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|39330266|10.3390/md22090385|49,115 experimentally validated AMPs|195,525 UniProt sequences (filtered by CD-HIT at 40% identity to positives)|CD-HIT at 40% pairwise identity for negative dataset, removed sequences with non-standard residues||True|fulltext|
|SSFGM-Model|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40462515|10.1186/s12864-020-06978-0|||||True|abstract|
|ACEP|APD database (mentioned in fulltext)|training_or_benchmark_unspecified|APD database (mentioned in fulltext)||described_no_link|40462515|10.1186/s12864-020-06978-0|||||True|fulltext|
|ACP-DL|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34880291|10.1038/s41598-021-02703-3|||||True|repository|
|MultiPep|multiple public databases (not specified in abstract)|training_or_benchmark_unspecified|multiple public databases (not specified in abstract)||described_no_link|34909478|10.1093/biomethods/bpab021|||||True|abstract|
|iAMP-2L|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35305010|10.1093/database/baab012|||||True|review|
|iAMPred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35305010|10.1093/database/baab012|||||True|review|
|AmPEP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|35305010|10.1093/database/baab012|3268 AMPs|166791 non-AMPs|CD-HIT at 90% identity|10-fold cross-validation (1:3 positive:negative ratio)|True|review|
|AntiBP2|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|35305010|10.1093/database/baab012|||||True|review|
|CAMPR3|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|35305010|10.1093/database/baab012|||||True|review|
|ADAM|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35305010|10.1093/database/baab012|||||True|review|
|DBAASP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35305010|10.1093/database/baab012|||||True|review|
|MLAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35305010|10.1093/database/baab012|||||True|review|
|CAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|35305010|10.1093/database/baab012|||||True|review|
|ClassAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35305010|10.1093/database/baab012|||||True|review|
|AVPpred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35305010|10.1093/database/baab012|||||True|review|
|AMPER|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|35305010|10.1093/database/baab012|||||True|review|
|EFC-FCBF|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35305010|10.1093/database/baab012|||||True|review|
|AMPlify|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|39557756|10.1007/s12602-024-10402-4|Known AMPs from databases|Non-AMP sequences (details in paper)|||True|search_result|
|E-CLEAP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|https://github.com/Wangsicheng52/E-CLEAP|described_no_link|39557756|10.1007/s12602-024-10402-4|1750|1750|ensured no duplicates between positive and negative sets, and within each set (method not specified)||True|review|
|UniproLcad|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39557756|10.1007/s12602-024-10402-4|||||True|review|
|TriStack|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39557756|10.1007/s12602-024-10402-4|||||True|review|
|iAMP-DL|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39557756|10.1007/s12602-024-10402-4|||||True|review|
|amp-gan|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39557756|10.1007/s12602-024-10402-4|||||True|review|
|AVPIden|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39557756|10.1007/s12602-024-10402-4|||||True|review|
|antibp|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39557756|10.1007/s12602-024-10402-4|||||True|review|
|ampsphere|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39557756|10.1007/s12602-024-10402-4|||||True|review|
|hydramp|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39557756|10.1007/s12602-024-10402-4|||||True|review|
|AMPDiscover|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34081438|10.1021/acs.jcim.1c00251|||||True|abstract|
|ESM2-AFPpred|DRAMP and APD3 databases (no direct download link provided in evidence)|training_or_benchmark_unspecified|DRAMP and APD3 databases (no direct download link provided in evidence)|not_reported_in_available_evidence|described_no_link|35724626|10.1093/bib/bbac226|1237 anti-Candida peptides|not specified|CD-HIT or similar (mentioned in text but not detailed)||True|fulltext|
|ANIA|DBAASP, dbAMP, DRAMP|training_or_benchmark_unspecified|DBAASP, dbAMP, DRAMP|not_reported_in_available_evidence|described_no_link|41664908|10.1093/bib/bbag023|8283 (S. aureus), 7582 (E. coli), 5621 (P. aeruginosa) AMPs with MIC values|N/A (regression task)|not_reported_in_available_evidence||True|fulltext|
|AI4AFP|CAMP, DRAMP, YADAMP, SATPdb, DBAASP (AFPs); UniProtKB/Swiss-Prot (non-AMPs); DBAASP (hemolysis data)|training_or_benchmark_unspecified|CAMP, DRAMP, YADAMP, SATPdb, DBAASP (AFPs); UniProtKB/Swiss-Prot (non-AMPs); DBAASP (hemolysis data)|not_reported_in_available_evidence|described_no_link|42146199|10.1021/acsomega.6c00049|3011 antifungal peptides|3011 (half random, half UniProt non-AMPs)|CD-HIT at 0.95 identity||True|fulltext|
|not_applicable|Collagen_derived_AMP_activity|training_or_benchmark_unspecified|in-house experimental data|not_reported_in_available_evidence|direct_url_found|41528266|10.1021/acs.jnatprod.5c01318|7 synthesized peptides (REI-26, LEL-28, TRR-26, LRS-21, SPE-22, GPE-19, GFD-30) plus controls|GEK-25 (predicted non-AMP)|N/A||False|fulltext|
|AI4AMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34783578|10.1128/msystems.00299-21|||||True|fulltext|
|Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|27870247|10.1002/minf.201600029|||||True|abstract|
|SAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|https://figshare.com/projects/Tabula_Muris_Transcriptomic_characterization_of_20_organs_and_tissues_from_Mus_musculus_at_single_cell_resolution/27733.|described_no_link|39573886|10.1093/bfgp/elae046|||||True|fulltext|
|DL-QSARES|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39921483|10.1002/advs.202412488|||||True|abstract|
|AI4AVP|https://github.com/LinTzuTang/AI4AVP_predictor (datasets from APD3, DRAMP, YADAMP, DBAASP, CAMP, AVPdb, UniProt/SwissProt)|training_or_benchmark_unspecified|https://github.com/LinTzuTang/AI4AVP_predictor (datasets from APD3, DRAMP, YADAMP, DBAASP, CAMP, AVPdb, UniProt/SwissProt)|https://github.com/LinTzuTang/AI4AVP_predictor (datasets from APD3, DRAMP, YADAMP, DBAASP, CAMP, AVPdb, UniProt/SwissProt)|direct_url_found|37626205|10.1109/JBHI.2021.3130825|2934 AVPs|17184 non-AVP peptides (Swiss-Prot + random)|CD-HIT at 95% identity||False|fulltext|
|PepForge|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39705302|10.64898/2026.05.29.728379|||||True|abstract|
|Al-Omari 2024 AMP prediction model|https://dbaasp.org|training_or_benchmark_unspecified|https://dbaasp.org|https://dbaasp.org|direct_url_found|39705302|10.1371/journal.pone.0315477|1360 peptides with anti-E. coli activity|not explicitly mentioned|||False|fulltext|
|BBATProt|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41212592|10.1093/bib/bbaf593|||||True|fulltext|
|AMAP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|30831306|10.1016/j.compbiomed.2019.02.018|||||True|abstract|
|AMP|https://github.com/researchprotein/amp|training_or_benchmark_unspecified|https://github.com/researchprotein/amp|https://github.com/researchprotein/amp|direct_url_found|38972032|10.1007/s12539-024-00640-z|not specified|not specified|not specified||False|abstract|
|Deep-AmPEP30|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|32464552|10.1109/INDCON.2011.6139332|||||True|abstract|
|EBAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40906555|10.1016/j.celrep.2025.116215|||||True|abstract|
|DLFea4AMPGen|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41093853|10.1002/adma.202307680|||||True|abstract|
|AMP-BERT|https://github.com/GIST-CSBL/AMP-BERT.|training_or_benchmark_unspecified|https://github.com/GIST-CSBL/AMP-BERT.|https://github.com/GIST-CSBL/AMP-BERT.|direct_url_found|36461699|10.1002/pro.4529|||||False|fulltext|
|COMDEL|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39234615|10.1016/j.apsb.2024.05.003|||||True|fulltext|
|C. acnes-targeted AMP generation pipeline (activity classifier)|https://dbaasp.org/|training_or_benchmark_unspecified|https://dbaasp.org/|https://dbaasp.org/|direct_url_found|38402320|10.1038/s41598-024-55205-3|||||False|fulltext|
|BERT-based AMP recognition model|Six AMP datasets (not specified in abstract) and a new constructed AMP dataset|training_or_benchmark_unspecified|Six AMP datasets (not specified in abstract) and a new constructed AMP dataset||described_no_link|34037687|10.1093/bib/bbab200|||||True|abstract|
|Dong2024_AMP_activity_classifier|DBAASP|training_or_benchmark_unspecified|DBAASP database|https://dbaasp.org/|direct_url_found|38402320|10.1038/s41598-024-55205-3|8884 active AMPs|4009 inactive + 4875 pseudo-inactive peptides|||False|fulltext|
|Zhang2021_BERT_AMP|UniProt and six AMP datasets|training_or_benchmark_unspecified|UniProt and multiple AMP datasets|not_reported_in_available_evidence|direct_url_found|34037687|10.1093/bib/bbab200|||||False|abstract|
|AmpGPT2|COMPASS database (https://compass.imi.uni-muenster.de)|training_or_benchmark_unspecified|COMPASS database (https://compass.imi.uni-muenster.de)|https://compass.imi.uni-muenster.de/data.json|direct_url_found|42174216|10.1038/s44259-026-00218-3|75,381 unique AMP sequences|not_applicable|deduplication performed across the 9 databases, resulting in 75,381 unique sequences||False|fulltext|
|AMP-CapsNet|derived from UniProt and previous study [31]; positive: 1085 AMPs, negative: 1316 non-AMPs|training_or_benchmark_unspecified|derived from UniProt and previous study [31]; positive: 1085 AMPs, negative: 1316 non-AMPs|not_reported_in_available_evidence|described_no_link|41654884|10.1186/s44342-026-00067-6|1085|1316|duplicate peptide sequences removed||True|fulltext|
|deepAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41753681|10.3390/microorganisms14020394|||||True|fulltext|
|AMP-RL|PeptideAtlas, DBAASP v3 (no direct links provided)|training_or_benchmark_unspecified|PeptideAtlas, DBAASP v3 (no direct links provided)|not_reported_in_available_evidence|described_no_link|37992451|10.1016/j.sbi.2023.102733|1,725,301 unique peptides|not_applicable|||True|fulltext|
|PepCVAE|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37992451|10.1016/j.sbi.2023.102733|||||True|review|
|PrefixProt|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37992451|10.1016/j.sbi.2023.102733|||||True|review|
|MoFormer|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37992451|10.1016/j.sbi.2023.102733|||||True|review|
|HMAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37992451|10.1016/j.sbi.2023.102733|||||True|review|
|AMP-Designer|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37992451|10.1016/j.sbi.2023.102733|||||True|review|
|AMP-MIC|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|29679519|10.1002/cmdc.201800204|630,683 peptides|N/A (unsupervised pretraining)|not_reported_in_available_evidence||True|fulltext|
|AP_Sin|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38416364|10.1038/s41467-018-03746-3|||||True|fulltext|
|AMP-Detector|Peptide Atlas (used for discovery)|training_or_benchmark_unspecified|Peptide Atlas (used for discovery)||described_no_link|39201537|10.3389/fmicb.2018.00323|||||True|fulltext|
|AMP-RNNpro|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38839785|10.1016/j.csbj.2022.07.043|||||True|fulltext|
|not directly linked|ADAM (Antimicrobial Peptide Database)|training_or_benchmark_unspecified|mentioned in fulltext of PMID 38839785 (extracted from MLACP 2.0 context, but not directly linked to AMP-RNNpro)|https://bioinformatics.cs.ntou.edu.tw/ADAM|direct_url_found|38839785|10.1016/j.csbj.2022.07.043|not_specified|not_specified|not_specified||False|fulltext|
|AMP-Distillation|APD3 and DADP databases, CD-HIT deduplication|training_or_benchmark_unspecified|APD3 and DADP databases, CD-HIT deduplication||described_no_link|42155201|10.1016/j.compbiolchem.2026.109129|||||True|abstract|
|iAMP-SeE|Dataset 1: DRAMP, dbAMP, CAMPr-4, AMPfun, ADAPTABLE (positive), UniProt (negative); Dataset 2: from deep-AMPpred (Zhao et al. 2024); Zenodo data: https://doi.org/10.5281/zenodo.17398951|training_or_benchmark_unspecified|Dataset 1: DRAMP, dbAMP, CAMPr-4, AMPfun, ADAPTABLE (positive), UniProt (negative); Dataset 2: from deep-AMPpred (Zhao et al. 2024); Zenodo data: https://doi.org/10.5281/zenodo.17398951|https://doi.org/10.5281/zenodo.17398951|direct_url_found|41913931|10.7717/peerj.20978|16,200 AMP sequences from DRAMP, dbAMP, CAMPr-4, AMPfun, ADAPTABLE|16,200 non-AMP sequences from UniProt|CD-HIT at 100% identity||False|fulltext|
|STAMP|Used three benchmark datasets including two previously published and a new curated dataset from DBAASP|training_or_benchmark_unspecified|Used three benchmark datasets including two previously published and a new curated dataset from DBAASP||described_no_link|42155201|10.64898/2026.05.28.728246|||||True|abstract|
|CF-AMP prediction|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|42020672|10.1101/2022.11.16.516845|||||True|abstract|
|AMP-DualTransnet|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|42020672|10.1016/j.nexres.2026.101536|||||True|abstract|
|AMP-FreqNet|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link||10.1145/3766671.3766835|||||True|metadata|
|Collaborative Filtering and Link Prediction model|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link||10.1021/acs.jcim.3c00137|||||True|metadata|
|Predictive and Interpretable ML Models|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link||10.1021/acsomega.3c08676.s001|||||True|metadata|
|AMP prediction ML model|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link||10.54985/peeref.2405p7278831|||||True|metadata|
|GAC-BiTCNN-AMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41844874|10.1038/s41598-026-43370-6|||||True|fulltext|
|CVAE-BIO|APD3 (http://aps.unmc.edu/)|training_or_benchmark_unspecified|APD3 (http://aps.unmc.edu/)|http://aps.unmc.edu/|direct_url_found|41849223|10.1093/bib/bbag115|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||False|fulltext|
|AMPGAN|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41463765|10.3390/antibiotics14121263|||||True|review|
|Macrel|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41463765|10.3390/antibiotics14121263|||||True|review|
|iAMPpred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41463765|10.3390/antibiotics14121263|||||True|review|
|AMP-GPT|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40193623|10.1038/s44386-026-00045-6|||||True|fulltext|
|MCL-AMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40193623|10.1038/s44386-026-00045-6|||||True|fulltext|
|MAPLE|Benchmark dataset: integrated from dbAMP, DBAASP, APD3, DRAMP, etc. (no single download link); 25,507 AMPs and 72,606 non-AMPs. Independent validation set: 24,582 AMPs, 36,653 non-AMPs.|training_or_benchmark_unspecified|Benchmark dataset: integrated from dbAMP, DBAASP, APD3, DRAMP, etc. (no single download link); 25,507 AMPs and 72,606 non-AMPs. Independent validation set: 24,582 AMPs, 36,653 non-AMPs.|not_reported_in_available_evidence|described_no_link|39792442|10.1021/acs.jcim.4c01913|25507 AMPs|72606 non-AMPs|CD-HIT used to remove duplicates and control redundancy (details in paper)||True|fulltext|
|PepVAE|https://github.com/zswitten/Antimicrobial-Peptides|training_or_benchmark_unspecified|https://github.com/zswitten/Antimicrobial-Peptides|https://github.com/zswitten/Antimicrobial-Peptides|direct_url_found|34659152|10.3389/fmicb.2021.725727|6,760 unique AMP sequences (original), 3,280 for E. coli after filtering|Not explicitly mentioned; MIC values used as regression target|Not described||False|fulltext|
|LMPred|https://github.com/williamdee1/LMPred_AMP_Prediction|training_or_benchmark_unspecified|https://github.com/williamdee1/LMPred_AMP_Prediction|https://github.com/williamdee1/LMPred_AMP_Prediction|direct_url_found|36699381|10.1101/2020.07.12.199554v3|N/A|N/A|Not described||False|fulltext|
|AMP prediction SVM-LZ|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|25802839|10.1093/nar/gkn823|||||True|abstract|
|DDM|https://github.com/kww567upup/DDM (data provided in repository)|training_or_benchmark_unspecified|https://github.com/kww567upup/DDM (data provided in repository)|https://github.com/kww567upup/DDM (data provided in repository)|direct_url_found|41692989|10.1093/bioinformatics/btag077|||||False|fulltext|
|UniAMP|not_reported_in_available_evidence (dataset constructed from public AMP databases, no direct download link)|training_or_benchmark_unspecified|not_reported_in_available_evidence (dataset constructed from public AMP databases, no direct download link)||described_no_link|39799358|10.1186/s12859-025-06033-3|||||True|fulltext|
|not_reported_in_available_evidence|esm-AxP-GDL|training_or_benchmark_unspecified|GitHub repository|https://github.com/cicese-biocom/esm-AxP-GDL|direct_url_found|41594075|10.3390/antibiotics15010039|40,251 non-toxic dual-activity peptides|not_reported_in_available_evidence|not_reported_in_available_evidence||False|fulltext|
|AMP Scanner|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38129980|10.1002/mbo3.1393|||||True|review|
|AMPScanner vr.2|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37851665|10.1371/journal.pone.0292947|||||True|fulltext|
|PepGen 1.0|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40643674|10.1007/s00284-025-04346-3|||||True|fulltext|
|AmPepGen|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40643674|10.1007/s00284-025-04346-3|||||True|fulltext|
|AMP-SEMiner|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40445833|10.1016/j.celrep.2025.115773|||||True|fulltext|
|Unnamed AMP predictor from DRAMP 2.0|DRAMP database (http://dramp.cpu-bioinfor.org/)|training_or_benchmark_unspecified|DRAMP database (http://dramp.cpu-bioinfor.org/)|http://dramp.cpu-bioinfor.org/|direct_url_found|31409791|10.1038/s41597-019-0154-y|19,899 AMP entries (5,084 general, 14,739 patent, 76 clinical)|Not explicitly described; the database contains only AMPs|Compared with APD and CAMP; 70.56% non-overlapping sequences||False|fulltext|
|AMP toxicity prediction model (hybrid)|DBAASP database|training_or_benchmark_unspecified|DBAASP database|https://dbaasp.org/|described_no_link|34758751|10.1186/s12859-021-04468-y|Toxic AMPs (based on HC50, CC50, MIC thresholds)|Non-toxic AMPs|Records with unnatural residues or D-amino acids removed; concentration units converted||True|fulltext|
|CalcAMP|https://doi.org/10.5281/zenodo.7588702|training_or_benchmark_unspecified|https://doi.org/10.5281/zenodo.7588702|https://doi.org/10.5281/zenodo.7588702|direct_url_found|37107088|10.3390/antibiotics12040725|AMPs with activity below threshold; numbers not explicitly given|Confirmed inactive peptides (Non-AMPs) above threshold|Not described||False|fulltext|
|ANN-based AMP prediction model (Torrent et al. 2011)|CAMP database (http://www.camp.bicnirrh.res.in/) and Uniprot; no direct download link provided|training_or_benchmark_unspecified|CAMP database (http://www.camp.bicnirrh.res.in/) and Uniprot; no direct download link provided||direct_url_found|21347392|10.1371/journal.pone.0016968|||||False|fulltext|
|Deep learning regression model for antimicrobial peptide design (Witten & Witten 2019)|GRAMPA database; not directly linked but likely included in the GitHub repository|training_or_benchmark_unspecified|GRAMPA database; not directly linked but likely included in the GitHub repository||described_no_link|21347392|10.1101/692681|||||True|abstract|
|AMP-zGSM|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|21347392|10.5220/0014457300004070|total 3145 peptides|not specified|not_reported_in_available_evidence||True|abstract|
|Torrent-2011-ANN|CAMP (Collection of Anti-Microbial Peptides)|training_or_benchmark_unspecified|http://www.camp.bicnirrh.res.in/|not_reported_in_available_evidence|direct_url_found|21347392|10.1371/journal.pone.0016968|1157 AMPs|991 non-AMPs (from Uniprot)|not_reported_in_available_evidence||False|fulltext|
|Witten-2019-CNN|GRAMPA (Giant Repository of AMP Activities)|training_or_benchmark_unspecified|Possibly included in the GitHub repository|not_reported_in_available_evidence|direct_url_found|21347392|10.1101/692681|||not_reported_in_available_evidence||False|abstract|
|AMP0|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|32750857|10.1109/TCBB.2020.2999399|||||True|abstract|
|sAMPpred-GAT|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|https://github.com/HongWuL/sAMPpred-GAT/ (likely includes datasets)|described_no_link|36342186|10.1093/bioinformatics/btac715|||||True|abstract|
|PyAMPA|AMPlify dataset, Liu et al. CPP database, AMPDeep hemolytic database, ToxinPred toxicity database, GRAMPA database (https://github.com/zswitten/Antimicrobial-Peptides)|training_or_benchmark_unspecified|AMPlify dataset, Liu et al. CPP database, AMPDeep hemolytic database, ToxinPred toxicity database, GRAMPA database (https://github.com/zswitten/Antimicrobial-Peptides)|not_reported_in_available_evidence|direct_url_found|38934543|10.1128/msystems.01358-23|3338 AMPs|3338 non-AMPs|not_reported_in_available_evidence||False|fulltext|
|AMPA|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40410382|10.1038/s44320-025-00120-6|||||True|fulltext|
|AntiBP3|not_reported_in_available_evidence (training data compiled from public databases, no direct download link provided)|training_or_benchmark_unspecified|not_reported_in_available_evidence (training data compiled from public databases, no direct download link provided)|https://doi.org/10.5281/zenodo.19911030|described_no_link|38391554|10.3390/antibiotics13020168|GP: 930; GN: 1455; GV: 8985|GP: 1860; GN: 2910; GV: 17970|non-redundant ABPs; non-ABPs randomly selected from Swiss-Prot/UniProt excluding ABPs||True|fulltext|
|AMPActiPred|not_reported_in_available_evidence (elaborate dataset constructed from public sources, no direct download link)|training_or_benchmark_unspecified|not_reported_in_available_evidence (elaborate dataset constructed from public sources, no direct download link)||described_no_link|38723168|10.1002/pro.5006|||||True|fulltext|
|APEX|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|39754551|10.1111/1751-7915.70072|19,762 known AMPs|not specified||not applicable|True|mixed|
|AMPfinder|dbAMP database|training_or_benchmark_unspecified|dbAMP database||described_no_link|39540425|10.1093/nar/gkae1019|||||True|fulltext|
|AMPpredictor|dbAMP database|training_or_benchmark_unspecified|dbAMP database||described_no_link|39540425|10.1093/nar/gkae1019|||||True|fulltext|
|AMPfinder, AMPpredictor, AMPActiPred|dbAMP 3.0|training_or_benchmark_unspecified|dbAMP database|https://awi.cuhk.edu.cn/dbAMP/|direct_url_found|39540425|10.1093/nar/gkae1019|33,065 AMPs, 2,453 antimicrobial proteins||||False|fulltext|
|AMPBAN|https://github.com/baiwenhuim/ampban (dataset in repository)|training_or_benchmark_unspecified|https://github.com/baiwenhuim/ampban (dataset in repository)|https://github.com/baiwenhuim/ampban (dataset in repository)|direct_url_found||10.64898/2026.01.20.700468|||||False|abstract|
|Generative AMP pipeline (VINCI)|AMPSphere, DBAASP (links not provided)|training_or_benchmark_unspecified|AMPSphere, DBAASP (links not provided)||described_no_link||10.64898/2026.06.16.732639|||||True|abstract|
|AMPBenchmark|single positive data set and 11 negative data sampling methods|training_or_benchmark_unspecified|single positive data set and 11 negative data sampling methods|not_reported_in_available_evidence|direct_url_found|38416364|10.1101/2022.05.30.493946||not_reported (11 sampling methods)|||False|abstract|
|AMPCLGPT|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link||10.1101/2025.03.07.642021|||||True|abstract|
|CAmidPred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link||10.21203/rs.3.rs-7764304/v1|||||True|abstract|
|PepMCP|MemAMPdb (described in paper, no explicit link)|training_or_benchmark_unspecified|MemAMPdb (described in paper, no explicit link)|not_reported_in_available_evidence|described_no_link||10.64898/2026.02.01.703163|>500 membrane-lytic AMPs||||True|abstract|
|iMFP-LG|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|39585308|10.1093/gpbjnl/qzae084|not specified in available evidence|not specified in available evidence|not reported in available evidence||True|fulltext|
|Deep learning model for AMP discovery from ruminant gastrointestinal microbiomes|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|39756573|10.1016/j.jare.2025.01.005|27,192 potential secretory AMP candidates identified; 39 synthesized|not specified|||True|abstract|
|Deep learning model for AMP discovery from protist genomes (BERT+CNN)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|40958742|10.1021/acs.jcim.5c01196|3133 novel candidate AMPs identified|not specified|||True|abstract|
|not_applicable (existing ML models used)|Centenarian gut microbiome metagenomic data|training_or_benchmark_unspecified|Healthy individuals: centenarians (n=20), older adults (n=15), young (n=15)|not_reported_in_available_evidence|direct_url_found|39207726|10.1093/gerona/glae218|Potential AMPs identified in gut microbiome|not specified|||False|abstract|
|amPEPpy|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|33135060|10.1093/bioinformatics/btaa917|||||True|abstract|
|panCleave|Training and test data (MEROPS substrates) available in the panCleave repository (https://gitlab.com/machine-biology-group-public/pancleave)|training_or_benchmark_unspecified|Training and test data (MEROPS substrates) available in the panCleave repository (https://gitlab.com/machine-biology-group-public/pancleave)||direct_url_found|37516110|10.1016/j.chom.2023.07.001|||||False|fulltext|
|Bacteria-specific ML models for E. coli AMP activity|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|36912047|10.1021/acs.jcim.2c01551|||||True|abstract|
|XGBoost AMP prediction model (Bhangu2025)|http://cabgrid.res.in:8080/amppred/about.html (and other AMP databases)|training_or_benchmark_unspecified|http://cabgrid.res.in:8080/amppred/about.html (and other AMP databases)|http://cabgrid.res.in:8080/amppred/about.html (and other AMP databases)|direct_url_found|40529865|10.1002/smsc.202400579|984 AMPs|984 non-AMPs|||False|fulltext|
|Multiple DL models reviewed (e.g., AMP-BERT, Deep-AmPEP30, etc.)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|36290108|10.3390/antibiotics11101451|||||True|review|
|not specified|BMXC7|training_or_benchmark_unspecified|OSF|https://doi.org/10.17605/OSF.IO/BMXC7|direct_url_found|29889579|10.1080/14787210.2018.1483720|not specified|not specified|||False|metadata|
|not applicable (benchmark dataset)|Zenodo dataset for peptide benchmark|training_or_benchmark_unspecified|Zenodo|https://doi.org/10.5281/zenodo.19388783|direct_url_found|33774670|10.1093/bib/bbab083|not specified|not specified|||False|fulltext|
|StarPep tool|StarPep|training_or_benchmark_unspecified|Integrated from 42 databases|http://mobiosd-hub.com/starpep/|direct_url_found|39858924|10.3390/microorganisms13010156|over 22,600 AMPs||||False|review|
|AMPGAN v3|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|https://github.com/marszzibros/AMPGANv3|described_no_link|42364293|10.1016/j.jmgm.2026.109497|||||True|abstract|
|PepAnno|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|42228741|10.1371/journal.pcbi.1014369|||||True|abstract|
|AMPGP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40825014|10.1021/acs.jcim.5c00647|||||True|abstract|
|AmpGram|Training data not detailed; benchmarked on APD3 and DAMPD datasets|training_or_benchmark_unspecified|Training data not detailed; benchmarked on APD3 and DAMPD datasets|not_reported_in_available_evidence|described_no_link|32560350|10.3390/ijms21124310|||||True|fulltext|
|AMPScanner V2|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|38877295|10.1002/2211-5463.13847|||||True|review|
|ampir|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38877295|10.1002/2211-5463.13847|||||True|review|
|Ensemble-AMPPred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38877295|10.1002/2211-5463.13847|||||True|review|
|CancerGram|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38877295|10.1002/2211-5463.13847|||||True|review|
|PPTPP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38877295|10.1002/2211-5463.13847|||||True|review|
|MLBP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38877295|10.1002/2211-5463.13847|||||True|review|
|Deep2Pep|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38877295|10.1002/2211-5463.13847|||||True|review|
|Pore-Forming_AMP_SVM|https://github.com/ComputBiophys/Pore%E2%80%90Forming_AMP_SVM (training data included)|training_or_benchmark_unspecified|https://github.com/ComputBiophys/Pore%E2%80%90Forming_AMP_SVM (training data included)|https://github.com/ComputBiophys/Pore%E2%80%90Forming_AMP_SVM (training data included)|direct_url_found|41391039|10.1002/advs.202516470|||||False|fulltext|
|CG-AMP|AMPlify and DAMP benchmark datasets|training_or_benchmark_unspecified|AMPlify and DAMP benchmark datasets|not_reported_in_available_evidence|described_no_link|41286313|10.1038/s41598-025-29666-z|||||True|fulltext|
|AmpHGT|XUAMP, AMPDiscover, NCAA datasets|training_or_benchmark_unspecified|XUAMP, AMPDiscover, NCAA datasets||described_no_link|40598389|10.1186/s12915-025-02253-4|||||True|fulltext|
|SGAC|not_reported_in_available_evidence (paper states 'publicly available AMP and non-AMP datasets')|training_or_benchmark_unspecified|not_reported_in_available_evidence (paper states 'publicly available AMP and non-AMP datasets')||described_no_link|41662353|10.1093/bib/bbag038|||||True|fulltext|
|TP-LMMSG|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41978380|10.1093/bib/bbag107|||||True|review|
|PGAT-ABPp|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41755839|10.1021/jacsau.5c01520|||||True|review|
|Bidirectional LSTM AMP classification model (Wang2021)|CAMP, DBAASP, DRAMP, YADAMP, UniProt (as described in Methods)|training_or_benchmark_unspecified|CAMP, DBAASP, DRAMP, YADAMP, UniProt (as described in Methods)|not_reported_in_available_evidence|described_no_link|33810011|10.3390/biom11030471|AMPs with MIC ≤100 µM against E. coli, length ≤20, after deduplication|Random UniProt sequences without AMP keywords, and AMP database sequences with MIC >100 µM against E. coli|Duplicate sequences from multiple databases included only once||True|fulltext|
|AMPScanner, CAMP (used for prediction)|Equine milk lactoferrin peptides (56 unique peptides)|training_or_benchmark_unspecified|LC-MS/MS of tryptic hydrolyzate of equine milk lactoferrin; in vitro antimicrobial activity tested against E. coli, S. aureus, P. aeruginosa.|not_reported_in_available_evidence|direct_url_found|42071989|10.3390/ani16081223|not explicitly reported (some peptides showed antimicrobial activity in vitro)|not explicitly reported|novelty check against APD3, DBAASP, and BLASTp||False|fulltext|
|PrMFTP|not_reported_in_available_evidence (constructed from 22 therapeutic peptide datasets; no direct download link provided in evidence)|training_or_benchmark_unspecified|not_reported_in_available_evidence (constructed from 22 therapeutic peptide datasets; no direct download link provided in evidence)|not_reported_in_available_evidence|described_no_link|36094961|10.1371/journal.pcbi.1010511|ABP: 2154, ACP: 861, etc. (see Table 1), total peptides with labels|not explicitly defined; multi-label setting, each class is binary|removed non-standard amino acids, length 5-50bp, classes with <40 peptides excluded||True|fulltext|
|DeepAFP|not_reported_in_available_evidence (DeepAFP-Main dataset, curated, no direct link provided)|training_or_benchmark_unspecified|not_reported_in_available_evidence (DeepAFP-Main dataset, curated, no direct link provided)|not_reported_in_available_evidence|described_no_link|37595093|10.1002/pro.4758|not reported in evidence||||True|fulltext|
|AMPpred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37914524|10.24272/j.issn.2095-8137.2023.246|||||True|fulltext|
|not specific|APD (Antimicrobial Peptide Database)|training_or_benchmark_unspecified|curated from literature|https://aps.unmc.edu/AP/|direct_url_found|37914524|10.24272/j.issn.2095-8137.2023.246|more than 3,500 AMPs cataloged|N/A|||False|fulltext|
|AMPpred-AAIW|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|37120707|10.1142/S0219720023500063|||||True|abstract|
|MIC prediction ensemble model (BiLSTM-CNN-MBM)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39262770|10.48550/arXiv.1810.11363|||||True|abstract|
|AMPpred-EL|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35576825|10.1016/j.compbiomed.2022.105577|||||True|abstract|
|AMPpred-MFA|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link||10.1021/acs.jcim.3c01017.s001|||||True|metadata|
|Multifunctional AMP Design Framework (FBGAN-enhanced)|Integrated from GRAMPA, APD3, ADAM, CAMPR4, UniProt|training_or_benchmark_unspecified|Integrated from GRAMPA, APD3, ADAM, CAMPR4, UniProt||described_no_link|40806517|10.3390/ijms26157387|||||True|abstract|
|AMPpredMFA|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40806517|10.3390/ijms26157387|||||True|review|
|sAMP-pred-GAT|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40806517|10.3390/ijms26157387|||||True|review|
|AMP-META|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40806517|10.3390/ijms26157387|||||True|review|
|MBC-attention|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40806517|10.3390/ijms26157387|||||True|review|
|EnDL-HemoLyt|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40806517|10.3390/ijms26157387|||||True|review|
|SenseXAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40806517|10.3390/ijms26157387|||||True|review|
|multiple|DBAASP|training_or_benchmark_unspecified|Review: AI-Driven Antimicrobial Peptide Discovery: Mining and Generation|not_reported_in_available_evidence|direct_url_found|40459283|10.1021/acs.accounts.0c00594|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence||False|review|
|AniAMPpred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34259329|10.1093/bib/bbab242|||||True|abstract|
|Appred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|39247292|10.1016/j.heliyon.2024.e36163|||||True|fulltext|
|AMPs-Net|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37521317|10.3389/fbinf.2023.1216362|||||True|review|
|LABAMPs|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37521317|10.3389/fbinf.2023.1216362|||||True|review|
|LSTM-based AMP classifier/generator|not reported (likely from public databases)|training_or_benchmark_unspecified|not reported (likely from public databases)||described_no_link|33810011|10.1016/j.diagmicrobio.2004.02.008|||||True|fulltext|
|AMPScanner|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34165973|10.1021/acs.jcim.1c00175|||||True|review|
|AMPSpeciesSpecific|https://github.com/bzlee-bio/AMPSpeciesSpecific (may contain data)|training_or_benchmark_unspecified|https://github.com/bzlee-bio/AMPSpeciesSpecific (may contain data)|https://github.com/bzlee-bio/AMPSpeciesSpecific (may contain data)|direct_url_found|39766503|10.3390/antibiotics13121113|||||False|fulltext|
|PepNet|not_reported_in_available_evidence (described as AMP and AIP test sets from previous studies; likely included in Zenodo records)|training_or_benchmark_unspecified|not_reported_in_available_evidence (described as AMP and AIP test sets from previous studies; likely included in Zenodo records)|not_reported_in_available_evidence|direct_url_found|39341947|10.1038/s42003-024-06911-1|||||False|fulltext|
|BPFun|https://github.com/291357657/BPFun (data included in repository)|training_or_benchmark_unspecified|https://github.com/291357657/BPFun (data included in repository)|https://github.com/291357657/BPFun (data included in repository)|direct_url_found|40691539|10.1186/s12859-025-06190-5|2409 AMPs, etc.|not explicitly reported|CD-HIT at 0.9 threshold||False|fulltext|
|LLAMP|https://github.com/GIST-CSBL/LLAMP (data included); DBAASP v3 for MIC data|training_or_benchmark_unspecified|https://github.com/GIST-CSBL/LLAMP (data included); DBAASP v3 for MIC data|https://github.com/GIST-CSBL/LLAMP (data included); DBAASP v3 for MIC data|direct_url_found|40676915|10.1093/bib/bbaf343|~1.7 million peptide-MIC pairs|not applicable (regression)|||False|fulltext|
|CL-ACP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|34670488|10.1186/s12859-021-04433-9|||||True|fulltext|
|AMPTrans-lstm|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|36618982|10.1016/j.csbj.2022.12.029|36,088 AMP sequences|747,352 non-AMP sequences (PDB + UniProt)|||True|fulltext|
|CSAMPPRED|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35988923|10.1093/bib/bbac343|||||True|fulltext|
|Thomas et al. 2009 AMP prediction model|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|29379261|10.6026/97320630013415|||||True|fulltext|
|ANN-based AMP prediction model (ref [4])|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|29379261|10.6026/97320630013415|||||True|fulltext|
|Multiple alignment based AMP predictor (ref [5])|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|29379261|10.6026/97320630013415|||||True|fulltext|
|Two-level fuzzy K-NN model (ref [7])|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|29379261|10.6026/97320630013415|||||True|fulltext|
|Sequence alignment-SVM-LZ complexity model (ref [8])|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|29379261|10.6026/97320630013415|||||True|fulltext|
|Anti-Hepatitis Peptides predictor (ref [9])|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|29379261|10.6026/97320630013415|||||True|fulltext|
|AmpClass|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|39383429|10.1590/0001-3765202420230756|15945|12535|not specified||True|fulltext|
|AMPScannerV2|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35988923|10.1093/bib/bbac343|||||True|fulltext|
|Gabere&Noble AMP predictor|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35988923|10.1093/bib/bbac343|||||True|fulltext|
|Wang et al. AMP predictor|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35988923|10.1093/bib/bbac343|||||True|fulltext|
|Witten&Witten AMP predictor|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|35988923|10.1093/bib/bbac343|||||True|fulltext|
|Unnamed CVAE-diffusion AMP generator|UniProt (uniprotkb_reviewed_true_2024_12_17.fasta) for pretraining; GRAMPA for fine-tuning and MIC training|training_or_benchmark_unspecified|UniProt (uniprotkb_reviewed_true_2024_12_17.fasta) for pretraining; GRAMPA for fine-tuning and MIC training||described_no_link|41460918|10.1371/journal.pcbi.1013833|||||True|fulltext|
|Malebary-Khan AMP predictor|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|38391554|10.32604/cmc.2021.015041|||||True|abstract|
|Anticancer-Peptides-CNN|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34880291|10.1038/s41598-021-02703-3|||||True|repository|
|APIN|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|31870282|10.1093/bioinformatics/btx679|||||True|abstract|
|SeqGAN-BERT-MLP AMP identifier (Cao et al. 2023)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|36857616|10.1093/bib/bbad058|||||True|abstract|
|Co-AMPpred GitHub repository|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|CoAMPpred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|2020-peptidomics|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AAGP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|MetagenomicDC|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|deep-belief-network|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|acp-ope|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|2022-iAMP-DL|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AFP_DL|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AFP_DL-QSARES|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|ANIA_github|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|PC6-protein-encoding-method|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|BAGEL4|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|LinearDisplay|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|msaconverter|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|LysePred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AI4AVP_predictor|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMP-researchprotein|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|learning_sequence_motifs|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMP-BERT GitHub repository|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|LightGBM|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|shap|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|COMPASS database|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMP-RNNpro web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|deep_AMPpred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|ADAM_web_server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|ampsphere_web_server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|MAPLE GitHub repository|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Antimicrobial-Peptides|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|LMPred_AMP_Prediction|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|CDPfold|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|DDM GitHub|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|UniAMP web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|PepProtGraphAnalyzer|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|esm-AxP-GDL|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|esm|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|E-CLEAP GitHub repository|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMP Scanner v2|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|[<br>"41315055",<br>"40891852"<br>]|[<br>"10.1007/s00248-025-02620-2",<br>"10.1128/spectrum.01504-25"<br>]|||||True|github_search|
|AMPScanner vr.2 web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|PepGen 1.0 web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|CalcAMP GitHub repository|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Deep-AmPEP30 web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMP toxicity prediction code|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMP0 webserver|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMPA web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AntiBP3 GitLab|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AntiBP3 Web Server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AntiBP3 PyPI|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|dbAMP 3.0 web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|StarPep|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AmpGram R package|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|PepNet web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Antimicrobial Peptide Scanner vr.2 web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMPScanner vr.2 web server (alternate)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|ACPred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|AMPfun|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|AntiCP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|AntiCP2.0|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|HAPPENN|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|HemoPred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|ToxinPred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|ToxIBTL|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|AllerTop|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|AllergenFP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|AllerCatPro|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41155367|10.3390/ijms262010077|||||True|fulltext|
|Deep learning hybrid model (unnamed)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41731616|10.1186/s40168-025-02326-0|||||True|fulltext|
|ACPred, AMPfun, AntiCP, AntiCP2.0, iAMPpred, Macrel, HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP, AllerCatPro|Cathelicidin-derived peptides (virtual screening set)|benchmark|described in PMID 41155367|not_reported_in_available_evidence|described_no_link|41155367|10.3390/ijms262010077|8 peptides (AL-38, LL-37, RK-31, KS-30, KR-20, FK-16, FK-13, KR-12)|not_reported_in_available_evidence|not_reported_in_available_evidence|not_applicable|True|fulltext|
|Deep learning hybrid model (unnamed), Macrel|Marine biofilm metagenomic AMP candidates|benchmark|generated in PMID 41731616|not_reported_in_available_evidence|described_no_link|41731616|10.1186/s40168-025-02326-0|3,054,472 candidate AMPs (1048 high-confidence)|not_reported_in_available_evidence|not_reported_in_available_evidence|not_reported_in_available_evidence|True|fulltext|
|AxPEP3|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34867843|not_reported_in_available_evidence|||||True|fulltext|
|RF-AmPEP30|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34867843|not_reported_in_available_evidence|||||True|fulltext|
|CAMPR34|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34867843|not_reported_in_available_evidence|||||True|fulltext|
|CLASSAMP5|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34867843|not_reported_in_available_evidence|||||True|fulltext|
|DBAASP6|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|34867843|not_reported_in_available_evidence|||||True|fulltext|
|APSvr.2|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37523405|not_reported_in_available_evidence|||||True|fulltext|
|DBAASPv3.0|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37523405|not_reported_in_available_evidence|||||True|fulltext|
|CAMPR3(RF)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|28203715|10.1093/bioinformatics/btx081|||||True|review|
|CAMPR3(SVM)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|28203715|10.1093/bioinformatics/btx081|||||True|review|
|BAGEL3|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|28203715|10.1093/bioinformatics/btx081|||||True|review|
|BACTIBASE|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|28203715|10.1093/bioinformatics/btx081|||||True|review|
|AMP prediction server (biosino)|CAMP database (http://www.camp.bicnirrh.res.in/) and UniProt|training_or_benchmark_unspecified|CAMP database (http://www.camp.bicnirrh.res.in/) and UniProt|not_reported_in_available_evidence|direct_url_found|21533231|10.1371/journal.pone.0018476|870 AMPs|8661 non-AMPs|CD-HIT at 70% sequence identity|jackknife test|False|fulltext|
|multiple (used in Paper 1)|APD3 (Antimicrobial Peptide Database 3)|training|https://aps.unmc.edu/AP/ (not explicitly in extracted links)|not_reported_in_available_evidence|source_database_named|34867843|10.3389/fmicb.2021.715246|Dataset 1 (594 AMPs with activity against Gram-negative bacteria), Dataset 2 (299 AMPs with activity only against Gram-negative)||||True|fulltext|
|multiple (used in Paper 2)|Lophotrochozoan AMP library (from multiple databases)|external_test|APD, CAMP, DBAASP, DRAMP, ADAM, YADAMP, InverPep, etc.|not_reported_in_available_evidence|source_database_named|37523405|10.1371/journal.ppat.1011508|Lophotrochozoan AMPs from multiple phyla||||True|fulltext|
|multiple (used in Paper 2 pipeline)|Helminth predicted peptide dataset|benchmark|WormBase ParaSite (WBPS) predicted proteomes|not_reported_in_available_evidence|source_database_named|37523405|10.1371/journal.ppat.1011508|>16,000 AMP-like peptides from 127 helminth species||duplicate proteins removed, cutoffs by length, signal peptide, etc.||True|fulltext|
|ADAM (prediction tool)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|37523405|10.1371/journal.ppat.1011508|||||True|fulltext|
|ADMETlab 3|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|42276016|10.1016/j.ultsonch.2026.107920|||||True|fulltext|
|Multi-label weighted KNN-MLR model|APD database (May 2016) filtered to 2222 AMPs with 5 activities; APD3 available at https://aps.unmc.edu/AP/|training_or_benchmark_unspecified|APD database (May 2016) filtered to 2222 AMPs with 5 activities; APD3 available at https://aps.unmc.edu/AP/|https://aps.unmc.edu/AP/|direct_url_found|28526820|10.1038/s41598-017-01986-9|2222 AMPs|none (multi-label dataset, no explicit negative class)|not described|leave-one-out cross-validation|False|fulltext|
|cdGAN|APD3 + UniProt (2600 AMPs, 2600 non-AMPs)|training_or_benchmark_unspecified|APD3 + UniProt (2600 AMPs, 2600 non-AMPs)|https://github.com/aretiz/amp_de_novo_design_cdGAN|described_no_link|41137855|10.1093/bib/bbaf500|2600 AMPs (10-50 residues)|2600 non-AMPs (10-50 residues)|MMseqs2 clustering at 50% sequence identity for non-AMPs; average pairwise similarity 0.2|not explicitly described (likely used for GAN training)|True|fulltext|
|AMP-GSM|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|41072192|10.3390/app13085106|||||True|abstract|
|ISCAPE|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|41072192|10.1016/j.jmgm.2025.109188|||||True|abstract|
|AMP MIC predictor (CNN/RNN)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|37938588|10.1038/s41467-023-42434-9|~5000|not applicable (regression)|||True|fulltext|
|AxPEP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|41315055|10.1007/s00248-025-02620-2|||||True|fulltext|
|AMPGenix|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40891852|10.1128/spectrum.01504-25|||||True|fulltext|
|Macrel, AxPEP, AMP Scanner V2|PRJNA600247|external_test|NCBI BioProject|https://www.ncbi.nlm.nih.gov/bioproject/PRJNA600247|source_database_named|41315055|10.1007/s00248-025-02620-2|||||True|fulltext|
|StackAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|29374199|10.1109/tai.2024.3421176|||||True|metadata|
|AMPlify_bal|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40891852|10.1128/spectrum.01504-25|||||True|mixed|
|AMPlify_imbal|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|40891852|10.1128/spectrum.01504-25|||||True|mixed|
|PeptideRanker|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)|training_or_benchmark_unspecified|BIOPEP, PeptideDB, APD2, CAMP (positive); UniProt secreted and non-secreted sequences (negative control)|not_reported_in_available_evidence|described_no_link|23056189|10.1371/journal.pone.0045012|0 (negative control set)|N/A|70% sequence similarity reduction|five-fold cross-validation with 70% sequence similarity reduction|True|fulltext|
|HydraAMP|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|MetaPepticon|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|Venomics artificial intelligence|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|WeightedEnsemble_L3 (Anti_Cp)|https://github.com/xubocheng/Anti_Cp.git|training_or_benchmark_unspecified|https://github.com/xubocheng/Anti_Cp.git|https://github.com/xubocheng/Anti_Cp.git|direct_url_found|38266820|10.1016/j.jare.2024.01.023|||||False|fulltext|
|WeightedEnsemble_L3|DBAASP-derived AMP activity dataset (MRSA focus)|training|DBAASP database|https://github.com/xubocheng/Anti_Cp.git|direct_url_found|38266820|10.1016/j.jare.2024.01.023|high activity (MIC <= 32 μg/ml)|low activity (32 < MIC <= 128 μg/ml) and no activity (MIC > 128 μg/ml)|not_reported_in_available_evidence|80% train, 20% test|False|fulltext|
|hydramp (conda-feedstock)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|hydramp (pytorch port)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|AMPlify (AWS Amplify JS)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|AMPlify (AWS Amplify CLI)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|AMPlify (Jekyll AMP theme)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|Macrel (BigDataBiology)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|Macrel (MacReloader)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|Macrel (macrelay)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|AmPEP (amPEPpy)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|AmPEP (Ampep_Python)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|AmPEP (ShirleyWISiu)|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|23056189|10.1371/journal.pone.0045012|||||True|github_search|
|PLUM|Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository|training_or_benchmark_unspecified|Composite from CAMPR4, ADAM, APD3, GRAMPA, UniProtKB, and a non-AMP dataset from Ma et al. (2022); no direct download link provided, but data likely included in the GitHub repository|not_reported_in_available_evidence|described_no_link|42124643|10.64898/2026.02.21.707214|17,456 AMPs|58,775 non-AMPs|Removed duplicates and sequences already present in the AMP dataset during generation pipeline|not_reported_in_available_evidence|True|fulltext|
|APD3|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence|not_reported_in_available_evidence|described_no_link|33996914|10.3389/fmolb.2021.669431|||||True|review|
|AVCpred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|33996914|10.3389/fmolb.2021.669431|||||True|review|
|ApexGO|not_reported_in_available_evidence (VAE training data not specified, APEX trained on in-house peptides)|training_or_benchmark_unspecified|not_reported_in_available_evidence (VAE training data not specified, APEX trained on in-house peptides)||described_no_link|42206144|10.1038/s42256-026-01237-5|||||True|fulltext|
|c_AMPs-prediction|https://github.com/mayuefine/c_AMPs-prediction|training_or_benchmark_unspecified|https://github.com/mayuefine/c_AMPs-prediction|https://github.com/mayuefine/c_AMPs-prediction|direct_url_found|41164228|10.3389/fvets.2025.1689589|2,820,488 potential AMPs predicted||redundancy removed by Perl script, details not specified||False|fulltext|
|AMPlify GitHub|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AmPEP web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|AMPer web server|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|CatBoost AMP predictor|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Two_Level_Ensemble-classifier-chain|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|amp_de_novo_design_cdGAN|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|MAPLE GitHub|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|kneaddata|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|VirSorter2|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|COGclassifier|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Anti_Cp|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Anti_Cp.git|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|PLUM GitHub|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Antimicrobial|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Urchin|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|allenCCF|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|phy|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|iblapps|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Lab|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Npx|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|soft-neighbors-supported-clustering|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|DeepSeaQuence_biofilms|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|FMT-MetagenomicData|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|TransDecoder|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|macrel2020benchmark|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|nov-fams-pipeline|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|aro|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|StackEnPred|not_reported_in_available_evidence|training_or_benchmark_unspecified|not_reported_in_available_evidence||described_no_link|||||||True|github_search|
|Multi-label WKnn-MLR|APD (May 2016) filtered multi-label dataset|training|||source_database_named|||||||||

## Dataset Links

|dataset_name|url|source|linked_model|dataset_status|evidence|source_pmid|source_doi|
|---|---|---|---|---|---|---|---|
|Co-AMPpred benchmark dataset (from DEEP-AmPEP30)|https://github.com/onkarS23/CoAMPpred|DEEP-AmPEP30 study [32]|Co-AMPpred|direct_url_found|fulltext|34330209|10.1186/s12859-021-04305-2|
|2020-peptidomics|https://github.com/ErikHartman/2020-peptidomics|Wound peptidome paper||direct_url_found|fulltext|33613550|10.3389/fimmu.2020.620707|
|m9.figshare.31099765.|https://doi.org/10.6084/m9.figshare.31099765.|||direct_url_found|dataset_repository|28892365|10.1021/acs.bioconjchem.7b00368|
|AAGP.|https://github.com/saptawtf/AAGP.|||direct_url_found|fulltext|40781463|10.1038/s41598-025-12759-0|
|iAMPCN|https://github.com/joy50706/iAMPCN|||direct_url_found|fulltext|39330266|10.3390/md22090385|
|master,|https://github.com/joy50706/iAMPCN/tree/master,|||direct_url_found|fulltext|39330266|10.3390/md22090385|
|dryad.p745m.|http://doi.org/10.5061/dryad.p745m.|||direct_url_found|regex_fulltext_or_metadata|26729502|10.1111/eva.12202|
|ACP-DL|https://github.com/haichengyi/ACP-DL|||direct_url_found|regex_fulltext_or_metadata|34880291|10.1038/s41598-021-02703-3|
|Anticancer-Peptides-CNN|https://github.com/mrzResearchArena/Anticancer-Peptides-CNN|||direct_url_found|regex_fulltext_or_metadata|34880291|10.1038/s41598-021-02703-3|
|MetagenomicDC|https://github.com/IcarPA-TBlab/MetagenomicDC|||direct_url_found|regex_fulltext_or_metadata|30066629|10.1186/s12859-018-2182-6|
|deep-belief-network.|https://github.com/albertbup/deep-belief-network.|||direct_url_found|regex_fulltext_or_metadata|30066629|10.1186/s12859-018-2182-6|
|acp-ope|https://github.com/khanhlee/acp-ope|||direct_url_found|regex_fulltext_or_metadata|36642410|10.1093/bib/bbac630|
|Nerita versicolor AMP candidates||PMID 36835264||described_no_link|fulltext|36835264|10.3390/ijms24043852|
|Pomacea poeyana AMP candidates||PMID 33113998||described_no_link|fulltext|33113998|10.3390/biom10111473|
|DRAMP_APD3_anti-Candida|not_reported_in_available_evidence|DRAMP and APD3 databases|ESM2-AFPpred|direct_url_found|fulltext|35724626|10.1093/bib/bbac226|
|AFP_DL|https://github.com/DongYin521/AFP_DL||ESM2-AFPpred|direct_url_found|fulltext|35724626|10.1093/bib/bbac226|
|AFP_DL‐QSARES|https://github.com/DongYin521/AFP_DL‐QSARES||ESM2-AFPpred|direct_url_found|fulltext|35724626|10.1093/bib/bbac226|
|ANIA.|https://github.com/SilverGojo4/ANIA.|not_reported_in_available_evidence|ANIA|direct_url_found|fulltext|41664908|10.1093/bib/bbag023|
|AI4AMP_predictor|https://github.com/LinTzuTang/AI4AMP_predictor|||direct_url_found|regex_fulltext_or_metadata|34783578|10.1128/msystems.00299-21|
|PC6-protein-encoding-method|https://github.com/LinTzuTang/PC6-protein-encoding-method|||direct_url_found|regex_fulltext_or_metadata|34783578|10.1128/msystems.00299-21|
|SAMP|https://github.com/wan-mlab/SAMP|GitHub|SAMP|direct_url_found|regex_fulltext_or_metadata|39573886|10.1093/bfgp/elae046|
|AI4AVP_dataset|https://github.com/LinTzuTang/AI4AVP_predictor|APD3, DRAMP, YADAMP, DBAASP, CAMP, AVPdb, UniProt/SwissProt|AI4AVP|direct_url_found|fulltext|37626205|10.1109/JBHI.2021.3130825|
|DBAASP|https://dbaasp.org|Database of Antimicrobial Activity and Structure of Peptides|Al-Omari 2024 AMP prediction model|direct_url_found|fulltext|39705302|10.1371/journal.pone.0315477|
|AMP training dataset|https://github.com/researchprotein/amp|GitHub repository of the AMP model|AMP|direct_url_found|abstract|38972032|10.1007/s12539-024-00640-z|
|learning_sequence_motifs.|https://github.com/p-koo/learning_sequence_motifs.||AMP|direct_url_found|abstract|38972032|10.1093/nar/gkab1080|
|AMP-BERT dataset|https://github.com/GIST-CSBL/AMP-BERT.|AMP-BERT GitHub repository|AMP-BERT|direct_url_found|fulltext|36461699|10.1002/pro.4529|
|treexplainer-study|https://github.com/suinleelab/treexplainer-study|review paper||direct_url_found|fulltext|36290108|10.1038/s42256-019-0138-9|
|LightGBM|https://github.com/Microsoft/LightGBM|review paper||direct_url_found|fulltext|36290108|10.1038/s42256-019-0138-9|
|shap|https://github.com/slundberg/shap|review paper||direct_url_found|fulltext|36290108|10.1038/s42256-019-0138-9|
|COMPASS|https://compass.imi.uni-muenster.de/data.json|aggregated from 9 public AMP databases (Bactibase, YADAMP, APD3, DRAMP, CAMP3, DBAASP, LAMP2, dbAMP, UniProt)|AmpGPT2|direct_url_found|fulltext|42174216|10.1038/s44259-026-00218-3|
|zenodo.13999503.|https://doi.org/10.5281/zenodo.13999503.|||direct_url_found|regex_fulltext_or_metadata|29679519|10.1002/cmdc.201800204|
|AMP-Designer|https://github.com/jkwang93/AMP-Designer|||direct_url_found|regex_fulltext_or_metadata|29679519|10.1002/cmdc.201800204|
|ADAM (Antimicrobial Peptide Database)|https://bioinformatics.cs.ntou.edu.tw/ADAM|mentioned in fulltext of PMID 38839785 (extracted from MLACP 2.0 context, but not directly linked to AMP-RNNpro)|not directly linked|direct_url_found|fulltext|38839785|10.1016/j.csbj.2022.07.043|
|iAMP-SeE Dataset (Zenodo)|https://doi.org/10.5281/zenodo.17398951|Zenodo|iAMP-SeE|direct_url_found|fulltext|41913931|10.7717/peerj.20978|
|APD3|http://aps.unmc.edu/|Antimicrobial Peptide Database|CVAE-BIO|direct_url_found|fulltext|41849223|10.1093/bib/bbag115|
|scan2030|https://github.com/scan2030||CVAE-BIO|direct_url_found|repository|41849223|10.1093/bib/bbag115|
|GRAMPA (modified)|https://github.com/zswitten/Antimicrobial-Peptides|Aggregated from APD, DADP, DBAASP, DRAMP, YADAMP|PepVAE|direct_url_found|fulltext|34659152|10.3389/fmicb.2021.725727|
|LMPred independent dataset|https://github.com/williamdee1/LMPred_AMP_Prediction|Created by the authors|LMPred|direct_url_found|fulltext|36699381|10.1101/2020.07.12.199554v3|
|LMPred_AMP_Prediction.\\nSUPPLEMENTARY|https://github.com/williamdee1/LMPred_AMP_Prediction.\\nSUPPLEMENTARY|not_reported_in_available_evidence|LMPred|direct_url_found|mixed|36699381|10.1101/2020.07.12.199554v3|
|CDPfold.|https://github.com/zhangch994/CDPfold.|not_reported_in_available_evidence|LMPred|direct_url_found|mixed|36699381|10.1101/2020.07.12.199554v3|
|DDM AMP dataset|https://github.com/kww567upup/DDM|GitHub repository for DDM model|DDM|direct_url_found|fulltext|41692989|10.1093/bioinformatics/btag077|
|PepProtGraphAnalyzer|https://github.com/cicese-biocom/PepProtGraphAnalyzer|||direct_url_found|regex_fulltext_or_metadata|41594075|10.3390/antibiotics15010039|
|esm-AxP-GDL|https://github.com/cicese-biocom/esm-AxP-GDL|GitHub repository|not_reported_in_available_evidence|direct_url_found|regex_fulltext_or_metadata|41594075|10.3390/antibiotics15010039|
|esm|https://github.com/facebookresearch/esm|||direct_url_found|regex_fulltext_or_metadata|41594075|10.3390/antibiotics15010039|
|E-CLEAP dataset|https://github.com/Wangsicheng52/E-CLEAP|Compiled from APD3, PlantPepDB, BaAMPs, BioPepDB (positive) and UniProt (negative)|E-CLEAP|direct_url_found|fulltext|38722967|10.1371/journal.pone.0300125|
|DRAMP 2.0|http://dramp.cpu-bioinfor.org/|DRAMP database|Unnamed AMP predictor from DRAMP 2.0|direct_url_found|fulltext|31409791|10.1038/s41597-019-0154-y|
|CalcAMP dataset|https://doi.org/10.5281/zenodo.7588702|Custom built from public AMP data|CalcAMP|direct_url_found|fulltext|37107088|10.3390/antibiotics12040725|
|CalcAMP.|https://github.com/CDDLeiden/CalcAMP.|GitHub|CalcAMP|direct_url_found|fulltext|37107088|10.3390/antibiotics12040725|
|sAMPpred-GAT|https://github.com/HongWuL/sAMPpred-GAT|||direct_url_found|metadata|36342186|10.1093/bioinformatics/btac715|
|.\\nSUPPLEMENTARY|https://github.com/HongWuL/sAMPpred-GAT/.\\nSUPPLEMENTARY|||direct_url_found|metadata|36342186|10.1093/bioinformatics/btac715|
|dbAMP 3.0|https://awi.cuhk.edu.cn/dbAMP/|dbAMP database|AMPfinder, AMPpredictor, AMPActiPred|direct_url_found|fulltext|39540425|10.1093/nar/gkae1019|
|battleamp-snakemake|https://github.com/szczurek-lab/battleamp-snakemake|||direct_url_found|regex_fulltext_or_metadata||10.64898/2026.06.19.733349|
|ampban|https://github.com/baiwenhuim/ampban|||direct_url_found|regex_fulltext_or_metadata||10.64898/2026.01.20.700468|
|PepMCP|https://github.com/ComputBiophys/PepMCP|||direct_url_found|regex_fulltext_or_metadata||10.64898/2026.02.01.703163|
|BMXC7|https://doi.org/10.17605/OSF.IO/BMXC7|OSF|not specified|direct_url_found|metadata|29889579|10.1080/14787210.2018.1483720|
|Zenodo dataset for peptide benchmark|https://doi.org/10.5281/zenodo.19388783|Zenodo|not applicable (benchmark dataset)|direct_url_found|fulltext|33774670|10.1093/bib/bbab083|
|AMP training data from amppred|http://cabgrid.res.in:8080/amppred/about.html|amppred web server|XGBoost AMP prediction model (Bhangu2025)|direct_url_found|fulltext|40529865|10.1002/smsc.202400579|
|StarPep|http://mobiosd-hub.com/starpep/|Integrated from 42 databases|StarPep tool|direct_url_found|review|39858924|10.3390/microorganisms13010156|
|AMPGAN v3 dataset|https://github.com/marszzibros/AMPGANv3|GitHub repository|AMPGAN v3|direct_url_found|abstract|42364293|10.1016/j.jmgm.2026.109497|
|27733.|https://figshare.com/projects/Tabula_Muris_Transcriptomic_characterization_of_20_organs_and_tissues_from_Mus_musculus_at_single_cell_resolution/27733.|figshare|SAMP|direct_url_found|chunk_summary|38712184|10.1128/aac.02340-16|
|SHARP|https://github.com/shibiaowan/SHARP|GitHub|SAMP|direct_url_found|chunk_summary|38712184|10.1128/aac.02340-16|
|Pore|https://github.com/ComputBiophys/Pore|||direct_url_found|regex_fulltext_or_metadata|41391039|10.1002/advs.202516470|
|Pore‐Forming_AMP_SVM.|https://github.com/ComputBiophys/Pore‐Forming_AMP_SVM.|||direct_url_found|regex_fulltext_or_metadata|41391039|10.1002/advs.202516470|
|iFeature|https://github.com/Superzchen/iFeature|||direct_url_found|regex_fulltext_or_metadata|30867681|10.1186/s13040-019-0196-x|
|MAPLE.|https://github.com/Harkool/MAPLE.|||direct_url_found|regex_fulltext_or_metadata|39927895|10.1021/acs.jcim.5c00006|
|SGAC.|https://github.com/wyxwyx46941930/SGAC.|||direct_url_found|regex_fulltext_or_metadata|41662353|10.1093/bib/bbag038|
|keras-multi-head|https://github.com/CyberZHG/keras-multi-head|||direct_url_found|regex_fulltext_or_metadata|35078402|10.1186/s12864-022-08310-4|
|AMPlify|https://github.com/bcgsc/AMPlify|||direct_url_found|regex_fulltext_or_metadata|35078402|10.1186/s12864-022-08310-4|
|keras_attention.|https://github.com/lzfelix/keras_attention.|||direct_url_found|regex_fulltext_or_metadata|35078402|10.1186/s12864-022-08310-4|
|APD (Antimicrobial Peptide Database)|https://aps.unmc.edu/AP/|curated from literature|not specific|direct_url_found|fulltext|37914524|10.24272/j.issn.2095-8137.2023.246|
|DBAASP|http://www.biomedicine.org.ge/dbaasp/|database|not specific|direct_url_found|fulltext|37914524|10.24272/j.issn.2095-8137.2023.246|
|LAMP|http://biotechlab.fudan.edu.cn/database/lamp|database|not specific|direct_url_found|fulltext|37914524|10.24272/j.issn.2095-8137.2023.246|
|Antifreeze-Peptide-Discovery.|https://github.com/imamabi/Antifreeze-Peptide-Discovery.|||direct_url_found|regex_fulltext_or_metadata|35576825|10.1016/j.compbiomed.2022.105577|
|SendongZhao.|https://github.com/SendongZhao.|||direct_url_found|regex_fulltext_or_metadata|36227057|10.1093/bioinformatics/btac675|
|AMPSpeciesSpecific dataset|https://github.com/bzlee-bio/AMPSpeciesSpecific|likely included in GitHub repository|AMPSpeciesSpecific|direct_url_found|fulltext|39766503|10.3390/antibiotics13121113|
|PepNet Zenodo 1322351661|https://zenodo.org/records/1322351661|Zenodo repository (likely code and data)|PepNet|direct_url_found|fulltext|39341947|10.1038/s42003-024-06911-1|
|PepNet Zenodo 1373425862|https://zenodo.org/records/1373425862|Zenodo repository (likely code and data)|PepNet|direct_url_found|fulltext|39341947|10.1038/s42003-024-06911-1|
|BPFun dataset|https://github.com/291357657/BPFun|GitHub repository; includes AMP, ACP, ADP, AHP, AIP, AAP, AOP peptides|BPFun|direct_url_found|fulltext|40691539|10.1186/s12859-025-06190-5|
|LLAMP dataset|https://github.com/GIST-CSBL/LLAMP|DBAASP v3, processed; included in GitHub|LLAMP|direct_url_found|fulltext|40676915|10.1093/bib/bbaf343|
|grampa.csv|https://github.com/zswitten/Antimicrobial-Peptides/blob/master/data/grampa.csv|file in Antimicrobial-Peptides repository|LLAMP|direct_url_found|fulltext|40676915|10.1093/bib/bbaf343|
|peptides_molecular_fingerprints_classification|https://github.com/scikit-fingerprints/peptides_molecular_fingerprints_classification|||direct_url_found|regex_fulltext_or_metadata|34037687|10.1093/bib/bbab200|
|AntiBP3 dataset|https://doi.org/10.5281/zenodo.19911030|Curated from APD3, AntiBP2, dbAMP 2.0, CAMPR3, DRAMP, ABP-Finder|AntiBP3|direct_url_found|repository|38391554|10.3390/antibiotics13020168|
|zenodo.5347031|https://doi.org/10.5281/zenodo.5347031|||direct_url_found|regex_fulltext_or_metadata|40410382|10.5281/zenodo.5347031|
|models?filter=beit|https://huggingface.co/models?filter=beit|||direct_url_found|regex_fulltext_or_metadata|40410382|10.5281/zenodo.5347031|
|models?filter=layoutlmv2|https://huggingface.co/models?filter=layoutlmv2|||direct_url_found|regex_fulltext_or_metadata|40410382|10.5281/zenodo.5347031|
|5347031|https://zenodo.org/record/5347031|||direct_url_found|regex_fulltext_or_metadata|40410382|10.5281/zenodo.5347031|
|https://github.com/onkarS23/CoAMPpred||||||||
|https://github.com/joy50706/iAMPCN||||||||
|https://github.com/DongYin521/AFP_DL||||||||
|https://github.com/SilverGojo4/ANIA.||||||||
|AI4AFP dataset|not reported|||||||
|zenodo.19462601|https://doi.org/10.5281/zenodo.19462601|DOI||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|Urchin|https://github.com/VirtualBrainLab/Urchin|GitHub||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|allenCCF|https://github.com/cortex-lab/allenCCF|GitHub||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|phy|https://github.com/cortex-lab/phy|GitHub||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|iblapps|https://github.com/int-brain-lab/iblapps|GitHub||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|atlaselectrophysiology|https://github.com/int-brain-lab/iblapps/tree/master/atlaselectrophysiology|GitHub||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|Lab|https://github.com/tortugar/Lab|GitHub||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|PySleep|https://github.com/tortugar/Lab/tree/master/PySleep|GitHub||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|Npx.|https://github.com/tortugar/Npx.|GitHub||direct_url_found|repository|40233747|10.1016/j.neuron.2025.03.020|
|APD3 + UniProt balanced dataset|https://github.com/aretiz/amp_de_novo_design_cdGAN|APD3 for AMPs, reviewed UniProt for non-AMPs, clustered with MMseqs2 at 50% identity|cdGAN|direct_url_found|fulltext|41137855|10.1093/bib/bbaf500|
|PRJNA600247|https://www.ncbi.nlm.nih.gov/bioproject/PRJNA600247|NCBI BioProject|Macrel, AxPEP, AMP Scanner V2|source_database_named|fulltext|41315055|10.1007/s00248-025-02620-2|
|PRJNA646512|https://www.ncbi.nlm.nih.gov/bioproject/PRJNA646512|NCBI BioProject|Macrel, AxPEP, AMP Scanner V2|source_database_named|fulltext|41315055|10.1007/s00248-025-02620-2|
|DBAASP-derived AMP activity dataset (MRSA focus)|https://github.com/xubocheng/Anti_Cp.git|DBAASP database|WeightedEnsemble_L3|direct_url_found|fulltext|38266820|10.1016/j.jare.2024.01.023|
|dataset_for_|https://github.com/zswitten/Antimicrobial|||direct_url_found|chunk_summary|42124643|10.64898/2026.02.21.707214|
|ConoServer|https://www.conoserver.org/|ConoServer|APEX|direct_url_found|fulltext|39764027|10.1101/2024.12.17.628923|
|ArachnoServer|https://arachnoserver.qfab.org/mainMenu.html|ArachnoServer|APEX|direct_url_found|fulltext|39764027|10.1101/2024.12.17.628923|
|ISOB|https://www.snakebd.com/|ISOB (Indigenous Snake Proteins)|APEX|direct_url_found|fulltext|39764027|10.1101/2024.12.17.628923|
|VenomZone|https://venomzone.expasy.org/|VenomZone (UniProtKB)|APEX|direct_url_found|fulltext|39764027|10.1101/2024.12.17.628923|
|FESNov antimicrobial peptide families|https://novelfams.cgmlab.org|Nature 2023 paper (doi:10.1038/s41586-023-06955-z)||direct_url_found|fulltext|38109938|10.1038/s41586-023-06955-z|
|nov-fams-pipeline|https://github.com/AlvaroRodriguezDelRio/nov-fams-pipeline.|GitHub repository linked to Nature 2023 paper||direct_url_found|repository|38109938|10.1038/s41586-023-06955-z|
|aro|https://github.com/arpcard/aro|GitHub repository linked to Nature 2023 paper||direct_url_found|repository|38109938|10.1038/s41586-023-06955-z|
|FMT donor fecal AMP candidates|https://github.com/pointwei/FMT-MetagenomicData|Fecal metagenomes from 120 FMT donors|c_AMPs-prediction|direct_url_found|fulltext|41164228|10.3389/fvets.2025.1689589|
|AMOR biofilm AMPs dataset|https://github.com/trongthucnguyen/DeepSeaQuence_biofilms|Arctic deep-sea hydrothermal vent biofilm metagenomes||direct_url_found|fulltext|42104260|10.1186/s12866-026-05098-1|

## Dataset Follow-up Tasks

|model_name|dataset_status|reason|next_action|source_pmid|source_doi|
|---|---|---|---|---|---|
|CTCM-Neo & ConformaX-PEP framework|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41859462|10.3389/fcimb.2026.1707267|
|A-CaMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|31870207|10.1080/07391102.2019.1708796|
|PCSPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40781463|10.1109/NEleX59773.2023.10421222|
|iAMPCN|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39330266|10.3390/md22090385|
|SSFGM-Model|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40462515|10.1186/s12864-020-06978-0|
|ACEP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40462515|10.1186/s12864-020-06978-0|
|ACP-DL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34880291|10.1038/s41598-021-02703-3|
|MultiPep|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34909478|10.1093/biomethods/bpab021|
|iAMP-2L|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|iAMPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|AmPEP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|AntiBP2|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|CAMPR3|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|ADAM|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|DBAASP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|MLAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|CAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|ClassAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|AVPpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|AMPER|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|EFC-FCBF|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35305010|10.1093/database/baab012|
|AMPlify|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|E-CLEAP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|UniproLcad|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|TriStack|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|iAMP-DL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|amp-gan|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|AVPIden|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|antibp|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|ampsphere|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|hydramp|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39557756|10.1007/s12602-024-10402-4|
|AMPDiscover|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34081438|10.1021/acs.jcim.1c00251|
|ESM2-AFPpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35724626|10.1093/bib/bbac226|
|ANIA|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41664908|10.1093/bib/bbag023|
|AI4AFP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42146199|10.1021/acsomega.6c00049|
|AI4AMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34783578|10.1128/msystems.00299-21|
|Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|27870247|10.1002/minf.201600029|
|SAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39573886|10.1093/bfgp/elae046|
|DL-QSARES|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39921483|10.1002/advs.202412488|
|PepForge|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39705302|10.64898/2026.05.29.728379|
|BBATProt|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41212592|10.1093/bib/bbaf593|
|AMAP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|30831306|10.1016/j.compbiomed.2019.02.018|
|Deep-AmPEP30|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|32464552|10.1109/INDCON.2011.6139332|
|EBAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40906555|10.1016/j.celrep.2025.116215|
|DLFea4AMPGen|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41093853|10.1002/adma.202307680|
|COMDEL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39234615|10.1016/j.apsb.2024.05.003|
|BERT-based AMP recognition model|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34037687|10.1093/bib/bbab200|
|AMP-CapsNet|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41654884|10.1186/s44342-026-00067-6|
|deepAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41753681|10.3390/microorganisms14020394|
|AMP-RL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37992451|10.1016/j.sbi.2023.102733|
|PepCVAE|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37992451|10.1016/j.sbi.2023.102733|
|PrefixProt|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37992451|10.1016/j.sbi.2023.102733|
|MoFormer|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37992451|10.1016/j.sbi.2023.102733|
|HMAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37992451|10.1016/j.sbi.2023.102733|
|AMP-Designer|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37992451|10.1016/j.sbi.2023.102733|
|AMP-MIC|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|29679519|10.1002/cmdc.201800204|
|AP_Sin|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38416364|10.1038/s41467-018-03746-3|
|AMP-Detector|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39201537|10.3389/fmicb.2018.00323|
|AMP-RNNpro|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38839785|10.1016/j.csbj.2022.07.043|
|AMP-Distillation|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42155201|10.1016/j.compbiolchem.2026.109129|
|STAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42155201|10.64898/2026.05.28.728246|
|CF-AMP prediction|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42020672|10.1101/2022.11.16.516845|
|AMP-DualTransnet|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42020672|10.1016/j.nexres.2026.101536|
|AMP-FreqNet|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.1145/3766671.3766835|
|Collaborative Filtering and Link Prediction model|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.1021/acs.jcim.3c00137|
|Predictive and Interpretable ML Models|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.1021/acsomega.3c08676.s001|
|AMP prediction ML model|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.54985/peeref.2405p7278831|
|GAC-BiTCNN-AMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41844874|10.1038/s41598-026-43370-6|
|AMPGAN|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41463765|10.3390/antibiotics14121263|
|Macrel|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41463765|10.3390/antibiotics14121263|
|iAMPpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41463765|10.3390/antibiotics14121263|
|AMP-GPT|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40193623|10.1038/s44386-026-00045-6|
|MCL-AMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40193623|10.1038/s44386-026-00045-6|
|MAPLE|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39792442|10.1021/acs.jcim.4c01913|
|AMP prediction SVM-LZ|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|25802839|10.1093/nar/gkn823|
|UniAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39799358|10.1186/s12859-025-06033-3|
|AMP Scanner|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38129980|10.1002/mbo3.1393|
|AMPScanner vr.2|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37851665|10.1371/journal.pone.0292947|
|PepGen 1.0|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40643674|10.1007/s00284-025-04346-3|
|AmPepGen|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40643674|10.1007/s00284-025-04346-3|
|AMP-SEMiner|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40445833|10.1016/j.celrep.2025.115773|
|AMP toxicity prediction model (hybrid)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34758751|10.1186/s12859-021-04468-y|
|Deep learning regression model for antimicrobial peptide design (Witten & Witten 2019)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|21347392|10.1101/692681|
|AMP-zGSM|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|21347392|10.5220/0014457300004070|
|AMP0|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|32750857|10.1109/TCBB.2020.2999399|
|sAMPpred-GAT|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|36342186|10.1093/bioinformatics/btac715|
|AMPA|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40410382|10.1038/s44320-025-00120-6|
|AntiBP3|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38391554|10.3390/antibiotics13020168|
|AMPActiPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38723168|10.1002/pro.5006|
|APEX|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39754551|10.1111/1751-7915.70072|
|AMPfinder|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39540425|10.1093/nar/gkae1019|
|AMPpredictor|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39540425|10.1093/nar/gkae1019|
|Generative AMP pipeline (VINCI)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.64898/2026.06.16.732639|
|AMPCLGPT|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.1101/2025.03.07.642021|
|CAmidPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.21203/rs.3.rs-7764304/v1|
|PepMCP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.64898/2026.02.01.703163|
|iMFP-LG|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39585308|10.1093/gpbjnl/qzae084|
|Deep learning model for AMP discovery from ruminant gastrointestinal microbiomes|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39756573|10.1016/j.jare.2025.01.005|
|Deep learning model for AMP discovery from protist genomes (BERT+CNN)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40958742|10.1021/acs.jcim.5c01196|
|amPEPpy|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|33135060|10.1093/bioinformatics/btaa917|
|Bacteria-specific ML models for E. coli AMP activity|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|36912047|10.1021/acs.jcim.2c01551|
|Multiple DL models reviewed (e.g., AMP-BERT, Deep-AmPEP30, etc.)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|36290108|10.3390/antibiotics11101451|
|AMPGAN v3|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42364293|10.1016/j.jmgm.2026.109497|
|PepAnno|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42228741|10.1371/journal.pcbi.1014369|
|AMPGP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40825014|10.1021/acs.jcim.5c00647|
|AmpGram|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|32560350|10.3390/ijms21124310|
|AMPScanner V2|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38877295|10.1002/2211-5463.13847|
|ampir|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38877295|10.1002/2211-5463.13847|
|Ensemble-AMPPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38877295|10.1002/2211-5463.13847|
|CancerGram|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38877295|10.1002/2211-5463.13847|
|PPTPP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38877295|10.1002/2211-5463.13847|
|MLBP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38877295|10.1002/2211-5463.13847|
|Deep2Pep|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38877295|10.1002/2211-5463.13847|
|CG-AMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41286313|10.1038/s41598-025-29666-z|
|AmpHGT|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40598389|10.1186/s12915-025-02253-4|
|SGAC|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41662353|10.1093/bib/bbag038|
|LMPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41978380|10.1093/bib/bbag107|
|TP-LMMSG|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41978380|10.1093/bib/bbag107|
|PGAT-ABPp|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41755839|10.1021/jacsau.5c01520|
|Bidirectional LSTM AMP classification model (Wang2021)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|33810011|10.3390/biom11030471|
|PrMFTP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|36094961|10.1371/journal.pcbi.1010511|
|DeepAFP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37595093|10.1002/pro.4758|
|AMPpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37914524|10.24272/j.issn.2095-8137.2023.246|
|AMPpred-AAIW|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37120707|10.1142/S0219720023500063|
|MIC prediction ensemble model (BiLSTM-CNN-MBM)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39262770|10.48550/arXiv.1810.11363|
|AMPpred-EL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35576825|10.1016/j.compbiomed.2022.105577|
|AMPpred-MFA|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper||10.1021/acs.jcim.3c01017.s001|
|Multifunctional AMP Design Framework (FBGAN-enhanced)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40806517|10.3390/ijms26157387|
|AMPpredMFA|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40806517|10.3390/ijms26157387|
|sAMP-pred-GAT|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40806517|10.3390/ijms26157387|
|AMP-META|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40806517|10.3390/ijms26157387|
|MBC-attention|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40806517|10.3390/ijms26157387|
|EnDL-HemoLyt|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40806517|10.3390/ijms26157387|
|SenseXAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40806517|10.3390/ijms26157387|
|AniAMPpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34259329|10.1093/bib/bbab242|
|Appred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39247292|10.1016/j.heliyon.2024.e36163|
|AMPs-Net|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37521317|10.3389/fbinf.2023.1216362|
|LABAMPs|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37521317|10.3389/fbinf.2023.1216362|
|LSTM-based AMP classifier/generator|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|33810011|10.1016/j.diagmicrobio.2004.02.008|
|AMPScanner|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34165973|10.1021/acs.jcim.1c00175|
|CL-ACP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34670488|10.1186/s12859-021-04433-9|
|AMPTrans-lstm|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|36618982|10.1016/j.csbj.2022.12.029|
|CSAMPPRED|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35988923|10.1093/bib/bbac343|
|Thomas et al. 2009 AMP prediction model|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|29379261|10.6026/97320630013415|
|ANN-based AMP prediction model (ref [4])|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|29379261|10.6026/97320630013415|
|Multiple alignment based AMP predictor (ref [5])|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|29379261|10.6026/97320630013415|
|Two-level fuzzy K-NN model (ref [7])|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|29379261|10.6026/97320630013415|
|Sequence alignment-SVM-LZ complexity model (ref [8])|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|29379261|10.6026/97320630013415|
|Anti-Hepatitis Peptides predictor (ref [9])|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|29379261|10.6026/97320630013415|
|AmpClass|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|39383429|10.1590/0001-3765202420230756|
|AMPScannerV2|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35988923|10.1093/bib/bbac343|
|Gabere&Noble AMP predictor|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35988923|10.1093/bib/bbac343|
|Wang et al. AMP predictor|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35988923|10.1093/bib/bbac343|
|Witten&Witten AMP predictor|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|35988923|10.1093/bib/bbac343|
|Unnamed CVAE-diffusion AMP generator|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41460918|10.1371/journal.pcbi.1013833|
|Malebary-Khan AMP predictor|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|38391554|10.32604/cmc.2021.015041|
|Anticancer-Peptides-CNN|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34880291|10.1038/s41598-021-02703-3|
|APIN|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|31870282|10.1093/bioinformatics/btx679|
|SeqGAN-BERT-MLP AMP identifier (Cao et al. 2023)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|36857616|10.1093/bib/bbad058|
|Verify exact composition and availability of Co-AMPpred dataset from GitHub; confirm negative samples are free of ACP/AFP contamination||||||
|Obtain the AI4AFP dataset (3011 AFPs) and confirm CD-HIT threshold; not for main AMP benchmark||||||
|Clarify negative samples for ESM2-AFPpred dataset; dataset only for antifungal extension||||||
|Check if the Nerita versicolor and Pomacea poeyana peptides are independent of training sets||||||
|Download or reconstruct the ANIA MIC dataset from DBAASP/dbAMP/DRAMP; for regression extension only||||||
|Locate the generative models training set (CAMPR4 etc.) and ensure it is not used for discriminative model testing||||||
|For all review-cited models, trace original papers to obtain dataset details||||||
|Verify if Co-AMPpred, ACEP, AMPlify, and other benchmark-ready models provide pre-trained weights or full training scripts||||||
|Seek permanent dataset archives (Zenodo/Figshare) for all key datasets||||||
|Investigate negative sample sourcing for Co-AMPpred and other datasets to avoid cross-contamination with anticancer/antifungal peptides||||||
|Reconstruct a unified non-redundant AMP/non-AMP benchmark dataset using DRAMP/APD3 with strict deduplication, excluding subclass contamination.||||||
|Obtain permanent DOIs for all datasets (e.g., Zenodo, Figshare) to ensure reproducibility.||||||
|Verify negative sample composition for Co-AMPpred and iAMPCN datasets to remove known functional peptides.||||||
|Resolve missing links for ANIA and AI4AFP datasets; contact authors if necessary.||||||
|Investigate sequence identity cutoff for short peptides (≤30 aa) – consider 30% or local alignment.||||||
|CVAE-BIO|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Co-AMPpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Co-AMPpred GitHub repository|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|CoAMPpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|2020-peptidomics|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AAGP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|MetagenomicDC|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|deep-belief-network|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|acp-ope|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|2022-iAMP-DL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AFP_DL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AFP_DL-QSARES|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|ANIA_github|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|PC6-protein-encoding-method|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|BAGEL4|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|LinearDisplay|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|msaconverter|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|LysePred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AI4AVP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AI4AVP_predictor|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMP-researchprotein|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|learning_sequence_motifs|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMP-BERT|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMP-BERT GitHub repository|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|LightGBM|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|shap|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AmpGPT2|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|COMPASS database|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMP-RNNpro web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|iAMP-SeE|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|deep_AMPpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|ADAM_web_server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|ampsphere_web_server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|MAPLE GitHub repository|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|PepVAE|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Antimicrobial-Peptides|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|LMPred_AMP_Prediction|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|CDPfold|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|DDM|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|DDM GitHub|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|UniAMP web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|PepProtGraphAnalyzer|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|esm-AxP-GDL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|esm|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|E-CLEAP GitHub repository|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMP Scanner v2|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|[<br>"41315055",<br>"40891852"<br>]|[<br>"10.1007/s00248-025-02620-2",<br>"10.1128/spectrum.01504-25"<br>]|
|AMPScanner vr.2 web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|PepGen 1.0 web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|CalcAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|CalcAMP GitHub repository|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Deep-AmPEP30 web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMP toxicity prediction code|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMP0 webserver|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMPA web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AntiBP3 GitLab|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AntiBP3 Web Server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AntiBP3 PyPI|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|dbAMP 3.0 web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMPBenchmark|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|StarPep|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AmpGram R package|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|PepNet|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|PepNet web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Antimicrobial Peptide Scanner vr.2 web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMPScanner vr.2 web server (alternate)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Verify download links for DRAMP/APD3 derived datasets||||||
|Check if Co-AMPpred training data is identical to original DEEP-AmPEP30||||||
|Obtain positive/negative splits for ANIA dataset||||||
|Verify positive/negative definitions for Co-AMPpred dataset||||||
|Extract exact data files from Co-AMPpred repository||||||
|Obtain exact data files for ESM2-AFPpred training set (if needed for reference)||||||
|Check if AI4AFP dataset is publicly downloadable||||||
|Confirm deduplication methods for ANIA dataset||||||
|Collect peptide sequences from Nerita and Pomacea papers for independent test||||||
|Construct a standard AMP binary classification dataset from APD3, DRAMP, DBAASP with clear negatives, CD-HIT 40% deduplication, and permanent links||||||
|Verify Co-AMPpred dataset composition (positive/negative counts, deduplication).||||||
|Obtain DRAMP/APD3 external dataset for ESM2-AFPpred.||||||
|Construct a unified benchmark dataset integrating multiple sources and applying CD-HIT at 0.9.||||||
|Check if 2020-peptidomics dataset can be used as independent test set.||||||
|Collect hemolysis data for safety evaluation.||||||
|Verify Co-AMPpred GitHub repo for actual data files, positive/negative composition, and CD-HIT parameters.||||||
|Attempt to obtain AI4AFP dataset via direct request to authors or check supplementary materials of the paper.||||||
|Monitor ANIA repository for possible future release of training data.||||||
|Curate a low-homology independent test set from dbAMP/APD3 with CD-HIT 40% against all training sets used in benchmark.||||||
|Search for AMPlify original paper (e.g., 2018 BMC Genomics) to confirm dataset and training details.||||||
|ACPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|AMPfun|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|AntiCP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|AntiCP2.0|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|HAPPENN|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|HemoPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|ToxinPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|ToxIBTL|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|AllerTop|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|AllergenFP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|AllerCatPro|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|Deep learning hybrid model (unnamed)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41731616|10.1186/s40168-025-02326-0|
|ACPred, AMPfun, AntiCP, AntiCP2.0, iAMPpred, Macrel, HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP, AllerCatPro|described_no_link|dataset source/link incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41155367|10.3390/ijms262010077|
|Deep learning hybrid model (unnamed), Macrel|described_no_link|dataset source/link incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41731616|10.1186/s40168-025-02326-0|
|AxPEP3|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34867843|not_reported_in_available_evidence|
|RF-AmPEP30|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34867843|not_reported_in_available_evidence|
|CAMPR34|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34867843|not_reported_in_available_evidence|
|CLASSAMP5|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34867843|not_reported_in_available_evidence|
|DBAASP6|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34867843|not_reported_in_available_evidence|
|APSvr.2|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37523405|not_reported_in_available_evidence|
|DBAASPv3.0|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37523405|not_reported_in_available_evidence|
|CAMPR3(RF)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|28203715|10.1093/bioinformatics/btx081|
|CAMPR3(SVM)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|28203715|10.1093/bioinformatics/btx081|
|BAGEL3|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|28203715|10.1093/bioinformatics/btx081|
|BACTIBASE|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|28203715|10.1093/bioinformatics/btx081|
|multiple (used in Paper 1)|source_database_named|dataset source/link incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|34867843|10.3389/fmicb.2021.715246|
|multiple (used in Paper 2)|source_database_named|dataset source/link incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37523405|10.1371/journal.ppat.1011508|
|multiple (used in Paper 2 pipeline)|source_database_named|dataset source/link incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37523405|10.1371/journal.ppat.1011508|
|AMP prediction server (biosino)|source_database_named|dataset source/link incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|21533231|10.1371/journal.pone.0018476|
|ADAM (prediction tool)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37523405|10.1371/journal.ppat.1011508|
|ADMETlab 3|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42276016|10.1016/j.ultsonch.2026.107920|
|cdGAN|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41137855|10.1093/bib/bbaf500|
|Multi-label weighted KNN-MLR model|source_database_named|dataset source/link incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|28526820|10.1038/s41598-017-01986-9|
|AMP-GSM|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41072192|10.3390/app13085106|
|ISCAPE|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41072192|10.1016/j.jmgm.2025.109188|
|AMP MIC predictor (CNN/RNN)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|37938588|10.1038/s41467-023-42434-9|
|AxPEP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41315055|10.1007/s00248-025-02620-2|
|AMPGenix|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40891852|10.1128/spectrum.01504-25|
|Macrel, AxPEP, AMP Scanner V2|source_database_named|dataset source/link incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|41315055|10.1007/s00248-025-02620-2|
|StackAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|29374199|10.1109/tai.2024.3421176|
|AMPlify_bal|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40891852|10.1128/spectrum.01504-25|
|AMPlify_imbal|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|40891852|10.1128/spectrum.01504-25|
|PeptideRanker|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|HydraAMP|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|MetaPepticon|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|Venomics artificial intelligence|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|hydramp (conda-feedstock)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|hydramp (pytorch port)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|AMPlify (AWS Amplify JS)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|AMPlify (AWS Amplify CLI)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|AMPlify (Jekyll AMP theme)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|Macrel (BigDataBiology)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|Macrel (MacReloader)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|Macrel (macrelay)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|AmPEP (amPEPpy)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|AmPEP (Ampep_Python)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|AmPEP (ShirleyWISiu)|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|23056189|10.1371/journal.pone.0045012|
|PLUM|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42124643|10.64898/2026.02.21.707214|
|APD3|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|33996914|10.3389/fmolb.2021.669431|
|AVCpred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|33996914|10.3389/fmolb.2021.669431|
|ApexGO|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|42206144|10.1038/s42256-026-01237-5|
|AMPlify GitHub|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AmPEP web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|AMPer web server|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|CatBoost AMP predictor|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Two_Level_Ensemble-classifier-chain|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|amp_de_novo_design_cdGAN|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|MAPLE GitHub|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|kneaddata|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|VirSorter2|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|COGclassifier|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Anti_Cp|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Anti_Cp.git|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|PLUM GitHub|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Antimicrobial|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Urchin|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|allenCCF|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|phy|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|iblapps|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Lab|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Npx|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|soft-neighbors-supported-clustering|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|c_AMPs-prediction|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|DeepSeaQuence_biofilms|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|FMT-MetagenomicData|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|TransDecoder|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|macrel2020benchmark|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|nov-fams-pipeline|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|aro|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|StackEnPred|described_no_link|dataset link/source missing or incomplete|search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper|||
|Construct an independent, balanced test set with positive AMPs from APD3 and negative non-AMP peptides from UniProt, deduplicated with CD-HIT (40% identity) and verified for no ACP/AIP/AVP contamination.||||||
|Audit negative sample composition in Co-AMPpred dataset: source, deduplication, and cross-contamination check.||||||
|Attempt to obtain or reconstruct Wang et al. 2011 dataset for AMP prediction server (biosino).||||||
|Verify APD3 version and access for reproducibility.||||||
|Design multi-distribution test matrices (1:1, 1:10, 1:100, low-homology independent set) using the constructed benchmark dataset.||||||

## Metrics

|metric_name|usage|source_pmid|source_doi|evidence|
|---|---|---|---|---|
|Accuracy|Co-AMPpred test set|34330209|10.1186/s12859-021-04305-2|Test accuracy 80.3% (Table 1), abstract reports 80.8%|
|AUROC|Co-AMPpred test set|34330209|10.1186/s12859-021-04305-2|AUROC 0.873 (Table 1)|
|MCC|Co-AMPpred test set|34330209|10.1186/s12859-021-04305-2|MCC 0.606 (Table 1)|
|AUPRC|CTCM-Neo & ConformaX-PEP held-out evaluation|41859462|10.3389/fcimb.2026.1707267|AUPRC ≈0.80|
|ECE|CTCM-Neo & ConformaX-PEP held-out evaluation|41859462|10.3389/fcimb.2026.1707267|ECE ≈0.03|
|predictive agreement (qualitative benchmarking)|Compared performance with 13 existing tools; models had best performance in all endpoints.|34081438|10.1021/acs.jcim.1c00251|Abstract: 'The models proposed are those with the best performance in all of the endpoints modeled, while most of the methods from the literature have weak-to-random predictive agreements.'|
|Precision|Model evaluation|35724626|10.1093/bib/bbac226|Tables 1 and 2 show Precision values.|
|Recall|Model evaluation|35724626|10.1093/bib/bbac226|Tables 1 and 2 show Recall values.|
|F1-score|Model evaluation|35724626|10.1093/bib/bbac226|Derived from Precision and Recall.|
|Pearson Correlation Coefficient (PCC)|MIC prediction performance|41664908|10.1093/bib/bbag023|ANIA achieved PCCs of 0.75–0.79 across all bacteria.|
|Mean Squared Error (MSE)|MIC prediction error|41664908|10.1093/bib/bbag023|ANIA achieved MSEs of 0.23–0.26.|
|Matthews Correlation Coefficient (MCC)|AFP classification|42146199|10.1021/acsomega.6c00049|AI4AFP achieved MCC of 0.89.|
|Sensitivity|AFP classification|42146199|10.1021/acsomega.6c00049|Performance reported in AI4AFP paper.|
|Specificity|AFP classification|42146199|10.1021/acsomega.6c00049|Performance reported in AI4AFP paper.|
|AUC-ROC|hemolysis prediction|42146199|10.1021/acsomega.6c00049|AUC-ROC of 0.90 for hemolysis model (cited from HAPPENN).|
|Diversity (normalized pairwise Levenshtein distance)|generated sequence diversity|42106831|10.1186/s13040-026-00558-w|Used to compare generative models.|
|Novelty (fraction not in reference set)|generated sequence novelty|42106831|10.1186/s13040-026-00558-w|Defined as fraction of sequences not occurring in training set.|
|Maximum Mean Discrepancy (MMD)|distribution similarity of peptide properties|42106831|10.1186/s13040-026-00558-w|Used to quantify property distribution similarity.|
|accuracy improvement|AMP prediction improvement over state-of-the-art|41212592|10.1093/bib/bbaf593|BBATProt improves accuracy by 2.96%-41.96% in antimicrobial peptide (AMP) prediction|
|AUC-PR|Deep-AmPEP30 achieves 85% AUC-PR.|32464552|10.1109/INDCON.2011.6139332|85% in area under the precision-recall curve (AUC-PR).|
|ROC AUC|activity classifier performance|38402320|10.1038/s41598-024-55205-3|LSTM paired with ProtTrans performed best ... ROC AUC = 0.872|
|AUC|model evaluation on test set|41654884|10.1186/s44342-026-00067-6|AUC score of 98.91% on the test set using with dipeptide Composition (DPC).|
|Activity probability (by CAMP, AMP Scanner, Macrel)|Used to evaluate generated AMP sequences; percentage of peptides identified as AMPs by all three predictors.|29679519|10.1002/cmdc.201800204|Fulltext: 'we used three different AMP classifiers: The predictor based on the collection of anti-microbial peptides (CAMP), AMP Scanner, and Macrel, to assess their potential for bioactivity... the percentage of the peptide sequences identified as AMPs by all three predictors simultaneously (with predicted values of 0.5 or higher).'|
|R^2|STAMP evaluation||10.64898/2026.05.28.728246|R^2 of 0.70.|
|RMSE|Evaluate MIC prediction regression models|34659152|10.3389/fmicb.2021.725727|Initial comparison of the eight different regression models ... was performed by calculating Root mean square error (RMSE) and R2 values|
|R²|Evaluate MIC prediction regression models|34659152|10.3389/fmicb.2021.725727|calculating Root mean square error (RMSE) and R2 values from actual MIC (log μM) vs. predicted|
|AUPR|model evaluation|38934543|10.1128/msystems.01358-23|The area under the ROC curve (AUROC) and area under the PR curve (AUPR) were used as an indicator of the classification performance.|
|MIC|experimental validation|38934543|10.1128/msystems.01358-23|The minimum inhibitory concentration (MIC) was defined as the lowest concentration of the test peptide at which microorganism growth was visibly absent.|
|Hemolysis (%)|experimental validation|38934543|10.1128/msystems.01358-23|Hemoglobin release was then measured as OD at 540 nm. Percent hemolysis was calculated.|
|GMean|Geometric mean used for multi-species ABP activity prediction in AMPActiPred|38723168|10.1002/pro.5006|AMPActiPred achieved an average GMean at 82.8% in identifying ABPs targeting 10 bacterial species.|
|PCC|Pearson correlation coefficient for activity level prediction in AMPActiPred|38723168|10.1002/pro.5006|AMPActiPred demonstrates robust predictive capabilities for ABP activity levels with an average PCC of 0.722.|
|Pearson correlation coefficient|node-level MCP prediction evaluation|39262770|10.64898/2026.02.01.703163|achieved a Pearson correlation coefficient of 0.883|
|Coverage|Multi-label classification performance for iMFP-LG|39585308|10.1093/gpbjnl/qzae084|iMFP-LG achieved coverage of 0.803 on MFBP and 0.730 on MFTP.|
|Absolute true|Evaluation metric for iMFP-LG|39585308|10.1093/gpbjnl/qzae084|iMFP-LG achieved absolute true of 0.788 on MFBP and 0.616 on MFTP.|
|Absolute false|Evaluation metric for iMFP-LG|39585308|10.1093/gpbjnl/qzae084|iMFP-LG achieved absolute false of 0.078 on MFBP and 0.032 on MFTP.|
|G-measure|Used in SAMP benchmarking|39573886|10.1101/gr.254557.119|Same as above.|
|Minimum Inhibitory Concentration (MIC)|global quantitative AMP activity regression module in the FBGAN-enhanced framework|40806517|10.3390/ijms26157387|optimizes computational predictions of Minimum Inhibitory Concentration (MIC) values|
|AUCROC|Appred and sAMPpred-GAT|39247292|10.1016/j.heliyon.2024.e36163|AUCROC 0.99 for Appred; sAMPpred-GAT AuC mentioned|
|average precision|evaluation of AMPs-Net model|35877911|10.3389/fmicb.2021.710199|outperforms the state-of-the-art method by 8.8% in average precision|
|absolute truth value|multi-label metric|40691539|10.1186/s12859-025-06190-5|BPFun absolute truth value 0.6573|
|Accuracy (Acc)|CL-ACP: average Acc 83.83% (ACP736), 87.92% (ACP240), 84.41% (ACP539)|34670488|10.1186/s12859-021-04433-9|Five-fold cross-validation average Acc reported.|
|Sensitivity (Sens)|CL-ACP: average Sens 82.93% (ACP736), 90.74% (ACP240), 77.48% (ACP539)|34670488|10.1186/s12859-021-04433-9|Five-fold cross-validation average Sens reported.|
|Specificity (Spec)|CL-ACP: average Spec 84.76% (ACP736), 84.76% (ACP240), 88.23% (ACP539)|34670488|10.1186/s12859-021-04433-9|Five-fold cross-validation average Spec reported.|
|Precision (Prec)|CL-ACP: average Prec 85.15% (ACP736), 88.41% (ACP240), 78.46% (ACP539)|34670488|10.1186/s12859-021-04433-9|Five-fold cross-validation average Prec reported.|
|MCC (for QSAR)|AMPTrans-lstm: used to evaluate the QSAR classifiers|36618982|10.1016/j.csbj.2022.12.029|MCC was used to evaluate the QSAR model.|
|Matthew Correlation Coefficient (MCC)|binary classification performance|29379261|10.6026/97320630013415|MCC value of 0.86 ... MCC of 0.73|
|ROC|binary classification performance|29379261|10.6026/97320630013415|using the most suitable performance measures like accuracy, Mathew Correlation Coefficient, ROC etc.|
|AUPRC|||||
|MCC|||||
|Recall/Sensitivity|||||
|Precision|||||
|ACC|||||
|Specificity|||||
|AUROC|||||
|F1|||||
|ECE|||||
|95% CI|||||
|Recall|||||
|Accuracy|||||
|F1-score|||||
|Hamming loss|Multi-label evaluation of AMP activity prediction|28526820|10.1038/s41598-017-01986-9|Paper reports Hamming loss, one-error, coverage, ranking loss, average precision.|
|One-error|Multi-label evaluation|28526820|10.1038/s41598-017-01986-9|Used in comparing multi-label algorithms.|
|Ranking loss|Multi-label evaluation|28526820|10.1038/s41598-017-01986-9|Reported in results.|
|Kappa|Model evaluation|29374199|10.1038/s41598-018-19752-w|Kappa statistic of 0.9.|
|False Positive Rate|model evaluation|23056189|10.1371/journal.pone.0045012|FPR reported.|
|external_test_activity|||||
|Experimental antimicrobial hit rate|Proportion of predicted peptides showing antimicrobial activity in vitro|39764027|10.1101/2024.12.17.628923|53 out of 58 synthesized VEPs (91.4%) exhibited activity against at least one pathogenic strain.|
|Optimization success rate|Percentage of ApexGO-optimized peptides with enhanced antimicrobial activity against Gram-negative pathogens|42206144|10.1038/s42256-026-01237-5|ApexGO achieved an 85% ground-truth experimental hit rate and a 72% success rate in enhancing antimicrobial activity against Gram-negative pathogens.|
|Recall (Sensitivity)|||||

## Papers

|title|pmid|pmcid|doi|year|role|open_fulltext_status|
|---|---|---|---|---|---|---|
|Co-AMPpred for in silico-aided predictions of antimicrobial peptides by integrating composition-based features.|34330209|PMC8325260|10.1186/s12859-021-04305-2|2021|original_model_paper||
|Bioinformatic Analysis of the Wound Peptidome Reveals Potential Biomarkers and Antimicrobial Peptides.|33613550|PMC7888259|10.3389/fimmu.2020.620707|2020|uncertain||
|Therapeutic peptide development revolutionized: Harnessing the power of artificial intelligence for drug discovery.|39605829|PMC11600032|10.1016/j.heliyon.2024.e40265|2024|review_or_secondary||
|Intelligent in-silico prioritization of antimalarial peptide candidates under explicit physicochemical windows via de novo CTCM-Neo generation and conformal-gated calibrated classification.|41859462|PMC12996230|10.3389/fcimb.2026.1707267|2026|original_model_paper||
|A-CaMP: a tool for anti-cancer and antimicrobial peptide generation.|31870207|PMC12806129|10.1080/07391102.2019.1708796|2021|original_model_paper||
|Recent Developments in Antimicrobial-Peptide-Conjugated Gold Nanoparticles.|28892365|PMC13284850|10.1021/acs.bioconjchem.7b00368|2017|review_or_secondary||
|Molecular Dynamics for Antimicrobial Peptide Discovery.|33558318|PMC6545919|10.1021/acs.biochem.9b00440|2021|review_or_secondary||
|Targeting Multidrug Resistance With Antimicrobial Peptide-Decorated Nanoparticles and Polymers.|35432230|PMC6934137|10.1021/acsami.5b12688|2022|review_or_secondary||
|AAGP integrates physicochemical and compositional features for machine learning-based prediction of anti-aging peptides|40781463|12334579|10.1038/s41598-025-12759-0|2025|uncertain||
|PCSPred: Prediction of Short Chain Antimicrobial Peptides using Machine Learning Algorithms|40781463||10.1109/NEleX59773.2023.10421222|2023|original_model_paper||
|Knowledge Discovery from Bioactive Peptide Data in the PepLab Database Through Quantitative Analysis and Machine Learning|||10.3390/sci7030122|2025|uncertain||
|Machine Learning-Driven Discovery and Evaluation of Antimicrobial Peptides from Crassostrea gigas Mucus Proteome|39330266|11432763|10.3390/md22090385|2024|benchmark_paper||
|Structural and functional evaluation of the palindromic alanine-rich antimicrobial peptide Pa-MAP2.|27063608|PMC11302733|10.1016/j.bbamem.2016.04.003|2016|uncertain||
|Liposomes encapsulating novel antimicrobial peptide Omiganan: Characterization and its pharmacodynamic evaluation in atopic dermatitis and psoriasis mice model.|35878872|PMC13113520|10.1016/j.ijpharm.2022.122045|2022|uncertain||
|Combination Effects of Antimicrobial Peptides.|26729502|PMC4380922|10.1111/eva.12202|2016|uncertain||
|Multimodal geometric learning for antimicrobial peptide identification by leveraging alphafold2-predicted structures and surface features.|40462515|PMC7455913|10.1186/s12864-020-06978-0|2025|original_model_paper||
|Active Learning A Neural Network Model For Gold Clusters &amp; Bulk From Sparse First Principles Training Data|||10.1002/cctc.202000774|2020|uncertain||
|An application of Random Forests to a genome-wide association dataset: Methodological considerations &amp; new findings|20546594|PMC2896336|10.1186/1471-2156-11-49|2010|uncertain||
|Deep learning models for bacteria taxonomic classification of metagenomic data|30066629|PMC6069770|10.1186/s12859-018-2182-6|2018|uncertain||
|ACP-MHCNN: an accurate multi-headed deep-convolutional neural network to predict anticancer peptides|34880291|PMC8654959|10.1038/s41598-021-02703-3|2021|original_model_paper||
|MultiPep: a hierarchical deep learning approach for multi-label classification of peptide bioactivities|34909478|PMC8665375|10.1093/biomethods/bpab021|2021|model_original||
|Prediction of anticancer peptides based on an ensemble model of deep learning and machine learning using ordinal positional encoding|36642410||10.1093/bib/bbac630|2023|unclear||
|Comparative Protein Structure Modeling Using MODELLER|||10.1002/cpbi.3|2016|unclear||
|SAGE Handbook of Mixed Methods in Social & Behavioral Research|||10.4135/9781506335193|2010|unclear||
|Adaptive peptide design.|24594327|PMC7547863|10.2533/chimia.2013.859|2013|unclear||
|Computational resources and tools for antimicrobial peptides.|27966278|PMC13296189|10.1002/psc.2947|2017|review_or_secondary||
|A review on antimicrobial peptides databases and the computational tools.|35305010|PMC7038045|10.1093/database/baab012|2022|review_or_secondary||
|Antibiotic discovery in the artificial intelligence era.|36447334|PMC13203839|10.1111/nyas.14930|2023|review_or_secondary||
|Progress in the Identification and Design of Novel Antimicrobial Peptides Against Pathogenic Microorganisms|39557756|PMC11925980|10.1007/s12602-024-10402-4|2024|review_or_secondary||
|Identification and Characterization of Three New Antimicrobial Peptides from the Marine Mollusk Nerita versicolor (Gmelin, 1791)|36835264|PMC9968088|10.3390/ijms24043852|2023|dataset_paper||
|New Antibacterial Peptides from the Freshwater Mollusk Pomacea poeyana (Pilsbry, 1927)|33113998|PMC7690686|10.3390/biom10111473|2020|dataset_paper||
|AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens|||10.21203/rs.3.rs-120780/v1|2020|original_model_paper||
|Alignment-Free Antimicrobial Peptide Predictors: Improving Performance by a Thorough Analysis of the Largest Available Data Set.|34081438|PMC13233393|10.1021/acs.jcim.1c00251|2021|original_model_paper||
|Antimicrobial Activity of Mesenchymal Stem Cells: Current Status and New Perspectives of Antimicrobial Peptide-Based Therapies.|28424688|PMC3707526|10.1007/s00109-009-0588-3|2017|review_or_secondary||
|Antimicrobial peptides: An alternative to traditional antibiotics.|38147812|PMC13295260|10.1016/j.ejmech.2023.116072|2024|review_or_secondary||
|The double-edged sword of probiotic supplementation on gut microbiota structure in|35951774|PMC6658209|10.1038/nmicrobiol.2017.57|2022|uncertain||
|Generating and screening de novo compounds against given targets using ultrafast deep learning models as core components|35724626|PMC11967820|10.1093/bib/bbac226|2022|original_model_paper||
|Evolutionary patterns of structural disorder and post-translational modifications in the 18.5 kDa myelin basic protein|||10.24193/subbbiol.2025.2.08|2025|uncertain||
|Depth-corrected multi-factor dissection of chromatin accessibility for scATAC-seq data with PACS.|39757254|PMC6836739|10.1101/gr.275223.121|2025|uncertain||
|Sequence Analysis|||10.1016/b978-0-12-809633-8.20106-4|2020|uncertain||
|ANIA: an inception-attention network for predicting minimum inhibitory concentration of antimicrobial peptides.|41664908|PMC12895073|10.1093/bib/bbag023|2026|original_model_paper||
|Harnessing Sequence Embedding and Ensemble Learning to Identify Antifungal Peptides with Low Hemolytic Risk.|42146199|PMC13177248|10.1021/acsomega.6c00049|2026|original_model_paper||
|α-Helical Peptides Encoded in Collagen Exhibit Antimicrobial Activity with Low Cytotoxicity.|41528266|PMC12836314|10.1021/acs.jnatprod.5c01318|2026|dataset_paper||
|Generative models for antimicrobial peptide design: auto-encoders and beyond.|42106831|PMC13181977|10.1186/s13040-026-00558-w|2026|benchmark_paper||
|AI4AMP: an Antimicrobial Peptide Predictor Using Physicochemical Property-Based Encoding Method and Deep Learning.|34783578|PMC7302108|10.1109/bibm.2016.7822515|2021|original_model_paper||
|AI4AMP: an Antimicrobial Peptide Predictor Using Physicochemical Property-Based Encoding Method and Deep Learning.|34783578|PMC8594441|10.1128/msystems.00299-21|2021|original_model_paper||
|Unifying antimicrobial peptide datasets for robust deep learning-based classification.|39281014|PMC9712827|10.1093/bioinformatics/btv180|2024|dataset_paper||
|Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships.|27870247|PMC12834174|10.1002/minf.201600029|2016|original_model_paper||
|LysePred: A Multiscale Convolutional Neural Network for Predicting Hemolytic Activity of Antimicrobial Peptides.|42338220|not_reported_in_available_evidence|10.1021/acssynbio.6c00173|2026|review_or_secondary||
|SAMP: Identifying antimicrobial peptides by an ensemble learning model based on proportionalized split amino acid composition.|39573886|PMC11631067|10.1093/bfgp/elae046|2024|original_model_paper||
|Deep Learning Combined with Quantitative Structure‒Activity Relationship Accelerates De Novo Design of Antifungal Peptides.|39921483|PMC11967820|10.1002/advs.202412488|2025|original_model_paper||
|AI4AMP: Sequence-based antimicrobial peptides predictor using physicochemical properties-based encoding method and deep learning|not_reported_in_available_evidence|not_reported_in_available_evidence|10.1101/2020.12.17.423359|2020|original_model_paper||
|Identification and Characterization of RK22, a Novel Antimicrobial Peptide from Hirudinaria manillensis against Methicillin Resistant Staphylococcus aureus.|37686259|PMC10487658|10.3390/ijms241713453|2023|review_or_secondary||
|Graph-Contrastive Convolutional Neural Network for Extracting and Classifying Peptide-Based Periodontal Immunomodulatory and Anti-Inflammatory Signatures.|41197436|PMC12637048|10.1016/j.identj.2025.103986|2026|review_or_secondary||
|In Silico Identification and Molecular Characterization of Lentilactobacillus hilgardii Antimicrobial Peptides with Activity Against Carbapenem-Resistant Acinetobacter baumannii.|41148698|PMC12561633|10.3390/antibiotics14101004|2025|dataset_paper||
|Intelligent De Novo Design of Novel Antimicrobial Peptides against Antibiotic-Resistant Bacteria Strains.|37047760|PMC10095442|10.3390/ijms24076788|2023|review_or_secondary||
|Review and perspective on bioinformatics tools using machine learning and deep learning for predicting antiviral peptides.|37626205|9710571|10.1109/JBHI.2021.3130825|2024|original_model_paper||
|Design methods for antimicrobial peptides with improved performance.|37914524|PMC4262413|10.1371/journal.pone.0114605|2023|review_or_secondary||
|Repositioning Antimicrobial Peptides Against WHO-Priority Fungi.|40884276|PMC7657096|10.1002/advs.202509567|2025|review_or_secondary||
|Antimicrobial peptides with cell-penetrating activity as prophylactic and treatment drugs.|36052730|PMC7937881|10.3389/fmicb.2021.616979|2022|review_or_secondary||
|Peptide-based drug design using generative AI.|41376388|PMC13060724|10.1039/d5cc04998a|2026|review_or_secondary||
|PepForge: Hierarchical HELM-Based Peptide Generation|||10.64898/2026.05.29.728379|2026|original_model_paper||
|Accelerating antimicrobial peptide design: Leveraging deep learning for rapid discovery.|39705302|PMC11661626|10.1371/journal.pone.0315477|2024|original_model_paper||
|BBATProt: a framework predicting biological function with enhanced feature extraction via interpretable deep learning.|41212592|PMC12599320|10.1093/bib/bbaf593|2025|original_model_paper||
|AMAP: Hierarchical multi-label prediction of biologically active and antimicrobial peptides.|30831306|PMC13233393|10.1016/j.compbiomed.2019.02.018|2019|original_model_paper||
|Antimicrobial peptides from ranid frogs: taxonomic and phylogenetic markers and a potential source of new therapeutic agents.|14726199|PMC13305825|10.1016/j.bbapap.2003.09.004|2004|review_or_secondary||
|From Data to Decisions: Leveraging Artificial Intelligence and Machine Learning in Combating Antimicrobial Resistance - a Comprehensive Review.|39088151|PMC8404696|10.1128/cmr.00050-19|2024|review_or_secondary||
|Antimicrobial Compounds from Microorganisms.|35326749|PMC6416458|10.2174/0929866525666181026160852|2022|review_or_secondary||
|Ensemble Machine Learning and Predicted Properties Promote Antimicrobial Peptide Identification.|38972032|6941814|10.1007/s12539-024-00640-z|2024|original_model_paper||
|Deep-AmPEP30: Improve Short Antimicrobial Peptides Prediction with Deep Learning.|32464552|PMC5210628|10.1109/INDCON.2011.6139332|2020|original_model_paper||
|EBAMP: An efficient de novo broad-spectrum antimicrobial peptide discovery framework.|40906555|PMC13310060|10.1016/j.celrep.2025.116215|2025|original_model_paper||
|DLFea4AMPGen de novo design of antimicrobial peptides by integrating features learned from deep learning models.|41093853|PMC8601225|10.1002/adma.202307680|2025|original_model_paper||
|AMP-BERT: Prediction of antimicrobial peptide function based on a BERT model.|36461699|PMC3765848|10.1002/pro.4529|2023|original_model_paper||
|Screening antimicrobial peptides and probiotics using multiple deep learning and directed evolution strategies.|39234615|PMC2935846|10.1016/j.apsb.2024.05.003|2024|original_model_paper||
|Novel antimicrobial peptides against Cutibacterium acnes designed by deep learning|38402320|not_reported_in_available_evidence|10.1038/s41598-024-55205-3|2024|original_model_paper||
|A novel antibacterial peptide recognition algorithm based on BERT|34037687|PMC13143419|10.1093/bib/bbab200|2021|original_model_paper||
|Recent Progress in the Discovery and Design of Antimicrobial Peptides Using Traditional Machine Learning and Deep Learning.|36290108|PMC7326367|10.1038/s42256-019-0138-9|2022|review_or_secondary||
|Discovery of naturally inspired antimicrobial peptides using deep learning.|40209356|PMC13161480|10.1016/j.bioorg.2025.108444|2025|uncertain||
|Contemporary data-driven innovations in peptide-based therapeutic design.|42153319|PMC13184531|10.1093/bib/bbag220|2026|review_or_secondary||
|Harnessing generative AI for predicting and optimizing antimicrobial peptides against drug-resistant infections.|42174216|PMC13230547|10.1038/s44259-026-00218-3|2026|original_model_paper||
|Artificial Intelligence as a Catalyst for Antimicrobial Discovery: From Predictive Models to De Novo Design.|41753681|PMC12943268|10.3390/microorganisms14020394|2026|review_or_secondary||
|AMP-CapsNet: a multi-view feature fusion approach for antimicrobial peptide prediction using capsule networks.|41654884|PMC12977703|10.1186/s44342-026-00067-6|2026|original_model_paper||
|Artificial intelligence-driven antimicrobial peptide discovery.|37992451|PMC13223582|10.1016/j.sbi.2023.102733|2023|review_or_secondary||
|Innate immunity in C. elegans.|33992157|PMC4850687|10.1016/bs.ctdb.2020.12.007|2021|uncertain||
|Crayfish immunity - Recent findings.|28502650|PMC13088477|10.1016/j.dci.2017.05.010|2018|uncertain||
|Methods to study Drosophila immunity.|24631888|PMC12767022|10.1016/j.ymeth.2014.02.023|2014|uncertain||
|Designing Anticancer Peptides by Constructive Machine Learning.|29679519|PMC13109931|10.1002/cmdc.201800204|2018|uncertain||
|De Novo Peptide and Protein Design Using Generative Adversarial Networks: An Update.|35128926|PMC13184531|10.1021/acs.jcim.1c01361|2022|review_or_secondary||
|Deep Learning Based Drug Screening for Novel Coronavirus 2019-nCov.|32488835|PMC2142462|10.1080/07391102.2017.1415822|2020|uncertain||
|Deep Learning-Enhanced Generation and Screening of Antihyperuricemic Peptides from Chickpea Proteins: from Multienzyme Optimization to Molecular Mechanisms.|42158992||10.1021/acs.jafc.6c00467|2026|uncertain||
|Machine Learning-Driven Discovery and Evaluation of Antimicrobial Peptides from Crassostrea gigas Mucus|39330266|PMC9917735|10.3390/ijms24032914|2024|uncertain||
|Machine Learning Accelerates De Novo Design of Antimicrobial Peptides|38416364|5902452|10.1038/s41467-018-03746-3|2024|original_model_paper||
|Protein Language Models and Machine Learning Facilitate the Identification of Antimicrobial Peptides|39201537|PMC5834480|10.3389/fmicb.2018.00323|2024|original_model_paper||
|AMP-RNNpro: a two-stage approach for identification of antimicrobials using probabilistic features|38839785|PMC9421197|10.1016/j.csbj.2022.07.043|2024|original_model_paper||
|AMP-distillation: A knowledge distillation framework for accurate and efficient antimicrobial peptide prediction.|42155201||10.1016/j.compbiolchem.2026.109129|2026|original_model_paper||
|A Deep Hypergraph Learning Model for Predicting Antimicrobial Combination Effects Across Bacterial Targets|||10.64898/2026.06.09.731104|2026|uncertain||
|iAMP-SeE: an antimicrobial peptide recognition model based on ESM2 feature extraction and hybrid attention mechanisms.|41913931|PMC13033287|10.7717/peerj.20978|2026|original_model_paper||
|Species- and Topic-aware Representation Learning for Antimicrobial Peptide Discovery|||10.64898/2026.05.28.728246|2026|original_model_paper||
|Progress in the Development of Antimicrobial Peptide Prediction Tools|||10.2174/1389203721666200117163802|2021|review_or_secondary||
|Predicting Antimicrobial Activity for Untested Peptide-Based Drugs Using Collaborative Filtering and Link Prediction|||10.1101/2022.11.16.516845|2022|original_model_paper||
|Artificial Intelligence-Driven Antimicrobial Peptide Discovery: Prediction, Generation, Mining and Optimization|42020672||10.1007/s12602-026-11013-x|2026|review_or_secondary||
|AMP-DualTransnet: An effective dual-pathway deep learning model for antimicrobial peptide prediction in black pepper|||10.1016/j.nexres.2026.101536|2026|original_model_paper||
|AMP-FreqNet:A Frequency-Domain Enhanced Model for Antimicrobial Peptide Prediction with Efficient Attention Mechanism|||10.1145/3766671.3766835|2025|original_model_paper||
|Predicting Antimicrobial Activity for Untested Peptide-Based Drugs Using Collaborative Filtering and Link Prediction|||10.1021/acs.jcim.3c00137|2023|original_model_paper||
|Accelerating Antimicrobial Peptide Discovery for WHO Priority Pathogens through Predictive and Interpretable Machine Learning Models|||10.1021/acsomega.3c08676.s001|2024|original_model_paper||
|Prediction of Antimicrobial Peptides Using Machine Learning Approach|||10.54985/peeref.2405p7278831|2024|original_model_paper||
|A generative explainable model for antimicrobial peptide prediction using bidirectional temporal convolutional neural network.|41844874|PMC13128923|10.1038/s41598-026-43370-6|2026|original_model_paper||
|Biochemical-knowledge-driven machine learning pipeline for generating potent antimicrobial peptides.|41849223|PMC12998437|10.1093/bib/bbag115|2026|original_model_paper||
|Harnessing Machine Learning Approaches for the Identification, Characterization, and Optimization of Novel Antimicrobial Peptides.|41463765|PMC12730010|10.3390/antibiotics14121263|2025|review_or_secondary||
|Artificial Intelligence-Driven Discovery and Optimization of Antimicrobial Peptides Targeting ESKAPE Pathogens and Multidrug-Resistant Fungi.|41900351|PMC13029496|10.3390/microorganisms14030591|2026|review_or_secondary||
|Deep learning-driven integrated pipeline for de novo design and synthesis of antimicrobial peptides||13267055|10.1038/s44386-026-00045-6|2026|original_model_paper||
|deep-AMPpred: A Deep Learning Method for Identifying Antimicrobial Peptides and Their Functional Activities.|39792442|PMC13274993|10.1021/acs.jcim.4c01913|2025|original_model_paper||
|Rapid Response Antimicrobial Peptide Design Strategy Driven by Meta-Learning for Emerging Drug-Resistant Pathogens.|40193623|PMC13267055|10.1021/acs.jmedchem.5c00188|2025|uncertain||
|Design and evaluation of octopromycin-derived peptides as multifunctional antimicrobial agents against multidrug-resistant pathogens.|41272087|PMC5574609|10.1038/s41579-022-00791-0|2025|uncertain||
|PepVAE: Variational Autoencoder Framework for Antimicrobial Peptide Generation and Activity Prediction|34659152|8515052|10.3389/fmicb.2021.725727|2021|original_model_paper||
|Structure, mechanism and crystallographic fragment screening of the SARS-CoV-2 NSP13 helicase|34381037|8358061|10.1038/s41467-021-25166-6|2021|uncertain||
|LMPred: predicting antimicrobial peptides using pre-trained language models and deep learning.|36699381|PMC6540740|10.1101/2020.07.12.199554v3|2022|original_model_paper||
|Prediction of antimicrobial peptides based on sequence alignment and support vector machine-pairwise algorithm utilizing LZ-complexity.|25802839|PMC2686604|10.1093/nar/gkn823|2015|original_model_paper||
|A dual diffusion model-based representation learning framework for antimicrobial peptides classification.|41692989|PMC12960902|10.1093/bioinformatics/btag077|2026|model_original||
|UniAMP: enhancing AMP prediction using deep neural networks with inferred information of peptides.|39799358|PMC11725221|10.1186/s12859-025-06033-3|2025|model_original||
|AMP-RNNpro: a two-stage approach for identification of antimicrobials using probabilistic features.|38839785|PMC11153637|10.1038/s41598-024-63461-6|2024|model_original||
|Leveraging Different Distance Functions to Predict Antiviral Peptides with Geometric Deep Learning from ESMFold-Predicted Tertiary Structures.|41594075|PMC12837384|10.3390/antibiotics15010039|2026|unclear (antiviral peptide prediction, not AMP)||
|E-CLEAP: An ensemble learning model for efficient and accurate identification of antimicrobial peptides.|38722967|PMC11081394|10.1371/journal.pone.0300125|2024|original_model_paper||
|Examining the functional space of gut microbiome-derived peptides.|38129980|PMC10714122|10.1002/mbo3.1393|2023|review_or_secondary||
|Deep learning-driven integrated pipeline for de novo design and synthesis of antimicrobial peptides|not_reported_in_available_evidence|PMC13267055|not_reported_in_available_evidence|2026|original_model_paper||
|Recent Applications of Artificial Intelligence in Discovery of New Antibacterial Agents.|39650228|PMC11624680|10.2147/aabc.s484321|2024|review_or_secondary||
|The investigation of antibacterial properties of peptides and protein hydrolysates derived from serum of Asian water monitor (Varanus salvator).|37851665|PMC10584125|10.1371/journal.pone.0292947|2023|benchmark_paper||
|Antimicrobial Peptide Identified via Machine Learning Presents Both Potent Antibacterial Properties and Low Toxicity toward Human Cells.|39203524|PMC11356914|10.3390/microorganisms12081682|2024|benchmark_paper||
|Antimicrobial Peptides Design Using Deep Learning and Rational Modifications: Activity in Bacteria, Candida albicans, and Cancer Cells.|40643674|PMC12254070|10.1007/s00284-025-04346-3|2025|benchmark_paper||
|Restraint of VP1 Protein of Foot and Mouth Disease Virus using Specific Antiviral Peptides: an in Silico Investigation.|38590669|PMC10998950|10.22092/ari.2023.78.5.1483|2023|uncertain||
|Unveiling the evolution of antimicrobial peptides in gut microbes via foundation-model-powered framework|40445833|PMC13310060|10.1016/j.celrep.2025.115773|2025|original_model_paper||
|Bioactive peptides for anticancer therapies|37206303|PMC4758372|10.12336/biomatertransl.2023.01.003|2023|uncertain||
|Development and evaluation of a chewing gum containing antimicrobial peptide GH12 for caries prevention|35917355|PMC12938800|10.1111/eos.12887|2022|uncertain||
|Antibacterial activity and cytocompatibility evaluation of the antimicrobial peptide Nal-P-113-loaded graphene oxide coating on titanium|36244736|PMC13046344|10.4012/dmj.2022-094|2022|uncertain||
|DRAMP 2.0, an updated data repository of antimicrobial peptides|31409791|PMC6692298|10.1038/s41597-019-0154-y|2019|dataset_paper||
|Deep-AmPEP30: Improve Short Antimicrobial Peptides Prediction with Deep Learning|32464552|PMC7256447|10.1016/j.omtn.2020.05.006|2020|original_model_paper||
|Prediction of antimicrobial peptides toxicity based on their physico-chemical properties using machine learning techniques|34758751|PMC8582201|10.1186/s12859-021-04468-y|2021|original_model_paper||
|CalcAMP: A New Machine Learning Model for the Accurate Prediction of Antimicrobial Activity of Peptides|37107088|PMC10135148|10.3390/antibiotics12040725|2023|original_model_paper||
|Connecting Peptide Physicochemical and Antimicrobial Properties by a Rational Prediction Model|21347392|PMC3036733|10.1371/journal.pone.0016968|2011|original_model_paper||
|Deep learning regression model for antimicrobial peptide design|||10.1101/692681|2019|original_model_paper||
|AMP-zGSM: A z-Scoring Enhanced Grouping--Scoring--Modeling Framework for Antimicrobial Peptide Prediction|||10.5220/0014457300004070|2026|original_model_paper||
|PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction|||10.1021/acs.jpclett.3c02398.s001|2023|uncertain||
|In Vivo Antibacterial Efficacy of Antimicrobial Peptides Modified Metallic Implants─Systematic Review and Meta-Analysis.|35412810|PMC7393194|10.1111/bph.15193|2022|review_or_secondary||
|AMP0: targeted antimicrobial peptide activity prediction using zero and few shot machine learning|32750857|PMC4879360|10.1109/TCBB.2020.2999399|2022|original_model_paper||
|Structural Classification Insights Into the Plant Defensive Peptides.|39161242||10.1002/prot.26736|2024|uncertain||
|sAMPpred-GAT: prediction of antimicrobial peptide by graph attention network and predicted peptide structure.|36342186|PMC3688957|10.1093/bioinformatics/btac715|2023|original_model_paper||
|Measurement of electrocardiograms in a bath through tap water utilizing capacitive coupling electrodes placed outside the bathtub wall.|28086891|PMC5234137|10.1186/s12938-016-0304-9|2017|unrelated||
|A genetically encoded biosensor reveals heterogenous cAMP dynamics coordinating growth and resuscitation in Mycobacterium tuberculosis|||10.64898/2026.06.02.729633|2026|unrelated||
|Gaming-based program for internet gaming disorder: feasibility and preliminary outcomes of a structured camp program.|42344677|PMC13288209|10.3389/fpsyt.2026.1825298|2026|unrelated||
|PyAMPA: a high-throughput prediction and optimization tool for antimicrobial peptides.|38934543|PMC11264690|10.1128/msystems.01358-23|2024|original_model_paper||
|BERT & Family Eat Word Salad: Experiments with Text Understanding|||10.1609/aaai.v35i14.17531|2021|unrelated||
|Transformers: State-of-the-Art Natural Language Processing|||10.5281/zenodo.5347031|2020|unrelated||
|Entreprises non cotées et différence d'évaluation entre LBO et M&A|||10.3917/g2000.296.0093|2012|unrelated||
|Mining the heparinome for cryptic antimicrobial peptides that selectively kill Gram-negative bacteria|40410382|PMC12223310|10.1038/s44320-025-00120-6|2025|review_or_secondary||
|AI Methods for Antimicrobial Peptides: Progress and Challenges.|39754551|PMC11702388|10.1111/1751-7915.70072|2025|review_or_secondary||
|AntiBP3: A Method for Predicting Antibacterial Peptides against Gram-Positive/Negative/Variable Bacteria.|38391554|PMC10885866|10.3390/antibiotics13020168|2024|original_model_paper||
|AMPActiPred: A three-stage framework for predicting antibacterial peptides and activity levels with deep forest.|38723168|PMC11081525|10.1002/pro.5006|2024|original_model_paper||
|Protease-Resistant, Broad-Spectrum Antimicrobial Peptides with High Antibacterial and Antifungal Activity.|40003651|PMC11856857|10.3390/life15020242|2025|uncertain||
|dbAMP 3.0: updated resource of antimicrobial activity and structural annotation of peptides in the post-pandemic era.|39540425|PMC11701527|10.1093/nar/gkae1019|2025|original_model_paper||
|Transcriptomic analysis of non-model Drosophilidae reveals novel AMP candidates.|41634709|PMC12958777|10.1186/s12915-026-02535-5|2026|dataset_paper||
|De Novo Design and In Silico Validation of a Cationic Antimicrobial Peptide Using an AI-Guided Framework for Membrane Thermodynamics and Hemolytic Toxicity|||10.21203/rs.3.rs-9615735/v1|2026|uncertain||
|BATTLE-AMP: Benchmarking Antimicrobial Peptide Predictors|||10.64898/2026.06.19.733349|2026|benchmark_paper||
|AMPBAN: A Deep Learning Framework Integrating Protein Sequence and Structural Features for Antimicrobial Peptide Prediction|||10.64898/2026.01.20.700468|2026|original_model_paper||
|Generating antimicrobial peptides via genomic transfer learning|||10.64898/2026.06.16.732639|2026|original_model_paper||
|How Machine Learning Helps in Combating Antimicrobial Resistance: A Review of AMP Analysis and Generation Methods|||10.1007/s10989-025-10716-z|2025|review_or_secondary||
|The impact of negative data sampling on antimicrobial peptide prediction|||10.1101/2022.05.30.493946|2022|benchmark_paper||
|Machine Learning Accelerates De Novo Design of Antimicrobial Peptides|38416364|PMC12729384|10.1007/s12539-024-00612-3|2024|uncertain||
|Machine learning assisted rational design of antimicrobial peptides based on human endogenous proteins and their applications for cosmetic preservative system optimization|38200054|10781772|10.1038/s41598-023-50832-8|2024|uncertain||
|Harnessing Generative Pre-trained Transformer for Antimicrobial Peptide Generation and MIC Prediction with Contrastive Learning|||10.1101/2025.03.07.642021|2025|original_model_paper||
|PepMCP: A Graph-Based Membrane Contact Probability Predictor for Membrane-Lytic Antimicrobial Peptides|||10.64898/2026.02.01.703163|2026|original_model_paper||
|Practical Machine Learning Framework for Designing and Predicting C-Amidated Antimicrobial Peptides|||10.21203/rs.3.rs-7764304/v1|2025|original_model_paper||
|New potential antimicrobial peptides with amazing symmetrical structure in fungi and insects|||10.64898/2025.12.21.695455|2025|uncertain||
|Integrated convolution and self-attention for improving peptide toxicity prediction|38696758|PMC11654579|10.1093/bioinformatics/btae297|2024|uncertain||
|iMFP-LG: Identify Novel Multi-functional Peptides Using Protein Language Models and Graph-based Deep Learning|39585308|PMC12011362|10.1093/gpbjnl/qzae084|2024|original_model_paper||
|Legume Plant Peptides as Sources of Novel Antimicrobial Molecules Against Human Pathogens|35755814|PMC9218685|10.3389/fmolb.2022.870460|2022|dataset_paper||
|De novo synthetic antimicrobial peptide design with a recurrent neural network|38988311|PMC11237553|10.1002/pro.5088|2024|uncertain||
|[Analysis of distribution and drug resistance of pathogens from the wounds of 1 310 thermal burn patients].|30481922|PMC11535417|10.3760/cma.j.issn.1009-2587.2018.11.016|2018|unclear||
|Unveiling novel antimicrobial peptides from the ruminant gastrointestinal microbiomes: A deep learning-driven approach yields an anti-MRSA candidate.|39756573|PMC9545273|10.1016/j.jare.2025.01.005|2025|original_model_paper||
|Deep Learning-Driven Discovery of Novel Antimicrobial Peptides from Large-Scale Protist Genomes and Experimental Characterization.|40958742|PMC12730010|10.1021/acs.jcim.5c01196|2025|original_model_paper||
|Antimicrobial Peptides From the Gut Microbiome of the Centenarians: Diversification of Biosynthesis and Youthful Development of Resistance Genes.|39207726||10.1093/gerona/glae218|2024|dataset_paper||
|amPEPpy 1.0: a portable and accurate antimicrobial peptide prediction tool.|33135060|PMC13289652|10.1093/bioinformatics/btaa917|2021|original_model_paper||
|Prospects for antimicrobial peptide-based immunotherapy approaches in Leishmania control.|29889579|PMC13184368|10.1080/14787210.2018.1483720|2018|review_or_secondary||
|Halocins, natural antimicrobials of Archaea: Exotic or special or both?|34509601|PMC13192219|10.1016/j.biotechadv.2021.107834|2021|review_or_secondary||
|Functional genomic analysis of the Drosophila immune response.|23707784|PMC5872677|10.1016/j.dci.2013.05.007|2014|review_or_secondary||
|Molecular de-extinction of ancient antimicrobial peptides enabled by machine learning|37516110|11625410|10.1016/j.chom.2023.07.001|2023|original_model_paper||
|Bacteria-Specific Feature Selection for Enhanced Antimicrobial Peptide Activity Predictions Using Machine-Learning Methods|36912047||10.1021/acs.jcim.2c01551|2023|original_model_paper||
|Comprehensive assessment of machine learning-based methods for predicting antimicrobial peptides|33774670|PMC13143419|10.1093/bib/bbab083|2021|benchmark_paper||
|Machine Learning‐Assisted Prediction and Generation of Antimicrobial Peptides|40529865|12168616|10.1002/smsc.202400579|2025|original_model_paper||
|Recent Progress in the Discovery and Design of Antimicrobial Peptides Using Traditional Machine Learning and Deep Learning|36290108|PMC9598685|10.3390/antibiotics11101451|2022|review_or_secondary||
|Aquatic Invertebrate Antimicrobial Peptides in the Fight Against Aquaculture Pathogens|39858924|PMC11767717|10.3390/microorganisms13010156|2025|review_or_secondary||
|Challenges and applications of artificial intelligence in infectious diseases and antimicrobial resistance|39843587|PMC11721440|10.1038/s44259-024-00068-x|2025|review_or_secondary||
|Automatic construction of molecular similarity networks for visual graph mining in chemical space of bioactive peptides: an unsupervised learning approach|33093586|PMC7582163|10.1038/s41598-020-75029-1|2020|review_or_secondary||
|Transcriptomic analysis of non-model Drosophilidae reveals novel AMP candidates|||10.21203/rs.3.rs-6856057/v1|2025|dataset_paper||
|Big Data & Deep Data: Minding the Challenges|||10.3233/978-1-61499-822-8-177|2017|uncertain||
|A mechanism-guided framework for prioritizing membrane-interaction anti-Vibrio peptides from peptidomics data.|42364293||10.1016/j.jmgm.2026.109497|2026|uncertain||
|Agentic Discovery of Non-Canonical Antimicrobial Peptides with AMPGAN v3||||2026|original_model_paper||
|From Innate Immunity to Cancer Therapy: Antimicrobial Peptides as Emerging Anticancer Agents.|42352904|PMC13299682|10.3390/ijms27125179|2026|review_or_secondary||
|PepAnno: A structure-aware deep learning framework for bioactive peptide prediction, structural visualization, and physicochemical profiling.|42228741||10.1371/journal.pcbi.1014369|2026|original_model_paper||
|AMPGP: Discovering Highly Effective Antimicrobial Peptides via Deep Learning.|40825014||10.1021/acs.jcim.5c00647|2025|original_model_paper||
|AI-driven discovery of antimicrobial peptides and derivatives: database and tools|||10.37349/eds.2026.1008161|2026|review_or_secondary||
|Antimicrobial Peptides Against Antimicrobial-Resistant Bacteria: Focus on Machine Learning|||10.2147/idr.s602699|2026|review_or_secondary||
|A Systematic Benchmark for Peptide Property Prediction|||10.64898/2026.02.09.704773|2026|benchmark_paper||
|SAMP: Identifying Antimicrobial Peptides by an Ensemble Learning Model Based on Proportionalized Split Amino Acid Composition.|38712184|PMC7050522|10.1128/aac.02340-16|2024|original_model_paper||
|SAMP: Identifying antimicrobial peptides by an ensemble learning model based on proportionalized split amino acid composition.|39573886|PMC11081394|10.1101/gr.254557.119|2024|original_model_paper||
|Proteomic Screening for Prediction and Design of Antimicrobial Peptides with AmpGram.|32560350|PMC7352166|10.3390/ijms21124310|2020|original_model_paper||
|Multifunctional Peptides from Equine Milk Lactoferrin: Evaluation of Antimicrobial Activity In Silico and In Vitro.|42071989|PMC8066137|10.1007/s10989-023-10541-2|2026|benchmark_paper||
|Structural information in therapeutic peptides: Emerging applications in biomedicine.|38877295|PMC11788753|10.1002/2211-5463.13847|2025|review_or_secondary||
|Mass Spectrometry-Based Peptidomics for the Discovery and Profiling of Endogenous Peptides in Crustacean Hemolymph.|42179631|PMC13191662|10.1021/acsomega.6c00679|2026|review_or_secondary||
|Unveiling the Bioactive Potential of the Invasive Jellyfish Phyllorhiza punctata Through Integrative Transcriptomic and Proteomic Analyses.|40867566|PMC12383608|10.3390/biom15081121|2025|review_or_secondary||
|Proteomic Screening for Prediction and Design of Antimicrobial Peptides with AmpGram.|32560350|PMC3293554|10.1074/jbc.M111.303602|2020|original_model_paper||
|Mechanism-Driven Screening of Membrane-Targeting and Pore-Forming Antimicrobial Peptides.|41391039|PMC12904036|10.1002/advs.202516470|2026|original_model_paper||
|GenPept-Curated-2025: A Benchmark Dataset for Antimicrobial Peptide Prediction with Homology-Controlled Partitioning|||10.64898/2026.04.25.720793|2026|dataset_paper||
|Antimicrobial peptide prediction based on contrastive learning and gated convolutional neural network.|41286313|PMC12749617|10.1038/s41598-025-29666-z|2025|original_model_paper||
|AmpHGT: expanding prediction of antimicrobial activity in peptides containing non-canonical amino acids using multi-view constrained heterogeneous graph transformer.|40598389|PMC12217533|10.1186/s12915-025-02253-4|2025|original_model_paper||
|Models and data of AMPlify: a deep learning tool for antimicrobial peptide prediction.|36732807|PMC6323992|10.1093/nar/gky1049|2023|original_model_paper||
|Encodings and models for antimicrobial peptide classification for multi-resistant pathogens.|30867681|PMC6658705|10.1186/s13040-019-0196-x|2019|review_or_secondary||
|Deep Learning for Antimicrobial Peptides: Computational Models and Databases.|39927895|PMC13274993|10.1021/acs.jcim.5c00006|2025|original_model_paper||
|Accelerating antimicrobial peptide design: Leveraging deep learning for rapid discovery.|39705302|PMC5846155|10.1007/s10989-023-10552|2024|uncertain||
|Multifunctional Peptides from Equine Milk Lactoferrin: Evaluation of Antimicrobial Activity In Silico and In Vitro.|42071989|PMC13113733|10.3390/ani16081223|2026|dataset_paper||
|Artificial intelligence drives the identification and screening of novel antibiotics and antimicrobial peptides.|41978380|PMC13076943|10.1093/bib/bbag107|2026|review_or_secondary||
|Harnessing AI for Antimicrobial Peptide Innovation against Multidrug Resistance.|41755839|PMC12933362|10.1021/jacsau.5c01520|2026|review_or_secondary||
|SGAC: a graph neural network framework for imbalanced and structure-aware AMP classification.|41662353|PMC12885103|10.1093/bib/bbag038|2026|original_model_paper||
|Deep Learning for Novel Antimicrobial Peptide Design|33810011|PMC8004669|10.3390/biom11030471|2021|original_model_paper||
|AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens|35078402|PMC8788131|10.1186/s12864-022-08310-4|2022|original_model_paper||
|Design of target specific peptide inhibitors using generative deep learning and molecular dynamics simulations|38383543|PMC10882002|10.1038/s41467-024-45766-2|2024|uncertain||
|iAMPCN: a deep-learning approach for identifying antimicrobial peptides and their functional activities|37369638|PMC10359087|10.1093/bib/bbad240|2023|original_model_paper||
|PrMFTP: Multi-functional therapeutic peptides prediction based on multi-head self-attention mechanism and class weight optimization|36094961|PMC9499272|10.1371/journal.pcbi.1010511|2022|original_model_paper||
|DeepAFP: An effective computational framework for identifying antifungal peptides based on deep learning|37595093|PMC10503419|10.1002/pro.4758|2023|original_model_paper||
|Design Methods for Antimicrobial Peptides with Improved Performance|37914524|PMC10802102|10.24272/j.issn.2095-8137.2023.246|2023|review_or_secondary||
|Characterization and Engineering Studies of a New Endolysin from the Propionibacterium acnes Bacteriophage PAC1 for the Development of a Broad-Spectrum Artilysin with Altered Specificity|37239874|PMC10218239|10.3390/ijms24108523|2023|uncertain||
|Diversity in penaeidin antimicrobial peptide form and function.|17716729|PMC168927|10.1016/j.dci.2007.06.009|2008|uncertain||
|Antimicrobial peptides recognition using weighted physicochemical property encoding.|37120707|PMC11584731|10.1142/S0219720023500063|2023|original_model_paper||
|Polymer-Antimicrobial Peptide Constructs with Tailored Drug-Release Behavior.|36839728|PMC9042309|10.1039/D1RA06231J|2023|uncertain||
|Marine Antimicrobial Peptides: An Emerging Nightmare to the Life-Threatening Pathogens.|37022565||10.1016/j.dci.2019.05.005|2024|review_or_secondary||
|An ensemble deep learning model for predicting minimum inhibitory concentrations of antimicrobial peptides against pathogenic bacteria.|39262770|PMC4702849|10.48550/arXiv.1810.11363|2024|original_model_paper||
|AMPpred-EL: An effective antimicrobial peptide prediction model based on ensemble learning.|35576825|PMC13044805|10.1016/j.compbiomed.2022.105577|2022|original_model_paper||
|Age-sex differences in the global burden of lower respiratory infections and risk factors, 1990-2019: results from the Global Burden of Disease Study 2019.|35964613|PMC8807230|10.1093/cid/ciab1051|2022|uncertain||
|Biomedical evidence engineering for data-driven discovery.|36227057|PMC11182767|10.1093/bioinformatics/btac675|2022|uncertain||
|Anti-hypertensive Peptide Predictor: A Machine Learning-Empowered Web Server for Prediction of Food-Derived Peptides with Potential Angiotensin-Converting EnzymeI Inhibitory Activity|||10.1021/acs.jafc.1c04555.s002|2021|unclear||
|Overcoming the Challenges in Machine Learning-Guided Antimicrobial Peptide Design|||10.17952/36eps.2022.207|2022|review_or_secondary||
|AMPpred-MFA: An Interpretable Antimicrobial Peptide Predictor with a Stacking Architecture, Multiple Features, and Multihead Attention|||10.1021/acs.jcim.3c01017.s001|2023|original_model_paper||
|Computational Design of Potentially Multifunctional Antimicrobial Peptide Candidates via a Hybrid Generative Model.|40806517|PMC12347886|10.3390/ijms26157387|2025|original_model_paper||
|AI-Driven Antimicrobial Peptide Discovery: Mining and Generation.|40459283|PMC12177927|10.1021/acs.accounts.0c00594|2025|review_or_secondary||
|Comparison between strip sampling and laser ablation methods to infer seasonal movements from intra-tooth strontium isotopes profiles in migratory caribou.|36869076|PMC9984400|10.1038/s41598-023-30222-w|2023|uncertain||
|Functional Plasticity Coupled With Structural Predispositions in Auditory Cortex Shape Successful Music Category Learning.|35837119|PMC9274125|10.3389/fnins.2022.897239|2022|uncertain||
|AniAMPpred: artificial intelligence guided discovery of novel antimicrobial peptides in animal kingdom|34259329||10.1093/bib/bbab242|2021|original_model_paper||
|Antiprotozoal peptide prediction using machine learning with effective feature selection techniques|39247292|11380031|10.1016/j.heliyon.2024.e36163|2024|original_model_paper||
|Geometric deep learning as a potential tool for antimicrobial peptide prediction|37521317|10374423|10.3389/fbinf.2023.1216362|2023|review_or_secondary||
|Antimicrobial peptide activity prediction using machine learning methods (Makine öğrenmesi yöntemleriyle antimikrobiyal peptit aktivite tahmini)||||2023|uncertain||
|Nanoengineered phosphorus doped graphitic carbon nitride based ultrasensitive biosensing platform for Swine flu detection.|37597493||10.1016/j.colsurfb.2023.113504|2023|uncertain||
|A repurposed AMP binding domain reveals mitochondrial protein AMPylation as a regulator of cellular metabolism.|40849408|PMC6930341|10.1038/s41467-025-63014-z|2025|uncertain||
|Machine learning-driven multifunctional peptide engineering for sustained ocular drug delivery.|37130851|PMC10154330|10.1016/S0039-6257(01)00211-9|2023|uncertain||
|Rational Discovery of Antimicrobial Peptides by Means of Artificial Intelligence.|35877911|PMC8406695|10.3389/fmicb.2021.710199|2022|original_model_paper||
|Deep learning improves antimicrobial peptide recognition.|29590297|PMC3166253|10.1093/bioinformatics/bty179|2018|original_model_paper||
|Deep Learning for Novel Antimicrobial Peptide Design.|33810011|PMC6628222|10.1016/j.diagmicrobio.2004.02.008|2021|original_model_paper||
|Structure-aware deep learning model for peptide toxicity prediction.|39196703|PMC4702905|10.1016/j.cub.2015.11.017|2024|uncertain||
|Research Advance in the Development of Antimicrobial Peptides Using Deep Learning.|40801287||10.1002/jcc.70203|2025|review_or_secondary||
|Computational Methods and Tools in Antimicrobial Peptide Research.|34165973|PMC13029496|10.1021/acs.jcim.1c00175|2021|review_or_secondary||
|Antimicrobial peptide defense in Drosophila.|9394624|PMC11731912|10.1002/bies.950191112|1997|uncertain||
|Antimicrobial peptides in the airway.|16909921|PMC12826059|10.1007/3-540-29916-5_6|2006|uncertain||
|Mechanisms of antimicrobial peptide action and resistance.|12615953|PMC13310060|10.1124/pr.55.1.2|2003|review_or_secondary||
|PA-Win2: In Silico-Based Discovery of a Novel Peptide with Dual Antibacterial and Anti-Biofilm Activity|39766503|PMC11672609|10.3390/antibiotics13121113|2024|uncertain||
|PepNet: an interpretable neural network for anti-inflammatory and antimicrobial peptides prediction using a pre-trained protein language model|39341947|PMC11438969|10.1038/s42003-024-06911-1|2024|original_model_paper||
|BPFun: a deep learning framework for bioactive peptide function prediction using multi-label strategy by transformer-driven and sequence rich intrinsic information|40691539|PMC12278619|10.1186/s12859-025-06190-5|2025|original_model_paper||
|AI-guided discovery and optimization of antimicrobial peptides through species-aware language model|40676915|PMC12271573|10.1093/bib/bbaf343|2025|original_model_paper||
|CL-ACP: a parallel combination of CNN and LSTM anticancer peptide recognition model|34670488|PMC8527680|10.1186/s12859-021-04433-9|2021|original_model_paper||
|Application of a deep generative model produces novel and diverse functional peptides against microbial resistance|36618982|PMC9804011|10.1016/j.csbj.2022.12.029|2022|original_model_paper||
|A Survey of Forex and Stock Price Prediction Using Deep Learning|||10.3390/asi4010009|2021|review_or_secondary||
|Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models|||10.1109/tpami.2021.3116668|2021|review_or_secondary||
|Recent trends in antimicrobial peptide prediction using machine learning techniques|29379261|5767919|10.6026/97320630013415|2017|review_or_secondary||
|AmpClass: an Antimicrobial Peptide Predictor Based on Supervised Machine Learning.|39383429|PMC12730010|10.1590/0001-3765202420230756|2024|original_model_paper||
|Benchmarks in antimicrobial peptide prediction are biased due to the selection of negative data|35988923|9487607|10.1093/bib/bbac343|2022|benchmark_paper||
|Advances in Antimicrobial Peptide Prediction Based on Machine Learning and Deep Learning|||||uncertain||
|Diversity and Molecular Evolution of Antimicrobial Peptides in Caecilian Amphibians|38535816|PMC10975883|10.3390/toxins16030150|2024|benchmark_paper||
|Occurrence and evolutionary conservation analysis of α‐helical cationic amphiphilic segments in the human proteome|37945538||10.1111/febs.16997|2023|uncertain||
|A novel generative framework for designing pathogen-targeted antimicrobial peptides with programmable physicochemical properties|41460918|PMC12747415|10.1371/journal.pcbi.1013833|2025|original_model_paper||
|Dual Activity Microbial Peptides Catalog|42260273||10.1038/s41597-026-07521-8|2026|dataset_paper||
|Identification of Antimicrobial Peptides Using Chou’s 5 Step Rule|||10.32604/cmc.2021.015041|2021|original_model_paper||
|AntiBP3: A hybrid method for predicting antibacterial peptides against gram-positive/negative/variable bacteria|||10.1101/2023.07.25.550443|2023|original_model_paper||
|AntiBP3: An improved method for predicting of antibacterial peptides using machine learning yechniques|||10.5281/zenodo.19911030|2026|dataset_paper||
|AntiBP3: An improved method for predicting of antibacterial peptides using machine learning yechniques|||10.5281/zenodo.19911031|2026|dataset_paper||
|Deep-learning-enabled antibiotic discovery through molecular de-extinction|38862735|PMC11310081|10.1038/s41551-024-01201-x|2024|original_model_paper||
|Discovering highly potent antimicrobial peptides with deep generative model HydrAMP|36922490|PMC10017685|10.1038/s41467-023-36994-z|2023|original_model_paper||
|Antimicrobial peptide identification using multi-scale convolutional network.|31870282|PMC6192215|10.1093/bioinformatics/btx679|2019|original_model_paper||
|Designing antimicrobial peptides using deep learning and molecular dynamic simulations.|36857616|PMC13289945|10.1093/bib/bbad058|2023|original_model_paper||
|Antimicrobial peptide prediction based on contrastive learning and gated convolutional neural network.|41286313|PMC7574553|10.1098/rsob.200004|2025|original_model_paper||
|Multi-CGAN: Deep Generative Model-Based Multiproperty Antimicrobial Peptide Design.|38135439|PMC13289945|10.1021/acs.jcim.3c01881|2024|uncertain||
|Virtual Screening of Cathelicidin-Derived Anticancer Peptides and Validation of Their Production in the Probiotic Limosilactobacillus fermentum KUB-D18 Using Genome-Scale Metabolic Modeling and Experimental Approaches.|41155367|PMC12563682|10.3390/ijms262010077|2025|benchmark_paper||
|Unlocking the unexplored AMPSphere in marine rare species.|41731616|PMC13036921|10.1186/s40168-025-02326-0|2026|benchmark_paper||
|Ab initio Designed Antimicrobial Peptides Against Gram-Negative Bacteria.|34867843|PMC8636942|10.3389/fmicb.2021.715246|2021|unclear (uses existing AMP prediction tools for validation)||
|Novel integrated computational AMP discovery approaches highlight diversity in the helminth AMP repertoire.|37523405|PMC10414684|10.1371/journal.ppat.1011508|2023|benchmark||
|Empirical comparison of web-based antimicrobial peptide prediction tools|28203715|PMC5860510|10.1093/bioinformatics/btx081|2017|benchmark||
|Prediction of Antimicrobial Peptides Based on Sequence Alignment and Feature Selection Methods|21533231|PMC3076375|10.1371/journal.pone.0018476|2011|model_original||
|Artificial Neural Network-Guided Discovery of Antioxidant Peptides from Peony (Paeonia ostii) Seed Meal: Peptidomics, Molecular Mechanism, and Cellular Validation||PMC12986296||2026|uncertain||
|Acoustic Cavitation-Induced Unfolding and Solubilization of Velvet Antler Protein for Antioxidant Peptide Release: Substrate Modification Kinetics, Quantum Chemistry, and Keap1/Nrf2-Associated Cellular Responses.|42276016|PMC13276608|10.1016/j.ultsonch.2026.107920|2026|uncertain||
|A time window for memory consolidation during NREM sleep revealed by cAMP oscillation.|40233747|PMC13270127|10.1016/j.neuron.2025.03.020|2025|uncertain||
|Bioactive Peptides from Yellowfin Tuna By-Products: Structural Characterization and Neuro-Related Activities in PC12 Cells.|42042034|PMC3403559|10.1021/acs.jafc.3c05718|2026|uncertain||
|Multi-label Learning for Predicting the Activities of Antimicrobial Peptides|28526820|5438384|10.1038/s41598-017-01986-9|2017|original_model_paper||
|Classifier-driven generative adversarial networks for enhanced antimicrobial peptide design|41137855|12553139|10.1093/bib/bbaf500|2025|original_model_paper||
|AMP-GSM: Prediction of Antimicrobial Peptides via a Grouping–Scoring–Modeling Approach|||10.3390/app13085106|2023|original_model_paper||
|Interpretable support vector classifier for reliable prediction of antibacterial activity of modified peptides against Escherichia coli.|41072192||10.1016/j.jmgm.2025.109188|2025|original_model_paper||
|Cell-free biosynthesis combined with deep learning accelerates de novo-development of antimicrobial peptides|37938588|PMC10632401|10.1038/s41467-023-42434-9|2023|original_model_paper||
|Bioactive Plasmid- and Phage-Encoded Antimicrobial Peptides (AMPs) in the Human Gut: A Metatranscriptome–Virome Profiling Reveals Exploratory Links to Metabolic Human Diseases|41315055|12775044|10.1007/s00248-025-02620-2|2025|benchmark_paper||
|Exploring phenotypic and genotypic diversity among methicillin-resistant, vancomycin-resistant, and sensitive Staphylococcus aureus.|39969287|PMC9598308|10.5772/intechopen.84411|2024|uncertain||
|Evaluation of LLM-generated peptide as foundation template for discovery of effective encrypted AMPs against clinical superbugs.|40891852|PMC12502564|10.1128/spectrum.01504-25|2025|benchmark_paper||
|Antimicrobial Peptide Arsenal Predicted from the Venom Gland Transcriptome of the Tropical Trap-Jaw Ant Odontomachus chelifer.|37235379|PMC10221683|10.3390/toxins15050345|2023|uncertain||
|AmPEP: Sequence-based prediction of antimicrobial peptides using distribution patterns of amino acid properties and random forest|29374199|PMC5785966|10.1038/s41598-018-19752-w|2018|original_model_paper||
|StackAMP: Stacking-Based Ensemble Classifier for Antimicrobial Peptide Identification|||10.1109/tai.2024.3421176|2024|original_model_paper||
|Towards the Improved Discovery and Design of Functional Peptides: Common Features of Diverse Classes Permit Generalized Prediction of Bioactivity|23056189|PMC3466233|10.1371/journal.pone.0045012|2012|model_original||
|PEP-FOLD: an online resource for de novo peptide structure prediction|19433514||10.1093/nar/gkp323|2009|uncertain||
|Deep-Learning Driven Identification of Novel Antimicrobial Peptides.|40801152|PMC8001998|10.3390/ijms22062857|2025|original_model_paper||
|Synthetic antimicrobial peptides: Combatting antimicrobial resistance for sustainable aquaculture.|40967516|PMC13203813|10.1016/j.micpath.2025.108029|2025|review_or_secondary||
|Mining the UniProtKB/Swiss-Prot database for antimicrobial peptides.|40100125|PMC1084323|10.1093/nar/gki524|2025|original_model_paper||
|Specifically targeted antimicrobial peptides synergize with bacterial-entrapping peptide against systemic MRSA infections.|38266820|PMC10541502|10.1016/j.jare.2024.01.023|2025|original_model_paper||
|Gut-targeted nanoparticles deliver specifically targeted antimicrobial peptides against Clostridium perfringens infections.|37774026|PMC10541502|10.1126/sciadv.adf8782|2023|original_model_paper||
|Antimicrobial activity of novel symmetrical antimicrobial peptides centered on a hydrophilic motif against resistant clinical isolates:|39382284|PMC163996|10.1128/AAC.41.8.1738|2024|uncertain||
|Table 1: The comparison of Macrel AMP classifier performance and state-of-art methods shows that Macrel is among the best methods across a range of metrics.|||10.7717/peerj-10555/table-1|2020|benchmark_paper||
|AntiBP2: improved version of antibacterial peptide prediction|20122190|PMC3009489|10.1186/1471-2105-11-s1-s19|2010|original_model_paper||
|Cat-PIPpred: Pro-Inflammatory Peptide Predictor Integrating CatBoost and Cross-Modal Feature Fusion|||10.3390/ijms262110484|2025|uncertain||
|A PLUM Job: Peptide modeLs for Understanding and engineering antiMicrobial therapeutics.|42124643|PMC11578372|10.64898/2026.02.21.707214|2026|original_model_paper||
|ProToxin, a Predictor of Protein Toxicity.|41150190|PMC6157185|10.1023/A:1022627411411|2025|uncertain||
|Identification of Antimicrobial Peptides Isolated From the Skin Mucus of African Catfish, Clarias gariepinus (Burchell, 1822)|34987491|PMC8721588|10.3389/fmicb.2021.794631|2021|original_model_paper||
|In silico Approaches for the Design and Optimization of Interfering Peptides Against Protein–Protein Interactions|33996914|PMC8113820|10.3389/fmolb.2021.669431|2021|review_or_secondary||
|Venomics AI: a computational exploration of global venoms for antibiotic discovery|39764027|PMC11702808|10.1101/2024.12.17.628923|2024|benchmark_paper|open_fulltext_found|
|A generative artificial intelligence approach for peptide antibiotic optimization.|42206144|PMC13201158|10.1038/s42256-026-01237-5|2026|model_original|open_fulltext_found|
|Peer Review #2 of "Macrel: antimicrobial peptide screening in genomes and metagenomes (v0.2)"|||10.7287/peerj.10555v0.2/reviews/2|2020|review_or_secondary||
|Functional and evolutionary significance of unknown genes from uncultivated taxa|38109938|PMC10849945|10.1038/s41586-023-06955-z|2023|dataset_paper||
|Detection of antimicrobial peptides from fecal samples of FMT donors using deep learning.|41164228|PMC12560166|10.3389/fvets.2025.1689589|2025|review_or_secondary||
|Arctic deep-sea hydrothermal microbiomes as a natural niche for novel antimicrobial peptides.|42104260|PMC13321764|10.1186/s12866-026-05098-1|2026|dataset_paper||

## Benchmark Implications

|topic|decision|reason|evidence|
|---|---|---|---|
|Model exclusion from main benchmark|Remove antifungal, anticancer, antimalarial, and regression models from the main AMP binary classification leaderboard|These models target different tasks and would compromise the specificity of the benchmark; they may be included in extended tracks|ESM2-AFPpred (antifungal), AI4AFP (antifungal), ACP-DL (anticancer), CTCM-Neo (antimalarial), ANIA (MIC regression)|
|Weight availability deadline|Models without pre-trained weights or verified training scripts are marked as 'weight dead' and cannot be benchmarked until resolved|Batch inference requires reproducible model weights; missing weights block standardization|iAMPCN, E-CLEAP, UniproLcad, TriStack, iAMP-DL, etc. lack weights|
|Webserver-only models|Downgrade webserver-only models (AMPDiscover, ADAM, CAMPR3, etc.) to non-reproducible; they are excluded from the main benchmark|Cannot be locally executed for large-scale standardized testing|No code or local binaries available|
|Dataset quality and contamination|Mandatory verification of negative sample composition for all datasets; datasets with potential cross-contamination (e.g., Co-AMPpred) require cleaning before use|Contaminated negatives can inflate performance and invalidate comparisons|Co-AMPpred dataset negative source unclear; AI4AFP/ESM2-AFPpred datasets are antifungal-specific|
|Representative model uniqueness|Each model can only represent one representation and one architecture category; SSFGM-Model assigned to structure/graph representation, not multimodal|Avoids duplication and ensures clear categorization|Critic review pointed out SSFGM-Model double representation|
|Metric weighting and evaluation protocol|Adopt AUPRC (0.35), MCC (0.35), Recall (0.15), Precision (0.15) as core weighted metrics; mandatory report Accuracy, Specificity, AUROC, F1; use fixed threshold from Max MCC on validation set; apply CD-HIT 0.6 for homology control|Balances sensitivity to imbalance with overall robustness, ensures comparability|Metrics expert proposal and Critic endorsement with adjustments|
|Main benchmark model selection|Only five models qualified for the main AMP binary classification benchmark: AMPlify, ACEP, iAMPCN, Co-AMPpred, AMPDiscover.|All other models were either subclass-specific, regression, generative, databases, or had insufficient evidence. This ensures benchmark purity.|Critic final review; all subclass models (AI4AFP, ESM2-AFPpred, AVPpred, etc.) moved to a separate subclass appendix.|
|Metric weighting|Primary ranking uses weighted AUPRC (0.35), MCC (0.30), Recall (0.20), Precision (0.15). AUROC and Accuracy are mandatory reported but not weighted.|Severe class imbalance demands focus on positive class performance; AUPRC and MCC are robust.|Metrics proposal approved by Critic with minor modifications (CI requirement, hard-label exclusion).|
|Dataset reconstruction|Current model-specific datasets rejected; a unified, non-redundant test set from DRAMP/APD3 must be built.|Existing datasets suffer from subclass contamination, outdated negative sets, and lack of permanent DOIs.|Critic dataset quality review; follow-up tasks defined.|
|Representative model adjustments|Replaced ESM2-AFPpred with AMPlify as PLM/Transformer representative; removed SSFGM-Model and AI4AFP from representatives due to evidence issues and subclass specificity.|Representatives must be general AMP models with reliable evidence.|Critic review of representative models.|
|GitHub missing-link enrichment|Models without repository links were searched by exact model name on GitHub before the global meeting; candidate repositories were saved as evidence_level=github_search.|Some models lacked GitHub links in literature evidence; repository evidence should be added to the evidence pool before deployment decisions.|83 GitHub enrichment records saved to data\github_missing_model_enrichment.json|
|Qwen-Max web-search enrichment|Qwen-Max web search was used as a supplemental missing-evidence layer for repositories, datasets, weights, web servers, and paper pages.|Structured databases and GitHub API may miss aliases, author pages, supplementary links, and new web evidence.|20 Qwen web enrichment records saved to data\qwen_web_enrichment.json|
|由于AMP数据通常高度不平衡，优先使用AUPRC和MCC作为核心排名指标。||||
|强制报告ACC、Specificity、AUROC、F1以支持与现有文献（如Co-AMPpred、CTCM-Neo、AI4AFP）的全面对标。||||
|阈值必须基于验证集校准，避免0.5固定阈值导致的不公平比较。||||
|多分布测试矩阵可评估模型在不同不平衡程度下的泛化能力，暴露仅依赖平衡集的性能虚高。||||
|低同源独立测试集考验模型对未见序列同源家族的预测能力，反映真实应用场景。||||
|当前候选模型污染严重，若直接用于基准测试，将导致排名不可信，且可能将非 AMP 模型误认为 AMP 预测器。||||
|数据集无负样本和永久链接，基准不可复现，且无法保证公平性。||||
|权重缺失模型无法批量推理，若强行纳入，需整体重训，失去原模型意义。||||
|代表模型缺失使基准比较无焦点，必须修复。||||
|Model selection|Only models with verified code, weights, and pure AMP binary classification task are eligible for main benchmark. ESM2-AFPpred, AI4AFP, ANIA, AMPDiscover, amp-gan, MultiPep and review-only models are excluded from benchmark-ready list.|Task mismatch, lack of reproducibility, or insufficient evidence.|Critic review and Scout filter records.|
|Dataset construction|No existing dataset is acceptable; must build a new standard AMP binary classification dataset from APD3/DRAMP/DBAASP with clear negative definition and CD-HIT deduplication.|All candidate datasets lack proper negatives, are task-specific, or have no public access.|Critic dataset quality decisions.|
|Metric framework|Adopt weighted primary metrics (AUPRC 0.35, MCC 0.30, Recall 0.20, Precision 0.15) with mandatory reporting of ACC, Specificity, AUROC, F1, ECE, 95% CI. Threshold determined per model on validation set.|Balances imbalanced evaluation and literature alignment.|Metrics proposal and Critic approval.|
|Representative models|Reselect representatives only from verified models. Co-AMPpred for traditional features, ACEP for sequence encoding, protein LM category currently empty.|Previous representatives either unverified or task-mismatched.|Critic representative model review.|
|CTCM-Neo在外部测试中Acc达92.86%~93.33%，AUROC≈0.93，AUPRC≈0.80，说明AUPRC较AUROC有更明显区分度，应作为主指标||||
|Co-AMPpred仅报告AUROC(0.873)和MCC(0.606)，缺乏AUPRC，反映过去指标不够全面，新benchmark需强制报告PR曲线||||
|AI4AFP使用MCC(0.89)和CD-HIT去重，证实MCC和去重策略的有效性||||
|多篇模型仅使用Accuracy，可能导致在不平衡数据上高估性能，因此需强制报告AUPRC/MCC||||
|当前候选模型池严重注水，实际可复现的纯 AMP 二分类模型可能不足 5 个||||
|数据集缺失将导致无法统一评测，必须先构建 curated benchmark dataset||||
|指标强制实施可能迫使部分模型重新训练，增加工作量||||
|需要开发自动化验证脚本，对每个仓库执行下载、权重检查、测试推理，否则无法保证质量||||
|Core metric selection|Weighted AUPRC (0.35), MCC (0.30), Recall (0.20), Precision (0.15) for imbalanced AMP data; ACC, Specificity, AUROC, F1 mandatory but excluded from ranking.|AUPRC and MCC are more robust for class imbalance, while reported metrics align with literature.|Metrics agent proposal and Critic approval.|
|Threshold optimization|Per-model threshold fixed on validation set via Max MCC or Youden Index, applied to test set; no default 0.5 threshold.|Prevents test set overfitting and ensures fair comparison.|Metrics agent proposal, Critic approved.|
|Dataset imbalance matrix|Test on 1:1, 1:10, 1:100 balanced sets and a low-homology independent set.|Evaluates model robustness under varying class imbalance and domain shift.|Metrics agent proposal, Critic approved.|
|Homology leakage control|Apply CD-HIT 40% identity cutoff and StratifiedGroupKFold by sequence family/species.|Prevents inflated performance due to sequence redundancy.|Metrics agent proposal, Critic approved.|
|Model evidence verification|All candidate models must have correct paper DOI/PMID, accessible code, and clear task definition. SSFGM-Model and ACEP paper info need immediate correction; ESM2-AFPpred excluded from main benchmark due to task mismatch.|Ensures benchmark integrity and reproducibility.|Critic review of paper info and task alignment.|
|Representative model selection|Co-AMPpred (traditional ML), iAMPCN (CNN), ACEP (hybrid) are conditionally accepted; AMPlify pending original paper. Multi-modal and PLM categories lack reliable representatives.|Need to cover architecture and representation diversity; SSFGM-Model evidence unsound, ESM2-AFPpred task-specific.|Critic review of representative models.|
|Model inclusion criteria|Models must provide pre-trained weights (or reproducible training script) and a batch inference script to be considered for main benchmark. All current candidates downgraded to needs_weight_check or needs_verification.|Without verified weights and batch inference, fair automated evaluation is impossible.|Critic review of provided repositories; none meet the full criteria.|
|Confidence intervals|95% confidence intervals via bootstrap must be reported for all core metrics.|To assess statistical significance of differences between models.|Critic suggestion.|
|Multi-distribution testing|Evaluate all models on 1:1, 1:10, 1:100 balanced/imbalanced test sets, plus a low-homology independent set (<40% identity to training).|Reveal model robustness under different class distributions.|Metrics proposal.|

## Open Questions

|question|reason|next_action|
|---|---|---|
|CTCM-Neo & ConformaX-PEP framework lacks code repository and full text; verification needed.|chunk_summary_uncertainty|manual_or_followup_search|
|Co-AMPpred web server not reported; may be available offline only.|chunk_summary_uncertainty|manual_or_followup_search|
|Applicability of antimalarial-specific model to general AMP benchmark needs evaluation.|chunk_summary_uncertainty|manual_or_followup_search|
|Code repository, web server, model weights, and dataset source not reported in available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|PCSPred full text unavailable; code and dataset details unknown.|chunk_summary_uncertainty|manual_or_followup_search|
|iAMPCN original model publication not identified; the article uses the model but does not provide training dataset details.|chunk_summary_uncertainty|manual_or_followup_search|
|AAGP and PepLab papers are not AMP prediction models, but they contain related datasets (anti-aging peptides, multi-class bioactive peptides).|chunk_summary_uncertainty|manual_or_followup_search|
|SSFGM-Model论文的全文缓存内容与ACEP模型描述不符，可能为系统错误，建议核实原始全文。|chunk_summary_uncertainty|manual_or_followup_search|
|SSFGM-Model使用的具体训练数据集和外部测试集未在摘要中明确，需查阅全文补充。|chunk_summary_uncertainty|manual_or_followup_search|
|Whether ACP-DL and ACP-MHCNN are exactly the same model or ACP-DL is an independent implementation.|chunk_summary_uncertainty|manual_or_followup_search|
|No explicit mention of training dataset or weights availability.|chunk_summary_uncertainty|manual_or_followup_search|
|The web server link may refer to the original ACP-MHCNN, not necessarily ACP-DL.|chunk_summary_uncertainty|manual_or_followup_search|
|The matched repositories include two unrelated bacterial metagenomic projects, likely due to imperfect search.|chunk_summary_uncertainty|manual_or_followup_search|
|Does MultiPep explicitly include antimicrobial peptide (AMP) classification among its 20 bioactivity classes?|chunk_summary_uncertainty|manual_or_followup_search|
|Are there any other AMP prediction models in this batch that were overlooked?|chunk_summary_uncertainty|manual_or_followup_search|
|Most models from the Ramazi et al. review (PMID 35305010) lack detailed architecture, training data, and code availability; they require full-text verification of original papers.|chunk_summary_uncertainty|manual_or_followup_search|
|The exact algorithm and input features for many models (e.g., iAMPred, AmPEP, AntiBP2, ClassAMP, AVPpred, AMPER, EFC-FCBF) are not specified in the available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|The ADAM web server URL was extracted from the 2024 review but not confirmed by direct testing; the 2022 review did not provide a URL.|chunk_summary_uncertainty|manual_or_followup_search|
|Several models from the 2024 review (E-CLEAP, UniproLcad, TriStack, iAMP-DL, amp-gan) have only repository links and no published papers with full details verified.|chunk_summary_uncertainty|manual_or_followup_search|
|Unclear whether the datasets from PMID 36835264 and 33113998 are independent of the training data used by the listed prediction tools.|chunk_summary_uncertainty|manual_or_followup_search|
|The role of the 2013 paper 'Adaptive peptide design' (PMID 24594327) in relation to the ADAM model is unclear.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext for PMID 34081438 was mismatched (lipid metabolism content); extraction relied on abstract and metadata. Verification of fulltext is needed.|chunk_summary_uncertainty|manual_or_followup_search|
|No dataset link or name provided for the training data; dataset description: 'largest experimentally validated nonredundant peptide data set' but no accession or download URL.|chunk_summary_uncertainty|manual_or_followup_search|
|Code repository not reported; only web server available.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext cache for PMID 34081438 contained content about lipid metabolism peptides and not the original article; only abstract and metadata were used for evidence extraction.|chunk_summary_uncertainty|manual_or_followup_search|
|No dataset identifier or download link is provided for the training data; described as 'largest experimentally validated nonredundant peptide data set' but not accessible.|chunk_summary_uncertainty|manual_or_followup_search|
|The other three papers in this chunk are not related to AMP prediction models; they were present in the search results but not used for model evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Direct download link for the training dataset (DRAMP/APD3 derived) is not provided in the evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Negative sample set composition and deduplication details are not fully specified.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights are not reported as separate downloadable resource.|chunk_summary_uncertainty|manual_or_followup_search|
|Web server is not available.|chunk_summary_uncertainty|manual_or_followup_search|
|Negative samples not specified; deduplication method not detailed.|chunk_summary_uncertainty|manual_or_followup_search|
|Independent test set details limited.|chunk_summary_uncertainty|manual_or_followup_search|
|No direct download link for the constructed dataset.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights not provided in the repository.|chunk_summary_uncertainty|manual_or_followup_search|
|Should the specific generative models (VAE, WAE, RNN, LM) compared in the benchmark paper be recorded as individual models?|chunk_summary_uncertainty|manual_or_followup_search|
|Are the collagen-derived AMP sequences and MICs directly usable as a benchmark dataset for existing AMP predictors?|chunk_summary_uncertainty|manual_or_followup_search|
|AI4AMP has two DOIs (10.1109/bibm.2016.7822515 and 10.1128/msystems.00299-21) and inconsistent metadata; likely a preprint/journal version mismatch.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext for PMID 34783578 (BIBM 2016) may be incorrect; the correct fulltext is PMC8594441 (mSystems 2021).|chunk_summary_uncertainty|manual_or_followup_search|
|Unifying antimicrobial peptide datasets (PMID 39281014) title/journal mismatch; may be a dataset description rather than a review.|chunk_summary_uncertainty|manual_or_followup_search|
|Sparse neural network model (PMID 27870247) fulltext was a review chapter, not original research article.|chunk_summary_uncertainty|manual_or_followup_search|
|E-CLEAP source paper and DOI remain unknown due to truncated chunk; record_keys include doi:10 1371 journal pone 0300125, doi:10 1002 mbo3 1393, pmcid:pmc13267055, doi:10 2147 aabc s484321 but no titles or models associated.|chunk_summary_uncertainty|manual_or_followup_search|
|No web server URL found for AI4AMP in fulltext; only pre-print URL (http://symbiosis.iis.sinica.edu.tw/PC_6/) is available.|chunk_summary_uncertainty|manual_or_followup_search|
|BAGEL4 is a tool, not an AMP prediction model; used in a dataset paper (PMID 41148698).|chunk_summary_uncertainty|manual_or_followup_search|
|AI4AMP web server URL not found in extracted links; needs verification from paper or external sources.|chunk_summary_uncertainty|manual_or_followup_search|
|Exact training dataset sources for AI4AMP and SAMP not detailed in evidence; may be in respective GitHub repositories or supplementary materials.|chunk_summary_uncertainty|manual_or_followup_search|
|SAMP's internal architecture (e.g., base classifiers) not specified in the provided excerpt; full text may clarify.|chunk_summary_uncertainty|manual_or_followup_search|
|Source paper is a review; the original model description may be in a different paper not identified here.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights not directly available.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights not reported in available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Al-Omari 2024 code not available; may limit benchmark inclusion.|chunk_summary_uncertainty|manual_or_followup_search|
|PepForge dataset source not explicitly reported.|chunk_summary_uncertainty|manual_or_followup_search|
|BBATProt dataset source not explicitly reported.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext verification needed for PMID 34081438; cached fulltext was mismatched.|chunk_summary_uncertainty|manual_or_followup_search|
|Training dataset not publicly accessible; no link or name provided.|chunk_summary_uncertainty|manual_or_followup_search|
|No code repository available; reproducibility limited.|chunk_summary_uncertainty|manual_or_followup_search|
|Full text for AMAP (PMID 30831306) was not available; evidence is based on abstract only.|chunk_summary_uncertainty|manual_or_followup_search|
|DOI 10.1128/cmr.00050-19 (PMID 39088151) shows a mismatch between the abstract (AI/ML in AMR) and the provided full-text excerpt.|chunk_summary_uncertainty|manual_or_followup_search|
|No dataset links or code repositories were identified for the AMAP model.|chunk_summary_uncertainty|manual_or_followup_search|
|Full text for AMAP (PMID 30831306) not available; evidence based on abstract only.|chunk_summary_uncertainty|manual_or_followup_search|
|DOI 10.1128/cmr.00050-19 (PMID 39088151) shows abstract on AI/ML in AMR but full-text excerpt describes Evolutionary Pathways in Antibiotic Resistance, mismatch noted.|chunk_summary_uncertainty|manual_or_followup_search|
|No dataset links or code repositories identified for AMAP model.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext content for all four papers did not match actual articles; real fulltexts needed to verify code availability, methods, and datasets.|chunk_summary_uncertainty|manual_or_followup_search|
|DOIs for several papers may be incorrect; correct DOIs should be verified.|chunk_summary_uncertainty|manual_or_followup_search|
|EBAMP and DLFea4AMPGen are de novo design methods, not pure prediction models; suitability for AMP prediction benchmarks needs clarification.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP model's dataset details (positive/negative samples, deduplication) require examination of the actual GitHub repository.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext not available for Zhang2021_BERT_AMP; details limited to abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|Dong2024_AMP_activity_classifier is part of a pipeline, not a standalone AMP predictor.|chunk_summary_uncertainty|manual_or_followup_search|
|Several repository and dataset entries from regex expansion may be irrelevant (e.g., treexplainer, LightGBM, shap) and are omitted from this compression.|chunk_summary_uncertainty|manual_or_followup_search|
|PMC IDs for some papers may be incorrect (e.g., PMC2935846 for COMDEL paper, PMC3765848 for AMP-BERT paper)|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights for AMP-BERT not available|chunk_summary_uncertainty|manual_or_followup_search|
|Repositories/datasets associated with the review paper (treexplainer, LightGBM, shap) are not directly linked to benchmark models|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-CapsNet code and dataset are not publicly available; the original reference [31] for the dataset is not identified in this batch.|chunk_summary_uncertainty|manual_or_followup_search|
|deepAMP original paper details (PMID, code) are missing; only mentioned in review, requiring full-text verification.|chunk_summary_uncertainty|manual_or_followup_search|
|AmpGPT2 model weights may be available from the code repository but not explicitly stated; further verification needed.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text verification needed for HydrAMP, PepCVAE, PrefixProt, MoFormer, HMAMP, and AMP-Designer to confirm code availability and exact model details.|chunk_summary_uncertainty|manual_or_followup_search|
|No direct links to PeptideAtlas or DBAASP v3 provided; these need to be sourced from the original publications.|chunk_summary_uncertainty|manual_or_followup_search|
|The review mentions AMP-RL as a new model; its benchmark suitability depends on independent evaluation and availability of trained weights.|chunk_summary_uncertainty|manual_or_followup_search|
|The fulltext for PMID 29679519 appears to be from a different paper (likely about AMP-Designer). The actual paper 'Designing Anticancer Peptides by Constructive Machine Learning' may not contain AMP-Designer. Need to verify correct source of AMP-MIC and AMP-Designer.|chunk_summary_uncertainty|manual_or_followup_search|
|Review paper (PMID:35128926) might mention AMP prediction models, but fulltext not available to extract details.|chunk_summary_uncertainty|manual_or_followup_search|
|iAMPCN 的原始文献和训练数据是什么？|chunk_summary_uncertainty|manual_or_followup_search|
|AP_Sin 和 AMP-Detector 是否公开了代码或数据集？|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-RNNpro 的 web 服务是否仍可访问，其训练数据来源？|chunk_summary_uncertainty|manual_or_followup_search|
|Full text needed for AMP-Distillation and STAMP to confirm datasets, code availability, and detailed metrics.|chunk_summary_uncertainty|manual_or_followup_search|
|Code and dataset links for STAMP are missing from the abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|The second paper (antimicrobial combination effects) is not an AMP prediction model; it may be excluded from benchmark.|chunk_summary_uncertainty|manual_or_followup_search|
|CF-AMP prediction: no code or data available, preprint, only abstract evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-DualTransnet: no abstract or full text, no code/data, limited information from journal article.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext content for all four papers did not match the actual articles; verification of real fulltexts needed to confirm code availability, detailed methods, and datasets.|chunk_summary_uncertainty|manual_or_followup_search|
|DOIs for several papers may be incorrect; correct DOIs should be verified from reliable sources.|chunk_summary_uncertainty|manual_or_followup_search|
|EBAMP and DLFea4AMPGen are de novo design methods rather than pure prediction models; their suitability for AMP prediction benchmarks should be clarified.|chunk_summary_uncertainty|manual_or_followup_search|
|For all four papers: are there any code repositories, web servers, or datasets released?|chunk_summary_uncertainty|manual_or_followup_search|
|What specific architectures, features, and performance metrics were used?|chunk_summary_uncertainty|manual_or_followup_search|
|Are the models actually novel AMP predictors or reviews/datasets?|chunk_summary_uncertainty|manual_or_followup_search|
|Can full texts be obtained to verify details?|chunk_summary_uncertainty|manual_or_followup_search|
|GAC-BiTCNN-AMP: code repository and dataset source not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|CVAE-BIO: GitHub link (scan2030) may not contain the full pipeline; model weights not available.|chunk_summary_uncertainty|manual_or_followup_search|
|HydrAMP, AMPGAN, Macrel, iAMPpred, AntiBP: only mentioned in review, original papers not verified.|chunk_summary_uncertainty|manual_or_followup_search|
|E-CLEAP, UniproLcad, TriStack, iAMP-DL, amp-gan, AVPIden, ADAM, ampsphere: only mentioned in review PMID 39557756, original papers not verified.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPlify: preprint status, need peer-reviewed publication verification.|chunk_summary_uncertainty|manual_or_followup_search|
|Two papers with DOIs 10.3390/ijms24043852 and 10.3390/biom10111473 have missing metadata.|chunk_summary_uncertainty|manual_or_followup_search|
|PMCID 13267055 is shared by two papers with different DOIs (10.1021/acs.jmedchem.5c00188 and 10.1038/s44386-026-00045-6). The evidence for AMP-GPT and MCL-AMP comes from the latter, but metadata attribution may be inconsistent.|chunk_summary_uncertainty|manual_or_followup_search|
|The paper titled 'deep-AMPpred' (PMID:39792442) describes MAPLE; 'deep-AMPpred' may be an alias for MAPLE or a separate model name used in the publication.|chunk_summary_uncertainty|manual_or_followup_search|
|No code or web server was found for AMP-GPT or MCL-AMP.|chunk_summary_uncertainty|manual_or_followup_search|
|Training data for AMP-GPT and MCL-AMP are not fully specified; only 'validated antimicrobial sequence datasets' mentioned.|chunk_summary_uncertainty|manual_or_followup_search|
|The full-text cache for PMID 25802839 (doi:10.1093/nar/gkn823) appears to correspond to a different article about APD2 database, not the SVM-LZ prediction method. Need to verify the actual paper content.|chunk_summary_uncertainty|manual_or_followup_search|
|PepVAE model weights are not publicly available; only 'available upon request'. Is it still a candidate for benchmark if code cannot be obtained?|chunk_summary_uncertainty|manual_or_followup_search|
|The LMPred paper's full-text excerpt contained unrelated RNA secondary structure content; verify that the actual paper matches the abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|Actual model architectures, features, training data, and performance unknown.|chunk_summary_uncertainty|manual_or_followup_search|
|Whether these papers are truly novel AMP predictors or reviews/datasets is uncertain.|chunk_summary_uncertainty|manual_or_followup_search|
|Availability of code repositories, web servers, or model weights not confirmed.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext content for all four papers in this batch did not match actual articles; verification of real fulltexts needed.|chunk_summary_uncertainty|manual_or_followup_search|
|EBAMP and DLFea4AMPGen are de novo design methods; suitability for AMP prediction benchmarks should be clarified.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP model's dataset details (positive/negative samples, deduplication) require examination of GitHub repository.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-RNNpro 存在两个可能来源/DOI 冲突 (10.1016/j.csbj.2022.07.043 和 10.1038/s41598-024-63461-6)，两者均指向同一 PMID 38839785，需核实正确版本|chunk_summary_uncertainty|manual_or_followup_search|
|DDM 论文发表于 2026 年，是否已正式发表？|chunk_summary_uncertainty|manual_or_followup_search|
|Antiviral peptide 论文 (PMID 41594075) 与 AMP 分类不直接相关，其模型是否适用于 AMP 需进一步确认|chunk_summary_uncertainty|manual_or_followup_search|
|Batch 84 provided AMP-RNNpro source DOI 10.1016/j.csbj.2022.07.043 and PMCID PMC9421197, which do not match the correct publication (DOI 10.1038/s41598-024-63461-6, PMCID PMC11153637) associated with PMID 38839785. This discrepancy may indicate a data error in the original chunk.|chunk_summary_uncertainty|manual_or_followup_search|
|The DDM paper is from 2026, which may be an error (future year) or a preprint; further verification needed.|chunk_summary_uncertainty|manual_or_followup_search|
|iAMPCN lacks any original paper, training data, or code; its validity as a benchmark candidate is uncertain.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-Detector's training data details are incomplete; only Peptide Atlas was mentioned for discovery, not for training.|chunk_summary_uncertainty|manual_or_followup_search|
|The antiviral peptide paper (PMID 41594075) is of unclear relevance to general AMP classification and was included only because of repository links.|chunk_summary_uncertainty|manual_or_followup_search|
|MCL-AMP: code and training data not reported in available evidence; needs full-text verification for repository and dataset links.|chunk_summary_uncertainty|manual_or_followup_search|
|AI4AMP, iAMPpred, AMP Scanner: only mentioned in review; need to locate original papers for code, data, and detailed method descriptions.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-GPT (generator) not included as a prediction model; its potential role in benchmark if considered as a generative AMP tool could be assessed later.|chunk_summary_uncertainty|manual_or_followup_search|
|Source code for AMPScanner vr.2 not found in available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|PepGen 1.0 web server link is a shortened URL, long-term accessibility unclear.|chunk_summary_uncertainty|manual_or_followup_search|
|The original publication of AMPScanner vr.2 not identified; only usage papers found.|chunk_summary_uncertainty|manual_or_followup_search|
|PMID 38590669 is about antiviral peptides, not directly related to AMP prediction models.|chunk_summary_uncertainty|manual_or_followup_search|
|Training data not explicitly described; likely trained on known AMP data and metagenomic sequences.|chunk_summary_uncertainty|manual_or_followup_search|
|The unnamed AMP predictor from DRAMP 2.0 may have been integrated later; check DRAMP website for any prediction tool.|chunk_summary_uncertainty|manual_or_followup_search|
|The AMP toxicity prediction model is not an AMP classifier but could be relevant for benchmark filtering; decide on inclusion policy.|chunk_summary_uncertainty|manual_or_followup_search|
|Deep-AmPEP30 web server availability should be confirmed; the paper states it supports genome screening, but no code repository found.|chunk_summary_uncertainty|manual_or_followup_search|
|CalcAMP provides separate models for Gram+ and Gram-; we need to decide how to handle them in a unified benchmark.|chunk_summary_uncertainty|manual_or_followup_search|
|PeptideBERT (doi:10.1021/acs.jpclett.3c02398.s001) is a supplementary material; unclear if it covers AMP prediction. Main article needed.|chunk_summary_uncertainty|manual_or_followup_search|
|For Witten & Witten 2019, GRAMPA dataset is not directly linked; GitHub repository should be inspected.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-zGSM paper year is 2026, possibly a metadata error or future publication; requires verification.|chunk_summary_uncertainty|manual_or_followup_search|
|Torrent et al. 2011 model uses 8 hand-crafted features; relevance as a modern benchmark may be limited.|chunk_summary_uncertainty|manual_or_followup_search|
|Witten & Witten 2019: GRAMPA dataset not directly linked; GitHub repository inspection required.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-zGSM paper year 2026 may be a metadata error or future publication; verification needed.|chunk_summary_uncertainty|manual_or_followup_search|
|Torrent et al. 2011 uses 8 hand-crafted features; limited modern relevance but could serve as historical baseline.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP0: full-text details on training data, evaluation, and code availability need verification.|chunk_summary_uncertainty|manual_or_followup_search|
|sAMPpred-GAT: exact dataset composition and deduplication methods not specified in abstract; full-text needed.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text of sAMPpred-GAT misaligned with provided excerpt (seems to be LAMP database paper); correct full-text should be retrieved.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPA training data and algorithm details not reported; only web server link available.|chunk_summary_uncertainty|manual_or_followup_search|
|PyAMPA model weights are not provided; only source code and datasets are available.|chunk_summary_uncertainty|manual_or_followup_search|
|Several regex-extracted datasets (e.g., 'PyAMPA.', 'PyAMPA.<h4') are likely noise and excluded.|chunk_summary_uncertainty|manual_or_followup_search|
|Relationship between PyAMPA and AMPA is unclear; they share the name 'AMPA' but appear to be different tools.|chunk_summary_uncertainty|manual_or_followup_search|
|Some non-AMP related papers were included in the retrieval batch (ECG, gaming, transformers) and are marked as unrelated.|chunk_summary_uncertainty|manual_or_followup_search|
|The specific prediction algorithm is not detailed in the available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Training data and external test signals are not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|Code repository and model weights are not available.|chunk_summary_uncertainty|manual_or_followup_search|
|APEX model details (architecture, code, dataset) need full-text verification of the original paper (Wan, Torres, et al. 2024).|chunk_summary_uncertainty|manual_or_followup_search|
|Are the training datasets of AntiBP3 and AMPActiPred publicly available? Current evidence does not provide download links.|chunk_summary_uncertainty|manual_or_followup_search|
|Does AMPActiPred have a code repository beyond the web server?|chunk_summary_uncertainty|manual_or_followup_search|
|The review mentions many other models (e.g., Martínez-Mauricio et al. 2024, Li et al. 2022, Shao et al. 2024) without explicit model names; these may be relevant for future inclusion after full-text review.|chunk_summary_uncertainty|manual_or_followup_search|
|APEX model details (architecture, code, dataset) require full-text verification of the original paper (Wan, Torres, et al. 2024).|chunk_summary_uncertainty|manual_or_followup_search|
|VINCI pipeline code repository link missing from abstract; full text needed to confirm availability.|chunk_summary_uncertainty|manual_or_followup_search|
|BATTLE-AMP benchmarked 10 model families (21 variants) but specific model names not listed in abstract; full text needed to identify candidate models.|chunk_summary_uncertainty|manual_or_followup_search|
|综述（10.1007/s10989-025-10716-z）可能提及多个AMP预测模型，但无全文，需进一步验证具体模型名称和代码可用性。|chunk_summary_uncertainty|manual_or_followup_search|
|AMPBenchmark使用的具体正数据集来源和11种负采样方法细节未知。|chunk_summary_uncertainty|manual_or_followup_search|
|后两篇AMP设计研究未提出新的AMP预测模型，仅应用现有工具，其相关性需进一步确认。|chunk_summary_uncertainty|manual_or_followup_search|
|Need full text for CAmidPred, AMPCLGPT, and PepMCP to verify web server, data, and details.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPCLGPT code availability unknown.|chunk_summary_uncertainty|manual_or_followup_search|
|MemAMPdb dataset URL not provided in abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights for iMFP-LG are not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset URLs for MFBP and MFTP are not available.|chunk_summary_uncertainty|manual_or_followup_search|
|Some repositories (CAPTP, AMPd-Up) were identified via regex matching with lower confidence.|chunk_summary_uncertainty|manual_or_followup_search|
|The chunk is named AMPd-Up but the main model extracted is iMFP-LG; AMPd-Up is a de novo design tool, not a prediction model, and is not included as a benchmark candidate.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext cache for PMID 34081438 contained content about lipid metabolism peptides and not the original article; extraction relied on abstract and metadata. Verification of fulltext is needed.|chunk_summary_uncertainty|manual_or_followup_search|
|No dataset link or name provided for the training data; dataset was described as 'largest experimentally validated nonredundant peptide data set' but no accession or download URL was mentioned.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text content for PMCIDs 9545273 and 12730010 appears unrelated (Aspergillus secondary metabolism vs. AMP discovery), suggesting incorrect PMCIDs or caching error.|chunk_summary_uncertainty|manual_or_followup_search|
|No specific model names were given for the deep learning models; described only as 'deep learning-based model' or 'BERT and CNN models'.|chunk_summary_uncertainty|manual_or_followup_search|
|None of the papers provided code, web server, or dataset download links in the provided evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|All models from review (PMID:35305010) need full-text verification of original papers for accurate architecture and availability.|chunk_summary_uncertainty|manual_or_followup_search|
|PepGen 1.0 code repository not found; only a shortened URL provided.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPScanner vr.2 architecture details limited; not original publication.|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset used for training models not reported in available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Paper PMID 38590669 (antiviral peptides) not directly related to AMP prediction, extracted models not from it.|chunk_summary_uncertainty|manual_or_followup_search|
|Full text of amPEPpy paper (PMID 33135060) appears to be mismatched in cache; verification needed to confirm dataset, features, and performance details.|chunk_summary_uncertainty|manual_or_followup_search|
|Is the GitHub repository for the benchmark paper (PMID 33774670) actually associated with the review, or does it belong to a different fingerprint-based study?|chunk_summary_uncertainty|manual_or_followup_search|
|No public code or model weights available for the Bacteria-specific (PMID 36912047) and XGBoost (PMID 40529865) models.|chunk_summary_uncertainty|manual_or_followup_search|
|Can the E. coli-specific model be generalized to other bacteria, or is it too narrow for a general AMP benchmark?|chunk_summary_uncertainty|manual_or_followup_search|
|AMPlify model details were extracted from a review; original paper not yet verified.|chunk_summary_uncertainty|manual_or_followup_search|
|Confidence values for models extracted from review articles are lower due to lack of primary source verification.|chunk_summary_uncertainty|manual_or_followup_search|
|Code for AMPfinder, AMPpredictor, AMPActiPred is not independently available.|chunk_summary_uncertainty|manual_or_followup_search|
|Method details for AMPfinder are unknown (method_family: unknown).|chunk_summary_uncertainty|manual_or_followup_search|
|External test performance of these models is not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|Relevance of the Big Data & Deep Data paper (doi:10.3233/978-1-61499-822-8-177) to AMP models is unclear.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPfinder、AMPpredictor 和 AMPActiPred 的独立代码库未公开，仅作为 dbAMP 3.0 网站的一部分提供。|chunk_summary_uncertainty|manual_or_followup_search|
|GAC-BiTCNN-AMP: 代码仓库未提供，需要确认是否开源；数据集具体来源不明。|chunk_summary_uncertainty|manual_or_followup_search|
|CVAE-BIO: GitHub 链接是否包含完整代码和训练好的模型权重？需要进一步检查。|chunk_summary_uncertainty|manual_or_followup_search|
|综述中提取的模型（HydrAMP, AMPGAN, Macrel, iAMPpred, AntiBP）均需找到原始论文、验证代码可用性和数据集，以确认benchmark资格。|chunk_summary_uncertainty|manual_or_followup_search|
|GAC-BiTCNN-AMP: code repository not provided; explicit dataset source unknown.|chunk_summary_uncertainty|manual_or_followup_search|
|CVAE-BIO: GitHub link (https://github.com/scan2030) unclear whether it contains full code and trained weights.|chunk_summary_uncertainty|manual_or_followup_search|
|HydrAMP, AMPGAN, Macrel, iAMPpred, AntiBP: only mentioned in review; need original papers to verify code availability, datasets, and benchmark eligibility.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPGAN v3: no DOI or PMID available; dataset details not reported; model weights not available.|chunk_summary_uncertainty|manual_or_followup_search|
|PepAnno: code repository not reported; training data details not available.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPGAN v3 paper lacks PMID/DOI; only Semantic Scholar id provided.|chunk_summary_uncertainty|manual_or_followup_search|
|Training data details (positive/negative samples, deduplication) for both models not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|PepAnno code availability unclear; no model weights provided.|chunk_summary_uncertainty|manual_or_followup_search|
|Anti-Vibrio paper and review paper are not directly AMP prediction models; their relevance is uncertain.|chunk_summary_uncertainty|manual_or_followup_search|
|What is the training dataset for AMPGP?|chunk_summary_uncertainty|manual_or_followup_search|
|Is AMPGP code or web server available?|chunk_summary_uncertainty|manual_or_followup_search|
|What are the four feature channels used in AMPGP prediction model?|chunk_summary_uncertainty|manual_or_followup_search|
|No full-text available; all details from abstract only.|chunk_summary_uncertainty|manual_or_followup_search|
|AmpGram web server and R package URLs are not provided in available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|DOI mismatch for AmpGram paper in some sources: 10.1074/jbc.M111.303602 vs correct 10.3390/ijms21124310.|chunk_summary_uncertainty|manual_or_followup_search|
|Several repository and dataset entries from regex matching may be false positives (e.g., SHARP, SAMP. links).|chunk_summary_uncertainty|manual_or_followup_search|
|Review-derived models (AMPScanner V2, ampir, etc.) lack performance details and need full-text verification of original papers.|chunk_summary_uncertainty|manual_or_followup_search|
|Some dataset entries (e.g., '27733.', 'SAMP') are likely code repositories or unrelated, requiring further validation.|chunk_summary_uncertainty|manual_or_followup_search|
|Possible DOI/PMID mismatches: SAMP paper PMID 39573886 linked to doi 10.1101/gr.254557.119 (Genome Research?) but journal is Briefings in functional genomics; same for PMID 38712184 with doi 10.1128/aac.02340-16.|chunk_summary_uncertainty|manual_or_followup_search|
|AmpGram paper PMID 32560350 has doi 10.1074/jbc.M111.303602 which does not match the title (likely incorrect).|chunk_summary_uncertainty|manual_or_followup_search|
|Many repository and dataset entries extracted by regex (SHARP, SAMP., etc.) may be false positives; confidence 0.6-0.7.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text for AmpGram paper was unrelated to the article; confirmation needed.|chunk_summary_uncertainty|manual_or_followup_search|
|AmpGram CRAN URL and web-server URL not reported in available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Model details for AmpGram (architecture, input features) missing from abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|DOI/PMCID inconsistencies: e.g., PMID 38712184 has DOI 10.1128/aac.02340-16 but PMC7050522 may belong to a different article; fulltext content for PMID 32560350 and 38712184 was reported as unrelated to the expected paper.|chunk_summary_uncertainty|manual_or_followup_search|
|AmpGram web-server URL is not reported in the abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|Some dataset entries (e.g., '27733.') seem to be artifacts from regex extraction and may not be genuine AMP-related datasets.|chunk_summary_uncertainty|manual_or_followup_search|
|The year of the benchmark paper (PMID 42071989) is 2026, which may be a typo or future date.|chunk_summary_uncertainty|manual_or_followup_search|
|Code availability for CG-AMP and AmpHGT not reported; further verification recommended.|chunk_summary_uncertainty|manual_or_followup_search|
|GenPept-Curated-2025 dataset URL not provided in available evidence; may be available via the preprint DOI.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text cache for PMCIDs 9545273 and 12730010 appears unrelated to the article titles/abstracts; may indicate incorrect PMCIDs or a caching error.|chunk_summary_uncertainty|manual_or_followup_search|
|No specific model names given in abstracts; names may be in full text.|chunk_summary_uncertainty|manual_or_followup_search|
|No code repositories, web servers, or dataset download links found in the provided evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Exact URL for AmpGram web server and R package is not provided in the extracted evidence; full URL retrieval from the original paper or supplementary material is needed.|chunk_summary_uncertainty|manual_or_followup_search|
|The review mentions many tools without performance details; full-text verification of the original papers is required to confirm their benchmark suitability.|chunk_summary_uncertainty|manual_or_followup_search|
|The peptidomics and proteomics papers (doi:10.1021/acsomega.6c00679, doi:10.3390/biom15081121) do not introduce new AMP prediction models and were omitted from the model list, but they may contain useful datasets or benchmark results if examined more closely.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text cache for PMID 36732807 (AMPlify) is incorrect; actual article needed to verify links and data availability.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text for PMID 30867681 (review) mismatched with iFeature paper; actual review text needed.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text for PMID 39927895 is MAPLE model paper, but title/abstract indicate a review; metadata may be mismatched.|chunk_summary_uncertainty|manual_or_followup_search|
|Full-text for PMID 39705302 is an unrelated JBC article; actual paper describing DL method for E. coli AMP prediction not available.|chunk_summary_uncertainty|manual_or_followup_search|
|SGAC paper's training and test datasets described as 'publicly available' but specific sources and download links not provided in excerpt.|chunk_summary_uncertainty|manual_or_followup_search|
|All models mentioned in reviews (LMPred, AMPlify, TP-LMMSG, sAMPpred-GAT, PGAT-ABPp) require verification against original papers.|chunk_summary_uncertainty|manual_or_followup_search|
|Code for Wang2021 model is in Supplementary Materials, unclear if publicly accessible.|chunk_summary_uncertainty|manual_or_followup_search|
|No web server reported for AMPlify or iAMPCN.|chunk_summary_uncertainty|manual_or_followup_search|
|The downloadable tool for DeepAFP is mentioned but URL not found in the available evidence; needs follow-up.|chunk_summary_uncertainty|manual_or_followup_search|
|Review-listed tools (iAMP-2L, AMPpred, AntiBP2, ClassAMP) require original publications for full verification.|chunk_summary_uncertainty|manual_or_followup_search|
|PrMFTP dataset is not directly downloadable; may need to contact authors for benchmark use.|chunk_summary_uncertainty|manual_or_followup_search|
|DeepAFP code repository not found; check supplementary material or contact authors.|chunk_summary_uncertainty|manual_or_followup_search|
|The fulltext retrieved for PMID 37120707 is unrelated to the AMP prediction model; model information is based solely on the abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|The fulltext retrieved for PMID 37120707 is unrelated to the AMP prediction model; it appears to be a different article. The model information is based solely on the abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext excerpt for PMID 39262770 appears unrelated to MIC prediction model; actual fulltext may contain different details.|chunk_summary_uncertainty|manual_or_followup_search|
|No code or dataset links found for AMPpred-EL or MIC ensemble model.|chunk_summary_uncertainty|manual_or_followup_search|
|GitHub repository 'Antifreeze-Peptide-Discovery' linked to AMPpred-EL paper but name suggests unrelated content.|chunk_summary_uncertainty|manual_or_followup_search|
|主论文 DOI 可能是 10.1021/acs.jcim.3c01017，补充材料 DOI 10.1021/acs.jcim.3c01017.s001 需确认并获取全文以验证模型细节。|chunk_summary_uncertainty|manual_or_followup_search|
|AMPpred-MFA 的代码、数据集、Web 服务器可用性未知，需要全文核查。|chunk_summary_uncertainty|manual_or_followup_search|
|三个工具均依赖 dbAMP 平台，无独立代码仓库或模型权重，无法脱离平台评估。|chunk_summary_uncertainty|manual_or_followup_search|
|训练数据具体划分、负样本、去重方法未报告。|chunk_summary_uncertainty|manual_or_followup_search|
|外部独立测试未提及。|chunk_summary_uncertainty|manual_or_followup_search|
|Code availability for the FBGAN-enhanced AMP design framework not reported; full text needed to verify.|chunk_summary_uncertainty|manual_or_followup_search|
|Details of models mentioned in the review (e.g., AMPlify, AMPpredMFA) require full-text review of original papers for accurate benchmarking.|chunk_summary_uncertainty|manual_or_followup_search|
|Are the datasets used in the FBGAN-based framework publicly available in a standardized format?|chunk_summary_uncertainty|manual_or_followup_search|
|AniAMPpred: The fulltext provided (PMC12620532) is a different article on AI and photodynamic therapy; the correct fulltext is needed to verify details.|chunk_summary_uncertainty|manual_or_followup_search|
|sAMPpred-GAT, AMPs-Net, LABAMPs: Only review-level evidence; original papers must be located and assessed for code/data availability.|chunk_summary_uncertainty|manual_or_followup_search|
|Turkish paper: No abstract or fulltext available; unclear if it describes a named model.|chunk_summary_uncertainty|manual_or_followup_search|
|Batch 103 includes three unrelated papers (biosensor, AMPylation, peptide drug delivery) that were retrieved but not relevant to AMP prediction.|chunk_summary_uncertainty|manual_or_followup_search|
|Exact composition and size of the AMPScanner training dataset unknown from provided evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|No code repository or web server for the Wang et al. 2021 LSTM model.|chunk_summary_uncertainty|manual_or_followup_search|
|DOI mismatch for the Wang et al. paper (10.1016/j.diagmicrobio.2004.02.008) may indicate indexing error.|chunk_summary_uncertainty|manual_or_followup_search|
|Full texts of the original papers for iAMPpred, AntiBP, and AMPScanner should be reviewed to obtain code, dataset, and web server URLs.|chunk_summary_uncertainty|manual_or_followup_search|
|Verify whether the mentioned models are still accessible and have been updated.|chunk_summary_uncertainty|manual_or_followup_search|
|Exact URL for AmpGram web server and R package is not provided in the extracted evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Many tools mentioned in review lack performance details or original source verification.|chunk_summary_uncertainty|manual_or_followup_search|
|The peptidomics/proteomics papers (doi:10.1021/acsomega.6c00679, doi:10.3390/biom15081121) do not introduce new AMP prediction models but may contain useful datasets.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPScanner vr.2 was not described in its original publication; architecture details are limited.|chunk_summary_uncertainty|manual_or_followup_search|
|PepGen 1.0 only has a shortened URL for its web server, no source code repository.|chunk_summary_uncertainty|manual_or_followup_search|
|One paper (PMID 38590669) was about antiviral peptides, not directly AMP prediction, and did not yield any model extraction.|chunk_summary_uncertainty|manual_or_followup_search|
|Original publication for AMPScanner vr.2, PepGen 1.0, and AmPepGen not identified; evidence from tool usage papers.|chunk_summary_uncertainty|manual_or_followup_search|
|PepGen 1.0 repository URL is a shortened link, may not be permanent.|chunk_summary_uncertainty|manual_or_followup_search|
|No information on training datasets or model weights for any of the models.|chunk_summary_uncertainty|manual_or_followup_search|
|One paper (PMID 38590669) focuses on antiviral peptides, not directly AMP prediction, model extraction uncertain.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPSpeciesSpecific details are from the PA-Win2 application paper, not the original model development paper; model architecture specifics may be incomplete.|chunk_summary_uncertainty|manual_or_followup_search|
|PepNet’s training and test datasets are described but not directly linked; they are assumed to be included in the Zenodo repositories.|chunk_summary_uncertainty|manual_or_followup_search|
|Some repository/dataset links (e.g., Antimicrobial-Peptides, grampa.csv) were extracted via regex and may be tangentially related or not directly endorsed by the original papers.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights are not reported for any of the four models in the available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|PepNet training and test datasets are not directly linked; likely contained in the provided Zenodo records.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPSpeciesSpecific training data may be present in its GitHub repository but was not explicitly verified.|chunk_summary_uncertainty|manual_or_followup_search|
|BPFun negative samples not explicitly reported; deduplication by CD-HIT at 0.9 threshold.|chunk_summary_uncertainty|manual_or_followup_search|
|LLAMP deduplication method not reported; data originates from DBAASP v3.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPTrans-lstm is primarily a generative model, not a standard classification benchmark; its QSAR component may be used for prediction but lacks standard external testing.|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset URLs or direct download links are not provided in the available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|AniAMPpred: fulltext provided (PMC12620532) is a different article on AI and photodynamic therapy; the correct fulltext is needed to verify details.|chunk_summary_uncertainty|manual_or_followup_search|
|Original code for most models not available|chunk_summary_uncertainty|manual_or_followup_search|
|Web server for CSAMPPRED and CLASSAMP not found|chunk_summary_uncertainty|manual_or_followup_search|
|AmpClass code not provided|chunk_summary_uncertainty|manual_or_followup_search|
|Full text of original papers for models mentioned in review is needed to verify details|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset sources for AmpClass not disclosed|chunk_summary_uncertainty|manual_or_followup_search|
|Negative sampling methods details for some models incomplete|chunk_summary_uncertainty|manual_or_followup_search|
|PeptideBERT (doi:10.1021/acs.jpclett.3c02398.s001) appears to be a transformer-based model for peptide property prediction, but the supplementary material contains no abstract or full text. It is unclear if it covers AMP prediction. The main article (likely doi:10.1021/acs.jpclett.3c02398) should be retrieved to verify.|chunk_summary_uncertainty|manual_or_followup_search|
|For Witten & Witten 2019, the GRAMPA dataset is not directly linked; the GitHub repository should be inspected for data availability.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-zGSM paper's year is 2026, which may be a metadata error or a future publication; requires verification.|chunk_summary_uncertainty|manual_or_followup_search|
|Torrent et al. 2011 model uses 8 hand-crafted features; its relevance as a modern benchmark may be limited but could serve as a historical baseline.|chunk_summary_uncertainty|manual_or_followup_search|
|The fulltext for the BERT paper (doi:10.1093/bib/bbab200) is not available; the provided fulltext is from a different paper (fingerprint-based peptide classification). The BERT model's details are limited to the abstract.|chunk_summary_uncertainty|manual_or_followup_search|
|Code repositories not found for most tools (AntiBP2, iAMP-2L, ClassAMP, etc.)|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset download links missing for PrMFTP and DeepAFP|chunk_summary_uncertainty|manual_or_followup_search|
|Needs original publications for review-listed tools (iAMP-2L, AMPpred, AntiBP2, ClassAMP) to verify details|chunk_summary_uncertainty|manual_or_followup_search|
|DeepAFP downloadable tool URL not provided in available evidence|chunk_summary_uncertainty|manual_or_followup_search|
|Several tools (AVPpred, AMPER, EFC-FCBF) lack architecture or input feature details|chunk_summary_uncertainty|manual_or_followup_search|
|PrMFTP dataset not directly downloadable; may require contacting authors|chunk_summary_uncertainty|manual_or_followup_search|
|Are the training datasets of AntiBP3 and AMPActiPred publicly available? Current evidence does not provide direct download links.|chunk_summary_uncertainty|manual_or_followup_search|
|Is the generative CVAE-diffusion model available as a tool or code?|chunk_summary_uncertainty|manual_or_followup_search|
|What is the exact name of the ML classifier from the 2021 CMC paper?|chunk_summary_uncertainty|manual_or_followup_search|
|Does the AntiBP3 standalone package include source code?|chunk_summary_uncertainty|manual_or_followup_search|
|Are the datasets used in the 2021 and 2023 papers publicly available?|chunk_summary_uncertainty|manual_or_followup_search|
|The review mentions many other models (e.g., Martínez-Mauricio et al. 2024, Li et al. 2022, Shao et al. 2024) without explicit model names.|chunk_summary_uncertainty|manual_or_followup_search|
|APEX model details (architecture, code, dataset) need full-text verification of original paper (Wan, Torres, et al. 2024).|chunk_summary_uncertainty|manual_or_followup_search|
|Training datasets of AntiBP3 and AMPActiPred are not directly downloadable; only compiled from public databases.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPActiPred code repository not confirmed beyond web server.|chunk_summary_uncertainty|manual_or_followup_search|
|The review mentions several other models (e.g., Martínez-Mauricio et al. 2024, Li et al. 2022, Shao et al. 2024) without explicit model names.|chunk_summary_uncertainty|manual_or_followup_search|
|The review mentions many other models without explicit model names; these may be relevant for future inclusion.|chunk_summary_uncertainty|manual_or_followup_search|
|Whether the anticancer peptide model can be adapted or fine-tuned for AMP prediction is unknown.|chunk_summary_uncertainty|manual_or_followup_search|
|The listed datasets may actually be code repositories, not standalone datasets.|chunk_summary_uncertainty|manual_or_followup_search|
|The fulltext excerpt for PMID 39262770 appears to be from a different paper (RefSeq database description), not the MIC prediction model. True model details may require the actual fulltext.|chunk_summary_uncertainty|manual_or_followup_search|
|No code or dataset links were found for either AMP model. Links to GitHub repositories for AMPpred-EL may exist but were not in the extracted content.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPA primary publication(s) not directly identified in this chunk; only a secondary application paper is available.|chunk_summary_uncertainty|manual_or_followup_search|
|Three unrelated papers (DOI:10.1609/aaai.v35i14.17531, 10.5281/zenodo.5347031, 10.3917/g2000.296.0093) were matched erroneously; they contain no antimicrobial peptide tool information.|chunk_summary_uncertainty|manual_or_followup_search|
|Datasets extracted from the Zenodo record (Transformers) are irrelevant to AMPA and likely represent regex false positives.|chunk_summary_uncertainty|manual_or_followup_search|
|No code repository, model weights, or training dataset found for AMPA.|chunk_summary_uncertainty|manual_or_followup_search|
|Exact composition and size of the AMPScanner training dataset unknown from available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|No code repository or web server for the LSTM-based classifier (Wang et al. 2021).|chunk_summary_uncertainty|manual_or_followup_search|
|DOI mismatch for the Wang et al. paper: 10.1016/j.diagmicrobio.2004.02.008 does not correspond to the 2021 paper; possible indexing error.|chunk_summary_uncertainty|manual_or_followup_search|
|PepGen 1.0 web server is provided as a shortened URL, stability and accessibility uncertain.|chunk_summary_uncertainty|manual_or_followup_search|
|Relevance of tAMPer (peptide toxicity prediction model, PMID 39196703) to antimicrobial peptide prediction is unclear.|chunk_summary_uncertainty|manual_or_followup_search|
|AniAMPpred: fulltext provided (PMC12620532) is a different article; correct fulltext needed to verify details.|chunk_summary_uncertainty|manual_or_followup_search|
|Turkish paper: no abstract or fulltext; unclear if it describes a named model.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext for all papers was incorrectly cached; details from fulltext are unreliable|chunk_summary_uncertainty|manual_or_followup_search|
|CG-AMP lacks publicly available code and model weights|chunk_summary_uncertainty|manual_or_followup_search|
|SeqGAN-BERT-MLP model not named, no code, insufficient detail for reproducibility|chunk_summary_uncertainty|manual_or_followup_search|
|APD3, AMPlify, DAMP dataset URLs not provided in abstracts|chunk_summary_uncertainty|manual_or_followup_search|
|Regex-identified repositories (lsgkm, Basset, Deopen) may not be directly related to APIN|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext caching error for all four papers; details on models, datasets, training data, and benchmarks may be incomplete|chunk_summary_uncertainty|manual_or_followup_search|
|No code or data repository found for CG-AMP and Multi-CGAN|chunk_summary_uncertainty|manual_or_followup_search|
|The SeqGAN-BERT-MLP model has no unique name, no code, and insufficient details to assess reproducibility|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset links for APD3, AMPlify, and DAMP not provided in abstracts; fulltext may contain references|chunk_summary_uncertainty|manual_or_followup_search|
|Does MultiPep explicitly include AMP as one of its 20 bioactivity classes?|||
|Is the SSFGM-Model fulltext cache correct? Evidence suggests it may be mismatched with ACEP paper.|||
|Are the datasets from Nerita versicolor and Pomacea poeyana independent of the training data used by CAMPR3 and iAMPpred?|||
|Can we obtain the original training datasets for review-cited models (iAMP-2L, CAMPR3, etc.) by contacting authors?|||
|Should we include antifungal (AI4AFP) and antimalarial (CTCM-Neo) models as separate tasks in the benchmark?|||
|For webserver-only models, can we implement local versions or API wrappers?|||
|What is the exact negative set composition for ESM2-AFPpred?|||
|Are the ANIA and AI4AFP datasets publicly accessible without restrictions?|||
|How to handle missing pre-trained weights for code-only models: require authors to provide or train from scratch?|||
|Is there a suitable general AMP binary classification PLM model to replace ESM2-AFPpred as Transformer/LLM representative?|||
|What is the best multimodal model for general AMP after removing SSFGM-Model and AI4AFP?|||
|Should we enforce a permanent dataset DOI requirement for all datasets used in the final benchmark?|||
|What is the optimal sequence identity cutoff for very short peptides (≤30 aa)?|||
|Should we require bootstrapped confidence intervals for all metrics, and how to handle models that cannot be easily retrained?|||
|How to handle models that only output hard labels (no probability) for AUPRC computation?|||
|Is AMPlify truly a Transformer-based model and does it provide pre-trained weights for batch inference?|||
|Can a generic AMP PLM model (non-subclass) be found to replace ESM2-AFPpred as the representative?|||
|What is the correct paper and repository for SSFGM-Model? Evidence is currently broken.|||
|Are there any other general AMP ensemble models that could replace AI4AFP as the pipeline representative?|||
|No confident GitHub repository found for CTCM-Neo & ConformaX-PEP framework|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PCSPred|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for MultiPep|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for iAMP-2L|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for iAMPred|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AmPEP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AntiBP2|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for CAMPR3|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for MLAMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ClassAMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AVPpred|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPER|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for EFC-FCBF|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AVPIden|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for antibp|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ampsphere|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPDiscover|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AI4AFP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ANIA_webserver|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AI4AFP_webserver|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for DL-QSARES|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AI4AVP_web_server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Al-Omari 2024 AMP prediction model|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMAP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMAP webserver|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Deep-AmPEP30|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for EBAMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for DLFea4AMPGen|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AxPEP web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for COMDEL|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for C. acnes-targeted AMP generation pipeline (activity classifier)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for BERT-based AMP recognition model|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AmpGPT2|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AmpGPT2 code repository|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for COMPASS database|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PepCVAE|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PrefixProt|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for MoFormer|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for HMAMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AP_Sin|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP-Detector|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP-RNNpro|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP-RNNpro web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP-Distillation|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for STAMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for CF-AMP prediction|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP-DualTransnet|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP-FreqNet|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP prediction ML model|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for GAC-BiTCNN-AMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPGAN|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Macrel|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for iAMPpred|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for scan2030 (potential CVAE-BIO code)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AVPIden_web_server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ADAM_web_server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for antibp_web_server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ampsphere_web_server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP-GPT|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for MCL-AMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP prediction SVM-LZ|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for UniAMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for UniAMP web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP Scanner|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP Scanner v2|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PepGen 1.0|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPScanner vr.2 web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PepGen 1.0 web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP toxicity prediction model (hybrid)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Deep-AmPEP30 web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP toxicity prediction code|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for DRAMP database website|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ESM2-AFPpred|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ANIA._github_duplicate|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for treexplainer-study|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for CVAE-BIO|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for LMPred GitHub repository|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for GRAMPA dataset repository|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AmPepGen GitHub repository|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ANN-based AMP prediction model (Torrent et al. 2011)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPpredictor|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPActiPred Web Server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Generative AMP pipeline (VINCI)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPCLGPT|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for iMFP-LG BioCode Tool|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Deep learning model for AMP discovery from ruminant gastrointestinal microbiomes|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Deep learning model for AMP discovery from protist genomes (BERT+CNN)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for panCleave|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Bacteria-specific ML models for E. coli AMP activity|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for XGBoost AMP prediction model (Bhangu2025)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for scan2030 GitHub (potential CVAE-BIO code)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AmpGram web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AmpGram R package on CRAN|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Bidirectional LSTM AMP classification model (Wang2021)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PrMFTP web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPpred-AAIW web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for MIC prediction ensemble model (BiLSTM-CNN-MBM)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPpred-EL|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Multifunctional AMP Design Framework (FBGAN-enhanced)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPpredMFA|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for EnDL-HemoLyt|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AniAMPpred|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AniAMPpred webserver|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Appred webserver|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for LSTM-based AMP classifier/generator|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PepNet Zenodo record 1|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PepNet Zenodo record 2|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for CSAMPPRED|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Thomas et al. 2009 AMP prediction model|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for ANN-based AMP prediction model (ref [4])|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Two-level fuzzy K-NN model (ref [7])|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Sequence alignment-SVM-LZ complexity model (ref [8])|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Anti-Hepatitis Peptides predictor (ref [9])|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for Co-AMPpred|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for CTCM-Neo & ConformaX-PEP framework|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for Co-AMPpred GitHub repository|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for CoAMPpred|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for 2020-peptidomics|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for A-CaMP|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for PCSPred|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for iAMPCN|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for AAGP|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for SSFGM-Model|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for ACEP|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for ACP-DL|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for Anticancer-Peptides-CNN|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for MetagenomicDC|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for deep-belief-network|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for MultiPep|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for acp-ope|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for iAMP-2L|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for iAMPred|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|Qwen-Max web search found no usable repository/dataset/paper evidence for AmPEP|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, or web server documentation|
|No confident GitHub repository found for Gabere&Noble AMP predictor|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Wang et al. AMP predictor|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Witten&Witten AMP predictor|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Malebary-Khan AMP predictor|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for SeqGAN-BERT-MLP AMP identifier (Cao et al. 2023)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|Does MultiPep explicitly include AMP as a bioactivity class?|||
|Are the datasets from Nerita versicolor and Pomacea poeyana independent of training data?|||
|Can we obtain pre-trained weights for benchmark-ready models like iAMPCN, ACEP, and AMPlify?|||
|Should we include generative models like amp-gan in the benchmark, or only classifiers?|||
|How to handle models with only webserver access (e.g., ADAM, CAMPR3) for systematic benchmarking?|||
|CD-HIT去重和分组的精确相似度阈值（0.3、0.4还是0.5）需根据域内共识确定。|||
|是否应纳入Balanced Accuracy或G-mean作为补充报告指标？|||
|对于仅输出概率的模型，如何强制其使用统一阈值？是否要求提供阈值选择代码？|||
|低同源独立测试集的建设标准：是否应限定为特定数据库版本或近期发表的新肽？|||
|如何批量获取仅 webserver 模型的预测？是否开发统一 API 包装器，但可能引入网络延迟和版本不可控？|||
|对于权重缺失的模型，是否允许使用作者提供的训练脚本在标准数据集上重新训练，并固定种子，以纳入基准？|||
|低同源独立测试集的具体构建标准：应以哪些数据库（如最新版 APD3、DRAMP）为基准？去重相似度阈值选 0.4 还是 0.3？|||
|重复模型 AntiBP2 与 antibp 如何区分？是否需排除其一？|||
|多分布测试矩阵中负样本的采样策略：从 Swiss-Prot 随机抽取、从非 AMP 肽集中随机筛选，还是采用特定去重方案？|||
|chief_agent_failed_or_timed_out|DeepSeek call failed: chief_agent: Request timed out.|rerun with smaller --chunk-target-size or lower --max-results|
|Does MultiPep explicitly include AMP as one of its bioactivity classes?|||
|Are the datasets from Co-AMPpred and iAMPCN independent of each other?|||
|Can we obtain pre-trained weights for ACEP, AMPlify, SSFGM-Model?|||
|What is the exact negative set composition for the AMP binary classification? Need to avoid ACP, AIP, AFP, etc. pollution.|||
|Which PLM-based AMP prediction models have code and weights available?|||
|How to handle models that only provide web servers but have high impact? Should we design a separate API-based benchmark?|||
|Is the SSFGM-Model repository genuinely the official code of the paper? Needs full-text verification.|||
|What is the minimum sample size requirement for independent test sets?|||
|Should we include ANIA as a regression task or adapt it for binary classification?|||
|How to handle web-server-only models in a reproducible benchmark?|||
|MultiPep 是否明确包含 AMP 类别？|||
|SSFGM-Model 的训练数据集和权重文件是否存在于代码仓库？|||
|AMPDiscover web server 是否仍可访问？|||
|AMPlify 的已发表论文是否包含权重和数据集？|||
|如何统一处理不同模型的正负样本定义（如来自 UniProt 的非 AMP 序列）？|||
|是否将抗真菌肽、抗病毒肽等子类别纳入统一 benchmark？|||
|缺少预训练权重的模型是否允许在 benchmark 中重新训练？|||
|需要为每个模型编写统一的输入输出接口脚本。|||
|是否需纳入MIC预测等回归任务指标（如PCC, MSE）？|||
|如何标准化不同长度肽段的评价？|||
|外部验证集是否应包含实验验证的新肽段（如伤口肽组学数据）？|||
|对于生成模型的评价指标（多样性、新颖性）是否并入主benchmark？|||
|阈值固定后，是否需要报告校准误差（ECE）以评估概率可靠性？|||
|Macrel 和 Ampir 等模型虽可批量运行，但 Scout 未收录，是否应手工补入？|||
|对于 AMPDiscover 等仅有 web server 的模型，是否允许通过 API 批量提交？若允许，需评估延迟与成本|||
|如何标准化多肽长度？是否要求所有模型接受变长输入？|||
|外部验证集是否应包含实验验证的阴性肽？|||
|MIC 回归任务是否需单独设立 benchmark 赛道？|||
|How should negative samples be defined? Should we include only non-antimicrobial peptides from the same proteome, random sequences, or specific non-AMP peptides?|||
|How to handle multi-label or multi-class AMP subcategories (antibacterial, antifungal, etc.) in the benchmark?|||
|Should the independent low-homology set be curated from entirely new databases (e.g., peptidomics data) to ensure novelty?|||
|What statistical test is most appropriate for comparing models across multiple imbalance ratios?|||
|How to integrate sequence-level uncertainty (e.g., conformal prediction) into the evaluation framework?|||
|What is the correct paper DOI for ACEP? SSFGM-Model's real paper?|||
|Is there a suitable general AMP PLM model to replace ESM2-AFPpred in the representative set?|||
|Can we obtain pre-trained weights or reliable training scripts for Co-AMPpred, iAMPCN, ACEP, and AMPlify?|||
|How to construct a unified benchmark dataset from APD3, DRAMP, dbAMP with CD-HIT 0.9 and clear provenance?|||
|What is the deep learning hybrid model used in PMID 41731616? It is referenced as [7] but not described in the available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|ACPred original paper not found; only usage evidence from a benchmark study.|chunk_summary_uncertainty|manual_or_followup_search|
|All listed webservers lack original paper links, code, or dataset details in this chunk.|chunk_summary_uncertainty|manual_or_followup_search|
|No training or test datasets for any of the tools were reported.|chunk_summary_uncertainty|manual_or_followup_search|
|Are Deep-AmPEP30 and RF-AmPEP30 part of the same AmPEP tool? The paper distinguishes them as two algorithms; further clarification needed.|chunk_summary_uncertainty|manual_or_followup_search|
|CAMPR34 vs CAMPR3: version difference? Possibly a typo or variant.|chunk_summary_uncertainty|manual_or_followup_search|
|Many tools lack code/webserver URLs in the provided evidence; full-text may contain references but not captured here.|chunk_summary_uncertainty|manual_or_followup_search|
|ADAM is primarily a database; its role as a prediction tool is ambiguous.|chunk_summary_uncertainty|manual_or_followup_search|
|The APSvr.2 webserver URL is inferred; confirm that it exactly matches the Feb2019 model.|chunk_summary_uncertainty|manual_or_followup_search|
|No AMP prediction model identified in this chunk; further search may be needed for AMP-related tools.|chunk_summary_uncertainty|manual_or_followup_search|
|Fulltext cache mismatch for PMID 42042034 (retrieved PMC3403559 about GABA sedation, not the tuna peptide paper).|chunk_summary_uncertainty|manual_or_followup_search|
|Chunk labeled as 'model_allenccf' but allenCCF is a neuroanatomical tool, not a model; no AMP model identified.|chunk_summary_uncertainty|manual_or_followup_search|
|All extracted repositories and datasets are from the Neuron paper, no connection to AMP tasks.|chunk_summary_uncertainty|manual_or_followup_search|
|Is the multi-label WKnn-MLR model available as a web server or code?|chunk_summary_uncertainty|manual_or_followup_search|
|Does the cdGAN classifier (ESM2-based) function as a standalone AMP prediction model worthy of separate benchmarking?|chunk_summary_uncertainty|manual_or_followup_search|
|Are the filtered datasets from both papers directly downloadable from the APD version used or from the GitHub repository?|chunk_summary_uncertainty|manual_or_followup_search|
|No code, web server, or dataset links available for either model.|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset details not specified; external validation missing.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP-GSM classifier type not specified.|chunk_summary_uncertainty|manual_or_followup_search|
|ISCAPE training dataset size and composition not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|Code repository and model weights not found in available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Training dataset source not explicitly named; described only as ~5000 known AMPs with MIC values.|chunk_summary_uncertainty|manual_or_followup_search|
|Exact architectures and training data for Macrel, AmPEP, AMPlify, AxPEP not available in the reviewed papers; need original publications.|chunk_summary_uncertainty|manual_or_followup_search|
|GitHub enrichment results for AmPEP, AntiBP2, PrefixProt, deep-AmPEP30, CAMPR3 require manual verification before deployment.|chunk_summary_uncertainty|manual_or_followup_search|
|The six AMP classifiers used in the ant venom study (PMID:37235379) are not named; no further evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPGenix code and weights not reported; access may be limited to authors.|chunk_summary_uncertainty|manual_or_followup_search|
|StackAMP lacks full text; cannot verify architecture, availability, or performance.|chunk_summary_uncertainty|manual_or_followup_search|
|Macrel, AMP Scanner V2, AMPlify_bal, AMPlify_imbal details are from a secondary source; original papers are needed.|chunk_summary_uncertainty|manual_or_followup_search|
|GitHub candidate repositories may be unofficial, outdated, or incorrectly matched; manual verification required before use.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPGenix code and weights are not reported; access may be limited to the original authors.|chunk_summary_uncertainty|manual_or_followup_search|
|StackAMP completely lacks details (input features, datasets, performance) due to missing full text; must be treated as low-confidence until verified.|chunk_summary_uncertainty|manual_or_followup_search|
|AmPEP code not found; only web server is accessible, limiting reproducibility.|chunk_summary_uncertainty|manual_or_followup_search|
|Original papers for CAMP, AntiBP2, and AMPer were not provided; only referenced in PMID 23056189.|chunk_summary_uncertainty|manual_or_followup_search|
|GitHub enrichment candidates are unverified and may include false positives (e.g., AWS Amplify repos).|chunk_summary_uncertainty|manual_or_followup_search|
|The PEP-FOLD paper is unrelated to AMP prediction and was included in the chunk by mistake.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPer code repository not found; only web server URL known.|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset details (sample sizes, exact composition) are not fully reported.|chunk_summary_uncertainty|manual_or_followup_search|
|Original papers for AMPer, CAMP, and AntiBP2 are not provided; details are based on the 2012 PLOS ONE paper, which may be incomplete.|chunk_summary_uncertainty|manual_or_followup_search|
|The exact training data size, positive/negative splits, and performance for AMPer, CAMP, and AntiBP2 are not described in this chunk.|chunk_summary_uncertainty|manual_or_followup_search|
|The PEP-FOLD paper (doi:10.1093/nar/gkp323) is unrelated to AMP prediction and appears as a false positive in the batch.|chunk_summary_uncertainty|manual_or_followup_search|
|The deep learning hybrid model in PMID 41731616 is not named and its details are missing.|chunk_summary_uncertainty|manual_or_followup_search|
|Original papers for AMPfun, AntiCP, AntiCP2.0, ACPred, iAMPpred, HAPPENN, HemoPred, ToxinPred, ToxIBTL, AllerTop, AllergenFP, AllerCatPro were not provided in this chunk.|chunk_summary_uncertainty|manual_or_followup_search|
|No model weights, code (except Macrel), or datasets were available for these tools in the given evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Exact architectures and code availability for Macrel, AmPEP, AMP Scanner V2, AMPlify_bal, AMPlify_imbal were not provided in the reviewed paper; need to retrieve from original sources.|chunk_summary_uncertainty|manual_or_followup_search|
|The six AMP classifiers used in the ant venom study are not named; verification of exact models would require full review of that paper's methods.|chunk_summary_uncertainty|manual_or_followup_search|
|HydrAMP original model paper not identified; architecture details missing|chunk_summary_uncertainty|manual_or_followup_search|
|HydrAMP code repository requires manual verification for authenticity|chunk_summary_uncertainty|manual_or_followup_search|
|AMPlify model weights not explicitly reported; code repository may contain them|chunk_summary_uncertainty|manual_or_followup_search|
|Several GitHub enrichment results (e.g., AWS Amplify, amperfy) are unrelated and were filtered out|chunk_summary_uncertainty|manual_or_followup_search|
|Training data for AMPlify beyond Swiss-Prot mining not described|chunk_summary_uncertainty|manual_or_followup_search|
|The paper (PMID 40100125) is a mining study using AMPlify, not necessarily the original AMPlify model publication; the true AMPlify original paper may be different.|chunk_summary_uncertainty|manual_or_followup_search|
|No model weights or pre-trained model file is provided in the repository or paper.|chunk_summary_uncertainty|manual_or_followup_search|
|Training dataset and exact architecture details are not described in the available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Exact architectures, training data, code, and weights for Macrel, AmPEP, AMP Scanner V2, AMPlify_bal, and AMPlify_imbal are not available from the reviewed 2025 paper; original publications must be located.|chunk_summary_uncertainty|manual_or_followup_search|
|AMPGenix code and model weights are not publicly reported; availability may be restricted.|chunk_summary_uncertainty|manual_or_followup_search|
|The six AMP classifiers used in the 2023 ant venom study are unnamed; their identities cannot be verified from the abstract alone.|chunk_summary_uncertainty|manual_or_followup_search|
|Macrel benchmark table (DOI:10.7717/peerj-10555/table-1) lacks full text; full comparison data not extracted.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights not explicitly provided in repository; may need retraining.|chunk_summary_uncertainty|manual_or_followup_search|
|Two papers describe slightly different dataset configurations (low activity class omitted in the enteric dataset).|chunk_summary_uncertainty|manual_or_followup_search|
|PMC10541502 is listed for both PMID 38266820 and 37774026; likely a data error, need PMCID verification.|chunk_summary_uncertainty|manual_or_followup_search|
|PeerJ 10555 table is only a metadata entry; full Macrel model details are missing and require the original article for benchmark extraction.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights not explicitly provided in the repository; need to check if trained model is included.|chunk_summary_uncertainty|manual_or_followup_search|
|Training data variations between the two papers may affect model generalizability and benchmark comparability.|chunk_summary_uncertainty|manual_or_followup_search|
|For review-only models, original papers must be retrieved to verify architectures, availability, and datasets.|chunk_summary_uncertainty|manual_or_followup_search|
|Web server http://amp.biosino.org/ may not be active or accessible; need verification.|chunk_summary_uncertainty|manual_or_followup_search|
|The benchmark dataset used by Wang et al. is not directly downloadable from the provided evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|AntiBP2 web server URL not found in available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|CAMP and AntiBP original papers not provided; details are from PeptideRanker paper or benchmark review.|chunk_summary_uncertainty|manual_or_followup_search|
|Models from the benchmark study (ADAM, CAMPR3 variants, MLAMP, DBAASP, BAGEL3, BACTIBASE) lack original papers and detailed information.|chunk_summary_uncertainty|manual_or_followup_search|
|GitHub candidates for AMPlify are false matches (AWS Amplify).|chunk_summary_uncertainty|manual_or_followup_search|
|Some GitHub repositories (hydramp, Macrel, AmPEP) need verification of official status and completeness.|chunk_summary_uncertainty|manual_or_followup_search|
|Deep learning hybrid model (referenced as [7] in PMID 41731616) is not identified; original paper needed.|chunk_summary_uncertainty|manual_or_followup_search|
|AntiCP and other tools' original papers not provided, only citations from 2025/2026 papers.|chunk_summary_uncertainty|manual_or_followup_search|
|None of the models have provided code or weights, except Macrel (code available).|chunk_summary_uncertainty|manual_or_followup_search|
|Batch inference capability for webservers is unknown.|chunk_summary_uncertainty|manual_or_followup_search|
|Original training datasets for these tools are not reported in the current evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|ProToxin paper (PMID 41150190) is about protein toxin prediction, not antimicrobial peptides; its full text appears mismatched. No AMP model extracted.|chunk_summary_uncertainty|manual_or_followup_search|
|Regex-matched GitHub repositories 'Antimicrobial' and 'Antimicrobial‐Peptides' may be unrelated to PLUM; confidence low.|chunk_summary_uncertainty|manual_or_followup_search|
|Weights for PLUM model not reported; training script may be available but weights not provided, requiring retraining for benchmarking.|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset direct download link not available; assumed to be in GitHub repository.|chunk_summary_uncertainty|manual_or_followup_search|
|Duplicate repository URL 'https://github.com/priyamayur/PLUM.' (with trailing dot) was ignored as likely erroneous.|chunk_summary_uncertainty|manual_or_followup_search|
|所有模型的代码和训练权重均未在现有证据中报告。|chunk_summary_uncertainty|manual_or_followup_search|
|APD3、AVPpred、AVCpred仅来自综述，缺乏原始论文验证。|chunk_summary_uncertainty|manual_or_followup_search|
|CAMPR3和Deep-AmPEP30的训练数据集未公开，无法评估泛化能力。|chunk_summary_uncertainty|manual_or_followup_search|
|Deep-AmPEP30的Web服务器是否仍可访问且功能正常需进一步验证。|chunk_summary_uncertainty|manual_or_followup_search|
|CAMPR3 and APD3 lack code and weight availability; no direct web server URL found in the evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Deep-AmPEP30 weights and source code are not reported; only the web server is mentioned.|chunk_summary_uncertainty|manual_or_followup_search|
|APD3 is only evidenced from a review paper; original paper and web server URL need verification.|chunk_summary_uncertainty|manual_or_followup_search|
|AVPpred and AVCpred are specialized for antiviral peptides and may not be suitable for a general AMP benchmark.|chunk_summary_uncertainty|manual_or_followup_search|
|No dataset links or training data details are provided for any of the models.|chunk_summary_uncertainty|manual_or_followup_search|
|Code and trained weights for APEX and ApexGO are not publicly available in the provided evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Training data for APEX (in-house peptides) is not publicly described, preventing reproducibility.|chunk_summary_uncertainty|manual_or_followup_search|
|ApexGO's VAE training data details are missing, and the exact peptide sequence dataset used is not specified.|chunk_summary_uncertainty|manual_or_followup_search|
|The exact URL for DBAASP database was not provided in the evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|No public code or weights for APEX or ApexGO.|chunk_summary_uncertainty|manual_or_followup_search|
|APEX training data (in-house peptides) not described.|chunk_summary_uncertainty|manual_or_followup_search|
|ApexGO VAE training data details missing.|chunk_summary_uncertainty|manual_or_followup_search|
|DBAASP database link not provided in evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|APEX training data (in-house peptides) not publicly described.|chunk_summary_uncertainty|manual_or_followup_search|
|Code and model weights for both APEX and ApexGO not available in this evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|DBAASP database URL not provided; exact source unclear.|chunk_summary_uncertainty|manual_or_followup_search|
|Code and trained weights for APEX and ApexGO are not available.|chunk_summary_uncertainty|manual_or_followup_search|
|DBAASP database URL not provided in evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|Macrel's full performance metrics, dataset composition, and availability of code/weights remain unknown (only peer review abstract).|chunk_summary_uncertainty|manual_or_followup_search|
|The exact role of the aro and nov-fams-pipeline repositories in AMP prediction is unclear.|chunk_summary_uncertainty|manual_or_followup_search|
|Validation on external benchmarks or experimental data is not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|AVCpred and AVPpred are specifically designed for antiviral peptides, not general AMP prediction; their inclusion in an AMP benchmark is questionable.|chunk_summary_uncertainty|manual_or_followup_search|
|All models lack reported code repositories, model weights, and dataset links in the provided evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|CAMPR3 and APD3 are mentioned as databases/predictors but their exact web server URLs and version details are not in this chunk.|chunk_summary_uncertainty|manual_or_followup_search|
|Deep-AmPEP30 web server URL is available, but code/weights availability remains unclear.|chunk_summary_uncertainty|manual_or_followup_search|
|APD3 and AVCpred/AVPpred originate from a review paper; original research papers are needed for full verification.|chunk_summary_uncertainty|manual_or_followup_search|
|Original model papers for Macrel, AxPEP, and AMP Scanner V2 not identified; only usage evidence from PMID:41315055.|chunk_summary_uncertainty|manual_or_followup_search|
|No training datasets or evaluation metrics reported for these tools in the available evidence.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP Scanner V2 lacks a public code repository; benchmark feasibility limited to web server testing.|chunk_summary_uncertainty|manual_or_followup_search|
|Second paper (PMID:39969287) is unrelated to AMP prediction and was excluded from compression.|chunk_summary_uncertainty|manual_or_followup_search|
|Additional repositories (kneaddata, VirSorter2, COGclassifier) mentioned in the study are analysis tools, not AMP models.|chunk_summary_uncertainty|manual_or_followup_search|
|Nine models (ADAM, CAMPR3(RF), CAMPR3(SVM), MLAMP, DBAASP, AntiBP, AntiBP2, BAGEL3, BACTIBASE) lack original papers in this evidence chunk; details, code, and weights unknown.|chunk_summary_uncertainty|manual_or_followup_search|
|AMP prediction server (biosino) provides web server but no code/weights; benchmarking status limited to webserver.|chunk_summary_uncertainty|manual_or_followup_search|
|Dataset for AMP prediction server is derived from CAMP and UniProt, but no direct download link; may need to be reconstructed.|chunk_summary_uncertainty|manual_or_followup_search|
|Original paper describing the c_AMPs-prediction model (Ma et al. (12)) is not included in this chunk; the source paper (PMID 41164228) only uses the model.|chunk_summary_uncertainty|manual_or_followup_search|
|Model weights are not provided in the repository, limiting reproducibility without retraining.|chunk_summary_uncertainty|manual_or_followup_search|
|Training data details (public databases) are vague; the exact composition and source of training AMP sequences are not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|The dataset_url for the model points to the code repository, not a separated dataset file; actual training data may not be directly accessible.|chunk_summary_uncertainty|manual_or_followup_search|
|PEP-FOLD (PMID 19433514) is not an AMP predictor and its role is uncertain.|chunk_summary_uncertainty|manual_or_followup_search|
|CAMP and AntiBP2 lack original papers and direct links; classification confidence is moderate.|chunk_summary_uncertainty|manual_or_followup_search|
|Several datasets (BIOPEP, PeptideDB, APD2, CAMP) are named but URLs are not reported.|chunk_summary_uncertainty|manual_or_followup_search|
|No confident GitHub repository found for Deep learning-based AMP discovery in cell-free systems|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PeptideRanker|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for PeptideRanker web server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Two-layer ensemble classifier chain for AMP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Multi-label weighted KNN-MLR model|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for WeightedEnsemble_L3 (Anti_Cp)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for FBGAN-kmers|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for FBGAN-ESM2|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AllergenFP|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Deep learning hybrid model (unnamed)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for DBAASP linear AMP prediction|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for DBAASP linear AMP prediction webserver|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for RF-AmPEP30|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for CAMPR34|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for DBAASPv3.0|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Antimicrobial Peptide Scanner (APSvr.2) webserver|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPlify_imbal|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMPGenix|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for FED_AMP_activity_model|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for AMP MIC predictor (CNN/RNN)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Macrel Source Code|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Macrel Benchmark Repository|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for Macrel Web Server|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for CAMPR3(RF)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|No confident GitHub repository found for CAMPR3(SVM)|github_missing_model_enrichment_no_hit|try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for Venomics artificial intelligence|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for Deep learning-based AMP discovery in cell-free systems|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for HydrAMP|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for AMPlify|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for AMPlify GitHub|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for Macrel|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for MetaPepticon|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for AmPEP|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for StackAMP|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for AmPEP web server|qwen_web_enrichment_no_hit|try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation|
|Can we obtain weights for Co-AMPpred, iAMPCN, ACEP?|||
|What is the exact negative set composition for Co-AMPpred dataset?|||
|Is there any PLM-based AMP classifier with publicly available code and weights?|||
|How to construct an independent test set with verified negative peptides?|||
|How to handle models that cannot be executed locally (e.g., webserver-only)?|||
|What is the correct paper and code for SSFGM-Model?|||
|Should we enforce a minimum sequence identity threshold for the external test set (e.g., 40%)?|||
|ACEP paper DOI mismatch: need to locate correct paper.|||
|Are Deep-AmPEP30 and RF-AmPEP30 from the same AmPEP30 suite?|||
|Can we obtain weights for iAMPCN and ACEP?|||
|How to handle models that output only class labels without probability scores?|||
|What is the acceptable minimum sequence identity for the independent test set (e.g., 40% vs 70%)?|||
|Should the multi-distribution test sets be constructed from the same source databases or should they be completely independent?|||
|How to define negative set for AMP prediction (e.g., random peptides from UniProt, or specific non-AMPs with known functions)?|||
