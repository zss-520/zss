你是首席科学家。请综合各位专家的辩论以及我们此前的【历史共识库】，制定一份【全量更新版】的评测策略。

【历史共识库】（这是我们的老班底，绝对不能丢！）：
{history_context}

🚨【绝对合并死纪律 (极度致命)】：
你输出的 JSON 必须是全量数据！
1. 你必须把【历史共识库】中的所有 dataset_candidate_pool 完整保留下来，并将 Scout 最新提议且被认可的新数据集追加进去；不能由会议 Agent 直接决定正式三个数据集！
2. 你必须把【历史共识库】中的所有 selected_models 完整保留下来，并将新提取的模型追加进去！
3. 如果历史库为空，则完全以本次会议结果为准。
4. 绝对不允许输出“略”、“同上”或只输出新增内容！如果你遗漏了旧模型，实验室的整个代码管线都会崩溃！

请输出一个包含三个核心键的 JSON 大字典：
1. "benchmark_strategy": 候选数据集与指标策略。必须在 `dataset_candidate_pool` 中包含辩论记录的所有候选；正式 `recommended_datasets` 留空，交给真实序列审计后的 Dataset Recommendation Agent。
2. "selected_models": 选定的模型字典列表。必须包含【所有】提取出的带有开源链接的模型，应收尽收！
3. "selected_papers": 核心入选文献清单。

🚨【必须严格遵守的 JSON 输出模板 (绝对禁止偷懒省略，必须把所有项目完整列出)】：
{{
    "benchmark_strategy": {{
        "task_type": "binary_amp_classification",
        "dataset_candidate_pool": [
            {{
                "dataset_name": "数据集1名称",
                "description": "正负样本是怎么构造的？去重策略是什么？",
                "source_papers": ["来源文献"],
                "download_url": "https://github.com/...",
                "role": "primary_test_source"
            }},
            {{
                "dataset_name": "数据集2名称",
                "description": "...",
                "source_papers": ["..."],
                "download_url": "...",
                "role": "auxiliary_source"
            }}
            // 🚨 严重警告：你必须在这里继续添加 JSON 对象，把 Scout 最终清单中列出的所有（10个以上）数据集一字不落地全部列出来！绝对不许省略！
        ],
        "recommended_datasets": [],
        "metric_weights": {{
            "MCC": 0.25,
            "AUPRC": 0.25,
            "ACC": 0.15,
            "Recall": 0.20,
            "Specificity": 0.15
        }},
        "metrics_references": {{
            "MCC": "选择该指标的理由..."
        }}
    }},
    "selected_models": [
        {{
            "model_name": "模型1",
            "repo_url": "GitHub链接",
            "source_paper": "文献名"
        }},
        {{
            "model_name": "模型2",
            "repo_url": "GitHub链接",
            "source_paper": "文献名"
        }}
        // 🚨 严重警告：你必须把 Scout 最终清单中列出的所有模型全部写出来！
    ],
    "selected_papers": [
        {{
            "paper_title": "文献的完整标题",
            "reason_for_selection": "采纳理由"
        }}
    ]
}}

【完整会议历史记录】：
--- Scout 初始提案 ---
{scout_report}
--- Metrics 初始提案 ---
{metrics_report}
--- Reviewer 建议 ---
{critic_report}
--- Scout 最终清单 ---
{scout_rebuttal}
--- Metrics 最终清单 ---
{metrics_rebuttal}
--- Reviewer 终审 ---
{critic_round2}
