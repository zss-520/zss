# meeting_orchestrator.py 修改说明

## 1. 增加 import

在 `meeting_orchestrator.py` 顶部 import 区域增加：

```python
from deep_research_literature_agent import (
    run_pubmed_deep_research_pipeline,
    build_context_from_deep_research_memory,
)
```

## 2. 替换旧的本地文献扫描流程

把原来的：

```python
stage1_summary = stage1_filter_model_papers(dir_model_papers)

if stage1_summary.get("approved_groups", 0) == 0 and stage1_summary.get("kept_on_error_groups", 0) == 0:
    print("!!! [Error] 一级筛选后没有剩余可精读的模型文献，流程终止。")
    return

print("\n>>> 正在全局扫描本地文献库，进入 Map-Reduce 处理管线...")
model_context = agentic_extract_from_papers(dir_model_papers)
benchmark_context = agentic_extract_from_papers(dir_benchmark_papers)
full_context = model_context + "\n" + benchmark_context

if len(full_context) < 50:
    print("!!! [Error] 未提取到有效干货！")
    return
```

替换为：

```python
stage1_summary = {
    "total_groups": 0,
    "approved_groups": 0,
    "rejected_groups": 0,
    "kept_on_error_groups": 0,
    "note": "新版流程使用 PubMed API + Qwen Deep Research Markdown Agent，不依赖本地 PDF/TXT 文献筛选。",
}

print("\n>>> 启动新版 PubMed + Qwen Deep Research 文献情报管线（不下载全文）...")

deep_result = run_pubmed_deep_research_pipeline(
    queries=[
        '("antimicrobial peptide"[Title/Abstract] OR AMP[Title/Abstract] OR "host defense peptide"[Title/Abstract]) '
        'AND (prediction[Title/Abstract] OR classification[Title/Abstract] OR identification[Title/Abstract]) '
        'AND ("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract] OR transformer[Title/Abstract] OR BERT[Title/Abstract] OR CNN[Title/Abstract])',

        '("antimicrobial peptide"[Title/Abstract] OR AMP[Title/Abstract]) '
        'AND (benchmark[Title/Abstract] OR dataset[Title/Abstract] OR database[Title/Abstract] OR evaluation[Title/Abstract] OR metrics[Title/Abstract])',
    ],
    max_results_per_query=30,
    year_from=past_year,
    year_to=current_year,
    batch_size=8,
    memory_md_path=Path("data/literature_deep_research_memory.md"),
    memory_index_path=Path("data/literature_deep_research_index.json"),
    raw_jsonl_path=Path("data/literature_deep_research_raw.jsonl"),
    agent_dir=Path("agents/deep_research"),
)

full_context = deep_result.get("memory_context", "")

print(f">>> Deep Research Agent 目录: {deep_result.get('agent_dir')}")
print(f">>> Deep Research 记忆文件: {deep_result.get('memory_md_path')}")
print(f">>> 本轮新处理文献数: {deep_result.get('num_new_records')}")
print(f">>> 本轮跳过已处理文献数: {deep_result.get('num_skipped_records')}")

if len(full_context) < 50:
    print("!!! [Error] 未提取到有效 Deep Research Markdown 记忆！")
    return
```
