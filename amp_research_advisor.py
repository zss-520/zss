# amp_research_advisor.py
# -*- coding: utf-8 -*-

"""
AMP 预测模型未来发展方向分析器。

功能：
1. 读取 data/results/*/eval_result.json
2. 读取 data/results/*/critic_individual.md
3. 读取 data/meeting_trace.md
4. 读取 data/benchmark_strategy.json 中由文献分析会议生成的动态 metric_weights
5. 读取 data/model_knowledge_db.json
6. 汇总跨数据集表现、模型稳定性、指标覆盖情况和失败模式
7. 调用 Qwen/OpenAI-compatible API 生成：
   - AMP 模型下一步发展方向报告
   - benchmark 改进建议
   - 数据集与指标改进建议
   - wet-lab 验证优先方向

注意：
- 本文件不负责跑模型；
- 不负责 HPC 自动搭建；
- 只负责“基于已有评测结果生成研究发展建议”。
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import MODEL_NAME, BENCHMARK_STRATEGY, TARGET_MODEL_NAMES
from iterative_weight_meeting import run_iterative_weight_meeting
from prompts import (
    AMP_RESEARCH_ADVISOR_SYSTEM_PROMPT,
    build_amp_research_advisor_prompt,
)
from run_meeting import _make_openai_client


# ========================================================
# 基础安全读取工具
# ========================================================

def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 读取 JSON 失败: {path} | {e}")
        return None


def _safe_read_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars]
    except Exception as e:
        print(f"⚠️ 读取文本失败: {path} | {e}")
        return ""


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# ========================================================
# 动态指标读取与对齐
# ========================================================

def load_meeting_metric_weights() -> Dict[str, float]:
    """
    读取文献分析会议最终给出的指标体系。

    优先级：
    1. data/benchmark_strategy.json 中的 metric_weights
    2. config.py 中已加载的 BENCHMARK_STRATEGY（同一会议文件的内存副本）
    3. 如果都没有，返回空字典，由 50 轮会议从等权先验开始更新
    """
    strategy_path = Path("data/benchmark_strategy.json")
    strategy = _safe_load_json(strategy_path)

    if isinstance(strategy, dict):
        weights = strategy.get("metric_weights", {})
        if isinstance(weights, dict) and weights:
            return _normalize_weight_dict(weights)

    if isinstance(BENCHMARK_STRATEGY, dict):
        weights = BENCHMARK_STRATEGY.get("metric_weights", {})
        if isinstance(weights, dict) and weights:
            return _normalize_weight_dict(weights)

    return {}


def _normalize_weight_dict(weights: Dict[str, Any]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for k, v in weights.items():
        fv = _to_float(v)
        if fv is not None and fv > 0:
            cleaned[str(k).strip()] = fv

    total = sum(cleaned.values())
    if total <= 0:
        return cleaned

    # 统一归一化，避免会议输出的权重不是严格 1
    return {k: round(v / total, 6) for k, v in cleaned.items()}


def _canonical_metric_key(metric_name: str) -> str:
    """
    用于内部对齐不同写法。
    不改变最终报告中的会议指标名称，只用于匹配 eval_result 里的字段。
    """
    s = str(metric_name).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.replace("-", "")

    aliases = {
        "acc": "accuracy",
        "accuracy": "accuracy",

        "aucpr": "auprc",
        "auprc": "auprc",
        "prauc": "auprc",
        "averageprecision": "auprc",

        "aucroc": "auroc",
        "rocauc": "auroc",
        "auroc": "auroc",

        "mcc": "mcc",
        "matthewscorrelationcoefficient": "mcc",
        "matthewscorrcoef": "mcc",

        "f1": "f1",
        "f1score": "f1",

        "recall": "recall",
        "sensitivity": "recall",

        "precision": "precision",

        "recall@spec95%": "recallatspec95",
        "recall@spec95": "recallatspec95",
        "recallatspec95": "recallatspec95",
        "recallatspecificity95": "recallatspec95",
        "recallatspecificity95%": "recallatspec95",

        "ef@1%": "efat1",
        "ef@1": "efat1",
        "ef1": "efat1",
        "efat1": "efat1",

        "ef@5%": "efat5",
        "ef@5": "efat5",
        "ef5": "efat5",
        "efat5": "efat5",
    }

    return aliases.get(s, s)


def _match_metric_weight(metric_name: str, metric_weights: Dict[str, float]) -> Tuple[Optional[str], float]:
    """
    将 eval_result 中的指标名匹配到会议指标名。
    返回：
    - meeting_metric_name
    - weight
    """
    metric_key = _canonical_metric_key(metric_name)

    for meeting_metric, weight in metric_weights.items():
        if _canonical_metric_key(meeting_metric) == metric_key:
            return meeting_metric, float(weight)

    return None, 0.0


def format_dynamic_metrics_for_prompt(metric_weights: Dict[str, float]) -> str:
    """
    生成给 prompt 看的动态指标描述。
    """
    if not metric_weights:
        return (
            "- 未在 data/benchmark_strategy.json 或 config.METRIC_WEIGHTS 中检测到会议指标权重。\n"
            "- 请基于 eval_result.json 中实际出现的指标进行分析，但必须明确说明当前缺少正式会议指标体系。"
        )

    lines = []
    for i, (metric, weight) in enumerate(metric_weights.items(), 1):
        lines.append(f"{i}. {metric}: 权重 {weight:.4f}")

    return "\n".join(lines)


# ========================================================
# 读取 eval_result 与 critic 报告
# ========================================================

def collect_eval_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    读取 data/results/{dataset}/eval_result.json，整理成长表。

    支持的 eval_result.json 格式：
    {
      "ModelA": {
        "AUC-PR": 0.82,
        "MCC": 0.61
      },
      "ModelB": {
        "AUC-PR": 0.76,
        "MCC": 0.55
      }
    }
    """
    rows: List[Dict[str, Any]] = []
    is_manual_import = (results_dir / "manual_import_manifest.json").exists()
    target_models = set() if is_manual_import else set(TARGET_MODEL_NAMES or [])

    if not results_dir.exists():
        return rows

    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir():
            continue

        eval_path = ds_dir / "eval_result.json"
        eval_data = _safe_load_json(eval_path)
        if not isinstance(eval_data, dict):
            continue

        dataset_name = ds_dir.name

        for model_name, metrics in eval_data.items():
            if target_models and str(model_name) not in target_models:
                continue
            if not isinstance(metrics, dict):
                continue

            for metric_name, raw_value in metrics.items():
                value = _to_float(raw_value)
                if value is None:
                    continue

                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": str(model_name),
                        "metric": str(metric_name),
                        "metric_key": _canonical_metric_key(str(metric_name)),
                        "value": value,
                    }
                )

    return rows


def collect_dataset_critics(results_dir: Path) -> Dict[str, str]:
    """
    读取每个数据集下的 critic_individual.md。
    """
    critics: Dict[str, str] = {}

    if not results_dir.exists():
        return critics

    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir():
            continue

        critic_path = ds_dir / "critic_individual.md"
        text = _safe_read_text(critic_path, max_chars=8000)
        if text:
            critics[ds_dir.name] = text

    return critics


# ========================================================
# 跨数据集统计与动态指标分析
# ========================================================

def summarize_model_performance(
    rows: List[Dict[str, Any]],
    metric_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    生成模型层面的性能摘要。

    包含两类信息：
    1. raw metric summary:
       每个模型在每个指标上的 mean/std/min/max/n_datasets。
    2. weighted_rank_score:
       根据会议动态指标做 rank-based 加权分。
       这样可以避免 EF@1% 这类尺度大于 1 的指标直接支配加权平均。
    """
    raw_summary = _summarize_raw_metrics(rows)
    weighted_rank_summary = _calculate_weighted_rank_scores(rows, metric_weights)

    return {
        "raw_metric_summary": raw_summary,
        "weighted_rank_summary": weighted_rank_summary,
    }


def _summarize_raw_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model_metric: Dict[str, Dict[str, List[float]]] = {}

    for row in rows:
        model = row["model"]
        metric = row["metric"]
        value = row["value"]
        by_model_metric.setdefault(model, {}).setdefault(metric, []).append(value)

    summary: Dict[str, Any] = {}

    for model, metric_map in by_model_metric.items():
        model_summary: Dict[str, Any] = {}

        for metric, values in metric_map.items():
            if not values:
                continue

            mean_v = statistics.mean(values)
            std_v = statistics.pstdev(values) if len(values) > 1 else 0.0

            model_summary[metric] = {
                "mean": round(mean_v, 6),
                "std": round(std_v, 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "n_values": len(values),
            }

        summary[model] = model_summary

    return summary


def _calculate_weighted_rank_scores(
    rows: List[Dict[str, Any]],
    metric_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    使用会议动态指标计算 rank-based 综合分。

    逻辑：
    - 对每个 dataset + metric 分别给模型排序；
    - 最好模型得 1，最差模型得 0；
    - 用会议指标权重加权；
    - 对每个模型只按它实际有结果的指标归一化。
    """
    if not rows or not metric_weights:
        return {
            "note": "缺少评测结果或会议动态指标，未计算 weighted_rank_score。",
            "ranked_models": [],
            "model_scores": {},
        }

    # group: (dataset, meeting_metric) -> [(model, value)]
    grouped: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}

    for row in rows:
        metric_name = row["metric"]
        meeting_metric, weight = _match_metric_weight(metric_name, metric_weights)
        if not meeting_metric or weight <= 0:
            continue

        key = (row["dataset"], meeting_metric)
        grouped.setdefault(key, []).append((row["model"], row["value"]))

    score_sum: Dict[str, float] = {}
    weight_sum: Dict[str, float] = {}
    evidence_count: Dict[str, int] = {}

    for (dataset, meeting_metric), model_values in grouped.items():
        if not model_values:
            continue

        weight = metric_weights.get(meeting_metric, 0.0)
        if weight <= 0:
            continue

        # 指标默认越大越好
        model_values = sorted(model_values, key=lambda x: x[1], reverse=True)

        n = len(model_values)
        if n == 1:
            model_rank_scores = [(model_values[0][0], 1.0)]
        else:
            model_rank_scores = []
            for rank_idx, (model, value) in enumerate(model_values):
                # rank_idx=0 -> 1.0, rank_idx=n-1 -> 0.0
                rank_score = 1.0 - (rank_idx / (n - 1))
                model_rank_scores.append((model, rank_score))

        for model, rank_score in model_rank_scores:
            score_sum[model] = score_sum.get(model, 0.0) + rank_score * weight
            weight_sum[model] = weight_sum.get(model, 0.0) + weight
            evidence_count[model] = evidence_count.get(model, 0) + 1

    model_scores: Dict[str, Any] = {}
    for model, s in score_sum.items():
        ws = weight_sum.get(model, 0.0)
        if ws <= 0:
            continue

        model_scores[model] = {
            "weighted_rank_score": round(s / ws, 6),
            "covered_weight_sum": round(ws, 6),
            "evidence_count": evidence_count.get(model, 0),
        }

    ranked = sorted(
        model_scores.items(),
        key=lambda x: x[1]["weighted_rank_score"],
        reverse=True,
    )

    return {
        "note": (
            "weighted_rank_score 是基于会议动态指标的 rank-based 加权分。"
            "它用于比较模型相对表现，避免不同指标尺度不一致导致直接加权失真。"
        ),
        "ranked_models": [
            {
                "rank": i + 1,
                "model": model,
                **data,
            }
            for i, (model, data) in enumerate(ranked)
        ],
        "model_scores": model_scores,
    }


def summarize_metric_coverage(
    rows: List[Dict[str, Any]],
    metric_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    检查会议指标在实际 eval_result 中的覆盖情况。
    """
    actual_metric_keys = {_canonical_metric_key(row["metric"]) for row in rows}

    coverage: Dict[str, Any] = {}

    for meeting_metric, weight in metric_weights.items():
        key = _canonical_metric_key(meeting_metric)
        is_present = key in actual_metric_keys

        coverage[meeting_metric] = {
            "weight": weight,
            "present_in_eval_results": is_present,
        }

    extra_metrics = []
    for row in rows:
        matched_metric, _ = _match_metric_weight(row["metric"], metric_weights)
        if matched_metric is None:
            extra_metrics.append(row["metric"])

    return {
        "meeting_metric_coverage": coverage,
        "extra_metrics_not_in_meeting_weights": sorted(set(extra_metrics)),
    }


# ========================================================
# 构建 AI 分析上下文
# ========================================================

def build_development_context(
    results_dir: Path = Path("data/results"),
    output_dir: Optional[Path] = None,
    weight_rounds: int = 50,
    weight_seed: int = 20260716,
) -> Dict[str, Any]:
    rows = collect_eval_results(results_dir)
    critics = collect_dataset_critics(results_dir)
    initial_metric_weights = load_meeting_metric_weights()
    manual_import_manifest = _safe_load_json(results_dir / "manual_import_manifest.json") or {}
    target_model_names = list(TARGET_MODEL_NAMES or [])

    iterative_meeting: Dict[str, Any] = {}
    metric_weights = initial_metric_weights
    if rows:
        iterative_meeting = run_iterative_weight_meeting(
            rows=rows,
            output_dir=output_dir or results_dir,
            rounds=weight_rounds,
            seed=weight_seed,
            initial_weights=initial_metric_weights,
        )
        metric_weights = iterative_meeting.get("final_weights_median", {})

    performance_summary = summarize_model_performance(rows, metric_weights)
    metric_coverage = summarize_metric_coverage(rows, metric_weights)

    meeting_trace = _safe_read_text(Path("data/meeting_trace.md"), max_chars=18000)
    benchmark_strategy = _safe_load_json(Path("data/benchmark_strategy.json")) or {}
    model_db = _safe_load_json(Path("data/model_knowledge_db.json")) or {}

    models_in_db = model_db.get("models", []) if isinstance(model_db, dict) else []
    papers_in_db = model_db.get("papers", []) if isinstance(model_db, dict) else []

    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    actual_metrics = sorted({row["metric"] for row in rows})

    # 控制上下文大小，防止请求过大
    compact_rows = rows[:1000]

    context = {
        "task": "AMP binary classification benchmark and future direction analysis",
        "results_dir": str(results_dir),
        "target_model_names": target_model_names,
        "scope_note": (
            "Only models in target_model_names are in scope for this report. "
            "Ignore stale critic text or historical result files mentioning non-target models."
            if target_model_names
            else "No TARGET_MODEL_NAMES filter is active; all valid eval_result models are in scope."
        ),
        "available_datasets": datasets,
        "evaluated_models": models,
        "actual_metrics_in_eval_results": actual_metrics,
        "meeting_metric_weights": metric_weights,
        "initial_literature_meeting_weights": initial_metric_weights,
        "iterative_weight_meeting": {
            key: value for key, value in iterative_meeting.items() if key != "round_records"
        },
        "manual_import_manifest": manual_import_manifest if isinstance(manual_import_manifest, dict) else {},
        "metric_coverage": metric_coverage,
        "num_result_rows": len(rows),
        "performance_rows_preview": compact_rows,
        "performance_summary": performance_summary,
        "dataset_critic_reports": critics,
        "benchmark_strategy": benchmark_strategy,
        "meeting_trace_preview": meeting_trace,
        "model_knowledge_db_summary": {
            "num_models": len(models_in_db),
            "num_papers": len(papers_in_db),
            "models_preview": models_in_db[:30],
            "papers_preview": papers_in_db[:30],
        },
        "analysis_focus": [
            "Use the median metric weights and model ranking from the 50-round evidence meeting.",
            "Do not apply model-specific priority bonuses or manually force a Top3.",
            "Respect target_model_names. Do not discuss non-target models as failed in the current run.",
            "Identify model families that generalize well across datasets.",
            "Identify models that are dataset-sensitive or unstable.",
            "Infer possible benchmark/data issues such as leakage, easy negatives, or distribution shift.",
            "Recommend next-generation AMP prediction model directions.",
            "Recommend better benchmark construction and metric protocols.",
            "Suggest wet-lab validation priorities based on screening-oriented metrics.",
        ],
    }

    return context


# ========================================================
# 报告生成
# ========================================================

def _offline_future_directions_report(context: Dict[str, Any], error: Optional[Exception] = None) -> str:
    """Generate a deterministic local report when the LLM API is unavailable."""
    rows = context.get("performance_rows_preview", [])
    by_model: Dict[str, Dict[str, List[float]]] = {}
    datasets_by_model: Dict[str, set] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or "").strip()
        metric = str(row.get("metric") or "").strip()
        value = _to_float(row.get("value"))
        dataset = str(row.get("dataset") or "").strip()
        if not model or not metric or value is None:
            continue
        by_model.setdefault(model, {}).setdefault(metric, []).append(value)
        if dataset:
            datasets_by_model.setdefault(model, set()).add(dataset)

    weights = context.get("meeting_metric_weights") or {}
    if not isinstance(weights, dict):
        weights = {}
    meeting_summary = context.get("iterative_weight_meeting")
    if not isinstance(meeting_summary, dict):
        meeting_summary = {}
    meeting_ranking = meeting_summary.get("final_ranking")
    if not isinstance(meeting_ranking, list):
        meeting_ranking = []
    resource_gate = meeting_summary.get("resource_gate")
    if not isinstance(resource_gate, dict):
        resource_gate = {}

    def metric_mean(model: str, metric: str) -> Optional[float]:
        values = by_model.get(model, {}).get(metric, [])
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None

    def metric_std(model: str, metric: str) -> float:
        values = by_model.get(model, {}).get(metric, [])
        values = [v for v in values if v is not None]
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    def metric_weighted_score(model: str) -> float:
        total = 0.0
        used = 0.0
        for metric, weight in weights.items():
            m = metric_mean(model, str(metric))
            if m is None:
                continue
            w = _to_float(weight) or 0.0
            total += m * w
            used += w
        return total / used if used else 0.0

    scored = []
    ranking_lookup = {
        str(row.get("model")): row
        for row in meeting_ranking
        if isinstance(row, dict) and row.get("model")
    }
    for model in sorted(by_model):
        meeting_row = ranking_lookup.get(model, {})
        score = _to_float(meeting_row.get("median_score"))
        if score is None:
            score = metric_weighted_score(model)
        scored.append(
            {
                "model": model,
                "score": score,
                "mean_score": _to_float(meeting_row.get("mean_score")),
                "score_iqr": _to_float(meeting_row.get("score_iqr")),
                "top3_frequency": _to_float(meeting_row.get("top3_frequency")),
                "auprc": metric_mean(model, "AUPRC"),
                "mcc": metric_mean(model, "MCC"),
                "recall": metric_mean(model, "Recall"),
                "precision": metric_mean(model, "Precision"),
                "auroc": metric_mean(model, "AUROC"),
                "datasets": len(datasets_by_model.get(model, set())),
                "stability_penalty": metric_std(model, "AUPRC") + metric_std(model, "MCC"),
            }
        )
    scored.sort(
        key=lambda x: (
            x["score"],
            x["auprc"] if x["auprc"] is not None else -1,
            x["mcc"] if x["mcc"] is not None else -1,
            x["datasets"],
            -x["stability_penalty"],
        ),
        reverse=True,
    )
    top3 = scored[:3]

    def fmt(v: Any) -> str:
        fv = _to_float(v)
        return f"{fv:.4f}" if fv is not None else "NA"

    lines = [
        "# AMP 模型未来发展方向报告",
        "",
        "> 本报告由本地确定性兜底逻辑生成：没有把评测数据发送到外部 LLM API。",
        "",
        "## 当前评测概况",
        "",
        f"- 结果目录：`{context.get('results_dir')}`",
        f"- 数据集数量：{len(context.get('available_datasets') or [])}",
        f"- 有效参评模型数量：{len(by_model)}",
        f"- 实际指标：{', '.join(context.get('actual_metrics_in_eval_results') or [])}",
        f"- 权重更新轮数：{meeting_summary.get('rounds', 0)}",
        f"- 最终指标权重（50 轮中位数）：{weights}",
        "- 模型特定优先加分：未启用",
        f"- 资源资格规则：{resource_gate.get('policy_mode', '未记录')}（在性能评分前统一执行）",
        f"- 因实测资源超限排除：{', '.join(resource_gate.get('excluded_models') or []) or '无'}",
        f"- 尚缺资源测量但暂时保留：{len(resource_gate.get('flagged_models') or [])} 个模型",
    ]
    if error is not None:
        lines.append(f"- LLM 报告生成失败原因：`{type(error).__name__}: {error}`")
    lines.extend([
        "",
        "## 跨数据集综合排名",
        "",
        "| Rank | Model | Median score | Mean score | Score IQR | Top3 frequency | AUPRC | MCC | Recall | Precision | AUROC | Datasets |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for i, row in enumerate(scored[:10], 1):
        lines.append(
            "| " + " | ".join([
                str(i),
                row["model"],
                fmt(row["score"]),
                fmt(row["mean_score"]),
                fmt(row["score_iqr"]),
                fmt(row["top3_frequency"]),
                fmt(row["auprc"]),
                fmt(row["mcc"]),
                fmt(row["recall"]),
                fmt(row["precision"]),
                fmt(row["auroc"]),
                str(row["datasets"]),
            ]) + " |"
        )

    lines.extend(["", "## Top3 集成学习候选模型推荐", ""])
    if len(top3) < 3:
        lines.append("当前不足以组成 Top3；建议继续补齐至少 3 个有完整概率输出和跨数据集结果的模型后再训练 ensemble。")
    else:
        names = [x["model"] for x in top3]
        lines.append(f"推荐 Top3：**{names[0]}**, **{names[1]}**, **{names[2]}**。")
        lines.extend(["", "| Rank | Model | 推荐理由 |", "|---:|---|---|"])
        for i, row in enumerate(top3, 1):
            lines.append(
                f"| {i} | {row['model']} | 50 轮中位综合分 {fmt(row['score'])}、IQR {fmt(row['score_iqr'])}、进入 Top3 频率 {fmt(row['top3_frequency'])}；AUPRC {fmt(row['auprc'])}，MCC {fmt(row['mcc'])}，Recall {fmt(row['recall'])}，Precision {fmt(row['precision'])}；覆盖 {row['datasets']} 个数据集。 |"
            )
        lines.extend([
            "",
            "### 为什么推荐集成学习",
            "",
            "这组模型的互补性主要来自 Precision/Recall 取舍和跨数据集稳定性差异。AUPRC 更高的模型更适合做候选排序主干，Recall 更高的模型适合扩大候选召回，MCC/Precision 更高的模型适合在后处理阶段压低假阳性。单模型通常只能固定在一个阈值策略上，而 ensemble 可以把排序能力、召回能力和阈值决策拆开组合。",
            "",
            "### 推荐集成策略",
            "",
            "- **首选 rank averaging / soft voting**：当前已有多个模型的概率输出，最容易无训练集泄漏地实现，可先按跨数据集 AUPRC/MCC 作为权重做加权平均。",
            "- **可选 stacking/meta-classifier**：如果后续有独立验证集，可用 Logistic Regression 或 LightGBM 作为二层模型；没有验证集时不建议在测试集上调参。",
            "- **high-recall candidate union**：用于湿实验筛选前置阶段，先用高 Recall 模型并集扩大候选，再用高 Precision/MCC 模型二次排序。",
            "- **不建议只做 hard voting**：hard voting 会丢失概率排序信息，对 AUPRC 和早期富集不友好。",
        ])

    lines.extend([
        "",
        "## 下一步建议",
        "",
        "1. 保留当前手动预测文件作为 raw evidence，并固定 `final_results_with_predictions.csv` 作为后续复现实验输入。",
        "2. 为 Top3 候选构建独立验证集，避免在测试集上选择 ensemble 权重或阈值。",
        "3. 优先报告 AUPRC、MCC、Recall、Precision 和覆盖率；如果用于湿实验筛选，额外加入 top-k enrichment 或 Recall@Precision。",
        "4. 对高 Recall 但 Precision 较弱的模型做二阶段过滤，对高 Precision 但 Recall 较弱的模型做候选排序校准。",
    ])
    return "\n".join(lines) + "\n"


def generate_amp_future_directions_report(
    results_dir: Path = Path("data/results"),
    output_dir: Path = Path("data/results"),
    weight_rounds: int = 50,
    weight_seed: int = 20260716,
    use_llm_report: bool = True,
) -> Path:
    """
    主入口：生成 AMP 模型未来发展方向报告。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_development_context(
        results_dir=results_dir,
        output_dir=output_dir,
        weight_rounds=weight_rounds,
        weight_seed=weight_seed,
    )
    metric_weights = context.get("meeting_metric_weights", {})

    context_path = output_dir / "amp_development_context.json"
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)

    if context["num_result_rows"] == 0:
        report = """# AMP 预测模型下一步发展方向分析报告

## 当前状态

未在 `data/results/*/eval_result.json` 中检测到可用的跨数据集评测结果。

## 下一步

请先完成至少一个数据集上的模型评测，并确保结果保存为：

```text
data/results/{dataset_name}/eval_result.json
data/results/{dataset_name}/critic_individual.md
```

之后重新运行：

```bash
python amp_research_advisor.py
```

系统将基于跨数据集结果、会议指标体系和 Critic 报告生成模型发展方向分析报告。
"""
        report_path = output_dir / "amp_future_directions_report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"⚠️ 未检测到 eval_result.json，已生成占位报告: {report_path}")
        return report_path

    dynamic_metrics_text = format_dynamic_metrics_for_prompt(metric_weights)
    context_json = json.dumps(context, ensure_ascii=False, indent=2)

    prompt = build_amp_research_advisor_prompt(
        context_json=context_json,
        dynamic_metrics_text=dynamic_metrics_text,
    )

    if use_llm_report:
        try:
            client = _make_openai_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": AMP_RESEARCH_ADVISOR_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
            )
            report = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"WARNING: LLM report generation failed; writing offline deterministic report: {exc}")
            report = _offline_future_directions_report(context, error=exc)
    else:
        report = _offline_future_directions_report(context)

    report_path = output_dir / "amp_future_directions_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n[OK] AMP 未来发展方向报告已生成: {report_path}")
    print(f"[OK] 分析上下文已保存: {context_path}")
    print("[OK] 本次报告使用的会议动态指标：")
    print(dynamic_metrics_text)

    return report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="基于一次 benchmark run 的结果生成研究建议")
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--weight-rounds", type=int, default=50)
    parser.add_argument("--weight-seed", type=int, default=20260716)
    parser.add_argument("--offline-report", action="store_true", help="Generate the report locally without calling an external LLM.")
    args = parser.parse_args()
    raise SystemExit(
        0
        if generate_amp_future_directions_report(
            results_dir=args.results_dir,
            output_dir=args.output_dir or args.results_dir,
            weight_rounds=args.weight_rounds,
            weight_seed=args.weight_seed,
            use_llm_report=not args.offline_report,
        )
        else 1
    )
