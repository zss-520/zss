#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMP Benchmark 项目统一运行菜单

用途：
- 统一启动 benchmark、模型入库、文献检索和研究报告。
- 不用反复记各脚本的长命令和环境变量。

运行：
    python amp_benchmark_menu.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from workflow_guards import model_readiness_issues
from benchmark_portfolio import build_benchmark_portfolio
from dataset_gate import dataset_gate_issues

ROOT = Path(__file__).resolve().parent
LITERATURE_MAIN = ROOT / "deep_research_literature_agent.py"
LLM_TOP_JOURNAL_MAIN = ROOT / "llm_top_journal_model_pipeline.py"
SCIENTIFIC_MODEL_EVIDENCE_MAIN = ROOT / "scientific_model_evidence.py"
LITERATURE_AGENT_EVALUATION_MAIN = ROOT / "literature_agent_evaluation.py"
BENCHMARK_MAIN = ROOT / "main.py"
ONBOARDING_MAIN = ROOT / "new_model_onboarding.py"
ADVISOR_MAIN = ROOT / "amp_research_advisor.py"
DATASET_GATE_MAIN = ROOT / "dataset_gate.py"
DATASET_RECOMMENDER_MAIN = ROOT / "dataset_recommendation_agent.py"
DATA_DIR = ROOT / "data"
REGISTRY_PATH = DATA_DIR / "local_registry.json"


def _current_year() -> int:
    # 按本地机器时间；如果机器时间不准，可用菜单里的自定义年份。
    return _dt.datetime.now().year


def _print_header() -> None:
    print("\n" + "=" * 78)
    print(" AMP Benchmark 项目统一运行菜单")
    print("=" * 78)
    print("当前目录:", ROOT)
    print("入口脚本: benchmark / onboarding / literature / advisor")
    print("建议首次使用先选择 5 查看状态，再按 6 的推荐顺序运行。")
    print("=" * 78)


def _ask(prompt: str, default: Optional[str] = None) -> str:
    if default is None:
        text = input(f"{prompt}: ").strip()
    else:
        text = input(f"{prompt} [{default}]: ").strip()
        if not text:
            text = default
    return text


def _ask_int(prompt: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("请输入整数。")
            continue
        if min_value is not None and value < min_value:
            print(f"不能小于 {min_value}。")
            continue
        if max_value is not None and value > max_value:
            print(f"不能大于 {max_value}。")
            continue
        return value


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "是", "1", "true"}


def _quote_cmd(cmd: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(shlex.quote(x) for x in cmd)


def _run(cmd: List[str], *, env_overrides: Optional[dict[str, str]] = None) -> int:
    print("\n即将运行命令:")
    print(_quote_cmd(cmd))
    print("-" * 78)
    if not _ask_yes_no("确认运行", True):
        print("已取消。")
        return 130
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print("-" * 78)
    print("命令退出码:", proc.returncode)
    return proc.returncode


def _base_cmd() -> List[str]:
    return [sys.executable, str(LITERATURE_MAIN)]


def _recent_search(year_from: int, year_to: int, *, label: str) -> int:
    print(f"\n模式: {label}")
    max_results = _ask_int("每个来源 max-results，越大越全面但越慢", 50, 1, 500)
    batch_size = _ask_int("DeepSeek 提取 batch-size，全文较长建议 2-4", 4, 1, 20)
    citation_seed = _ask_int("citation-seed-limit，引用/相似文章扩展数量", 8, 0, 100)
    max_chunks = _ask_int("max-chunks，最多压缩多少个 evidence chunk", 120, 1, 1000)
    use_qwen = _ask_yes_no("是否同时启用 Qwen3.7-Max 联网补漏（会消耗百炼额度）", False)
    qwen_n = 10
    if use_qwen:
        qwen_n = _ask_int("Qwen3.7-Max 本轮最多补漏多少个模型", 10, 1, 200)
    force_github = _ask_yes_no("是否强制 GitHub 补链失败/低置信模型重搜", True)

    cmd = _base_cmd() + [
        "--year-from", str(year_from),
        "--year-to", str(year_to),
        "--max-results", str(max_results),
        "--batch-size", str(batch_size),
        "--citation-seed-limit", str(citation_seed),
        "--max-chunks", str(max_chunks),
    ]
    if force_github:
        cmd.append("--force-github-enrichment")
    if use_qwen:
        cmd += ["--qwen-web-enrichment", "--qwen-web-provider", "dashscope_qwen37max_search", "--qwen-web-model", "qwen3.7-max", "--qwen-web-max-models", str(qwen_n)]
    return _run(cmd)


def _github_only() -> int:
    max_models = _ask_int("GitHub 补链最多处理多少个模型", 80, 1, 1000)
    repos_per_model = _ask_int("每个模型最多保留几个候选仓库", 3, 1, 20)
    refresh_all = _ask_yes_no("是否全部重搜 GitHub 缓存（通常不要）", False)
    cmd = _base_cmd() + [
        "--use-existing-meeting",
        "--force-github-enrichment",
        "--github-enrich-max-models", str(max_models),
        "--github-enrich-repos-per-model", str(repos_per_model),
    ]
    if refresh_all:
        cmd.append("--refresh-all-github-enrichment")
    return _run(cmd)


def _qwen_only() -> int:
    max_models = _ask_int("Qwen3.7-Max 联网补漏最多处理多少个模型", 20, 1, 200)
    force = _ask_yes_no("是否重搜失败/低置信 Qwen 缓存", True)
    refresh_all = False
    if force:
        refresh_all = _ask_yes_no("是否全部重搜 Qwen 缓存（会消耗更多额度，通常不要）", False)
    cmd = _base_cmd() + [
        "--use-existing-meeting",
        "--qwen-web-enrichment",
        "--qwen-web-provider", "dashscope_qwen37max_search",
        "--qwen-web-model", "qwen3.7-max",
        "--qwen-web-max-models", str(max_models),
    ]
    if force:
        cmd.append("--force-qwen-web-enrichment")
    if refresh_all:
        cmd.append("--refresh-all-qwen-web-enrichment")
    return _run(cmd)


def _github_then_qwen() -> int:
    gh_n = _ask_int("GitHub 补链最多处理多少个模型", 80, 1, 1000)
    qw_n = _ask_int("Qwen3.7-Max 联网补漏最多处理多少个模型", 20, 1, 200)
    cmd = _base_cmd() + [
        "--use-existing-meeting",
        "--force-github-enrichment",
        "--github-enrich-max-models", str(gh_n),
        "--qwen-web-enrichment",
        "--qwen-web-provider", "dashscope_qwen37max_search",
        "--qwen-web-model", "qwen3.7-max",
        "--qwen-web-max-models", str(qw_n),
    ]
    return _run(cmd)


def _comprehensive_architecture_search() -> int:
    year = _current_year()
    year_from = _ask_int("comprehensive year-from", year - 10, 1900, year + 1)
    year_to = _ask_int("comprehensive year-to", year, year_from, year + 1)
    use_qwen = _ask_yes_no("Use Qwen web enrichment for journal IF/citations/repos? This may consume quota", True)
    qwen_n = 80
    if use_qwen:
        qwen_n = _ask_int("Qwen web max models", 80, 1, 300)
    cmd = _base_cmd() + [
        "--comprehensive-architecture-search",
        "--year-from", str(year_from),
        "--year-to", str(year_to),
        "--max-results", "100",
        "--max-queries", "80",
        "--citation-seed-limit", "30",
        "--max-chunks", "240",
        "--force-github-enrichment",
        "--github-enrich-max-models", "200",
        "--github-enrich-repos-per-model", "5",
    ]
    if use_qwen:
        cmd += [
            "--qwen-web-enrichment",
            "--qwen-web-provider", "dashscope_qwen37max_search",
            "--qwen-web-model", "qwen3.7-max",
            "--qwen-web-max-models", str(qwen_n),
        ]
    return _run(cmd)


def _rebuild_memory() -> int:
    cmd = _base_cmd() + ["--refresh-memory-views-only"]
    return _run(cmd)


def _meeting_only_no_search() -> int:
    """Run a new global literature meeting from the existing compact evidence."""
    compact_pool = DATA_DIR / "compact_evidence_pool.json"
    if not compact_pool.is_file():
        print("找不到 data/compact_evidence_pool.json；请先至少完成一次文献检索。")
        return 2
    print("\n只重新运行文献会议：")
    print("  - 读取现有 compact evidence/chunk summaries")
    print("  - 不重新检索论文、不抓全文、不压缩 chunk")
    print("  - 不运行 GitHub/Qwen 联网补链")
    print("  - 会重新调用会议大模型并更新 literature memory")
    return _run(
        _base_cmd()
        + [
            "--resume-global-only",
            "--no-github-enrichment",
            "--resume-note",
            "menu_meeting_only_no_search",
        ]
    )


def _latest_model_incremental_search() -> int:
    year = _current_year()
    return _recent_search(
        year - 1,
        year,
        label=f"最新模型增量搜索（含 online-first，{year-1}-{year}）",
    )


def _full_reprocess() -> int:
    print("\n警告: 全量重跑会重新处理旧文章，耗时和调用量都更高。")
    if not _ask_yes_no("确认你真的要全量重跑", False):
        print("已取消全量重跑。")
        return 130
    year_from = _ask_int("year-from，建议只填近几年起始年，避免过大", _current_year() - 3, 1900, _current_year())
    year_to = _ask_int("year-to", _current_year(), year_from, _current_year() + 1)
    max_results = _ask_int("max-results", 80, 1, 500)
    batch_size = _ask_int("batch-size", 4, 1, 20)
    cmd = _base_cmd() + [
        "--reprocess",
        "--year-from", str(year_from),
        "--year-to", str(year_to),
        "--max-results", str(max_results),
        "--batch-size", str(batch_size),
        "--citation-seed-limit", "10",
        "--force-github-enrichment",
    ]
    if _ask_yes_no("是否同时启用 Qwen3.7-Max 联网补漏", False):
        cmd += ["--qwen-web-enrichment", "--qwen-web-provider", "dashscope_qwen37max_search", "--qwen-web-model", "qwen3.7-max", "--qwen-web-max-models", "20"]
    return _run(cmd)


def _llm_top_journal_pipeline() -> int:
    print("\nLLM 100 模型候选流程:")
    print("  1. 仅离线提名：5 批 × 20（不启用网页搜索，结果隔离）")
    print("  2. 仅联网核验：Crossref / OpenAlex / GitHub / 官方 JIF 清单")
    print("  3. 仅合并：只把已核验条目加入证据池并重排")
    print("  4. 完整流程：提名 -> 核验 -> 合并 -> 重排")
    stage_choice = _ask("输入阶段编号", "1")
    stage_map = {"1": "nominate", "2": "verify", "3": "integrate", "4": "all"}
    stage = stage_map.get(stage_choice)
    if not stage:
        print("无效编号。")
        return 2
    cmd = [sys.executable, str(LLM_TOP_JOURNAL_MAIN), "--stage", stage]
    if stage in {"nominate", "all"}:
        cmd += ["--target", "100", "--batch-size", "20"]
    if stage in {"verify", "all"}:
        limit = _ask_int("本次最多核验多少个；0 表示核验所有剩余候选", 20, 0, 100)
        cmd += ["--verify-limit", str(limit)]
    return _run(cmd)


def _required_scientific_models() -> int:
    print("\nRequired model scientific gate:")
    print("  seed claims -> Crossref/OpenAlex -> primary publisher page -> official GitHub/data -> evidence pool -> rerank")
    print("  Models: C_AMPs-predict, AMPSorter (ProteoGPT classifier), HMD-AMP")
    return _run([sys.executable, str(SCIENTIFIC_MODEL_EVIDENCE_MAIN), "--stage", "all"])


def _evaluate_literature_agent() -> int:
    print("\nEvaluate literature retrieval, meeting error filtering, metadata correction, and final-list contamination.")
    return _run([sys.executable, str(LITERATURE_AGENT_EVALUATION_MAIN)])


def _new_model_onboarding() -> int:
    if not ONBOARDING_MAIN.exists():
        print("找不到 new_model_onboarding.py，请确认已使用 v6.1 包。")
        return 2
    cmd = [sys.executable, str(ONBOARDING_MAIN)]
    return _run(cmd)


def _read_local_env() -> dict[str, str]:
    """Read local .env without importing project dependencies or printing secrets."""
    path = ROOT / ".env"
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in values:
                values[key] = value.strip()
    except OSError as exc:
        print(f"[WARN] 无法读取 .env 状态: {exc}")
    return values


def _load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        obj = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return [row for row in obj if isinstance(row, dict)] if isinstance(obj, list) else []
    except Exception as exc:
        print(f"[ERROR] 注册表读取失败: {exc}")
        return []


def _run_benchmark() -> int:
    if not BENCHMARK_MAIN.exists():
        print("找不到 main.py。")
        return 2
    dataset_issues = dataset_gate_issues(ROOT)
    if dataset_issues:
        print("数据集准备门禁未通过:")
        for issue in dataset_issues:
            print(" -", issue)
        print("请先选择菜单 8，完成数据集清单、下载、校验、标准化和泄漏检查。")
        return 2
    rows = _load_registry()
    if not rows:
        print("没有可用模型注册记录，请先选择 2 执行新模型入库。")
        return 2

    print("\n已注册模型:")
    portfolio_preview = build_benchmark_portfolio(
        rows,
        max_models=min(10, len(rows)),
    )
    portfolio_names = [
        str(row.get("model_name") or "")
        for row in portfolio_preview.get("selected_models", [])
    ]
    preview_by_name = {
        str(row.get("model_name") or "").casefold(): row
        for row in portfolio_preview.get("selected_models", [])
    }
    for i, row in enumerate(rows, 1):
        name = str(row.get("model_name") or f"unnamed_{i}")
        status = str(row.get("hpc_env_status") or "unknown")
        smoke = str(row.get("hpc_smoke_test") or "unknown")
        inferred = preview_by_name.get(name.casefold(), {})
        role = (
            row.get("benchmark_role_label")
            or inferred.get("benchmark_role_label")
            or "未分层"
        )
        print(f" {i:>2}. {name:<30} role={role:<14} HPC={status:<26} smoke={smoke}")

    counts = portfolio_preview.get("role_counts", {})
    print(
        "分层建议组合："
        f"经典基线={counts.get('classic_baseline', 0)}，"
        f"近期 SOTA 候选={counts.get('recent_sota_candidate', 0)}，"
        f"缺口={len(portfolio_preview.get('gaps', []))}"
    )
    for gap in portfolio_preview.get("gaps", []):
        names = ", ".join(gap.get("recommended_search_names", [])[:6])
        detail = names or ", ".join(gap.get("missing", [])[:6])
        print(f"  - {gap.get('type')}: {detail}")

    env_target = _read_local_env().get("TARGET_MODEL_NAMES", "").strip()
    default_target = env_target or str(rows[0].get("model_name") or "ALL")
    raw = _ask(
        "输入模型编号或名称，多个用逗号分隔；ALL=全部；PORTFOLIO=分层建议组合",
        default_target,
    )
    if raw.strip().upper() == "ALL":
        selected = ["ALL"]
    elif raw.strip().upper() == "PORTFOLIO":
        selected = portfolio_names
    else:
        by_name = {str(row.get("model_name") or "").casefold(): str(row.get("model_name") or "") for row in rows}
        selected: list[str] = []
        invalid: list[str] = []
        for token in [x.strip() for x in raw.split(",") if x.strip()]:
            if token.isdigit() and 1 <= int(token) <= len(rows):
                name = str(rows[int(token) - 1].get("model_name") or "")
            else:
                name = by_name.get(token.casefold(), "")
            if name and name not in selected:
                selected.append(name)
            elif not name:
                invalid.append(token)
        if invalid or not selected:
            print("无效模型选择:", ", ".join(invalid) or raw)
            return 2

    if not selected:
        print("分层建议组合为空，请先更新文献证据和候选模型。")
        return 2

    print("本轮目标模型:", ", ".join(selected))
    selected_rows = rows if selected == ["ALL"] else [
        row for row in rows if str(row.get("model_name") or "") in selected
    ]
    issues = model_readiness_issues(selected_rows)
    allow_unverified = False
    if issues:
        print("\n运行门禁未通过:")
        for issue in issues:
            print(
                f" - {issue['model_name']}: HPC={issue['hpc_env_status']}, "
                f"smoke={issue['hpc_smoke_test']}"
            )
        allow_unverified = _ask_yes_no(
            "仅用于诊断：是否明确绕过门禁并继续运行", default=False
        )
        if not allow_unverified:
            print("已取消。请先在新模型入库流程中完成 HPC 部署与 smoke test。")
            return 2
    return _run(
        [sys.executable, str(BENCHMARK_MAIN)],
        env_overrides={
            "TARGET_MODEL_NAMES": ",".join(selected),
            "ALLOW_UNVERIFIED_MODELS": "1" if allow_unverified else "0",
        },
    )


def _run_advisor() -> int:
    if not ADVISOR_MAIN.exists():
        print("找不到 amp_research_advisor.py。")
        return 2
    latest_path = DATA_DIR / "runs" / "latest.json"
    if not latest_path.exists():
        print("没有找到 data/runs/latest.json，请先运行一次 benchmark。")
        return 2
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        run_id = str(latest["run_id"])
    except Exception as exc:
        print(f"最新 run 指针无效: {exc}")
        return 2
    results_dir = DATA_DIR / "runs" / run_id / "results"
    return _run(
        [
            sys.executable,
            str(ADVISOR_MAIN),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(results_dir),
        ]
    )


def _status() -> int:
    print("\n项目状态:")
    env_values = _read_local_env()
    print(f"[{'OK' if (ROOT / '.env').exists() else '缺失'}] .env 本地配置")
    for key in ["OPENAI_API_KEY", "DASHSCOPE_API_KEY", "HPC_HOST", "HPC_USER", "HPC_PASS"]:
        print(f"[{'已设置' if env_values.get(key) else '未设置'}] {key}")

    rows = _load_registry()
    ready = sum(1 for row in rows if row.get("hpc_env_status") == "ready" and row.get("hpc_smoke_test") == "passed")
    unverified = sum(1 for row in rows if row.get("hpc_env_status") == "setup_complete_unverified")
    print(f"\n模型注册表: {len(rows)} 个模型；ready={ready}；未验证={unverified}")

    datasets_dir = DATA_DIR / "datasets"
    datasets = [p for p in datasets_dir.iterdir() if p.is_dir()] if datasets_dir.exists() else []
    print(f"数据集目录: {len(datasets)} 个")
    for path in datasets[:12]:
        print(f"  - {path.name}")

    recommendation_path = DATA_DIR / "dataset_agent_recommendation.json"
    if recommendation_path.is_file():
        try:
            recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
            print(
                "Dataset Agent:",
                recommendation.get("formal_selection_status", "unknown"),
                f"候选池={recommendation.get('candidate_pool_size', 0)}",
            )
        except Exception as exc:
            print(f"Dataset Agent 推荐清单读取失败: {exc}")
    else:
        print("Dataset Agent: 尚未运行")

    gate_issues = dataset_gate_issues(ROOT, datasets)
    if gate_issues:
        print("数据集门禁: 未通过")
        for issue in gate_issues[:8]:
            print(f"  - {issue}")
    else:
        print("数据集门禁: 已通过（标准化文件 SHA256 与 manifest 一致）")

    latest_path = DATA_DIR / "runs" / "latest.json"
    results_dir = DATA_DIR / "results"
    latest_report = None
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            run_id = str(latest["run_id"])
            results_dir = DATA_DIR / "runs" / run_id / "results"
            latest_report = results_dir / "amp_future_directions_report.md"
            print(f"最新 run: {run_id}；状态={latest.get('status', 'unknown')}")
        except Exception as exc:
            print(f"最新 run 指针读取失败: {exc}")
    eval_files = sorted(results_dir.glob("*/eval_result.json")) if results_dir.exists() else []
    print(f"最新运行评测结果: {len(eval_files)} 份 eval_result.json")

    print("\n常用输出文件:")
    files = [
        DATA_DIR / "literature_deep_research_memory.md",
        DATA_DIR / "literature_deep_research_memory.json",
        DATA_DIR / "compact_evidence_pool.json",
        DATA_DIR / "github_missing_model_enrichment.json",
        DATA_DIR / "github_enrichment_pending_models.txt",
        DATA_DIR / "github_enrichment_run_report.json",
        DATA_DIR / "qwen_web_enrichment.json",
        DATA_DIR / "qwen_web_enrichment_pending_models.txt",
        DATA_DIR / "qwen_web_enrichment_run_report.json",
        latest_report or DATA_DIR / "results" / "amp_future_directions_report.md",
    ]
    for f in files:
        if f.exists():
            size = f.stat().st_size
            mtime = _dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[存在] {f.relative_to(ROOT)}  {size:,} bytes  {mtime}")
        else:
            print(f"[缺失] {f.relative_to(ROOT)}")
    return 0


def _recommended_workflow() -> int:
    print("""
推荐运行顺序
--------------
首次部署：
  1) 选择 5 检查 .env、数据集和 registry
  2) 选择 3 更新文献证据、数据集清单和候选模型
  3) 选择 8 下载并准备数据集，通过 SHA256 与泄漏门禁
  4) 选择 2 下载新模型、部署 HPC 并完成 smoke test
  5) 选择 1 运行正式 benchmark
  6) 选择 4 汇总结果并生成研究发展建议

日常评测已有模型：
  1) 选择 5 确认数据集门禁已通过且目标模型为 ready/passed
  2) 选择 1，按编号选择一个或多个模型
  3) 运行结束后选择 4 更新研究报告

安全提示：
  - 不要提交 .env，也不要在日志中打印密钥。
  - 未通过 smoke test 的模型不要进入正式 benchmark。
  - 全量文献重跑和 ALL 模型评测都可能耗时较长。
""".strip())
    return 0


def _run_automated_tests() -> int:
    return _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])


def _run_dataset_gate() -> int:
    if not DATASET_GATE_MAIN.exists() or not DATASET_RECOMMENDER_MAIN.exists():
        print("找不到 Dataset Agent 或 dataset_gate.py。")
        return 2
    print("\n数据集 Agent 与准备门禁将依次执行:")
    print("  证据候选池 -> Agent 推荐 -> 真实序列审计 -> 自动选择 1 平衡 + 2 不平衡")
    print("  -> 生成清单 -> 下载/复用 -> SHA256 -> 安全解压 -> 标准化 -> 泄漏检查 -> manifest")
    rc = _run([sys.executable, str(DATASET_RECOMMENDER_MAIN), "recommend"])
    if rc != 0:
        return rc
    recommendation_path = DATA_DIR / "dataset_agent_recommendation.json"
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    if not recommendation.get("strategy_written"):
        print("\nDataset Agent 尚未找到 3 个同时通过全部科学规则的数据集。")
        print("已生成优先下载/审计候选与阻断原因，未使用人工清单继续正式门禁。")
        print("查看: data/dataset_agent_recommendation.md")
        return 2
    strategy_path = ROOT / str(recommendation["strategy_path"])
    print("没有官方 SHA256 的现有文件会在首轮建立 TOFU 基线，后续运行严格复核。")
    allow_overlap = _ask_yes_no(
        "是否允许不同独立测试集包含相同序列（仍会记录警告）",
        False,
    )
    require_expected = _ask_yes_no(
        "是否要求每个原始下载文件都必须预先提供官方 SHA256（严格模式）",
        False,
    )
    cmd = [sys.executable, str(DATASET_GATE_MAIN), "prepare", "--strategy", str(strategy_path)]
    if allow_overlap:
        cmd.append("--allow-cross-dataset-overlap")
    if require_expected:
        cmd.append("--require-expected-sha256")
    return _run(cmd)


def _refresh_dataset_recommendation() -> int:
    """Refresh candidate ranking only; never download or run the dataset gate."""
    if not DATASET_RECOMMENDER_MAIN.exists():
        print("找不到 dataset_recommendation_agent.py。")
        return 2
    print("\n只更新最合适的 3 个数据集推荐：")
    print("  - 读取当前 literature memory、证据池、核验种子和本地数据审计")
    print("  - 更新候选池、优先下载/审计 Top 3 和正式选择状态")
    print("  - 不下载、不解压，也不运行完整数据门禁")
    rc = _run([sys.executable, str(DATASET_RECOMMENDER_MAIN), "recommend"])
    if rc == 0:
        print("\n推荐结果：data/dataset_agent_recommendation.md")
        print("候选明细：data/dataset_candidate_pool.md")
        recommendation_path = DATA_DIR / "dataset_agent_recommendation.json"
        if recommendation_path.is_file():
            recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
            if recommendation.get("meeting_shortlist_status") != "ready_for_acquisition":
                print("文献会议尚未给出完整的 3 个数据集候选。请先进入文献菜单 16 重新开会，再运行主菜单 9。")
    return rc


def _print_literature_menu() -> None:
    year = _current_year()
    print("\n文献与候选模型更新:")
    print(f"  1. 搜索今年最新出现的模型 ({year})")
    print(f"  2. 搜索近 2 年新模型 ({year-1}-{year})")
    print(f"  3. 搜索近 3 年新模型 ({year-2}-{year})")
    print(f"  4. 搜索近 5 年新模型 ({year-4}-{year})")
    print("  5. 自定义年份范围搜索新模型")
    print("  6. 只做 GitHub 缺失链接补链")
    print("  7. 只做 Qwen3.7-Max 联网补漏")
    print("  8. GitHub 补链 + Qwen3.7-Max 联网补漏")
    print("  9. 架构全覆盖检索：按架构推荐 3-5 个模型，并按 IF/引用量排序")
    print(" 10. 不联网、不搜索，只用现有证据重新生成 memory")
    print(" 11. 全量重跑/重建证据池（谨慎）")
    print(" 12. 查看文献输出状态")
    print(" 13. LLM 分 5 批提名 100 个模型 -> 联网核验 -> 证据池 -> 重排")
    print("  0. 返回主菜单")
    print(" 14. Verify 3 required models from primary scientific sources -> evidence pool -> rerank")
    print(" 15. Evaluate literature meeting Agent (wrong-model detection / leakage / traceability)")
    print(" 16. 只重新运行文献会议（使用现有证据；不重新搜索、不联网补链）")
    print(f" 17. 最新模型增量搜索（推荐；{year-1}-{year}，含 online-first）")


def _literature_menu() -> int:
    while True:
        _print_literature_menu()
        choice = _ask("输入文献菜单编号", "3")
        year = _current_year()
        if choice == "0":
            return 0
        if choice == "1":
            _recent_search(year, year, label=f"只搜索今年新模型 {year}")
        elif choice == "2":
            _recent_search(year - 1, year, label=f"搜索近 2 年新模型 {year-1}-{year}")
        elif choice == "3":
            _recent_search(year - 2, year, label=f"搜索近 3 年新模型 {year-2}-{year}")
        elif choice == "4":
            _recent_search(year - 4, year, label=f"搜索近 5 年新模型 {year-4}-{year}")
        elif choice == "5":
            yf = _ask_int("year-from", year - 3, 1900, year + 1)
            yt = _ask_int("year-to", year, yf, year + 1)
            _recent_search(yf, yt, label=f"自定义年份范围 {yf}-{yt}")
        elif choice == "6":
            _github_only()
        elif choice == "7":
            _qwen_only()
        elif choice == "8":
            _github_then_qwen()
        elif choice == "9":
            _comprehensive_architecture_search()
        elif choice == "10":
            _rebuild_memory()
        elif choice == "11":
            _full_reprocess()
        elif choice == "12":
            _status()
        elif choice == "13":
            _llm_top_journal_pipeline()
        elif choice == "14":
            _required_scientific_models()
        elif choice == "15":
            _evaluate_literature_agent()
        elif choice == "16":
            _meeting_only_no_search()
        elif choice == "17":
            _latest_model_incremental_search()
        else:
            print("无效编号。")

        if not _ask_yes_no("是否留在文献菜单继续操作", True):
            return 0


def _print_menu() -> None:
    print("\n请选择功能:")
    print("  1. 运行正式 benchmark（可选择一个/多个/全部模型）")
    print("  2. 新模型自动入库 + 下载 + HPC 环境 + smoke test")
    print("  3. 文献检索 / GitHub 补链 / Qwen 补漏 / memory 更新")
    print("  4. 基于现有评测结果生成研究发展建议报告")
    print("  5. 查看项目配置、模型、数据集和输出状态")
    print("  6. 查看推荐运行顺序与安全提示")
    print("  7. 运行自动化测试（门禁 / SLURM / 协议 / manifest）")
    print("  8. Dataset Agent 自动推荐 3 个数据集并执行完整门禁")
    print("  9. 只更新最合适的 3 个数据集推荐（不下载、不运行门禁）")
    print("  0. 退出")


def main() -> int:
    required = [BENCHMARK_MAIN, LITERATURE_MAIN, LLM_TOP_JOURNAL_MAIN, SCIENTIFIC_MODEL_EVIDENCE_MAIN, ONBOARDING_MAIN, ADVISOR_MAIN, DATASET_GATE_MAIN, DATASET_RECOMMENDER_MAIN]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        print("缺少项目入口脚本:", ", ".join(missing))
        return 2
    while True:
        _print_header()
        _print_menu()
        choice = _ask("输入菜单编号", "5")
        if choice == "0":
            print("退出。")
            return 0
        if choice == "1":
            _run_benchmark()
        elif choice == "2":
            _new_model_onboarding()
        elif choice == "3":
            _literature_menu()
        elif choice == "4":
            _run_advisor()
        elif choice == "5":
            _status()
        elif choice == "6":
            _recommended_workflow()
        elif choice == "7":
            _run_automated_tests()
        elif choice == "8":
            _run_dataset_gate()
        elif choice == "9":
            _refresh_dataset_recommendation()
        else:
            print("无效编号。")

        if not _ask_yes_no("是否返回主菜单继续操作", True):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
