#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMP Benchmark 交互式运行菜单

用途：
- 不用反复记 deep_research_literature_agent.py 的长命令。
- 按菜单选择：搜索近几年新模型、GitHub 补链、Qwen3.7-Max 联网补漏、重建 memory 等。

运行：
    python amp_benchmark_menu.py
"""
from __future__ import annotations

import datetime as _dt
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "deep_research_literature_agent.py"
DATA_DIR = ROOT / "data"


def _current_year() -> int:
    # 按本地机器时间；如果机器时间不准，可用菜单里的自定义年份。
    return _dt.datetime.now().year


def _print_header() -> None:
    print("\n" + "=" * 78)
    print(" AMP Benchmark 文献/模型更新菜单")
    print("=" * 78)
    print("当前目录:", ROOT)
    print("主脚本:", MAIN.name)
    print("说明: 只想更新最新模型时，选择 1/2/3/4/5；不要使用全量重跑。")
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


def _run(cmd: List[str]) -> int:
    print("\n即将运行命令:")
    print(_quote_cmd(cmd))
    print("-" * 78)
    if not _ask_yes_no("确认运行", True):
        print("已取消。")
        return 130
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print("-" * 78)
    print("命令退出码:", proc.returncode)
    return proc.returncode


def _base_cmd() -> List[str]:
    return [sys.executable, str(MAIN)]


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


def _rebuild_memory() -> int:
    cmd = _base_cmd() + ["--use-existing-meeting"]
    return _run(cmd)


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


def _status() -> int:
    print("\n常用输出文件检查:")
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
    ]
    for f in files:
        if f.exists():
            size = f.stat().st_size
            mtime = _dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[存在] {f.relative_to(ROOT)}  {size:,} bytes  {mtime}")
        else:
            print(f"[缺失] {f.relative_to(ROOT)}")
    print("\n建议优先打开: data/literature_deep_research_memory.md")
    return 0


def _print_menu() -> None:
    year = _current_year()
    print("\n请选择功能:")
    print(f"  1. 只搜索今年新模型 ({year})")
    print(f"  2. 搜索近 2 年新模型 ({year-1}-{year})")
    print(f"  3. 搜索近 3 年新模型 ({year-2}-{year})")
    print(f"  4. 搜索近 5 年新模型 ({year-4}-{year})")
    print("  5. 自定义年份范围搜索新模型")
    print("  6. 只做 GitHub 缺失链接补链")
    print("  7. 只做 Qwen3.7-Max 联网补漏")
    print("  8. GitHub 补链 + Qwen3.7-Max 联网补漏")
    print("  9. 不联网、不搜索，只用现有证据重新生成 memory")
    print(" 10. 全量重跑/重建证据池（谨慎）")
    print(" 11. 查看输出文件状态")
    print("  0. 退出")


def main() -> int:
    if not MAIN.exists():
        print("找不到 deep_research_literature_agent.py，请把本脚本放在项目根目录。")
        return 2
    while True:
        _print_header()
        _print_menu()
        choice = _ask("输入菜单编号", "1")
        year = _current_year()
        if choice == "0":
            print("退出。")
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
            _rebuild_memory()
        elif choice == "10":
            _full_reprocess()
        elif choice == "11":
            _status()
        else:
            print("无效编号。")

        if not _ask_yes_no("是否返回菜单继续操作", False):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
