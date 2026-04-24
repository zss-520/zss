import json
import os
import re
from typing import Dict, List, Tuple
import shutil
import PyPDF2
import datetime
import time
from literature_filter.paper_type_classifier import classify_paper_from_title_abstract
from literature_filter.search_query_templates import QueryTemplates
from pathlib import Path
from openai import OpenAI
import subprocess
from config import HPC_HOST, HPC_USER, HPC_PASS, HPC_TARGET_DIR
from config import MODEL_NAME
from benchmark_designer import (
    fetch_from_semantic_scholar,
    fetch_and_save_pmc_papers,
    fetch_from_arxiv,
    fetch_from_openalex,
    fetch_from_serpapi_scholar,
    validate_strategy,
    snowball_from_seed_paper
)
# 确保文件顶部有这些导入：
from database_manager import ingest_new_paper, add_models_to_knowledge_db
from dataset_fetcher import fetch_datasets, fetch_models
import requests
from bs4 import BeautifulSoup
# 引入所有会议 Prompt
from prompts import (
    PAPER_PREPROCESSOR_PROMPT,
    MULTI_AGENT_SCOUT_PROMPT,
    MULTI_AGENT_METRICS_PROMPT,
    MULTI_AGENT_CRITIC_PROMPT,
    MULTI_AGENT_SCOUT_REBUTTAL_PROMPT,
    MULTI_AGENT_METRICS_REBUTTAL_PROMPT,
    MULTI_AGENT_CRITIC_ROUND2_PROMPT,
    MULTI_AGENT_CHIEF_PROMPT
)
def sync_models_to_hpc():
    """仅同步本地下载的模型源码至超算节点，跳过数据集"""
    print(f"\n>>> 📡 [HPC Sync] 正在启动模型代码同步程序，目标节点: {HPC_HOST}...")
    
    # 确保远程的 data/models 目录存在
    mkdir_cmd = f"sshpass -p '{HPC_PASS}' ssh {HPC_USER}@{HPC_HOST} 'mkdir -p {HPC_TARGET_DIR}/data/models'"
    subprocess.run(mkdir_cmd, shell=True)

    # 🚨 核心修改：源路径和目标路径精确锁定为 data/models/
    sync_cmd = f"sshpass -p '{HPC_PASS}' rsync -avz ./data/models/ {HPC_USER}@{HPC_HOST}:{HPC_TARGET_DIR}/data/models/"
    
    try:
        result = subprocess.run(sync_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ [HPC Sync] 代码同步成功！模型已存放至: {HPC_TARGET_DIR}/data/models")
        else:
            print(f"⚠️ [HPC Sync] 同步警告: {result.stderr}")
    except Exception as e:
        print(f"❌ [HPC Sync] 关键失败: {e}")

def run_literature_search(past_year, current_year, dir_model_papers, dir_benchmark_papers):
    """阶段一：触发文献检索引擎（新版：更严格的模型检索词）"""
    print("\n========== [Phase 1] 情报收集：多引擎联合检索与滚雪球 ==========")

    queries = QueryTemplates().build()

    pmc_model_query = f"{queries['pmc_model_query']} AND PUB_YEAR:[{past_year} TO {current_year}]"
    pmc_benchmark_query = f"({queries['benchmark_query']}) AND OPEN_ACCESS:y AND PUB_YEAR:[{past_year} TO {current_year}]"
    s2_model_query = queries["s2_model_query"]
    arxiv_expanded_query = "%22antimicrobial+peptide%22+AND+(classification+OR+identification+OR+prediction)+AND+(deep+learning+OR+transformer+OR+bert+OR+cnn)"
    openalex_expanded_query = queries["openalex_model_query"]
    serpapi_model_query = queries["serpapi_model_query"]

    dir_model_papers.mkdir(parents=True, exist_ok=True)
    dir_benchmark_papers.mkdir(parents=True, exist_ok=True)

    fetch_and_save_pmc_papers("AMP 预测模型(PMC)", pmc_model_query, dir_model_papers, max_results=20)
    fetch_from_semantic_scholar("AMP 预测模型(S2)", s2_model_query, dir_model_papers, max_results=8)
    fetch_and_save_pmc_papers("AMP 金标准数据集(PMC)", pmc_benchmark_query, dir_benchmark_papers, max_results=6)
    fetch_from_arxiv("AMP 最新预印本模型", arxiv_expanded_query, dir_model_papers, max_results=6)
    fetch_from_openalex("AMP 预测模型(兜底)", openalex_expanded_query, dir_model_papers, max_results=12)
    fetch_from_serpapi_scholar("AMP 预测模型(SerpApi)", serpapi_model_query, dir_model_papers, max_results=8)
    snowball_from_seed_paper(
        "Comprehensive assessment of machine learning-based methods for predicting antimicrobial peptides",
        dir_model_papers,
        max_results=8,
    )
# ========================================================
# 🚨 记忆缓存辅助函数
# ========================================================
CACHE_FILE = Path("data/extracted_reports_cache.json")

def load_extraction_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_extraction_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

# ========================================================
# 🚨 一级筛选：本地文献标题/摘要分类器
# ========================================================
STAGE1_FILTER_LOG = Path("data/stage1_filter_results.json")
REJECTED_MODEL_PAPERS_DIR = Path("data/rejected_model_papers")


def _safe_read_text(file_path: Path) -> str:
    """尽可能鲁棒地读取 txt/pdf 文本。"""
    try:
        if file_path.suffix.lower() == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if file_path.suffix.lower() == ".pdf":
            text = ""
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
    except Exception as e:
        print(f"       ⚠️ [读取失败] {file_path.name}: {e}")
    return ""


def _group_paper_files(directory: Path) -> Dict[str, List[Path]]:
    """按 stem 分组；同名 pdf/txt 视作同一篇文献。"""
    groups: Dict[str, List[Path]] = {}
    if not directory.exists():
        return groups

    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in {".txt", ".pdf"}:
            groups.setdefault(f.stem, []).append(f)
    return groups


def _pick_preferred_file(group: List[Path]) -> Path:
    """优先使用 PDF；没有 PDF 再使用 TXT。"""
    pdfs = [f for f in group if f.suffix.lower() == ".pdf"]
    txts = [f for f in group if f.suffix.lower() == ".txt"]
    if pdfs:
        return pdfs[0]
    return txts[0]


def _extract_title_and_abstract(raw_text: str, fallback_title: str) -> Tuple[str, str]:
    """
    从本地文献文本里粗提 title 和 abstract。
    这是一级筛选用，不追求完美，只追求稳定。
    """
    if not raw_text.strip():
        return fallback_title, ""

    text = raw_text.replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 1) 标题：优先取前几行中最像标题的一行
    title = fallback_title
    for ln in lines[:12]:
        low = ln.lower()
        if len(ln) < 15:
            continue
        if low in {"abstract", "introduction", "keywords", "background"}:
            continue
        if len(ln) > 300:
            continue
        title = ln
        break

    lowered = text.lower()

    # 2) 摘要：优先找 abstract 段
    abstract = ""
    match = re.search(
        r"\babstract\b[:\s]*([\s\S]{0,5000}?)(?:\n\s*(?:introduction|background|materials and methods|methods|results|keywords|1\.|i\.)\b)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        abstract = match.group(1).strip()

    # 3) 如果没找到 abstract，退化为前 2500 字符
    if not abstract:
        compact = " ".join(text.split())
        abstract = compact[:2500]

    # 清洗一下长度
    abstract = " ".join(abstract.split())[:3000]
    return title[:300], abstract


def _safe_move_to_rejected(file_path: Path, rejected_dir: Path) -> Path:
    """
    把文件移动到 rejected 目录，若重名则自动追加编号。
    """
    rejected_dir.mkdir(parents=True, exist_ok=True)
    target = rejected_dir / file_path.name

    if not target.exists():
        shutil.move(str(file_path), str(target))
        return target

    stem = file_path.stem
    suffix = file_path.suffix
    i = 1
    while True:
        candidate = rejected_dir / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            shutil.move(str(file_path), str(candidate))
            return candidate
        i += 1


def stage1_filter_model_papers(
    model_papers_dir: Path,
    rejected_dir: Path = REJECTED_MODEL_PAPERS_DIR,
    log_path: Path = STAGE1_FILTER_LOG,
) -> dict:
    """
    一级筛选：
    - 输入：data/papers 中已经落盘的 txt/pdf 文献
    - 行为：按 stem 去重 -> 抽 title/abstract -> 分类器判断
    - 输出：
        1) 不通过的论文移动到 data/rejected_model_papers/
        2) 写出 data/stage1_filter_results.json
        3) 返回 summary
    """
    model_papers_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_paper_files(model_papers_dir)
    if not groups:
        summary = {
            "total_groups": 0,
            "approved_groups": 0,
            "rejected_groups": 0,
            "kept_on_error_groups": 0,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "approved": [], "rejected": [], "kept_on_error": []}, f, ensure_ascii=False, indent=2)
        return summary

    approved_records = []
    rejected_records = []
    kept_on_error_records = []

    print("\n========== [Phase 1.5] 一级筛选：标题/摘要模型文献过滤 ==========")
    print(f">>> 共发现 {len(groups)} 组候选模型文献，开始本地一级筛选...")

    for idx, (stem, group) in enumerate(sorted(groups.items()), 1):
        preferred = _pick_preferred_file(group)
        print(f"    -> [{idx}/{len(groups)}] 正在筛选: {preferred.name}")

        try:
            raw_text = _safe_read_text(preferred)
            if not raw_text.strip():
                print("       ⚠️ 文本为空，保守放行。")
                kept_on_error_records.append({
                    "group_stem": stem,
                    "files": [str(p) for p in group],
                    "reason": "empty_text_fail_open",
                })
                continue

            title, abstract = _extract_title_and_abstract(raw_text, fallback_title=preferred.stem)
            result = classify_paper_from_title_abstract(title=title, abstract=abstract)

            record = {
                "group_stem": stem,
                "title": title,
                "files": [str(p) for p in group],
                "preferred_file": str(preferred),
                "abstract_preview": abstract[:500],
                "filter": result.to_dict(),
            }

            if result.should_download_full_text:
                print(f"       ✅ 保留：{result.confidence} | {result.paper_title[:80]}")
                approved_records.append(record)
            else:
                print(f"       ❌ 剔除：{result.reject_reason or 'classifier_rejected'} | {result.paper_title[:80]}")
                moved_files = []
                for fp in group:
                    if fp.exists():
                        new_path = _safe_move_to_rejected(fp, rejected_dir)
                        moved_files.append(str(new_path))
                record["moved_to"] = moved_files
                rejected_records.append(record)

        except Exception as e:
            print(f"       ⚠️ 一级筛选异常，保守放行: {e}")
            kept_on_error_records.append({
                "group_stem": stem,
                "files": [str(p) for p in group],
                "reason": f"classifier_error_fail_open: {e}",
            })

    summary = {
        "total_groups": len(groups),
        "approved_groups": len(approved_records),
        "rejected_groups": len(rejected_records),
        "kept_on_error_groups": len(kept_on_error_records),
    }

    log_payload = {
        "summary": summary,
        "approved": approved_records,
        "rejected": rejected_records,
        "kept_on_error": kept_on_error_records,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=2)

    print("\n>>> [一级筛选完成] 结果如下：")
    print(f"    - 原始候选组数: {summary['total_groups']}")
    print(f"    - 保留进入全文精读: {summary['approved_groups']}")
    print(f"    - 移入 rejected_model_papers: {summary['rejected_groups']}")
    print(f"    - 因异常保守放行: {summary['kept_on_error_groups']}")
    print(f"    - 详细日志已保存: {log_path}")

    return summary

# ========================================================
# 🚨 核心升级：前置 Agent 逐篇精读 (带秒速记忆缓存 + 断点续传)
# ========================================================
def agentic_extract_from_papers(directory: Path) -> str:
    """让 Agent 逐一读取全文，过滤水文，浓缩干货 (带记忆缓存)"""
    if not directory.exists(): return ""
    client = OpenAI()
    combined_reports = []
    
    # 1. 读取历史记忆
    cache = load_extraction_cache()
    
    # 2. 抓取所有合法文件
    all_files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in {".txt", ".pdf"}]
    if not all_files: return ""
    
    # 3. 按文件名分组去重，优先保留 PDF
    file_groups = {}
    for f in all_files:
        file_groups.setdefault(f.stem, []).append(f)
        
    files_to_process = []
    for stem, group in file_groups.items():
        pdfs = [f for f in group if f.suffix.lower() == ".pdf"]
        txts = [f for f in group if f.suffix.lower() == ".txt"]
        
        if pdfs:
            files_to_process.append(pdfs[0])
        elif txts:
            files_to_process.append(txts[0])
            
    print(f"\n>>> 🧠 [Pre-processing] 去重整理后锁定 {len(files_to_process)} 篇独立文献，启动前置 Agent (带记忆缓存)...")
    
    # 4. 开始逐篇精读
    for i, file_path in enumerate(files_to_process, 1):
        filename = file_path.name
        
        # 🚨 核心记忆机制：如果该文献之前读过，直接从缓存提取结果，不消耗 Token！
        if filename in cache:
            report = cache[filename]
            print(f"    -> [{i}/{len(files_to_process)}] ⚡ [命中缓存] 跳过精读: {filename[:30]}... ", end="")
            if "【无关键开源情报】" not in report:
                combined_reports.append(f"--- 📄 来源文献: {filename} ---\n{report}\n")
                print("✅ 提取到干货！")
            else:
                # 🚨 动态捕获具体死因 (缓存)
                match = re.search(r'【无关键开源情报：(.*?)】', report)
                if match:
                    reason = match.group(1).strip()
                else:
                    reason = report.replace('【无关键开源情报】', '').replace('：', '').strip()[:40]
                    if not reason:
                        reason = "未找到开源链接或不符合条件"
                print(f"⏭️ 过滤死因 (缓存): {reason}")
        # 如果缓存中没有，再动用大模型精读
        try:
            content = ""
            if file_path.suffix.lower() == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif file_path.suffix.lower() == ".pdf":
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text: content += page_text + "\n"
            
            # 如果文件为空或者损坏，标记为水文并存入缓存
            if not content.strip(): 
                cache[filename] = "【无关键开源情报】"
                save_extraction_cache(cache)
                continue
            
            # 追求最高准确率：不进行掐头去尾，直接硬塞（单篇最高放宽到 10 万字符）
            clean_content = " ".join(content.split())
            safe_content = clean_content[:100000]
            
            print(f"    -> [{i}/{len(files_to_process)}] 🤖 Agent 正在精读: {filename[:30]}... ", end="", flush=True)
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": PAPER_PREPROCESSOR_PROMPT},
                    {"role": "user", "content": f"文献文件名: {filename}\n文献内容:\n{safe_content}"}
                ],
                temperature=0.1
            )
            report = response.choices[0].message.content.strip()
            
            # 🚨 核心记忆机制：将结果(不管是干货还是水文)写回本地字典
            cache[filename] = report
            save_extraction_cache(cache)
            
            if "【无关键开源情报】" not in report:
                combined_reports.append(f"--- 📄 来源文献: {filename} ---\n{report}\n")
                print("✅ 提取到干货！")
            else:
                # 🚨 动态捕获具体死因 (新请求)
                match = re.search(r'【无关键开源情报：(.*?)】', report)
                if match:
                    reason = match.group(1).strip()
                else:
                    reason = report.replace('【无关键开源情报】', '').replace('：', '').strip()[:40]
                    if not reason:
                        reason = "未找到开源链接或不符合条件"
                print(f"⏭️ 过滤死因: {reason}")
                
        except Exception as e:
            error_str = str(e)
            print(f"❌ 读取或提取出错: {error_str}")
            # 熔断机制：即使中途断了，前面已经存进 cache 的文献下次依然秒读！
            if "insufficient_quota" in error_str or "FreeTierOnly" in error_str or "429" in error_str:
                print("\n!!! [系统熔断] API 额度已耗尽或触发严重限流！")
                print(f">>> [止损机制] 强制停止后续 {len(files_to_process) - i} 篇文章的解析，直接带着已经提取好的干货去开会！")
                break
            
    return "\n".join(combined_reports)
def load_historical_consensus() -> str:
    """加载历史会议记录与注册表，转化为 Agent 可读的上下文"""
    history_str = ""
    
    # 读取历史基准策略
    strategy_path = Path("data/benchmark_strategy.json")
    if strategy_path.exists():
        try:
            with open(strategy_path, "r", encoding="utf-8") as f:
                strat = json.load(f)
                history_str += "【现有基准数据集】:\n"
                for ds in strat.get("recommended_datasets", []):
                    history_str += f"- {ds.get('dataset_name')} (URL: {ds.get('download_url')})\n"
                history_str += "【现有评测指标】:\n"
                history_str += str(strat.get("metric_weights", {})) + "\n\n"
        except Exception: pass

    # 读取历史模型注册表
    registry_path = Path("data/local_registry.json")
    # 如果你主要依靠 database_manager 的话，也可以读 data/model_knowledge_db.json
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
                history_str += "【现有入库模型名单】:\n"
                for m in registry:
                    history_str += f"- {m.get('model_name')} (URL: {m.get('repo_url')})\n"
        except Exception: pass

    if not history_str.strip():
        return "当前团队暂无历史共识，本次是一次从0到1的奠基会议。"
    
    return history_str
def call_llm_with_retry(client, model, messages, temperature=0.2, response_format=None, max_retries=5):
    """带指数退避和自动重试的 LLM 调用保护罩"""
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model, 
                "messages": messages, 
                "temperature": temperature
            }
            if response_format:
                kwargs["response_format"] = response_format
                
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            # 捕获限流 (429) 或 服务端拥挤/内部错误 (500/503/Too many requests)
            if "Too many requests" in error_str or "503" in error_str or "500" in error_str or "429" in error_str:
                wait_time = (2 ** attempt) * 3  # 指数退避：3秒, 6秒, 12秒, 24秒...
                print(f"      ⚠️ [API限流保护] 服务端拥挤，正在静默等待 {wait_time} 秒后重试 ({attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                # 如果是其他致命错误（如 API Key 错误），直接抛出
                raise e
                
    raise Exception("❌ [API 彻底崩溃] 达到最大重试次数，请检查账户并发配额或稍后重试。")
def hold_multi_agent_meeting(full_context: str):
    """阶段二 & 三：带有 Rebuttal 对抗机制的圆桌会议 (带抗压重试机制)"""
    print("\n========== [Phase 2 & 3] 多智能体圆桌辩论会 (增量模式) ==========")
    client = OpenAI()
    
    # 1. 获取历史记忆
    history_context = load_historical_consensus()
    print("    -> 🧠 成功加载团队历史共识库。")
    
    trace_log = "# 🧠 AMP 预测基准测试规划会议记录\n\n"
    trace_log += f"## 📚 历史共识基线\n```\n{history_context}\n```\n\n"
    
    # [Round 1]
    print("    -> [Agent 1] 数据与模型侦察员正在对比历史并提取新实体...")
    scout_report = call_llm_with_retry(
        client, MODEL_NAME, 
        [{"role": "user", "content": MULTI_AGENT_SCOUT_PROMPT.format(history_context=history_context, full_context=full_context)}], 
        temperature=0.2
    )
    trace_log += f"## 🕵️ Agent 1 (Scout) 增量提案\n{scout_report}\n\n"

    print("    -> [Agent 2] 指标精算师正在分析评测体系...")
    metrics_report = call_llm_with_retry(
        client, MODEL_NAME, 
        [{"role": "user", "content": MULTI_AGENT_METRICS_PROMPT.format(full_context=full_context)}], 
        temperature=0.2
    )
    trace_log += f"## 📐 Agent 2 (Metrics) 初版提案\n{metrics_report}\n\n"

    # [Round 1 Critic]
    print("    -> [Agent 3] 评论家正在进行初轮火力输出 (Criticizing)...")
    critic_report = call_llm_with_retry(
        client, MODEL_NAME, 
        [{"role": "user", "content": MULTI_AGENT_CRITIC_PROMPT.format(scout_report=scout_report, metrics_report=metrics_report)}], 
        temperature=0.4
    )
    trace_log += f"## ⚖️ Agent 3 (Critic) 深度质疑\n{critic_report}\n\n"

    # [Round 2 Rebuttal]
    print("    -> [Agent 1] 数据侦察员正在进行辩护 (Rebuttal)...")
    scout_rebuttal = call_llm_with_retry(
        client, MODEL_NAME, 
        [{"role": "user", "content": MULTI_AGENT_SCOUT_REBUTTAL_PROMPT.format(scout_report=scout_report, critic_report=critic_report)}], 
        temperature=0.3
    )
    trace_log += f"## 🛡️ Agent 1 (Scout) 辩护与修正\n{scout_rebuttal}\n\n"

    print("    -> [Agent 2] 指标精算师正在进行辩护 (Rebuttal)...")
    metrics_rebuttal = call_llm_with_retry(
        client, MODEL_NAME, 
        [{"role": "user", "content": MULTI_AGENT_METRICS_REBUTTAL_PROMPT.format(metrics_report=metrics_report, critic_report=critic_report)}], 
        temperature=0.3
    )
    trace_log += f"## 🛡️ Agent 2 (Metrics) 辩护与修正\n{metrics_rebuttal}\n\n"

    # [Round 2 Critic Round 2]
    print("    -> [Agent 3] 评论家正在进行终审点评...")
    critic_round2 = call_llm_with_retry(
        client, MODEL_NAME, 
        [{"role": "user", "content": MULTI_AGENT_CRITIC_ROUND2_PROMPT.format(scout_rebuttal=scout_rebuttal, metrics_rebuttal=metrics_rebuttal)}], 
        temperature=0.2
    )
    trace_log += f"## ⚖️ Agent 3 (Critic) 终审点评\n{critic_round2}\n\n"

    # [Final Decision]
    print("    -> [Agent 4] 首席架构师正在综合全局，执行全量版本合并...")
    chief_prompt = MULTI_AGENT_CHIEF_PROMPT.format(
        history_context=history_context, 
        scout_report=scout_report, 
        metrics_report=metrics_report, 
        critic_report=critic_report,
        scout_rebuttal=scout_rebuttal, 
        metrics_rebuttal=metrics_rebuttal, 
        critic_round2=critic_round2
    )
    chief_response = call_llm_with_retry(
        client, MODEL_NAME, 
        [
            {"role": "system", "content": "你只输出严格合法的 JSON 格式。"},
            {"role": "user", "content": chief_prompt}
        ], 
        temperature=0.1, 
        response_format={"type": "json_object"}
    )
    
    final_data = json.loads(chief_response)
    return final_data, trace_log
def detect_url_migrations(strategy_data: dict) -> dict:
    """Phase 3.5: 仅嗅探数据集学术主页是否搬家。遇到网络错误绝对放行，不剔除任何数据！"""
    print("\n========== [Phase 3.5] 学术数据库搬家嗅探 (死链免死/GitHub免检) ==========")

    def get_migrated_url(url: str, item_name: str) -> str:
        # 空链接直接放行
        if not url or url.lower() in ["无", "none", "null", "无链接"]: 
            return url
        
        # 🚨 GitHub/Gitee 绝对免检，防误杀！
        if "github.com" in url.lower() or "gitee.com" in url.lower():
            return url

        print(f"    -> 🩺 嗅探搬家: [{item_name}] {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        try:
            res = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            
            # 1. 软重定向嗅探 (网页还能打开/甚至返回404页面，但里面写了已搬家)
            content_type = res.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type:
                html_lower = res.text.lower()
                # 寻找搬家关键词
                if len(html_lower) < 5000 and any(kw in html_lower for kw in ['moved to', 'new address', 'has been migrated', 'new site', 'redirecting']):
                    print("       💡 [软重定向嗅探] 发现原站已搬家提示，正在提取新链接...")
                    soup = BeautifulSoup(res.text, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if href.startswith('http') and href != url:
                            print(f"       ✅ [新地址替换成功] -> {href}")
                            return href # 返回新地址
            
            # 2. 硬重定向追踪 (URL 发生了自动跳转)
            if res.url != url and res.status_code < 400:
                print(f"       ✅ [硬重定向追踪] 已自动更新至 -> {res.url}")
                return res.url
                
            # 3. 其他情况（包括遇到 403, 404, 500 报错）
            # 🚨 绝对不踢出！原样返回老链接！
            return url

        except Exception as e:
            # 🚨 遇到网络超时或连接失败，也绝对不踢出！原样放行！
            print(f"       ⚠️ [网络异常] 无法嗅探，原链接免死放行 ({str(e)[:30]})")
            return url

    # 仅遍历数据集并更新 URL，不删除任何元素
    for ds in strategy_data.get("recommended_datasets", []):
        name = ds.get("dataset_name", "Unknown")
        old_url = ds.get("download_url", "")
        
        # 覆写为新地址（如果有搬家的话），否则保持原样
        ds["download_url"] = get_migrated_url(old_url, name)

    return strategy_data
def generate_markdown_table(strategy_data: dict, selected_models: list, selected_papers: list) -> str:
    """阶段四辅助：将 JSON 转化为易读的 Markdown 表格"""
    table = "\n## 📜 最终会议决议 (Final Consensus)\n"
    
    # 提取全局处理流水线步骤
    pipeline_steps = strategy_data.get("dataset_processing_steps", [])
    pipeline_str = " ➔ ".join(pipeline_steps) if pipeline_steps else "默认流水线"

    table += "\n### 📊 选定数据集清单\n"
    # 确保表头与内容对应
    table += "| 数据集名称 | 角色 | 处理流水线 (Pipeline) | 下载链接 | 详细处理规范 (Strategy) |\n"
    table += "|---|---|---|---|---|\n"
    
    for ds in strategy_data.get("recommended_datasets", []):
        if not isinstance(ds, dict): continue
        # 获取详细描述，并增加截断长度以显示更多细则
        description = str(ds.get('description', '无'))
        table += f"| {ds.get('dataset_name', '未知')} | {ds.get('role', '未知')} | {pipeline_str} | {ds.get('download_url', '无')} | {description[:200]}... |\n"
    
    # --- 指标、模型、文献部分保持原样 ---
    table += "\n### ⚖️ 评测指标选取\n"
    table += "| 指标 | 权重 | 选取理由 |\n"
    table += "|---|---|---|\n"
    metrics = strategy_data.get("metric_weights", {})
    refs = strategy_data.get("metrics_references", {})
    for m, w in metrics.items():
        table += f"| {m} | {w} | {str(refs.get(m, '无'))[:50]}... |\n"

    table += "\n### 🤖 拟提取复现模型清单\n"
    table += "| 模型名称 | GitHub/代码链接 | 来源文献 |\n"
    table += "|---|---|---|\n"
    for mod in selected_models:
        if not isinstance(mod, dict): continue
        table += f"| {mod.get('model_name', '未知')} | {mod.get('repo_url', '无链接')} | {str(mod.get('source_paper', '未知'))[:30]}... |\n"

    table += "\n### 📚 核心采纳文献清单\n"
    table += "| 文献标题 | 采纳理由 (核心贡献) |\n"
    table += "|---|---|\n"
    if selected_papers:
        for paper in selected_papers:
            if not isinstance(paper, dict): continue
            table += f"| {str(paper.get('paper_title', ''))[:50]}... | {str(paper.get('reason_for_selection', ''))} |\n"
    else:
        table += "| 暂无数据 | AI 未返回核心文献清单 |\n"

    return table

def main():
    current_year = datetime.datetime.now().year
    past_year = current_year - 5
    dir_model_papers = Path("data/papers")
    dir_benchmark_papers = Path("data/benchmark_papers")

    print("\n" + "=" * 60 + "\n========== [System Start] AMP 基准评估规划管线 ==========\n" + "=" * 60)
    search_input = input("🤔 是否需要联网执行最新的文献检索？(y/n) [默认: y]: ").strip().lower()

    if search_input not in ["n", "no"]:
        run_literature_search(past_year, current_year, dir_model_papers, dir_benchmark_papers)

    # ========================================================
    # 🚨 新增：对 data/papers 做一级筛选
    # ========================================================
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

    # 1. 召开圆桌会议并获取过程记录
    final_data, trace_log = hold_multi_agent_meeting(full_context)

    strategy_data = validate_strategy(final_data.get("benchmark_strategy", {}))
    selected_models = final_data.get("selected_models", [])
    selected_papers = final_data.get("selected_papers", [])
    strategy_data = detect_url_migrations(strategy_data)

    # 将清洗干净的数据覆盖回 final_data（保证后续存入的 JSON 是纯净的）
    final_data["benchmark_strategy"] = strategy_data

    # 2. 生成结果表格
    table_output = generate_markdown_table(strategy_data, selected_models, selected_papers)

    # 🚨 核心改进：把一级筛选摘要也附加到会议记录中
    filter_summary_md = (
        "\n## 🧹 一级筛选摘要 (Stage-1 Paper Filter)\n"
        f"- 原始候选组数: {stage1_summary.get('total_groups', 0)}\n"
        f"- 保留进入全文精读: {stage1_summary.get('approved_groups', 0)}\n"
        f"- 被移动到 rejected_model_papers: {stage1_summary.get('rejected_groups', 0)}\n"
        f"- 因异常保守放行: {stage1_summary.get('kept_on_error_groups', 0)}\n"
        f"- 详细日志: {STAGE1_FILTER_LOG}\n"
    )

    # 3. 保存会议记录
    full_meeting_record = trace_log + filter_summary_md + table_output
    trace_path = Path("data/meeting_trace.md")
    trace_path.parent.mkdir(exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(full_meeting_record)

    print(f"\n    -> [存档完成] 完整的会议辩论记录与最终决议表格已保存至: {trace_path}")

    # 4. 结果公示
    print("\n" + "=" * 60 + "\n========== [Phase 4] 首席科学家决策结果公示 ==========\n" + "=" * 60)
    print(filter_summary_md)
    print(table_output)

    # 5. 人机确认
    print("\n" + "=" * 60)
    user_input = input("🤔 专家团队已完成交叉验证。是否授权将新模型录入【情报知识库】并开始下载 GitHub 代码？(y/n) [默认: y]: ").strip().lower()
    if user_input in ["n", "no"]:
        print("    -> 🛑 已取消后续入库与下载流程。")
        return final_data

    print("\n========== [Phase 5] 情报入库与代码拉取 ==========")

    # 【核心 1】: 将新模型加入未复现的知识库 (model_knowledge_db.json)
    if selected_models:
        add_models_to_knowledge_db(selected_models)

    # 【核心 2】: 根据这个名单，自动去下载 GitHub 代码到 data/models 目录下
    fetch_models(selected_models)

    # 【核心 3】: 一并把会议决定的新 Benchmark 数据集也下载了
    fetch_datasets()
    sync_models_to_hpc()

    print("\n" + "🚀" * 15)
    print("✅ 第一阶段【文献情报获取与代码拉取】大功告成！")
    print("   -> 最新模型情报已存入: data/model_knowledge_db.json")
    print("   -> 被一级筛掉的候选文献已移入: data/rejected_model_papers/")
    print("   -> 一级筛选日志已保存: data/stage1_filter_results.json")
    print("   -> 请在准备好后，调用 Vanguard (探路者) 对下载的代码进行环境复现与扫描。")
    print("   -> 只有 Vanguard 验证通过的模型，才有资格进入 local_registry.json！")

    return final_data

if __name__ == "__main__":
    main()