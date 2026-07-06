# resume_global_meeting_from_chunks.py
# -*- coding: utf-8 -*-
"""
只续跑最后的全局会议 / 或只把已完成的全局会议结果写入 memory。

适用场景：
1. 你已经完成了搜索、全文获取、evidence 提取、chunk 压缩。
2. data/compact_evidence_pool.json 或 data/chunk_summaries/ 已经存在。
3. 程序在最后 Global Meeting 或 Write Memory 阶段中断。

常用命令：
    # 重新跑最后一次全局会议，然后安全写入 memory
    python resume_global_meeting_from_chunks.py

    # 如果上次全局会议已经跑完，只是在 Write Memory 阶段报错，直接复用最近一次 raw meeting，不再调用 DeepSeek
    python resume_global_meeting_from_chunks.py --use-existing-meeting
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
DEFAULT_AGENT_FILE = ROOT / "deep_research_literature_agent.py"


def now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_dumps(obj: Any, indent: int = 2, **kwargs: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, **kwargs)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(obj, 2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Any]:
    out: List[Any] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps(obj, None) + "\n")


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def normalize_key(x: Any) -> str:
    s = str(x or "").strip().lower()
    return " ".join(s.replace("_", "-").split())


def stable_hash(obj: Any) -> str:
    try:
        txt = json_dumps(obj, 2, sort_keys=True)
    except Exception:
        txt = str(obj)
    return hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()[:16]


def import_agent_module(agent_file: Path):
    if not agent_file.exists():
        raise FileNotFoundError(f"没有找到 {agent_file}。请在项目根目录运行本脚本。")
    spec = importlib.util.spec_from_file_location("amp_agent_runtime", str(agent_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 {agent_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["amp_agent_runtime"] = module
    spec.loader.exec_module(module)
    return module


def load_compact_evidence_pool(data_dir: Path) -> Dict[str, Any]:
    compact_fp = data_dir / "compact_evidence_pool.json"
    pool = read_json(compact_fp, None)
    if isinstance(pool, dict) and ensure_list(pool.get("chunk_summaries")):
        return pool

    # 如果 compact_evidence_pool.json 不存在或为空，就从 data/chunk_summaries 重建一个轻量 pool。
    chunk_dir = data_dir / "chunk_summaries"
    summaries: List[Dict[str, Any]] = []

    jsonl = chunk_dir / "chunk_summaries.jsonl"
    if jsonl.exists():
        for item in read_jsonl(jsonl):
            if isinstance(item, dict):
                summaries.append(item)

    if not summaries and chunk_dir.exists():
        for fp in sorted(chunk_dir.glob("*.json")):
            if fp.name.startswith("_chunk_index"):
                continue
            item = read_json(fp, None)
            if isinstance(item, dict):
                summaries.append(item)

    if not summaries:
        raise FileNotFoundError(
            "没有找到可用 chunk summaries。需要存在 data/compact_evidence_pool.json "
            "或 data/chunk_summaries/chunk_summaries.jsonl。"
        )

    evidence_pool = read_json(data_dir / "evidence_pool.json", {})
    pool = {
        "created_at": now_str(),
        "compression_mode": "resume_from_chunk_summaries",
        "paper_count": evidence_pool.get("paper_count"),
        "source_counts": evidence_pool.get("source_counts"),
        "evidence_batch_count": evidence_pool.get("evidence_batch_count"),
        "chunk_count": len(summaries),
        "chunk_summary_count": len(summaries),
        "chunk_summaries": summaries,
        "paper_overview": evidence_pool.get("paper_overview") or evidence_pool.get("papers", [])[:300],
    }
    write_json(compact_fp, pool)
    return pool


def latest_global_meeting_raw(data_dir: Path, module: Any = None) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    if module is not None and hasattr(module, "GLOBAL_MEETING_RAW_JSONL"):
        candidates.append(Path(getattr(module, "GLOBAL_MEETING_RAW_JSONL")))
    candidates.extend([
        data_dir / "deepseek_meeting_raw.jsonl",
        data_dir / "global_meeting_raw.jsonl",
    ])
    seen = set()
    for fp in candidates:
        fp = fp if fp.is_absolute() else ROOT / fp
        if str(fp) in seen:
            continue
        seen.add(str(fp))
        rows = read_jsonl(fp)
        for row in reversed(rows):
            if isinstance(row, dict) and ("chief_agent" in row or "final_data" in row):
                return row
    return None


def load_records_for_index(data_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for fp in [data_dir / "normalized_papers.jsonl", data_dir / "raw_candidates.jsonl", data_dir / "pubmed_records.jsonl"]:
        for item in read_jsonl(fp):
            if isinstance(item, dict):
                records.append(item)
    # 去重
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in records:
        key = r.get("candidate_key") or r.get("doi") or r.get("pmid") or r.get("title") or stable_hash(r)
        key = normalize_key(key)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def clean_index_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        s = str(v).strip()
        return s or None
    if isinstance(v, dict):
        for k in ["candidate_key", "pmid", "PMID", "doi", "DOI", "title", "url", "id", "paper_id", "openalex_id", "semantic_scholar_id"]:
            if v.get(k):
                return str(v.get(k)).strip()
        return "dict:" + stable_hash(v)
    if isinstance(v, (list, tuple, set)):
        return "list:" + stable_hash(list(v))
    return str(v).strip() or None


def fallback_merge_items(existing: List[Any], incoming: List[Any], section: str) -> List[Any]:
    result = list(existing or [])
    seen: Dict[str, int] = {}

    def key_for(item: Any) -> str:
        if isinstance(item, dict):
            for k in ["canonical_name", "model_name", "url", "dataset_url", "dataset_name", "metric_name", "doi", "pmid", "title", "topic", "question"]:
                if item.get(k):
                    return section + ":" + normalize_key(item.get(k))
        return section + ":" + stable_hash(item)

    for i, item in enumerate(result):
        seen[key_for(item)] = i
    for item in incoming:
        key = key_for(item)
        if key in seen and isinstance(result[seen[key]], dict) and isinstance(item, dict):
            merged = dict(result[seen[key]])
            for k, v in item.items():
                if v not in (None, "", [], {}):
                    if k in merged and merged[k] not in (None, "", [], {}) and merged[k] != v:
                        if isinstance(merged[k], list):
                            merged[k].extend(ensure_list(v))
                        else:
                            merged[k] = [merged[k]] + ensure_list(v)
                    else:
                        merged[k] = v
            result[seen[key]] = merged
        elif key not in seen:
            seen[key] = len(result)
            result.append(item)
    return result


def render_simple_memory_md(memory: Dict[str, Any], run_info: Dict[str, Any]) -> str:
    lines = [
        "# AMP Literature Deep Research Memory",
        "",
        f"- Updated: {now_str()}",
        f"- Resume mode: {run_info.get('mode')}",
        "",
    ]
    for section in ["models", "repositories", "datasets", "dataset_links", "metrics", "papers", "benchmark_implications", "open_questions"]:
        items = ensure_list(memory.get(section))
        lines.append(f"## {section}")
        lines.append("")
        if not items:
            lines.append("- 暂无")
        else:
            for item in items[:300]:
                if isinstance(item, dict):
                    name = item.get("model_name") or item.get("canonical_name") or item.get("dataset_name") or item.get("metric_name") or item.get("title") or item.get("topic") or item.get("question") or stable_hash(item)
                    lines.append(f"- **{name}**: `{json_dumps(item, None)[:1200]}`")
                else:
                    lines.append(f"- {str(item)[:1200]}")
        lines.append("")
    return "\n".join(lines)


def safe_merge_final(module: Any, final_data: Dict[str, Any], records: List[Dict[str, Any]], run_info: Dict[str, Any]) -> None:
    memory_json = Path(getattr(module, "MEMORY_JSON", DATA_DIR / "literature_deep_research_memory.json"))
    memory_md = Path(getattr(module, "MEMORY_MD", DATA_DIR / "literature_deep_research_memory.md"))
    index_json = Path(getattr(module, "INDEX_JSON", DATA_DIR / "literature_deep_research_index.json"))

    default_memory = {"models": [], "repositories": [], "datasets": [], "dataset_links": [], "metrics": [], "papers": [], "benchmark_implications": [], "open_questions": [], "runs": []}
    default_index = {"processed_keys": [], "processed_pmids": [], "processed_dois": [], "processed_titles": []}
    memory = read_json(memory_json, default_memory)
    if not isinstance(memory, dict):
        memory = default_memory
    index = read_json(index_json, default_index)
    if not isinstance(index, dict):
        index = default_index

    merge_func = getattr(module, "merge_items", None)
    for section in ["models", "repositories", "datasets", "dataset_links", "metrics", "papers", "benchmark_implications", "open_questions"]:
        existing = ensure_list(memory.get(section))
        incoming = ensure_list(final_data.get(section)) if isinstance(final_data, dict) else []
        try:
            if callable(merge_func):
                memory[section] = merge_func(existing, incoming, section)
            else:
                memory[section] = fallback_merge_items(existing, incoming, section)
        except Exception:
            memory[section] = fallback_merge_items(existing, incoming, section)

    memory.setdefault("runs", []).append(run_info)

    for r in records:
        if not isinstance(r, dict):
            continue
        if r.get("candidate_key"):
            index.setdefault("processed_keys", []).append(r.get("candidate_key"))
        if r.get("pmid"):
            index.setdefault("processed_pmids", []).append(str(r.get("pmid")))
        if r.get("doi"):
            index.setdefault("processed_dois", []).append(normalize_key(r.get("doi")))
        if r.get("title"):
            index.setdefault("processed_titles", []).append(normalize_key(r.get("title")))

    # 关键修复：index 里如果混入 dict/list，先安全转换成字符串再去重。
    for k in list(index.keys()):
        cleaned = []
        for v in ensure_list(index.get(k)):
            s = clean_index_value(v)
            if s:
                cleaned.append(s)
        index[k] = sorted(set(cleaned))

    write_json(memory_json, memory)
    write_json(index_json, index)

    render_func = getattr(module, "render_memory_md", None)
    try:
        if callable(render_func):
            md = render_func(memory, run_info)
        else:
            md = render_simple_memory_md(memory, run_info)
    except Exception:
        md = render_simple_memory_md(memory, run_info)
    memory_md.parent.mkdir(parents=True, exist_ok=True)
    memory_md.write_text(md, encoding="utf-8")

    print(f"✅ Memory JSON written: {memory_json}")
    print(f"✅ Memory MD written: {memory_md}")
    print(f"✅ Index JSON written: {index_json}")


def build_memory_context(module: Any) -> Dict[str, Any]:
    try:
        mm = module.MemoryManager()
        return mm.context()
    except Exception:
        memory = read_json(DATA_DIR / "literature_deep_research_memory.json", {})
        if not isinstance(memory, dict):
            return {}
        return {k: ensure_list(memory.get(k))[:80] for k in ["models", "repositories", "datasets", "metrics", "papers", "benchmark_implications", "open_questions"]}


def main() -> None:
    p = argparse.ArgumentParser(description="Resume only global meeting from existing chunk summaries")
    p.add_argument("--agent-file", default="deep_research_literature_agent.py")
    p.add_argument("--provider", default="dashscope")
    p.add_argument("--provider-config", default="llm_providers.json")
    p.add_argument("--meeting-agent-dir", default="agents/deepseek_meeting")
    p.add_argument("--use-existing-meeting", action="store_true", help="不重新调用 DeepSeek，直接复用 data/deepseek_meeting_raw.jsonl 里最近一次 chief_agent 结果写入 memory。")
    p.add_argument("--run-note", default="resume_global_meeting_from_chunks")
    args = p.parse_args()

    module = import_agent_module(Path(args.agent_file))
    compact_pool = load_compact_evidence_pool(DATA_DIR)
    records = load_records_for_index(DATA_DIR)

    raw_meeting: Dict[str, Any]
    final_data: Dict[str, Any]

    if args.use_existing_meeting:
        raw = latest_global_meeting_raw(DATA_DIR, module)
        if not raw:
            raise FileNotFoundError("没有找到已完成的 global meeting raw。请去掉 --use-existing-meeting 重新跑全局会议。")
        raw_meeting = raw
        final = raw.get("chief_agent") or raw.get("final_data")
        if not isinstance(final, dict):
            raise RuntimeError("最近一次 global meeting raw 里没有可用的 chief_agent/final_data JSON。请去掉 --use-existing-meeting 重新跑全局会议。")
        final_data = final
        print("✅ 已复用最近一次 global meeting raw，不重新调用 DeepSeek。")
    else:
        print("========== [Resume Global Meeting Only] ==========")
        print(f">>> Chunk summaries: {len(ensure_list(compact_pool.get('chunk_summaries')))}")
        print(">>> 不重新搜索、不重新抓全文、不重新压缩 chunk。")
        llm = module.DeepSeekChatLLM(provider=args.provider, config_path=Path(args.provider_config))
        loader = module.AgentMDLoader(Path(args.meeting_agent_dir))
        final_data, raw_meeting = module.global_meeting(llm, loader, compact_pool, build_memory_context(module))

    run_info = {
        "time": now_str(),
        "mode": "resume_global_meeting_only_from_chunk_summaries",
        "note": args.run_note,
        "use_existing_meeting": bool(args.use_existing_meeting),
        "chunk_summary_count": len(ensure_list(compact_pool.get("chunk_summaries"))),
        "compact_evidence_pool": "data/compact_evidence_pool.json",
        "record_count_for_index": len(records),
    }
    print("    -> [Safe Write Memory] 写入 MD + JSON 长期记忆，并修复 index 中的 dict/list 去重问题...")
    safe_merge_final(module, final_data, records, run_info)
    print("✅ 续跑完成。")


if __name__ == "__main__":
    main()
