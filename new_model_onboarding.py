#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新模型自动入库 / 下载 / HPC 部署 / 评测流水线。

典型运行：
    python new_model_onboarding.py

流程：
1. 从 data/literature_deep_research_memory.json/md、compact_evidence_pool、Qwen/GitHub 补链结果中汇总候选模型。
2. 展示前 20 个候选模型，标明代码/数据/权重证据是否完整、是否已在 registry 中。
3. 用户选择一个新模型；缺少证据时可选择 Qwen 联网补全或人工手动补全。
4. 下载模型到 data/models/{model_slug}。
5. Agent 读取 README / requirements / environment.yml / 代码文件，生成 registry 记录和环境安装方案。
6. 上传模型到超算，创建 conda 虚拟环境，安装依赖，执行 mini FASTA smoke test。
7. 写入 data/local_registry.json。
8. 可选择自动调用 main.py 对该新模型执行后续评测任务。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent_md_loader import AgentMDLoader
from benchmark_portfolio import build_benchmark_portfolio

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import requests
except Exception:
    requests = None  # type: ignore

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS_DIR = DATA / "models"
REGISTRY_PATH = DATA / "local_registry.json"
ONBOARDING_PROMPT_DIR = ROOT / "agents" / "model_onboarding"
_ONBOARDING_PROMPTS = AgentMDLoader(ONBOARDING_PROMPT_DIR)

NOT_REPORTED = {"", "not_reported_in_available_evidence", "not provided", "not_available", "none", "null", "unknown", "无"}
URL_RE = re.compile(r"https?://[^\s\]\)\}\>\"']+", re.I)
GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I)
DEPENDENCY_NAME_MAP = {"sklearn": "scikit-learn", "cv2": "opencv-python", "pil": "Pillow", "yaml": "PyYAML", "bio": "biopython"}
STDLIB_DEPENDENCIES = {"argparse", "collections", "csv", "datetime", "functools", "glob", "hashlib", "json", "logging", "math", "os", "pathlib", "pickle", "random", "re", "shutil", "subprocess", "sys", "tempfile", "time", "typing"}
SAFE_SETUP_PREFIXES = ("pip ", "python -m pip ", "conda ", "wget ", "curl ", "bash ", "sh ", "python ")
BLOCKED_COMMAND_FRAGMENTS = ("rm -rf", "rm -r ", "sudo ", "mkfs", "shutdown", "reboot", "dd if=", ":(){", "chmod -r 777")


def _safe_name(text: Any, fallback: str = "model") -> str:
    s = str(text or "").strip()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("._-")
    return (s[:80] or fallback)


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _clean_url(u: Any) -> str:
    if not u:
        return ""
    s = str(u).strip().strip(".,;，。；)）]")
    if s.lower() in NOT_REPORTED:
        return ""
    return s


def _first_url(*values: Any, github_only: bool = False) -> str:
    for v in values:
        if isinstance(v, dict):
            u = _first_url(*v.values(), github_only=github_only)
            if u:
                return u
        elif isinstance(v, list):
            u = _first_url(*v, github_only=github_only)
            if u:
                return u
        else:
            text = str(v or "")
            if not text or text.lower().strip() in NOT_REPORTED:
                continue
            rgx = GITHUB_RE if github_only else URL_RE
            m = rgx.search(text)
            if m:
                return _clean_url(m.group(0))
            if text.startswith("http") and not github_only:
                return _clean_url(text)
    return ""


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return default


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_registry() -> List[Dict[str, Any]]:
    data = _load_json(REGISTRY_PATH, [])
    return data if isinstance(data, list) else []


def upsert_registry(row: Dict[str, Any]) -> None:
    rows = load_registry()
    name = str(row.get("model_name") or "").strip()
    if not name:
        raise ValueError("registry row 缺少 model_name")
    done = False
    for i, old in enumerate(rows):
        if str(old.get("model_name") or "").strip() == name:
            merged = dict(old)
            merged.update({k: v for k, v in row.items() if v not in [None, "", [], {}]})
            rows[i] = merged
            done = True
            break
    if not done:
        rows.append(row)
    _save_json(REGISTRY_PATH, rows)


def _recursive_model_rows(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        keys = {k.lower() for k in obj.keys()}
        if any(k in keys for k in ["model_name", "canonical_name", "matched_model_name", "name"]):
            text = " ".join(str(obj.get(k, "")) for k in obj.keys()).lower()
            if "amp" in text or "peptide" in text or obj.get("model_name") or obj.get("matched_model_name"):
                yield obj
        for v in obj.values():
            yield from _recursive_model_rows(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from _recursive_model_rows(x)


def _row_model_name(row: Dict[str, Any]) -> str:
    for k in ["model_name", "canonical_name", "matched_model_name", "name", "linked_model"]:
        v = row.get(k)
        if v and str(v).strip().lower() not in NOT_REPORTED:
            return str(v).strip().strip(".")
    return ""


def _merge_candidate(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if v in [None, "", [], {}] or str(v).strip().lower() in NOT_REPORTED:
            continue
        if k not in out or out.get(k) in [None, "", [], {}] or str(out.get(k)).strip().lower() in NOT_REPORTED:
            out[k] = v
        elif k in {"sources", "risk_flags", "blocking_issues"}:
            vals = []
            for x in [out.get(k), v]:
                vals.extend(x if isinstance(x, list) else [x])
            out[k] = sorted({str(x) for x in vals if x})
    return out


def candidate_from_row(row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    name = _row_model_name(row)
    if not name or len(name) < 2:
        return None
    repo = _first_url(row.get("code_repository_url"), row.get("repository_url"), row.get("repo_url"), row.get("github_url"), row.get("url"), row.get("evidence"), github_only=True)
    if not repo:
        repo = _first_url(row.get("repository_candidates"), github_only=True)
    dataset = _first_url(row.get("dataset_source_or_link"), row.get("dataset_url"), row.get("dataset_candidates"))
    weights = _first_url(row.get("model_weights_url"), row.get("weights_url"), row.get("weight_candidates"))
    web = _first_url(row.get("web_server_url"), row.get("web_server_candidates"))
    doi = str(row.get("source_doi") or row.get("doi") or "").strip()
    pmid = str(row.get("source_pmid") or row.get("pmid") or "").strip()
    try:
        score = float(row.get("deployment_selection_score") or row.get("confidence") or row.get("score") or 0.0)
    except Exception:
        score = 0.0
    return {
        "model_name": name,
        "code_repository_url": repo,
        "dataset_source_or_link": dataset,
        "model_weights_url": weights,
        "web_server_url": web,
        "source_doi": doi,
        "source_pmid": pmid,
        "source_journal": row.get("source_journal") or row.get("journal") or "",
        "publication_year": row.get("publication_year") or row.get("source_year") or row.get("year"),
        "citation_count": row.get("citation_count") or row.get("cited_by_count"),
        "journal_impact_factor": row.get("journal_impact_factor") or row.get("impact_factor"),
        "method_family": row.get("method_family") or row.get("architecture") or row.get("model_family") or "",
        "architecture_category": row.get("architecture_category") or "",
        "representation_category": row.get("representation_category") or "",
        "benchmark_role": row.get("benchmark_role") or "",
        "benchmark_role_label": row.get("benchmark_role_label") or "",
        "benchmark_roles": row.get("benchmark_roles") or [],
        "benchmark_role_reason": row.get("benchmark_role_reason") or "",
        "deployment_rank": row.get("deployment_rank"),
        "task_type": row.get("task_type") or "",
        "evidence_level": row.get("evidence_level") or source,
        "blocking_issues": row.get("blocking_issues") or [],
        "needs_manual_verification": row.get("needs_manual_verification"),
        "candidate_reason": row.get("candidate_reason") or row.get("reason") or row.get("evidence") or "",
        "score": score,
        "sources": [source],
    }


def parse_markdown_candidates(md_path: Path) -> List[Dict[str, Any]]:
    if not md_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "|" not in line:
            continue
        if "---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        urls = [u.strip(".,;。") for u in URL_RE.findall(line)]
        if not urls and not any("not_reported" in c for c in cells):
            continue
        # 选择最像模型名的 cell：避开类别、URL、长说明。
        model = ""
        for c in cells:
            if c.startswith("http") or "not_reported" in c or len(c) > 80:
                continue
            if any(bad in c.lower() for bad in ["类别", "source", "dataset", "evidence", "reason", "pmid", "doi"]):
                continue
            if re.search(r"[A-Za-z]", c):
                model = c
                break
        if not model:
            continue
        repo = next((u for u in urls if "github.com" in u.lower()), "")
        dataset = next((u for u in urls if "github.com" not in u.lower()), "")
        rows.append({
            "model_name": model.strip(),
            "code_repository_url": repo,
            "dataset_source_or_link": dataset,
            "evidence_level": "memory_md_table",
            "sources": ["memory_md"],
            "score": 0.0,
        })
    return rows


def load_candidates(limit: int = 20, include_registered: bool = True) -> List[Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = {}
    json_files = [
        DATA / "literature_deep_research_memory.json",
        DATA / "compact_evidence_pool.json",
        DATA / "qwen_web_enrichment.json",
        DATA / "github_missing_model_enrichment.json",
        DATA / "model_knowledge_db.json",
    ]
    for fp in json_files:
        obj = _load_json(fp, None)
        if obj is None:
            continue
        for row in _recursive_model_rows(obj):
            cand = candidate_from_row(row, fp.name)
            if not cand:
                continue
            key = _norm(cand["model_name"])
            candidates[key] = _merge_candidate(candidates.get(key, {}), cand)
    for md in [DATA / "literature_deep_research_memory.md", ROOT / "literature_deep_research_memory.md"]:
        for cand in parse_markdown_candidates(md):
            key = _norm(cand["model_name"])
            candidates[key] = _merge_candidate(candidates.get(key, {}), cand)

    registry_names = {_norm(r.get("model_name")) for r in load_registry()}
    out = []
    for c in candidates.values():
        if not c.get("model_name"):
            continue
        c["already_registered"] = _norm(c.get("model_name")) in registry_names
        if not include_registered and c["already_registered"]:
            continue
        # 证据得分：优先有代码仓库，其次数据/权重/DOI。
        ev_score = 0
        ev_score += 100 if c.get("code_repository_url") else 0
        ev_score += 25 if c.get("dataset_source_or_link") else 0
        ev_score += 20 if c.get("model_weights_url") else 0
        ev_score += 10 if c.get("source_doi") or c.get("source_pmid") else 0
        raw_roles = c.get("benchmark_roles") or [c.get("benchmark_role")]
        if not isinstance(raw_roles, list):
            raw_roles = [raw_roles]
        roles = {str(role) for role in raw_roles if role}
        ev_score += 50 if "recent_sota_candidate" in roles else 0
        ev_score += 40 if "classic_baseline" in roles else 0
        ev_score += 15 if "architecture_representative" in roles else 0
        ev_score += float(c.get("score") or 0)
        ev_score -= 80 if c.get("already_registered") else 0
        c["evidence_score"] = ev_score
        out.append(c)

    portfolio = build_benchmark_portfolio(
        out,
        max_models=max(20, limit),
        score_fn=lambda row: float(row.get("evidence_score") or 0.0),
    )
    inferred_by_name = {
        _norm(row.get("model_name")): row
        for row in portfolio.get("selected_models", [])
    }
    for candidate in out:
        inferred = inferred_by_name.get(_norm(candidate.get("model_name")))
        if not inferred:
            continue
        existing_roles = candidate.get("benchmark_roles") or []
        if not isinstance(existing_roles, list):
            existing_roles = [existing_roles]
        inferred_roles = inferred.get("benchmark_roles") or []
        merged_roles = list(dict.fromkeys([*existing_roles, *inferred_roles]))
        newly_added = set(merged_roles) - set(existing_roles)
        candidate["benchmark_roles"] = merged_roles
        candidate["benchmark_role"] = candidate.get("benchmark_role") or inferred.get("benchmark_role")
        candidate["benchmark_role_label"] = candidate.get("benchmark_role_label") or inferred.get("benchmark_role_label")
        candidate["benchmark_role_reason"] = candidate.get("benchmark_role_reason") or inferred.get("benchmark_role_reason")
        candidate["publication_year"] = candidate.get("publication_year") or inferred.get("publication_year")
        candidate["evidence_score"] += 50 if "recent_sota_candidate" in newly_added else 0
        candidate["evidence_score"] += 40 if "classic_baseline" in newly_added else 0
        candidate["evidence_score"] += 15 if "architecture_representative" in newly_added else 0
    out.sort(key=lambda x: (x.get("evidence_score", 0), str(x.get("model_name"))), reverse=True)
    return out[:limit]


def print_candidates(cands: List[Dict[str, Any]]) -> None:
    print("\n候选新模型 Top 列表：")
    print("-" * 110)
    for i, c in enumerate(cands, 1):
        flags = []
        flags.append("代码✅" if c.get("code_repository_url") else "代码❌")
        flags.append("数据✅" if c.get("dataset_source_or_link") else "数据❌")
        flags.append("权重✅" if c.get("model_weights_url") else "权重❌")
        flags.append("已入库" if c.get("already_registered") else "未入库")
        repo = c.get("code_repository_url") or ""
        role = c.get("benchmark_role_label") or c.get("benchmark_role") or "未分层"
        year = c.get("publication_year") or "?"
        print(f"{i:>2}. {c.get('model_name'):<28} [{role} | {year} | {' | '.join(flags)}] score={c.get('evidence_score',0):.1f}")
        if repo:
            print(f"    repo: {repo}")
    print("-" * 110)


def _ask(prompt: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(f"{prompt}: ").strip()
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def _yes(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true", "是"}


def try_qwen_fill(cand: Dict[str, Any]) -> Dict[str, Any]:
    print("\n>>> 尝试调用 deep_research_literature_agent 的 Qwen3.7-Max 联网补漏...")
    try:
        import deep_research_literature_agent as agent
        rows = agent.search_qwen_web_for_model_evidence([cand], max_models=1, force=True, refresh_all=False)
        if rows:
            row = rows[-1]
            repo = _first_url(row.get("repository_candidates"), github_only=True)
            dataset = _first_url(row.get("dataset_candidates"))
            weights = _first_url(row.get("weight_candidates"))
            web = _first_url(row.get("web_server_candidates"))
            if repo:
                cand["code_repository_url"] = repo
            if dataset:
                cand["dataset_source_or_link"] = dataset
            if weights:
                cand["model_weights_url"] = weights
            if web:
                cand["web_server_url"] = web
            for k in ["source_doi", "source_pmid", "source_journal", "citation_count", "journal_impact_factor"]:
                if row.get(k) and not cand.get(k):
                    cand[k] = row.get(k)
            print(">>> Qwen 补漏完成。")
        else:
            print(">>> Qwen 没有返回可用补漏结果。")
    except Exception as e:
        print(f"⚠️ Qwen 补漏失败：{e}")
    return cand


def manual_fill(cand: Dict[str, Any]) -> Dict[str, Any]:
    print("\n>>> 手动补全证据。直接回车表示保持原值/跳过。")
    for key, label in [
        ("code_repository_url", "代码仓库 URL，GitHub/GitLab/Gitee/Zenodo"),
        ("dataset_source_or_link", "数据集 URL"),
        ("model_weights_url", "模型权重 URL"),
        ("web_server_url", "Web server URL"),
        ("source_doi", "论文 DOI"),
        ("source_pmid", "PMID"),
    ]:
        old = str(cand.get(key) or "")
        val = _ask(label, old if old else "")
        if val:
            cand[key] = val
    return cand


def download_model(cand: Dict[str, Any]) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = cand.get("model_name") or "model"
    target = MODELS_DIR / _safe_name(name)
    local = str(cand.get("local_model_dir") or "").strip()
    if local:
        local_path = Path(local)
        if not local_path.is_absolute():
            local_path = ROOT / local_path
        if local_path.exists() and local_path.is_dir() and any(local_path.iterdir()):
            print(f">>> Using existing local model directory from registry: {local_path}")
            return local_path

    if target.exists() and any(target.iterdir()):
        print(f">>> Local model directory already exists, skip download: {target}")
        return target

    url = _clean_url(cand.get("code_repository_url") or cand.get("repo_url") or "")
    if not url:
        raise RuntimeError("缺少 code_repository_url，无法自动下载模型。")

    if target.exists() and any(target.iterdir()):
        print(f">>> 本地模型目录已存在，跳过下载: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    if "github.com" in url.lower() or "gitlab.com" in url.lower() or "gitee.com" in url.lower():
        clean = re.sub(r"/(tree|blob)/.*$", "", url).rstrip(".")
        print(f">>> git clone {clean} -> {target}")
        res = subprocess.run(["git", "clone", clean, str(target)], text=True, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(f"git clone 失败：{res.stderr or res.stdout}")
        return target

    if "zenodo.org" in url.lower():
        try:
            from tool_executor import ToolRegistry
            ok = ToolRegistry.download_zenodo_dataset(url, target)
            if ok:
                return target
        except Exception as e:
            print(f"⚠️ Zenodo API 下载失败，转普通下载：{e}")

    if requests is None:
        raise RuntimeError("缺少 requests，无法下载非 git 链接。请 pip install requests")
    print(f">>> 下载 {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    filename = urllib.parse.urlparse(url).path.split("/")[-1] or "download.bin"
    target.mkdir(parents=True, exist_ok=True)
    fp = target / filename
    fp.write_bytes(r.content)
    if zipfile.is_zipfile(fp):
        with zipfile.ZipFile(fp) as z:
            z.extractall(target)
    elif filename.endswith((".tar.gz", ".tgz")):
        with tarfile.open(fp, "r:gz") as t:
            t.extractall(target)
    return target


def collect_repo_context(repo_dir: Path, max_chars: int = 32000) -> str:
    parts: List[str] = []
    patterns = ["README*", "readme*", "requirements*.txt", "environment*.yml", "environment*.yaml", "setup.py", "pyproject.toml", "*.md"]
    seen = set()
    for pat in patterns:
        for fp in sorted(repo_dir.glob(pat)):
            if fp in seen or not fp.is_file():
                continue
            seen.add(fp)
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            parts.append(f"\n===== FILE: {fp.relative_to(repo_dir)} =====\n{txt[:8000]}")
    py_list = []
    for fp in sorted(repo_dir.rglob("*.py"))[:120]:
        rel = str(fp.relative_to(repo_dir))
        if any(x in rel.lower() for x in ["__pycache__", ".venv", "site-packages"]):
            continue
        if re.search(r"(predict|infer|inference|test|eval|main|run)", fp.name, re.I):
            py_list.append(rel)
    if py_list:
        parts.append("\n===== PYTHON CANDIDATE SCRIPTS =====\n" + "\n".join(py_list[:80]))
    tree = []
    for fp in sorted(repo_dir.rglob("*"))[:400]:
        rel = str(fp.relative_to(repo_dir))
        if any(x in rel.lower() for x in [".git", "__pycache__", ".venv", "site-packages"]):
            continue
        tree.append(rel + ("/" if fp.is_dir() else ""))
    parts.append("\n===== REPO TREE PREVIEW =====\n" + "\n".join(tree[:200]))
    text = "\n".join(parts)
    return text[:max_chars]


def _parse_requirements(repo_dir: Path) -> List[str]:
    reqs = []
    for fp in list(repo_dir.glob("requirements*.txt"))[:3]:
        for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if "git+" in line or "http" in line:
                continue
            reqs.append(line.split(";")[0].strip())

    # v6.2: requirements 可能残留老包；再扫描候选运行脚本的 import，补关键运行依赖。
    import_map = {
        "torch": "torch",
        "tensorflow": "tensorflow",
        "keras": "keras",
        "sklearn": "scikit-learn",
        "Bio": "biopython",
        "yaml": "PyYAML",
        "rdflib": "rdflib",
        "networkx": "networkx",
        "tensorboardX": "tensorboardX",
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
    }
    for fp in sorted(repo_dir.rglob("*.py"))[:160]:
        rel = str(fp.relative_to(repo_dir)).lower()
        if any(x in rel for x in ["__pycache__", ".venv", "site-packages"]):
            continue
        if not re.search(r"(predict|infer|inference|test|eval|main|run)", fp.name, re.I):
            continue
        txt = fp.read_text(encoding="utf-8", errors="ignore")[:12000]
        for mod, pkg in import_map.items():
            if re.search(rf"(^|\n)\s*(import|from)\s+{re.escape(mod)}\b", txt):
                reqs.append(pkg)
    return sorted(set(reqs))[:100]


def heuristic_registry(cand: Dict[str, Any], repo_dir: Path) -> Dict[str, Any]:
    name = str(cand.get("model_name") or repo_dir.name)
    deps = _parse_requirements(repo_dir)
    scripts = []
    for fp in sorted(repo_dir.rglob("*.py")):
        rel = str(fp.relative_to(repo_dir))
        if re.search(r"(predict|infer|inference|test|eval|main|run)", fp.name, re.I):
            scripts.append(rel)
    script = scripts[0] if scripts else ""
    cmd = f"python {script} --input {{fasta_path}} --output {{output_dir}}/predictions.csv" if script else ""
    # 猜测 python 版本
    pyver = os.getenv("DEFAULT_PYTHON_VERSION", "3.9")
    env_yml = next(iter(repo_dir.glob("environment*.yml")), None) or next(iter(repo_dir.glob("environment*.yaml")), None)
    if env_yml:
        m = re.search(r"python\s*[=><!~]+\s*([0-9]+(?:\.[0-9]+)?)", env_yml.read_text(encoding="utf-8", errors="ignore"), re.I)
        if m:
            pyver = m.group(1)
    return {
        "model_name": name,
        "env_name": "env_" + _safe_name(name).lower().replace("-", "_").replace(".", "_"),
        "repo_url": cand.get("code_repository_url") or cand.get("repo_url") or "",
        "local_model_dir": str(repo_dir.relative_to(ROOT)) if repo_dir.is_relative_to(ROOT) else str(repo_dir),
        "dependencies": deps,
        "python_version": pyver,
        "inference_cmd_template": cmd,
        "skip_env_setup": False,
        "dataset_source_or_link": cand.get("dataset_source_or_link") or "",
        "model_weights_url": cand.get("model_weights_url") or "",
        "source_doi": cand.get("source_doi") or "",
        "source_pmid": cand.get("source_pmid") or "",
        "source_journal": cand.get("source_journal") or "",
        "publication_year": cand.get("publication_year"),
        "citation_count": cand.get("citation_count"),
        "journal_impact_factor": cand.get("journal_impact_factor"),
        "architecture_category": cand.get("architecture_category") or "",
        "representation_category": cand.get("representation_category") or "",
        "benchmark_role": cand.get("benchmark_role") or "",
        "benchmark_role_label": cand.get("benchmark_role_label") or "",
        "benchmark_roles": cand.get("benchmark_roles") or [],
        "benchmark_role_reason": cand.get("benchmark_role_reason") or "",
        "agent_registry_confidence": 0.35,
        "agent_registry_note": "fallback_heuristic; please verify inference_cmd_template" if cmd else "fallback_heuristic; no inference script identified",
    }


def _validated_dependencies(values: Any) -> List[str]:
    """Normalize Agent dependency suggestions; installation compatibility is checked again on HPC."""
    if not isinstance(values, list):
        return []
    output: List[str] = []
    for raw in values:
        dep = str(raw or "").strip().split(";")[0].strip()
        if not dep or dep.startswith(("-", "http://", "https://", "git+")):
            continue
        name_match = re.match(r"[A-Za-z0-9_.-]+", dep)
        if not name_match:
            continue
        name = name_match.group(0)
        if name.lower() in STDLIB_DEPENDENCIES:
            continue
        replacement = DEPENDENCY_NAME_MAP.get(name.lower())
        if replacement:
            dep = replacement + dep[len(name):]
        if dep not in output:
            output.append(dep)
    return output[:100]


def _validated_setup_commands(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    output: List[str] = []
    for raw in values:
        command = str(raw or "").strip()
        lowered = command.lower()
        if not command or "\n" in command or any(token in lowered for token in BLOCKED_COMMAND_FRAGMENTS):
            continue
        if not lowered.startswith(SAFE_SETUP_PREFIXES):
            continue
        if command not in output:
            output.append(command)
    return output[:30]


def validate_registry_record(
    candidate: Dict[str, Any],
    fallback: Dict[str, Any],
    evidence: Dict[str, Any],
    repo_dir: Path,
) -> Dict[str, Any]:
    """Apply authoritative path, dependency, command and type constraints to Agent output."""
    row = dict(fallback)
    if isinstance(candidate, dict):
        row.update({key: value for key, value in candidate.items() if value not in [None, "", [], {}]})

    model_name = str(evidence.get("model_name") or fallback.get("model_name") or repo_dir.name).strip()
    row["model_name"] = model_name
    row["env_name"] = "env_" + _safe_name(str(row.get("env_name") or model_name)).lower().replace("-", "_").replace(".", "_").removeprefix("env_")
    row["repo_url"] = evidence.get("code_repository_url") or evidence.get("repo_url") or fallback.get("repo_url") or ""

    resolved_repo = repo_dir.resolve()
    resolved_models = MODELS_DIR.resolve()
    if resolved_repo != resolved_models and resolved_models not in resolved_repo.parents:
        raise ValueError(f"local_model_dir must stay under {resolved_models}: {resolved_repo}")
    row["local_model_dir"] = str(repo_dir.relative_to(ROOT))

    version = str(row.get("python_version") or fallback.get("python_version") or "3.9").strip()
    row["python_version"] = version if re.fullmatch(r"\d+(?:\.\d+){1,2}", version) else "3.9"
    row["dependencies"] = _validated_dependencies(row.get("dependencies"))
    row["env_setup_commands"] = _validated_setup_commands(row.get("env_setup_commands"))
    row["readme_download_commands"] = _validated_setup_commands(row.get("readme_download_commands"))

    command = str(row.get("inference_cmd_template") or "").strip()
    if command and ("{fasta_path}" not in command or "{output_dir}" not in command or "\n" in command or any(token in command.lower() for token in BLOCKED_COMMAND_FRAGMENTS)):
        command = str(fallback.get("inference_cmd_template") or "").strip()
        row["agent_registry_note"] = (str(row.get("agent_registry_note") or "") + "; invalid Agent inference command replaced by deterministic fallback").strip("; ")
    row["inference_cmd_template"] = command
    row["skip_env_setup"] = False
    try:
        row["agent_registry_confidence"] = min(1.0, max(0.0, float(row.get("agent_registry_confidence", 0.0))))
    except (TypeError, ValueError):
        row["agent_registry_confidence"] = 0.0
    return row


def _llm_client_and_model():
    try:
        from openai import OpenAI
    except Exception:
        return None, ""
    model = os.getenv("ONBOARDING_LLM_MODEL") or os.getenv("MODEL_NAME") or "gpt-5.2"
    provider = os.getenv("ONBOARDING_LLM_PROVIDER", "auto").lower()
    if provider == "dashscope" or (provider == "auto" and model.startswith("qwen")):
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            return None, model
        return OpenAI(api_key=key, base_url=os.getenv("DASHSCOPE_OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")), model
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None, model
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=key, base_url=base_url), model
    return OpenAI(api_key=key), model


def agent_registry(cand: Dict[str, Any], repo_dir: Path) -> Dict[str, Any]:
    base = heuristic_registry(cand, repo_dir)
    context = collect_repo_context(repo_dir)
    client, model = _llm_client_and_model()
    if client is None:
        print("⚠️ 没有可用 LLM client，使用启发式 registry。")
        return validate_registry_record({}, base, cand, repo_dir)
    system = _ONBOARDING_PROMPTS.load_composed("repository_inspector_system")
    user = _ONBOARDING_PROMPTS.render(
        "repository_inspector_task",
        {
            "candidate_json": json.dumps(cand, ensure_ascii=False, indent=2),
            "repository_context": context,
        },
        composed=True,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or "{}"
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
        raw = re.sub(r"\s*```$", "", raw.strip())
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("LLM output is not object")
        return validate_registry_record(obj, base, cand, repo_dir)
    except Exception as e:
        print(f"⚠️ Agent registry 推断失败，使用启发式 fallback：{e}")
        return validate_registry_record({}, base, cand, repo_dir)


def confirm_registry(row: Dict[str, Any], interactive: bool = True) -> Dict[str, Any]:
    print("\n>>> 即将写入 data/local_registry.json 的模型记录：")
    print(json.dumps(row, ensure_ascii=False, indent=2))
    if not interactive:
        return row
    if not _yes("是否需要手动编辑关键字段", True):
        return row
    for key in ["model_name", "env_name", "python_version", "inference_cmd_template"]:
        row[key] = _ask(key, str(row.get(key) or ""))
    deps = _ask("dependencies，用逗号分隔", ",".join(row.get("dependencies") or []))
    row["dependencies"] = [x.strip() for x in deps.split(",") if x.strip()]
    extra = _ask("env_setup_commands，用 || 分隔；没有则空", " || ".join(row.get("env_setup_commands") or []))
    row["env_setup_commands"] = [x.strip() for x in extra.split("||") if x.strip()]
    row["skip_env_setup"] = False
    return row


def connect_hpc():
    import paramiko
    from config import HPC_HOST, HPC_PORT, HPC_USER, HPC_PASS
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    if os.getenv("HPC_ALLOW_UNKNOWN_HOST_KEY", "0").strip().lower() in {"1", "true", "yes", "y"}:
        print("[WARN] HPC_ALLOW_UNKNOWN_HOST_KEY 已启用；首次连接将信任并保存未知主机密钥。")
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    else:
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    ssh.connect(
        hostname=HPC_HOST,
        port=HPC_PORT,
        username=HPC_USER,
        password=HPC_PASS,
        timeout=15,
        auth_timeout=15,
        banner_timeout=15,
        gss_auth=False,
        gss_kex=False,
        look_for_keys=False,
        allow_agent=False,
    )
    return ssh


def deploy_to_hpc(row: Dict[str, Any], run_smoke: bool = True) -> bool:
    print("\n>>> 连接 HPC，上传模型并创建虚拟环境...")
    from hpc_model_ops import ensure_models_ready_on_hpc
    ssh = connect_hpc()
    try:
        results = ensure_models_ready_on_hpc(ssh, [row], mark_registry=True, run_smoke_test=run_smoke)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        ok = all(
            bool(x.get("env_setup_ok")) and (not run_smoke or bool(x.get("smoke_test_ok")))
            for x in results
        )
        if ok:
            print("\n>>> HPC 新模型部署和 smoke test 已通过，可以进入正式评测。")
        else:
            print("\n>>> HPC 新模型部署未通过，已停止在评测前。请先修复 env_setup_ok/smoke_test_ok。")
        return ok
    finally:
        ssh.close()


def clear_main_cache() -> None:
    d = DATA / "vlab_discussions"
    for name in ["stage1_code_cache.py", "stage1_observation.txt", "last_explored_dataset.txt", "stage1_context_for_stage2.txt"]:
        fp = d / name
        if fp.exists():
            fp.unlink()
    for fp in d.glob("stage1_obs_*.txt"):
        fp.unlink()
    for pattern in [
        "stage1_code_cache_*.py",
        "stage1_observation_*.txt",
        "last_explored_dataset_*.txt",
    ]:
        for fp in d.glob(pattern):
            fp.unlink()


def run_main_for_model(model_name: str) -> int:
    print(f"\n>>> 启动 main.py 对新模型进行后续评测: {model_name}")
    clear_main_cache()
    env = os.environ.copy()
    env["TARGET_MODEL_NAMES"] = model_name
    return subprocess.call([sys.executable, "main.py"], cwd=str(ROOT), env=env)


def choose_candidate(cands: List[Dict[str, Any]], model_name: Optional[str] = None) -> Dict[str, Any]:
    if model_name:
        key = _norm(model_name)
        for c in cands:
            if _norm(c.get("model_name")) == key:
                return c
        for row in load_registry():
            if _norm(row.get("model_name")) == key:
                cand = dict(row)
                if cand.get("repo_url") and not cand.get("code_repository_url"):
                    cand["code_repository_url"] = cand.get("repo_url")
                cand["already_registered"] = True
                return cand
        return {"model_name": model_name}
    print_candidates(cands)
    while True:
        raw = _ask("请选择要自动部署的新模型编号，或直接输入模型名")
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(cands):
                return cands[i - 1]
        elif raw:
            for c in cands:
                if _norm(c.get("model_name")) == _norm(raw):
                    return c
            for row in load_registry():
                if _norm(row.get("model_name")) == _norm(raw):
                    cand = dict(row)
                    if cand.get("repo_url") and not cand.get("code_repository_url"):
                        cand["code_repository_url"] = cand.get("repo_url")
                    cand["already_registered"] = True
                    return cand
            return {"model_name": raw}
        print("输入无效。")


def main() -> int:
    ap = argparse.ArgumentParser(description="新模型自动入库、下载、HPC 部署和评测")
    ap.add_argument("--limit", type=int, default=20, help="候选模型菜单数量")
    ap.add_argument("--model", default="", help="直接指定模型名，跳过编号选择")
    ap.add_argument("--auto-web-fill", action="store_true", help="缺证据时自动调用 Qwen 联网补全")
    ap.add_argument("--no-hpc", action="store_true", help="只下载和写 registry，不上传 HPC")
    ap.add_argument("--no-smoke", action="store_true", help="HPC 创建环境后不做 mini FASTA smoke test")
    ap.add_argument("--no-run-main", action="store_true", help="完成入库和部署后不自动运行 main.py")
    ap.add_argument("--non-interactive", action="store_true", help="尽量非交互；需要 --model 且证据足够")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cands = load_candidates(limit=args.limit, include_registered=True)
    cand = choose_candidate(cands, args.model or None)
    print("\n>>> 已选择模型：")
    print(json.dumps(cand, ensure_ascii=False, indent=2))

    if not cand.get("code_repository_url"):
        if args.auto_web_fill or (not args.non_interactive and _yes("缺少代码仓库，是否先 Qwen 联网补全", True)):
            cand = try_qwen_fill(cand)
    if not cand.get("code_repository_url") and not args.non_interactive:
        cand = manual_fill(cand)
    elif not cand.get("code_repository_url"):
        raise RuntimeError("非交互模式下缺少 code_repository_url，无法继续。")

    # 即使有 repo，也允许手动修正权重/数据等字段。
    if not args.non_interactive and _yes("是否手动检查/补全证据字段", True):
        cand = manual_fill(cand)

    repo_dir = download_model(cand)
    print(f">>> 模型已准备在本地目录: {repo_dir}")

    reg = agent_registry(cand, repo_dir)
    reg = confirm_registry(reg, interactive=not args.non_interactive)
    upsert_registry(reg)
    print(f">>> 已写入/更新 {REGISTRY_PATH.relative_to(ROOT)}")

    if args.no_hpc:
        print("\n>>> 已按 --no-hpc 仅完成下载和 registry 更新；不会启动 main.py 正式评测。")
        return 0

    hpc_ready = deploy_to_hpc(reg, run_smoke=not args.no_smoke)

    if not hpc_ready:
        print("\n>>> 新模型自动入库已完成，但 HPC 部署/小样本测试没有通过。")
        print("    已禁止自动进入 main.py，避免把失败模型带入正式 benchmark。")
        print("    修复后重新运行: python new_model_onboarding.py --model " + str(reg["model_name"]) + " --no-run-main")
        return 2

    if args.no_smoke:
        print("\n>>> 环境部署已完成，但 smoke test 被跳过；不会标记模型 ready，也不会启动正式评测。")
        return 0

    if not args.no_run_main:
        if args.non_interactive or _yes("是否立即自动运行 main.py 进入后续评测任务", True):
            return run_main_for_model(reg["model_name"])
    print("\n>>> 新模型自动入库流程完成，且 HPC 部署状态可用。")
    print(f"    下一步可手动运行: set TARGET_MODEL_NAMES={reg['model_name']} && python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
