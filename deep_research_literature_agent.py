# -*- coding: utf-8 -*-
"""
DeepSeek Multi-Source AMP Literature Agent
=========================================

功能：
1. 不只搜索 PubMed，同时搜索 PubMed / Europe PMC / Crossref / OpenAlex /
   Semantic Scholar / Europe PMC 预印本 / GitHub / DataCite / Zenodo。
2. 支持 PubMed Similar Articles、Semantic Scholar citation/reference 扩展、模型名称回搜。
3. 获取并保存开放全文：PMC EFetch XML、Europe PMC fullTextXML。
4. 所有原始搜索证据、全文 XML、全文文本、链接、结构化 evidence 都落盘。
5. 所有 evidence 收集完成后，先按模型名称 / 主题 / 来源分块压缩 evidence。
6. 全局多 Agent 会议只读取 chunk_summaries，降低超时风险。
7. Chief 不再删除候选模型：额外保留 all_candidate_models / benchmark_ready_models。
8. 新增 model_dataset_links / dataset_followup_tasks，避免数据集线索丢失。
9. literature_deep_research_memory.md 会保存接近 meeting_trace.md 风格的多 Agent 讨论过程。
10. 新增模型分类梳理和每类 1-2 个代表模型。
11. 全局会议前会对缺少 GitHub 链接的模型自动执行 GitHub 补链搜索。
12. GitHub 补链结果会作为独立证据写入 evidence pool / compact evidence pool，再进入 Agent 讨论。
13. 所有原始 evidence、全文缓存、chunk summary、GitHub 补链证据、最终 memory 都落盘。

安全边界：
- 只获取开放全文，不绕过付费墙。
- 没有开放全文时只保存 DOI / PubMed / Publisher / OA URL 等线索。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ------------------------- .env support -------------------------
try:
    from dotenv import load_dotenv
    _ROOT = Path(__file__).resolve().parent
    if (_ROOT / '.env').exists():
        load_dotenv(_ROOT / '.env', override=False)
    else:
        load_dotenv(override=False)
except Exception:
    pass

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# ------------------------- Paths -------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
SEARCH_CACHE_DIR = DATA_DIR / 'search_cache'
FULLTEXT_CACHE_DIR = DATA_DIR / 'fulltext_cache'
REPO_CACHE_DIR = DATA_DIR / 'repository_cache'
DATASET_CACHE_DIR = DATA_DIR / 'dataset_cache'
FAILED_DIR = DATA_DIR / 'failed_agent_outputs'
CHUNK_SUMMARIES_DIR = DATA_DIR / 'chunk_summaries'

RAW_CANDIDATES_JSONL = DATA_DIR / 'raw_candidates.jsonl'
NORMALIZED_PAPERS_JSONL = DATA_DIR / 'normalized_papers.jsonl'
NORMALIZED_REPOS_JSONL = DATA_DIR / 'normalized_repositories.jsonl'
NORMALIZED_DATASETS_JSONL = DATA_DIR / 'normalized_datasets.jsonl'
SEARCH_SUMMARY_JSON = DATA_DIR / 'multi_source_search_summary.json'
PUBMED_SEARCH_JSON = DATA_DIR / 'pubmed_search_results.json'
FULLTEXT_EVIDENCE_JSONL = DATA_DIR / 'pubmed_fulltext_key_evidence.jsonl'
EVIDENCE_POOL_JSON = DATA_DIR / 'evidence_pool.json'
EVIDENCE_POOL_MD = DATA_DIR / 'evidence_pool.md'
GLOBAL_MEETING_RAW_JSONL = DATA_DIR / 'deepseek_meeting_raw.jsonl'
CHUNK_SUMMARIES_JSONL = CHUNK_SUMMARIES_DIR / 'chunk_summaries.jsonl'
CHUNK_INDEX_JSON = CHUNK_SUMMARIES_DIR / '_chunk_index.json'
COMPACT_EVIDENCE_POOL_JSON = DATA_DIR / 'compact_evidence_pool.json'
COMPACT_EVIDENCE_POOL_MD = DATA_DIR / 'compact_evidence_pool.md'
MEMORY_JSON = DATA_DIR / 'literature_deep_research_memory.json'
MEMORY_MD = DATA_DIR / 'literature_deep_research_memory.md'
INDEX_JSON = DATA_DIR / 'literature_deep_research_index.json'
GITHUB_MISSING_MODEL_ENRICHMENT_JSONL = DATA_DIR / 'github_missing_model_enrichment.jsonl'
GITHUB_MISSING_MODEL_ENRICHMENT_JSON = DATA_DIR / 'github_missing_model_enrichment.json'
GITHUB_ENRICHMENT_PENDING_MODELS_TXT = DATA_DIR / 'github_enrichment_pending_models.txt'
GITHUB_ENRICHMENT_RUN_REPORT_JSON = DATA_DIR / 'github_enrichment_run_report.json'
QWEN_WEB_ENRICHMENT_JSONL = DATA_DIR / 'qwen_web_enrichment.jsonl'
QWEN_WEB_ENRICHMENT_JSON = DATA_DIR / 'qwen_web_enrichment.json'
QWEN_WEB_ENRICHMENT_PENDING_MODELS_TXT = DATA_DIR / 'qwen_web_enrichment_pending_models.txt'
QWEN_WEB_ENRICHMENT_RUN_REPORT_JSON = DATA_DIR / 'qwen_web_enrichment_run_report.json'

for p in [DATA_DIR, SEARCH_CACHE_DIR, FULLTEXT_CACHE_DIR, REPO_CACHE_DIR, DATASET_CACHE_DIR, FAILED_DIR, CHUNK_SUMMARIES_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------- Utilities -------------------------
def now_str() -> str:
    return _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def json_dumps(obj: Any, indent: int = 2, **kwargs: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, **kwargs)


def json_loads_safe(text: str, fallback: Any = None) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return fallback


def stable_hash(obj: Any) -> str:
    if not isinstance(obj, str):
        obj = json_dumps(obj, sort_keys=True)
    return hashlib.sha1(obj.encode('utf-8', errors='ignore')).hexdigest()[:16]


def normalize_key(text: Any) -> str:
    if text is None:
        return ''
    s = str(text).strip().lower()
    s = html.unescape(s)
    s = re.sub(r'https?://(www\.)?', '', s)
    s = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def safe_name(text: Any, fallback: str = 'item', max_len: int = 80) -> str:
    s = normalize_key(text).replace(' ', '_')
    s = re.sub(r'[^a-zA-Z0-9_\-]+', '_', s).strip('_')
    return (s[:max_len] or fallback)


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json_dumps(obj, 0) + '\n')


def write_jsonl(path: Path, rows: List[Any]) -> None:
    """Overwrite a JSONL file with one JSON object per line.

    v5.0: GitHub enrichment is a cache, not an append-only run log.
    Overwriting avoids duplicate rows making users think the same model was imported again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for obj in ensure_list(rows):
            f.write(json_dumps(obj, 0) + '\n')


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(obj, 2), encoding='utf-8')


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Any]:
    """读取 JSONL；坏行自动跳过，避免续跑时因为部分日志损坏而中断。"""
    out: List[Any] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def clean_index_value(v: Any) -> Optional[str]:
    """把 index 里的任意值安全转换成可 set 去重的字符串。"""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        s = str(v).strip()
        return s or None
    if isinstance(v, dict):
        for k in [
            'candidate_key', 'pmid', 'PMID', 'doi', 'DOI', 'pmcid', 'PMCID',
            'title', 'url', 'id', 'paper_id', 'openalex_id', 'semantic_scholar_id',
            'source_id', 'name', 'model_name', 'canonical_name',
        ]:
            if v.get(k):
                return str(v.get(k)).strip()
        return 'dict:' + stable_hash(v)
    if isinstance(v, (list, tuple, set)):
        return 'list:' + stable_hash(list(v))
    s = str(v).strip()
    return s or None


def clean_index_list(values: Any) -> List[str]:
    cleaned: List[str] = []
    for item in ensure_list(values):
        s = clean_index_value(item)
        if s:
            cleaned.append(s)
    return sorted(set(cleaned))


def trunc(s: Any, n: int = 6000) -> str:
    s = '' if s is None else str(s)
    return s if len(s) <= n else s[:n] + f'\n...[truncated {len(s)-n} chars]'


URL_RE = re.compile(r'https?://[^\s\]\)\}\>\"\']+', re.I)
DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.I)
GITHUB_RE = re.compile(r'https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', re.I)
DATA_RE = re.compile(r'https?://[^\s\]\)\}\>\"\']*(?:zenodo|figshare|dryad|dataverse|kaggle|datacite|osf\.io|github\.com|huggingface\.co)[^\s\]\)\}\>\"\']*', re.I)


def extract_links(text: str) -> Dict[str, List[str]]:
    urls = sorted(set(URL_RE.findall(text or '')))
    dois = sorted(set(DOI_RE.findall(text or '')))
    github = sorted(set(GITHUB_RE.findall(text or '')))
    dataset = sorted(set(DATA_RE.findall(text or '')))
    return {'urls': urls, 'dois': dois, 'github_urls': github, 'dataset_urls': dataset}


def text_from_xml(root: ET.Element) -> str:
    parts: List[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
        if elem.tail and elem.tail.strip():
            parts.append(elem.tail.strip())
    return re.sub(r'\s+', ' ', ' '.join(parts)).strip()


def sections_from_pmc_xml(xml_text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    try:
        root = ET.fromstring(xml_text.encode('utf-8', errors='ignore'))
    except Exception:
        return {}
    for sec in root.findall('.//sec'):
        title = ''.join(sec.findtext('title') or '').strip() or 'untitled_section'
        key = normalize_key(title)[:80] or 'untitled_section'
        body_parts = []
        for p in sec.findall('.//p'):
            txt = ''.join(p.itertext()).strip()
            if txt:
                body_parts.append(re.sub(r'\s+', ' ', txt))
        if body_parts:
            sections.setdefault(key, []).append('\n'.join(body_parts))
    return {k: '\n\n'.join(v)[:50000] for k, v in sections.items()}


# ------------------------- HTTP -------------------------
class HttpClient:
    def __init__(self, timeout: int = 45, retries: int = 3, delay: float = 0.4):
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.user_agent = os.getenv('AMP_BENCHMARK_USER_AGENT', 'amp-benchmark-literature-agent/1.0')

    def get_bytes(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> bytes:
        if params:
            qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
            url = url + ('&' if '?' in url else '?') + qs
        hdr = {'User-Agent': self.user_agent}
        if headers:
            hdr.update(headers)
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(url, headers=hdr)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                last_err = e
                # 404 is usually final; 429/5xx are retryable.
                if e.code == 404:
                    raise
                sleep = self.delay * attempt + random.random() * 0.2
                time.sleep(sleep)
            except Exception as e:
                last_err = e
                time.sleep(self.delay * attempt + random.random() * 0.2)
        raise RuntimeError(f'HTTP GET failed after {self.retries} retries: {url} :: {last_err}')

    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> str:
        data = self.get_bytes(url, params=params, headers=headers)
        return data.decode('utf-8', errors='replace')

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        txt = self.get_text(url, params=params, headers=headers)
        return json.loads(txt)


HTTP = HttpClient()


def cache_raw(source: str, name: str, payload: Any) -> Path:
    d = SEARCH_CACHE_DIR / source
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f'{safe_name(name, stable_hash(name))}_{stable_hash(payload)}.json'
    try:
        write_json(fp, payload)
    except Exception:
        fp.write_text(str(payload), encoding='utf-8', errors='ignore')
    return fp


# ------------------------- Agent MD loader -------------------------
class AgentMDLoader:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def load(self, name: str) -> str:
        p = self.base_dir / f'{name}.md'
        if not p.exists():
            raise FileNotFoundError(f'Agent prompt not found: {p}')
        return p.read_text(encoding='utf-8')


# ------------------------- LLM client -------------------------
class DeepSeekChatLLM:
    def __init__(self, provider: str = 'dashscope', config_path: Path = Path('llm_providers.json')):
        if OpenAI is None:
            raise RuntimeError('缺少 openai 包，请运行：pip install -U openai python-dotenv')
        cfg_all = read_json(config_path, {})
        if provider not in cfg_all:
            raise KeyError(f'Provider {provider!r} not found in {config_path}')
        cfg = cfg_all[provider]
        self.provider = provider
        self.cfg = cfg
        key_env = cfg.get('chat_api_key_env') or 'DASHSCOPE_API_KEY'
        api_key = os.getenv(key_env)
        if not api_key:
            raise RuntimeError(f'没有找到环境变量 {key_env}。请在 .env 中配置，例如：{key_env}=sk-xxx')
        self.model = cfg.get('chat_model') or cfg.get('meeting_model') or 'deepseek-v4-pro'
        self.client = OpenAI(api_key=api_key, base_url=cfg.get('chat_base_url'))
        self.timeout = int(cfg.get('timeout', 180))
        self.max_retries = int(cfg.get('max_retries', 4))
        self.use_response_format = bool(cfg.get('use_response_format', False))

    def chat(self, agent_name: str, system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: float = 0.1) -> str:
        model_name = model or self.model
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            print(f'       ↳ DeepSeek: agent={agent_name}, model={model_name}, attempt={attempt}/{self.max_retries}')
            try:
                resp = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                    temperature=temperature,
                    timeout=self.timeout,
                )
                return resp.choices[0].message.content or ''
            except Exception as e:
                last_err = e
                time.sleep(min(8, attempt * 2))
        raise RuntimeError(f'DeepSeek call failed: {agent_name}: {last_err}')

    def chat_json(self, agent_name: str, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> Any:
        raw = self.chat(agent_name, system_prompt, user_prompt, model=model)
        obj = parse_json_from_text(raw)
        if obj is not None:
            return obj
        repair_prompt = f"""
下面的模型输出不是合法 JSON。请只返回修复后的合法 JSON，不要解释，不要 Markdown。

原始输出：
{raw[:30000]}
"""
        fixed = self.chat(agent_name + '_json_repair', 'You repair invalid JSON. Output JSON only.', repair_prompt, model=model, temperature=0.0)
        obj = parse_json_from_text(fixed)
        if obj is None:
            fp = FAILED_DIR / f'{now_str().replace(":", "-").replace(" ", "_")}_{agent_name}.txt'
            fp.write_text(raw + '\n\n--- REPAIR ---\n\n' + fixed, encoding='utf-8')
            raise RuntimeError(f'JSON parse failed for {agent_name}. Raw saved: {fp}')
        return obj


class QwenMaxWebSearchLLM:
    """Alibaba Bailian / DashScope Qwen web-search helper.

    v5.7 默认使用百炼 OpenAI-compatible Responses API：
        qwen3.7-max + tools=[web_search, web_extractor]

    同时保留旧版 Chat Completions + extra_body={"enable_search": True}
    的兼容路径。llm_providers.json 中的 llm_api_mode / api_mode /
    qwen_web_api_mode 为 responses 时走 Responses API，否则走 chat。
    """
    def __init__(self, provider: str = 'dashscope_qwen37max_search', config_path: Path = Path('llm_providers.json'), model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError('缺少 openai 包，请运行：pip install -U openai python-dotenv')
        cfg_all = read_json(config_path, {})
        cfg = cfg_all.get(provider) or cfg_all.get('dashscope_qwen37max_search') or cfg_all.get('dashscope_qwen_search') or cfg_all.get('dashscope') or {}
        self.provider = provider
        self.cfg = cfg
        key_env = cfg.get('api_key_env') or cfg.get('chat_api_key_env') or 'DASHSCOPE_API_KEY'
        api_key = os.getenv(key_env)
        if not api_key:
            raise RuntimeError(f'没有找到环境变量 {key_env}。请在 .env 中配置，例如：{key_env}=sk-xxx')
        self.api_mode = str(cfg.get('qwen_web_api_mode') or cfg.get('llm_api_mode') or cfg.get('api_mode') or 'responses').lower()
        self.model = model or cfg.get('web_search_model') or cfg.get('qwen_web_search_model') or cfg.get('model') or ('qwen3.7-max' if self.api_mode == 'responses' else 'qwen-max')
        self.timeout = int(cfg.get('timeout', 240))
        self.max_retries = int(cfg.get('max_retries', 3))
        self.search_extra_body = cfg.get('web_search_extra_body') or cfg.get('search_extra_body') or ({'enable_search': True} if self.api_mode != 'responses' else None)
        self.tools = cfg.get('tools') or [{'type': 'web_search'}, {'type': 'web_extractor'}]
        self.instructions_prefix = str(cfg.get('instructions_prefix') or '').strip()
        base_url = self._resolve_base_url(cfg)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _resolve_base_url(self, cfg: Dict[str, Any]) -> str:
        # 优先环境变量，便于不同地域 / workspace 切换。
        base_url = (
            os.getenv(str(cfg.get('base_url_env') or 'DASHSCOPE_RESPONSES_BASE_URL'))
            or cfg.get('responses_base_url')
            or cfg.get('base_url')
            or cfg.get('chat_base_url')
            or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        )
        workspace_id = os.getenv(str(cfg.get('workspace_id_env') or 'DASHSCOPE_WORKSPACE_ID'), '')
        region = os.getenv(str(cfg.get('region_env') or 'DASHSCOPE_REGION'), cfg.get('region') or 'cn-beijing')
        base_url = str(base_url).replace('{WorkspaceId}', workspace_id).replace('{workspace_id}', workspace_id).replace('{Region}', region).replace('{region}', region)
        if '{' in base_url or '}' in base_url:
            raise RuntimeError(
                'Qwen Responses API base_url 仍包含未替换占位符。请在 .env 配置 DASHSCOPE_WORKSPACE_ID，'
                '或直接配置 DASHSCOPE_RESPONSES_BASE_URL=https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
            )
        return base_url

    def _extract_responses_text(self, resp: Any) -> str:
        text = getattr(resp, 'output_text', None)
        if text:
            return str(text)
        try:
            data = resp.model_dump() if hasattr(resp, 'model_dump') else resp
        except Exception:
            data = resp
        parts: List[str] = []
        if isinstance(data, dict):
            for item in ensure_list(data.get('output')):
                if isinstance(item, dict):
                    for c in ensure_list(item.get('content')):
                        if isinstance(c, dict):
                            val = c.get('text') or c.get('output_text') or c.get('content')
                            if isinstance(val, str):
                                parts.append(val)
                elif isinstance(item, str):
                    parts.append(item)
            if not parts and isinstance(data.get('content'), str):
                parts.append(data.get('content'))
        return '\n'.join([p for p in parts if p]).strip()

    def _chat_responses(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        if not hasattr(self.client, 'responses'):
            raise RuntimeError('当前 openai Python 包版本不支持 Responses API。请运行：pip install -U openai')
        full_system = system_prompt
        if self.instructions_prefix:
            full_system = self.instructions_prefix + '\n\n' + system_prompt
        kwargs = dict(
            model=self.model,
            input=[
                {'role': 'system', 'content': full_system},
                {'role': 'user', 'content': user_prompt},
            ],
            tools=self.tools,
            temperature=temperature,
            timeout=self.timeout,
        )
        # 部分 SDK / 百炼网关可能暂不接受 temperature 或 timeout；失败后降级重试。
        try:
            resp = self.client.responses.create(**kwargs)
        except TypeError:
            kwargs.pop('timeout', None)
            resp = self.client.responses.create(**kwargs)
        except Exception as first_err:
            msg = str(first_err)
            if 'temperature' in msg.lower():
                kwargs.pop('temperature', None)
                resp = self.client.responses.create(**kwargs)
            else:
                raise
        text = self._extract_responses_text(resp)
        if not text:
            raise RuntimeError(f'Qwen Responses API returned empty text: {resp}')
        return text

    def _chat_completions(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
            timeout=self.timeout,
        )
        if self.search_extra_body:
            kwargs['extra_body'] = self.search_extra_body
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ''

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        last_err: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            mode_label = 'Responses+web_search+web_extractor' if self.api_mode == 'responses' else 'ChatCompletions+enable_search'
            print(f'       ↳ Qwen-Web: mode={mode_label}, provider={self.provider}, model={self.model}, attempt={attempt}/{self.max_retries}', flush=True)
            try:
                if self.api_mode == 'responses':
                    return self._chat_responses(system_prompt, user_prompt, temperature=temperature)
                return self._chat_completions(system_prompt, user_prompt, temperature=temperature)
            except Exception as e:
                last_err = e
                time.sleep(min(8, attempt * 2))
        raise RuntimeError(f'Qwen web-search call failed: {last_err}')

    def chat_json(self, system_prompt: str, user_prompt: str) -> Any:
        raw = self.chat(system_prompt, user_prompt, temperature=0.0)
        obj = parse_json_from_text(raw)
        if obj is not None:
            return obj
        repair = f"""
下面的模型输出不是合法 JSON。请只返回修复后的合法 JSON，不要解释，不要 Markdown。

原始输出：
{raw[:30000]}
"""
        fixed = self.chat('You repair invalid JSON. Output JSON only.', repair, temperature=0.0)
        obj = parse_json_from_text(fixed)
        if obj is None:
            fp = FAILED_DIR / f'{now_str().replace(":", "-").replace(" ", "_")}_qwen_web_enrichment.txt'
            fp.write_text(raw + '\n\n--- REPAIR ---\n\n' + fixed, encoding='utf-8')
            raise RuntimeError(f'Qwen web enrichment JSON parse failed. Raw saved: {fp}')
        return obj


def parse_json_from_text(text: str) -> Any:
    text = (text or '').strip()
    if not text:
        return None
    # Remove code fences.
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I).strip()
    text = re.sub(r'\s*```$', '', text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try first JSON object or array.
    starts = [i for i in [text.find('{'), text.find('[')] if i >= 0]
    if not starts:
        return None
    start = min(starts)
    for end in range(len(text), start, -1):
        chunk = text[start:end].strip()
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None


# ------------------------- Query planning -------------------------
DEFAULT_QUERY_PLAN = {
    'pubmed': [
        {'name': 'broad_prediction', 'query': '("antimicrobial peptide" OR "antimicrobial peptides" OR (AMP AND peptide) OR "host defense peptide") AND (prediction OR predictor OR classifier OR classification OR identification OR recognition OR screening OR discrimination)'},
        {'name': 'ml_dl_precision', 'query': '("antimicrobial peptide"[tiab] OR "antimicrobial peptides"[tiab]) AND ("machine learning"[tiab] OR "deep learning"[tiab] OR "neural network"[tiab] OR CNN[tiab] OR LSTM[tiab] OR transformer[tiab] OR BERT[tiab] OR SVM[tiab] OR "support vector machine"[tiab] OR "random forest"[tiab] OR XGBoost[tiab]) AND (predictor[tiab] OR classifier[tiab] OR prediction[tiab] OR identification[tiab])'},
        {'name': 'software_server_tool', 'query': '("antimicrobial peptide"[tiab] OR "antimicrobial peptides"[tiab] OR "AMP prediction"[tiab]) AND ("web server"[tiab] OR "online tool"[tiab] OR software[tiab] OR standalone[tiab] OR platform[tiab] OR database[tiab])'},
        {'name': 'benchmark_dataset_metrics', 'query': '("antimicrobial peptide"[tiab] OR "antimicrobial peptides"[tiab] OR "AMP"[tiab]) AND (benchmark[tiab] OR dataset[tiab] OR "data set"[tiab] OR "independent test"[tiab] OR "external validation"[tiab] OR evaluation[tiab] OR comparison[tiab] OR metric[tiab])'},
        {'name': 'review_for_model_names', 'query': '("antimicrobial peptide"[tiab] OR "antimicrobial peptides"[tiab]) AND (predictor[tiab] OR classifier[tiab] OR "machine learning"[tiab] OR "deep learning"[tiab]) AND review[pt]'},
    ],
    'europe_pmc': [
        {'name': 'europepmc_broad', 'query': '"antimicrobial peptide" AND (prediction OR predictor OR classifier OR "machine learning" OR "deep learning" OR "web server")'},
        {'name': 'europepmc_dataset', 'query': '"antimicrobial peptide" AND (benchmark OR dataset OR "independent test" OR evaluation)'},
    ],
    'crossref': [
        {'name': 'crossref_broad', 'query': 'antimicrobial peptide prediction machine learning'},
        {'name': 'crossref_tool', 'query': 'antimicrobial peptide predictor web server'},
    ],
    'openalex': [
        {'name': 'openalex_broad', 'query': 'antimicrobial peptide prediction machine learning'},
        {'name': 'openalex_deep_learning', 'query': 'antimicrobial peptide deep learning classifier'},
    ],
    'semantic_scholar': [
        {'name': 's2_broad', 'query': 'antimicrobial peptide prediction machine learning'},
        {'name': 's2_tool', 'query': 'antimicrobial peptide predictor web server'},
    ],
    'preprint': [
        {'name': 'preprint_broad', 'query': '"antimicrobial peptide" AND (prediction OR predictor OR classifier OR "machine learning" OR "deep learning") AND SRC:PPR'},
    ],
    'github': [
        {'name': 'github_broad', 'query': '"antimicrobial peptide" prediction'},
        {'name': 'github_amp_ml', 'query': 'AMP prediction machine learning peptide'},
    ],
    'datacite': [
        {'name': 'datacite_dataset', 'query': 'antimicrobial peptide prediction dataset'},
    ],
    'zenodo': [
        {'name': 'zenodo_dataset', 'query': 'antimicrobial peptide prediction dataset'},
    ],
}


def llm_plan_queries(llm: DeepSeekChatLLM, loader: AgentMDLoader, max_queries: int = 20) -> Dict[str, List[Dict[str, str]]]:
    try:
        system = loader.load('pubmed_query_planner')
    except Exception:
        system = 'You plan literature search queries. Return JSON only.'
    user = f"""
请为“抗菌肽 AMP 预测模型 benchmark 系统”生成多源搜索 query。
目标：尽可能召回模型原始论文、benchmark 论文、数据集论文、web server / software 论文、预印本、代码仓库、数据集仓库。

返回严格 JSON，格式：
{{
  "pubmed": [{{"name":"...", "query":"..."}}],
  "europe_pmc": [{{"name":"...", "query":"..."}}],
  "crossref": [{{"name":"...", "query":"..."}}],
  "openalex": [{{"name":"...", "query":"..."}}],
  "semantic_scholar": [{{"name":"...", "query":"..."}}],
  "preprint": [{{"name":"...", "query":"..."}}],
  "github": [{{"name":"...", "query":"..."}}],
  "datacite": [{{"name":"...", "query":"..."}}],
  "zenodo": [{{"name":"...", "query":"..."}}]
}}

要求：
- PubMed query 不要全部加 NOT review，因为 review 可用于提取模型名称。
- PubMed query 中包含高召回 query 和高精度 query。
- 不要生成超过 {max_queries} 条总 query。
- 只返回 JSON。
"""
    try:
        obj = llm.chat_json('pubmed_query_planner', system, user)
        plan = normalize_query_plan(obj)
        if plan:
            return plan
    except Exception as e:
        print(f'    ⚠️ Query planner 失败，使用内置 query：{e}')
    return DEFAULT_QUERY_PLAN


def normalize_query_plan(obj: Any) -> Dict[str, List[Dict[str, str]]]:
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, List[Dict[str, str]]] = {}
    for source in DEFAULT_QUERY_PLAN:
        items = obj.get(source) or []
        normalized: List[Dict[str, str]] = []
        for i, it in enumerate(ensure_list(items)):
            if isinstance(it, str):
                q = it
                name = f'{source}_{i+1}'
            elif isinstance(it, dict):
                q = it.get('query') or it.get('q') or ''
                name = it.get('name') or f'{source}_{i+1}'
            else:
                continue
            if q.strip():
                normalized.append({'name': str(name), 'query': q.strip()})
        if normalized:
            out[source] = normalized
    return out


# ------------------------- Normalized candidates -------------------------
def candidate_key(c: Dict[str, Any]) -> str:
    for k in ['doi', 'pmid', 'pmcid', 'semantic_scholar_id', 'openalex_id']:
        if c.get(k):
            return f'{k}:{normalize_key(c[k])}'
    title = normalize_key(c.get('title'))
    year = str(c.get('year') or '')
    if title:
        return f'title:{title}:{year}'
    return 'hash:' + stable_hash(c)


def merge_candidate(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if v in [None, '', [], {}]:
            continue
        if k not in out or out[k] in [None, '', [], {}]:
            out[k] = v
        elif k in ['sources', 'raw_source_files', 'urls']:
            out[k] = sorted(set(ensure_list(out.get(k)) + ensure_list(v)))
        elif k == 'source_ids':
            merged = dict(out.get(k) or {})
            if isinstance(v, dict):
                merged.update({kk: vv for kk, vv in v.items() if vv})
            out[k] = merged
    return out


def dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    title_map: Dict[str, str] = {}
    for c in candidates:
        key = candidate_key(c)
        title_key = normalize_key(c.get('title'))
        if key.startswith('title:') and title_key in title_map:
            key = title_map[title_key]
        elif not key.startswith('title:') and title_key:
            title_map.setdefault(title_key, key)
        if key in merged:
            merged[key] = merge_candidate(merged[key], c)
        else:
            c = dict(c)
            c['candidate_key'] = key
            merged[key] = c
    return list(merged.values())


def looks_relevant(c: Dict[str, Any]) -> bool:
    txt = normalize_key(' '.join([str(c.get('title') or ''), str(c.get('abstract') or ''), ' '.join(ensure_list(c.get('keywords')))]))
    if not txt:
        return False
    amp_terms = ['antimicrobial peptide', 'antimicrobial peptides', 'amp', 'host defense peptide', 'antibacterial peptide', 'peptide']
    model_terms = ['prediction', 'predictor', 'classifier', 'classification', 'identification', 'recognition', 'machine learning', 'deep learning', 'neural', 'svm', 'random forest', 'xgboost', 'transformer', 'bert', 'web server', 'software', 'tool', 'benchmark', 'dataset', 'evaluation']
    has_amp = any(t in txt for t in amp_terms)
    has_model = any(t in txt for t in model_terms)
    return has_amp and has_model


# ------------------------- Source clients -------------------------
class PubMedClient:
    BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

    def __init__(self):
        self.email = os.getenv('PUBMED_TOOL_EMAIL') or os.getenv('NCBI_EMAIL') or ''
        self.tool = os.getenv('PUBMED_TOOL_NAME') or 'amp_benchmark_multisource'
        self.api_key = os.getenv('NCBI_API_KEY')

    def params(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        p = {'tool': self.tool, 'retmode': 'json'}
        if self.email:
            p['email'] = self.email
        if self.api_key:
            p['api_key'] = self.api_key
        p.update(extra)
        return p

    def esearch(self, query: str, retmax: int = 100, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[str]:
        term = query
        if year_from or year_to:
            yf = year_from or 1900
            yt = year_to or _dt.datetime.now().year
            term = f'({term}) AND ("{yf}"[Date - Publication] : "{yt}"[Date - Publication])'
        data = HTTP.get_json(f'{self.BASE}/esearch.fcgi', self.params({'db': 'pubmed', 'term': term, 'retmax': retmax, 'sort': 'relevance'}))
        return [str(x) for x in data.get('esearchresult', {}).get('idlist', [])]

    def efetch_records(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        xml_text = HTTP.get_text(f'{self.BASE}/efetch.fcgi', self.params({'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'xml'}))
        cache_raw('pubmed', 'efetch_' + '_'.join(pmids[:5]), {'pmids': pmids, 'xml': xml_text[:200000]})
        try:
            root = ET.fromstring(xml_text.encode('utf-8', errors='ignore'))
        except Exception:
            return []
        records = []
        for art in root.findall('.//PubmedArticle'):
            rec = self.parse_article(art)
            if rec:
                records.append(rec)
        return records

    def parse_article(self, art: ET.Element) -> Dict[str, Any]:
        pmid = ''.join(art.findtext('.//PMID') or '').strip()
        title = ''.join(art.findtext('.//ArticleTitle') or '').strip()
        abstract_parts = []
        for node in art.findall('.//Abstract/AbstractText'):
            label = node.attrib.get('Label')
            txt = ''.join(node.itertext()).strip()
            if txt:
                abstract_parts.append((label + ': ' if label else '') + txt)
        abstract = '\n'.join(abstract_parts)
        journal = ''.join(art.findtext('.//Journal/Title') or art.findtext('.//ISOAbbreviation') or '').strip()
        year = art.findtext('.//PubDate/Year') or art.findtext('.//ArticleDate/Year') or ''
        ids = {}
        for idn in art.findall('.//ArticleIdList/ArticleId'):
            typ = idn.attrib.get('IdType') or 'unknown'
            val = ''.join(idn.itertext()).strip()
            if val:
                ids[typ] = val
        doi = ids.get('doi')
        pmcid = ids.get('pmc') or ids.get('pmcid')
        authors = []
        for a in art.findall('.//Author')[:20]:
            last = a.findtext('LastName') or ''
            fore = a.findtext('ForeName') or ''
            coll = a.findtext('CollectiveName') or ''
            name = (fore + ' ' + last).strip() or coll.strip()
            if name:
                authors.append(name)
        keywords = [''.join(k.itertext()).strip() for k in art.findall('.//Keyword') if ''.join(k.itertext()).strip()]
        pub_types = [''.join(k.itertext()).strip() for k in art.findall('.//PublicationType') if ''.join(k.itertext()).strip()]
        urls = []
        if pmid:
            urls.append(f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/')
        if doi:
            urls.append(f'https://doi.org/{doi}')
        if pmcid:
            urls.append(f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/')
        return {
            'source_primary': 'pubmed',
            'sources': ['pubmed'],
            'source_ids': {'pubmed': pmid},
            'pmid': pmid,
            'pmcid': pmcid,
            'doi': doi,
            'title': html.unescape(title),
            'abstract': html.unescape(abstract),
            'journal': journal,
            'venue': journal,
            'year': int(year) if str(year).isdigit() else year,
            'authors': authors,
            'keywords': keywords,
            'publication_types': pub_types,
            'article_ids': ids,
            'urls': urls,
            'raw_source_files': [],
        }

    def pmcid_from_pmid(self, pmid: str) -> Optional[str]:
        try:
            xml = HTTP.get_text(f'{self.BASE}/elink.fcgi', self.params({'dbfrom': 'pubmed', 'db': 'pmc', 'id': pmid, 'retmode': 'xml'}))
            root = ET.fromstring(xml.encode('utf-8', errors='ignore'))
            pmc_id = root.findtext('.//LinkSetDb/Link/Id')
            if pmc_id:
                return 'PMC' + pmc_id if not str(pmc_id).upper().startswith('PMC') else str(pmc_id)
        except Exception:
            return None
        return None

    def similar_pmids(self, pmid: str, retmax: int = 20) -> List[str]:
        try:
            xml = HTTP.get_text(f'{self.BASE}/elink.fcgi', self.params({'dbfrom': 'pubmed', 'db': 'pubmed', 'id': pmid, 'cmd': 'neighbor_score', 'retmode': 'xml'}))
            root = ET.fromstring(xml.encode('utf-8', errors='ignore'))
            ids = []
            for link in root.findall('.//LinkSetDb/Link'):
                rid = link.findtext('Id')
                if rid and rid != pmid:
                    ids.append(rid)
                if len(ids) >= retmax:
                    break
            return ids
        except Exception:
            return []

    def fetch_pmc_xml(self, pmcid: str) -> Optional[str]:
        pmcid = str(pmcid).strip()
        numeric = re.sub(r'^PMC', '', pmcid, flags=re.I)
        try:
            return HTTP.get_text(f'{self.BASE}/efetch.fcgi', self.params({'db': 'pmc', 'id': numeric, 'retmode': 'xml'}))
        except Exception:
            return None


class EuropePMCClient:
    BASE = 'https://www.ebi.ac.uk/europepmc/webservices/rest'

    def search(self, query: str, page_size: int = 50) -> List[Dict[str, Any]]:
        data = HTTP.get_json(f'{self.BASE}/search', {'query': query, 'format': 'json', 'pageSize': page_size, 'resultType': 'core'})
        cache_raw('europe_pmc', query, data)
        out = []
        for r in data.get('resultList', {}).get('result', []) or []:
            doi = r.get('doi')
            pmid = r.get('pmid')
            pmcid = r.get('pmcid')
            urls = []
            if pmid:
                urls.append(f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/')
            if doi:
                urls.append(f'https://doi.org/{doi}')
            if pmcid:
                urls.append(f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/')
            out.append({
                'source_primary': 'europe_pmc', 'sources': ['europe_pmc'], 'source_ids': {'europe_pmc': r.get('id'), 'pmid': pmid},
                'pmid': pmid, 'pmcid': pmcid, 'doi': doi,
                'title': r.get('title'), 'abstract': r.get('abstractText') or '',
                'journal': r.get('journalTitle') or r.get('bookOrReportDetails'), 'venue': r.get('journalTitle'),
                'year': r.get('pubYear'), 'authors': [a.strip() for a in str(r.get('authorString') or '').split(',') if a.strip()][:20],
                'publication_types': ensure_list(r.get('pubType')), 'urls': urls,
                'is_open_access': r.get('isOpenAccess'), 'raw_source_files': [],
            })
        return out

    def fulltext_xml(self, pmcid: str) -> Optional[str]:
        if not pmcid:
            return None
        # Europe PMC accepts /PMC/<PMCID>/fullTextXML for PMC records.
        try:
            return HTTP.get_text(f'{self.BASE}/PMC/{pmcid}/fullTextXML')
        except Exception:
            try:
                no_pmc = re.sub(r'^PMC', '', pmcid, flags=re.I)
                return HTTP.get_text(f'{self.BASE}/PMC/{no_pmc}/fullTextXML')
            except Exception:
                return None


class CrossrefClient:
    BASE = 'https://api.crossref.org/works'

    def search(self, query: str, rows: int = 30) -> List[Dict[str, Any]]:
        mailto = os.getenv('CROSSREF_MAILTO') or os.getenv('PUBMED_TOOL_EMAIL')
        params = {'query.bibliographic': query, 'rows': rows}
        if mailto:
            params['mailto'] = mailto
        data = HTTP.get_json(self.BASE, params)
        cache_raw('crossref', query, data)
        out = []
        for item in data.get('message', {}).get('items', []) or []:
            title = (item.get('title') or [''])[0]
            doi = item.get('DOI')
            year = None
            parts = item.get('published-print', {}).get('date-parts') or item.get('published-online', {}).get('date-parts') or item.get('created', {}).get('date-parts')
            if parts and parts[0]:
                year = parts[0][0]
            authors = []
            for a in item.get('author', [])[:20]:
                authors.append(' '.join([a.get('given',''), a.get('family','')]).strip())
            urls = [item.get('URL')] if item.get('URL') else []
            if doi:
                urls.append(f'https://doi.org/{doi}')
            out.append({
                'source_primary': 'crossref', 'sources': ['crossref'], 'source_ids': {'crossref': item.get('DOI')},
                'doi': doi, 'title': title, 'abstract': item.get('abstract') or '',
                'journal': (item.get('container-title') or [''])[0], 'venue': (item.get('container-title') or [''])[0],
                'year': year, 'authors': authors, 'urls': [u for u in urls if u],
                'license': item.get('license'), 'raw_source_files': [],
            })
        return out


class OpenAlexClient:
    BASE = 'https://api.openalex.org/works'

    def search(self, query: str, rows: int = 30) -> List[Dict[str, Any]]:
        params = {'search': query, 'per-page': min(rows, 200)}
        mailto = os.getenv('OPENALEX_MAILTO') or os.getenv('PUBMED_TOOL_EMAIL')
        if mailto:
            params['mailto'] = mailto
        data = HTTP.get_json(self.BASE, params)
        cache_raw('openalex', query, data)
        out = []
        for item in data.get('results', []) or []:
            doi = item.get('doi')
            if doi:
                doi = doi.replace('https://doi.org/', '')
            ids = item.get('ids') or {}
            pmid = ids.get('pmid', '').replace('https://pubmed.ncbi.nlm.nih.gov/', '').strip('/') if ids.get('pmid') else None
            pmcid = ids.get('pmcid', '').replace('https://www.ncbi.nlm.nih.gov/pmc/articles/', '').strip('/') if ids.get('pmcid') else None
            abstract = inverted_index_to_text(item.get('abstract_inverted_index'))
            authors = []
            for au in item.get('authorships', [])[:20]:
                name = ((au.get('author') or {}).get('display_name'))
                if name:
                    authors.append(name)
            venue = ((item.get('primary_location') or {}).get('source') or {}).get('display_name')
            urls = [item.get('id')]
            landing = ((item.get('primary_location') or {}).get('landing_page_url'))
            pdf = ((item.get('primary_location') or {}).get('pdf_url'))
            for u in [landing, pdf, ids.get('doi'), ids.get('pmid'), ids.get('pmcid')]:
                if u:
                    urls.append(u)
            out.append({
                'source_primary': 'openalex', 'sources': ['openalex'], 'source_ids': {'openalex': item.get('id')},
                'openalex_id': item.get('id'), 'doi': doi, 'pmid': pmid, 'pmcid': pmcid,
                'title': item.get('title') or item.get('display_name'), 'abstract': abstract,
                'journal': venue, 'venue': venue, 'year': item.get('publication_year'), 'authors': authors,
                'urls': [u for u in urls if u], 'is_open_access': (item.get('open_access') or {}).get('is_oa'),
                'cited_by_count': item.get('cited_by_count'), 'raw_source_files': [],
            })
        return out


def inverted_index_to_text(idx: Any) -> str:
    if not isinstance(idx, dict):
        return ''
    words = []
    for word, positions in idx.items():
        for pos in positions:
            words.append((pos, word))
    return ' '.join(w for _, w in sorted(words))


class SemanticScholarClient:
    BASE = 'https://api.semanticscholar.org/graph/v1'

    def headers(self) -> Dict[str, str]:
        key = os.getenv('SEMANTIC_SCHOLAR_API_KEY') or os.getenv('S2_API_KEY')
        return {'x-api-key': key} if key else {}

    def search(self, query: str, rows: int = 20) -> List[Dict[str, Any]]:
        fields = 'paperId,title,abstract,year,authors,url,venue,externalIds,openAccessPdf,citationCount,referenceCount'
        data = HTTP.get_json(f'{self.BASE}/paper/search', {'query': query, 'limit': min(rows, 100), 'fields': fields}, headers=self.headers())
        cache_raw('semantic_scholar', query, data)
        out = []
        for p in data.get('data', []) or []:
            ext = p.get('externalIds') or {}
            doi = ext.get('DOI')
            pmid = ext.get('PubMed')
            pmcid = ext.get('PubMedCentral')
            urls = [p.get('url')]
            pdf = (p.get('openAccessPdf') or {}).get('url')
            if pdf:
                urls.append(pdf)
            if doi:
                urls.append(f'https://doi.org/{doi}')
            out.append({
                'source_primary': 'semantic_scholar', 'sources': ['semantic_scholar'], 'source_ids': {'semantic_scholar': p.get('paperId')},
                'semantic_scholar_id': p.get('paperId'), 'doi': doi, 'pmid': pmid, 'pmcid': pmcid,
                'title': p.get('title'), 'abstract': p.get('abstract') or '', 'journal': p.get('venue'), 'venue': p.get('venue'),
                'year': p.get('year'), 'authors': [(a or {}).get('name') for a in p.get('authors', []) if (a or {}).get('name')],
                'urls': [u for u in urls if u], 'citation_count': p.get('citationCount'), 'reference_count': p.get('referenceCount'),
                'raw_source_files': [],
            })
        return out

    def citations_and_references(self, paper_id: str, rows: int = 20) -> List[Dict[str, Any]]:
        fields = 'title,abstract,year,authors,url,venue,externalIds,openAccessPdf,citationCount'
        out = []
        for kind in ['citations', 'references']:
            try:
                data = HTTP.get_json(f'{self.BASE}/paper/{urllib.parse.quote(paper_id, safe=":")}/{kind}', {'limit': min(rows, 100), 'fields': fields}, headers=self.headers())
                cache_raw('semantic_scholar', f'{kind}_{paper_id}', data)
                for item in data.get('data', []) or []:
                    p = item.get('citingPaper') or item.get('citedPaper') or {}
                    ext = p.get('externalIds') or {}
                    out.append({
                        'source_primary': f'semantic_scholar_{kind}', 'sources': [f'semantic_scholar_{kind}'],
                        'source_ids': {'semantic_scholar': p.get('paperId')},
                        'semantic_scholar_id': p.get('paperId'), 'doi': ext.get('DOI'), 'pmid': ext.get('PubMed'), 'pmcid': ext.get('PubMedCentral'),
                        'title': p.get('title'), 'abstract': p.get('abstract') or '', 'journal': p.get('venue'), 'venue': p.get('venue'),
                        'year': p.get('year'), 'authors': [(a or {}).get('name') for a in p.get('authors', []) if (a or {}).get('name')],
                        'urls': [u for u in [p.get('url'), ((p.get('openAccessPdf') or {}).get('url'))] if u],
                        'raw_source_files': [],
                    })
            except Exception:
                continue
        return out


class GitHubClient:
    BASE = 'https://api.github.com'

    def headers(self) -> Dict[str, str]:
        h = {'Accept': 'application/vnd.github+json'}
        token = os.getenv('GITHUB_TOKEN')
        if token:
            h['Authorization'] = f'Bearer {token}'
        return h

    def search_repositories(self, query: str, rows: int = 20) -> List[Dict[str, Any]]:
        try:
            data = HTTP.get_json(f'{self.BASE}/search/repositories', {'q': query, 'per_page': min(rows, 100), 'sort': 'best-match'}, headers=self.headers())
            cache_raw('github', query, data)
        except Exception as e:
            print(f'    ⚠️ GitHub 搜索失败：{e}')
            return []
        out = []
        for item in data.get('items', []) or []:
            rec = {
                'name': item.get('full_name'), 'url': item.get('html_url'), 'description': item.get('description'),
                'stars': item.get('stargazers_count'), 'forks': item.get('forks_count'),
                'language': item.get('language'), 'updated_at': item.get('updated_at'),
                'source': 'github', 'query': query,
            }
            out.append(rec)
        return out


# ------------------------- GitHub missing-link enrichment -------------------------
def extract_github_url_from_any(v: Any) -> str:
    if v is None:
        return ''
    if isinstance(v, dict):
        for k in ['code_repository_url', 'repository_url', 'repo_url', 'url', 'html_url', 'github_url']:
            u = extract_github_url_from_any(v.get(k))
            if u:
                return u
        return ''
    if isinstance(v, (list, tuple, set)):
        for x in v:
            u = extract_github_url_from_any(x)
            if u:
                return u
        return ''
    m = GITHUB_RE.search(str(v))
    return m.group(0).rstrip('.,;') if m else ''


def model_has_github_repo(model: Dict[str, Any]) -> bool:
    if not isinstance(model, dict):
        return False
    for k in ['code_repository_url', 'repository_url', 'repo_url', 'url', 'github_url', 'evidence', 'dataset_source_or_link']:
        if extract_github_url_from_any(model.get(k)):
            return True
    return False


def collect_model_rows_for_github_search_from_evidence(evidence_batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in ensure_list(evidence_batches):
        if not isinstance(ev, dict):
            continue
        for m in ensure_list(ev.get('models')) + ensure_list(ev.get('all_candidate_models')) + ensure_list(ev.get('benchmark_ready_models')):
            if isinstance(m, dict):
                rows.append(m)
        for r in ensure_list(ev.get('repositories')):
            if isinstance(r, dict) and model_name_from_item(r):
                rows.append(r)
    return dedupe_models_by_name(rows)


def collect_model_rows_for_github_search_from_compact(compact_pool: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sm in ensure_list(compact_pool.get('chunk_summaries')):
        if not isinstance(sm, dict):
            continue
        for m in ensure_list(sm.get('models')) + ensure_list(sm.get('all_candidate_models')) + ensure_list(sm.get('benchmark_ready_models')):
            if isinstance(m, dict):
                rows.append(canonical_model_from_summary(m, sm) or m)
        for r in ensure_list(sm.get('repositories')):
            if isinstance(r, dict) and model_name_from_item(r):
                rows.append(r)
    for m in ensure_list(compact_pool.get('all_candidate_models')) + ensure_list(compact_pool.get('benchmark_ready_models')):
        if isinstance(m, dict):
            rows.append(m)
    return dedupe_models_by_name(rows)


NON_MODEL_GITHUB_ENRICHMENT_TERMS = {
    # Generic ML/statistical packages or explainability libraries. They can be dependencies,
    # but they are not deployable AMP models and should not consume GitHub search slots.
    'lightgbm', 'xgboost', 'random forest', 'randomforest', 'svm', 'support vector machine',
    'shap', 'treexplainer', 'treexplainer study', 'treexplainerstudy', 'scikit learn',
    'sklearn', 'tensorflow', 'pytorch', 'keras', 'numpy', 'pandas', 'matplotlib',
    'catboost', 'logistic regression', 'knn', 'k nearest neighbor', 'naive bayes',
}


def should_skip_github_enrichment_model(name: str, model: Dict[str, Any]) -> bool:
    if not name or len(name.strip()) < 3:
        return True
    name_key = normalize_key(name)
    name_compact = re.sub(r'[^a-z0-9]+', '', name_key)
    low = normalize_key(' '.join([name, str(model.get('task_type','')), str(model.get('blocking_issues','')), str(model.get('method_family',''))]))
    if name_key in {'amp', 'camp', 'dbaasp', 'apd', 'apd3', 'dramp', 'uniprot', 'not provided in evidence', 'not_provided_in_evidence'}:
        return True
    if name_key in NON_MODEL_GITHUB_ENRICHMENT_TERMS or name_compact in {re.sub(r'[^a-z0-9]+', '', x) for x in NON_MODEL_GITHUB_ENRICHMENT_TERMS}:
        return True
    if any(pat in low for pat in ['unnamed ', 'multiple ', 'various ', 'predictive and interpretable', 'collaborative filtering and link prediction model']):
        return True
    # Skip items whose task/method context strongly indicates they are generic algorithms/tools,
    # not named AMP predictors.
    if name_key in {'light gbm', 'tree explainer study'}:
        return True
    return False


# v4.9: GitHub 补链使用通用别名归一化 + fuzzy repo-name matching。
# 目标：解决不止 AMP Scanner v2 的问题，而是所有类似命名差异：
#   AMP Scanner v2 <-> amp-scanner-v2 <-> AMPScannerV2 <-> AMPScanner vr.2
#   Co-AMPpred <-> CoAMPpred
#   iAMP-SeE <-> iAMPSeE
#   sAMPpred-GAT <-> sAMP-pred-GAT
#   Deep-AmPEP30 <-> DeepAmPEP30
#   E-CLEAP <-> ECLEAP
GITHUB_MIN_CANDIDATE_SCORE = 0.15
GITHUB_HIGH_CONFIDENCE_SCORE = 0.70
GITHUB_MEDIUM_CONFIDENCE_SCORE = 0.45

KNOWN_GITHUB_REPO_HINTS: Dict[str, Dict[str, Any]] = {
    # 用户确认的正确仓库。作为补链 fallback 使用，仍会标记 source=github_known_hint，便于追溯。
    'amp scanner v2': {
        'name': 'dan-veltri/amp-scanner-v2',
        'url': 'https://github.com/dan-veltri/amp-scanner-v2',
        'description': 'AMP Scanner v2 repository; user-confirmed target repository for Antimicrobial Peptide Scanner v2.',
        'stars': 13,
        'language': None,
        'source': 'github_known_hint',
        'query': 'known_hint:AMP Scanner v2',
    },
    'antimicrobial peptide scanner v2': {
        'name': 'dan-veltri/amp-scanner-v2',
        'url': 'https://github.com/dan-veltri/amp-scanner-v2',
        'description': 'AMP Scanner v2 repository; user-confirmed target repository for Antimicrobial Peptide Scanner v2.',
        'stars': 13,
        'language': None,
        'source': 'github_known_hint',
        'query': 'known_hint:AMP Scanner v2',
    },
}

# 常见 AMP 模型的命名变体。这里不是硬编码“最终仓库”，只扩展搜索词和匹配别名。
# 真正链接仍来自 GitHub 搜索结果或已知 hint。
MODEL_ALIAS_OVERRIDES: Dict[str, List[str]] = {
    'amp scanner v2': ['AMP Scanner v2', 'AMPScanner V2', 'AMPScannerV2', 'AMPScanner vr.2', 'Antimicrobial Peptide Scanner v2', 'amp-scanner-v2', 'ampscanner-v2'],
    'antimicrobial peptide scanner v2': ['AMP Scanner v2', 'AMPScanner V2', 'AMPScannerV2', 'AMPScanner vr.2', 'amp-scanner-v2'],
    'co amppred': ['Co-AMPpred', 'CoAMPpred', 'Co AMPpred', 'coamppred'],
    'amp bert': ['AMP-BERT', 'AMPBERT', 'amp-bert'],
    'iam see': ['iAMP-SeE', 'iAMPSeE', 'iamp-see', 'iampsee'],
    'iamp see': ['iAMP-SeE', 'iAMPSeE', 'iamp-see', 'iampsee'],
    'samp pred gat': ['sAMPpred-GAT', 'sAMP-pred-GAT', 'samp-pred-gat', 'samppred-gat'],
    'samppred gat': ['sAMPpred-GAT', 'sAMP-pred-GAT', 'samp-pred-gat', 'samppred-gat'],
    'deep ampep30': ['Deep-AmPEP30', 'DeepAmPEP30', 'deep-ampep30', 'deepampep30'],
    'e cleap': ['E-CLEAP', 'ECLEAP', 'e-cleap', 'ecleap'],
    'ecleap': ['E-CLEAP', 'ECLEAP', 'e-cleap'],
    'uniprolcad': ['UniproLcad', 'UniProLcad', 'uniprolcad'],
    'iamp dl': ['iAMP-DL', 'iAMPDL', 'iamp-dl', 'iampdl'],
    'anti bp3': ['AntiBP3', 'Anti-BP3', 'antibp3'],
    'calcamp': ['CalcAMP', 'calc-amp', 'calcamp'],
    'amp z gsm': ['AMP-zGSM', 'AMP zGSM', 'amp-zgsm', 'ampzgsm'],
    'amp rnnpro': ['AMP-RNNpro', 'AMPRNNpro', 'amp-rnnpro'],
    'amp species specific': ['AMPSpeciesSpecific', 'AMP-Species-Specific', 'amp-species-specific'],
    'ampban': ['AMPBAN', 'AMP-BAN', 'ampban'],
}


def _split_camel_for_github(s: str) -> str:
    s = str(s or '')
    # Split lower->upper and acronym->word boundaries: AMPScannerV2 -> AMP Scanner V2.
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    s = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', s)
    s = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', s)
    return s


def github_name_key(s: str) -> str:
    """Lowercase model/repo name with punctuation and camelCase normalized."""
    s = _split_camel_for_github(str(s or ''))
    s = s.replace('_', ' ').replace('-', ' ').replace('/', ' ')
    s = re.sub(r'\bvr\s*\.?\s*(\d+)\b', r'v\1', s, flags=re.I)
    s = re.sub(r'\bversion\s*(\d+)\b', r'v\1', s, flags=re.I)
    s = re.sub(r'[^A-Za-z0-9]+', ' ', s).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def github_compact_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', github_name_key(s))


def github_slug_key(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', github_name_key(s)).strip('-')


def github_model_tokens(s: str) -> List[str]:
    key = github_name_key(s)
    stop = {'the', 'model', 'predictor', 'prediction', 'classification', 'classifier', 'framework', 'tool', 'server', 'method'}
    toks = [t for t in key.split() if len(t) >= 2 and t not in stop]
    # Keep version tokens and AMP tokens because they are important for models such as AMP Scanner v2.
    return toks


def github_alias_variants(model_name: str) -> List[str]:
    """Return broad aliases for GitHub search/scoring.

    This function is intentionally generic. It builds variants from punctuation,
    camelCase, compact form, hyphen slug, version markers, and curated common AMP aliases.
    """
    raw = str(model_name or '').strip().strip('"')
    can = canonicalize_model_name(raw)
    base_items = [raw, can, _split_camel_for_github(raw), _split_camel_for_github(can)]
    out: List[str] = []
    for item in base_items:
        item = str(item or '').strip()
        if not item:
            continue
        key = github_name_key(item)
        compact = github_compact_key(item)
        slug = github_slug_key(item)
        out.extend([item, key, compact, slug])
        # Version forms.
        if re.search(r'\bv\s*\d+\b', key):
            out.append(re.sub(r'\bv\s*(\d+)\b', r'v\1', key))
            out.append(re.sub(r'\bv\s*(\d+)\b', r'vr.\1', key))
            out.append(re.sub(r'\bv\s*(\d+)\b', r'version \1', key))
        # AMP hyphen forms.
        out.append(re.sub(r'\bamp\b\s+', 'AMP-', key, flags=re.I))
        out.append(key.replace(' amp ', ' AMP '))
    key = github_name_key(can or raw)
    compact = github_compact_key(can or raw)
    # Add override variants if any override key is contained either way.
    for ok, aliases in MODEL_ALIAS_OVERRIDES.items():
        ok_key = github_name_key(ok)
        ok_compact = github_compact_key(ok)
        if ok_key and (ok_key in key or key in ok_key or ok_compact == compact or (ok_compact and ok_compact in compact) or (compact and compact in ok_compact)):
            out.extend(aliases)
    # Generate token-joined variants, e.g. i amp see -> iampsee, amp scanner v2 -> amp-scanner-v2.
    toks = github_model_tokens(can or raw)
    if toks:
        out.extend([''.join(toks), '-'.join(toks), '_'.join(toks), ' '.join(toks)])
        if toks[0] == 'amp' and len(toks) >= 2:
            out.extend(['AMP-' + '-'.join(toks[1:]), 'AMP' + ''.join(toks[1:])])
    seen: set = set()
    clean: List[str] = []
    for x in out:
        x = str(x or '').strip().strip('-_')
        if not x or len(x) < 3:
            continue
        k = github_name_key(x) + '|' + github_compact_key(x)
        if k in seen:
            continue
        seen.add(k)
        clean.append(x)
    return clean[:60]


def github_search_queries_for_model(model_name: str) -> List[str]:
    aliases = github_alias_variants(model_name)
    queries: List[str] = []
    # Exact/slug/name-only queries first. These are critical for repositories whose
    # description is sparse but repo slug is exact.
    for a in aliases[:20]:
        a = a.strip()
        if not a:
            continue
        slug = github_slug_key(a)
        compact = github_compact_key(a)
        queries.append(f'{slug} in:name')
        queries.append(f'{compact} in:name')
        queries.append(f'"{a}"')
        if '-' in a or re.search(r'v\d+', a.lower()):
            queries.append(a)
    # Bio-context searches.
    for a in aliases[:12]:
        queries.append(f'"{a}" antimicrobial peptide')
        queries.append(f'"{a}" AMP prediction')
        queries.append(f'{github_slug_key(a)} antimicrobial peptide')
    # Fallback broader searches: model name without strict phrase, plus GitHub-style terms.
    raw_key = github_name_key(model_name)
    raw_slug = github_slug_key(model_name)
    raw_compact = github_compact_key(model_name)
    queries.extend([
        raw_key,
        raw_slug,
        raw_compact,
        f'{raw_key} github',
        f'{raw_key} peptide predictor',
        f'{raw_key} AMP classifier',
    ])
    seen: set = set()
    out: List[str] = []
    for q in queries:
        q = re.sub(r'\s+', ' ', str(q).strip())
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
    # More queries but still bounded to avoid API overuse.
    return out[:32]


def _repo_short_name(repo: Dict[str, Any]) -> str:
    full = str(repo.get('name') or '').strip()
    return full.split('/')[-1] if '/' in full else full


def _token_overlap_score(a_tokens: List[str], b_text: str) -> float:
    if not a_tokens:
        return 0.0
    b_key = github_name_key(b_text)
    b_compact = github_compact_key(b_text)
    hits = 0
    weighted_total = 0.0
    weighted_hits = 0.0
    for t in a_tokens:
        weight = 1.0
        if t == 'amp' or re.fullmatch(r'v\d+', t):
            weight = 1.25
        weighted_total += weight
        if t in b_key.split() or t in b_key or t in b_compact:
            hits += 1
            weighted_hits += weight
    return weighted_hits / max(weighted_total, 1e-6)


def github_repo_match_score(model_name: str, repo: Dict[str, Any]) -> float:
    aliases = github_alias_variants(model_name)
    alias_keys = [github_name_key(a) for a in aliases if a]
    alias_compacts = [github_compact_key(a) for a in aliases if a]
    alias_slugs = [github_slug_key(a) for a in aliases if a]
    repo_full_raw = str(repo.get('name') or '')
    repo_short_raw = _repo_short_name(repo)
    repo_full = github_name_key(repo_full_raw)
    repo_short = github_name_key(repo_short_raw)
    url = github_name_key(repo.get('url') or '')
    desc = github_name_key(repo.get('description') or '')
    text_raw = ' '.join([str(repo.get('name','')), str(repo.get('description','')), str(repo.get('url',''))])
    text = github_name_key(text_raw)
    compact_repo_short = github_compact_key(repo_short_raw)
    compact_repo_full = github_compact_key(repo_full_raw)
    compact_text = github_compact_key(text_raw)
    slug_repo_short = github_slug_key(repo_short_raw)

    score = 0.0
    # Strong repo-name evidence. Exact slug/compact repo short name should be high confidence
    # even when description is empty and star count is low.
    if any(c and c == compact_repo_short for c in alias_compacts):
        score += 0.88
    elif any(sl and sl == slug_repo_short for sl in alias_slugs):
        score += 0.86
    elif any(c and (c in compact_repo_short or compact_repo_short in c) for c in alias_compacts if len(c) >= 5):
        score += 0.74
    elif any(k and k in repo_short for k in alias_keys if len(k) >= 5):
        score += 0.70
    elif any(c and c in compact_repo_full for c in alias_compacts if len(c) >= 5):
        score += 0.64
    elif any(c and c in compact_text for c in alias_compacts if len(c) >= 5):
        score += 0.46

    # Fuzzy-ish token overlap on repo short name and full text.
    primary = canonicalize_model_name(model_name)
    tokens = github_model_tokens(primary)
    repo_overlap = _token_overlap_score(tokens, repo_short_raw)
    text_overlap = _token_overlap_score(tokens, text_raw)
    if repo_overlap >= 0.999 and len(tokens) >= 2:
        score += 0.36
    elif repo_overlap >= 0.75:
        score += 0.26
    elif repo_overlap >= 0.50:
        score += 0.16
    elif text_overlap >= 0.75:
        score += 0.16
    elif text_overlap >= 0.50:
        score += 0.10

    # Bio/implementation context. Not mandatory if repo-name match is exact, but it helps
    # disambiguate weak/short model names.
    if any(k in text for k in ['antimicrobial peptide', 'antimicrobial peptides', 'amp prediction', 'amp predictor', 'peptide prediction', 'amp classifier', 'amp scanner']):
        score += 0.18
    elif 'amp' in text and any(k in text for k in ['peptide', 'predict', 'scanner', 'classifier', 'antimicrobial']):
        score += 0.12
    if any(k in text for k in ['dataset', 'data', 'benchmark', 'model', 'predictor', 'classification', 'trained', 'weights', 'server', 'inference']):
        score += 0.10

    # If repo owner/name appears in url exactly like a known alias, reward URL evidence.
    if any(c and c in compact_text for c in alias_compacts if len(c) >= 8):
        score += 0.05
    # User-confirmed hint for AMP Scanner v2; kept as a narrow additional reward only.
    if 'github com dan veltri amp scanner v2' in url or 'dan veltri amp scanner v2' in text or 'dan veltri ampscannerv2' in github_compact_key(text_raw):
        score += 0.25

    try:
        stars = int(repo.get('stars') or 0)
        if stars >= 50:
            score += 0.08
        elif stars >= 10:
            score += 0.05
        elif stars >= 1:
            score += 0.02
    except Exception:
        pass

    # Penalize very generic model names unless repo/name evidence is strong.
    generic_compacts = {'amp', 'camp', 'apd', 'apd3', 'dramp', 'dbaasp'}
    if github_compact_key(model_name) in generic_compacts and score < 0.75:
        score *= 0.35

    return round(min(score, 1.0), 3)

def github_confidence_label(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if s >= GITHUB_HIGH_CONFIDENCE_SCORE:
        return 'high_confidence_repo'
    if s >= GITHUB_MEDIUM_CONFIDENCE_SCORE:
        return 'medium_confidence_repo'
    if s >= GITHUB_MIN_CANDIDATE_SCORE:
        return 'low_confidence_repo'
    return 'no_candidate'


def load_existing_github_enrichment() -> Dict[str, List[Dict[str, Any]]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    obj = read_json(GITHUB_MISSING_MODEL_ENRICHMENT_JSON, [])
    rows: List[Any] = obj if isinstance(obj, list) else ensure_list(obj.get('items') if isinstance(obj, dict) else obj)
    if not rows:
        rows = read_jsonl(GITHUB_MISSING_MODEL_ENRICHMENT_JSONL)
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get('model_name') or row.get('matched_model_name')
        key = normalize_key(canonicalize_model_name(name))
        if key:
            by_model.setdefault(key, []).append(row)
    return by_model


def write_github_enrichment(rows: List[Dict[str, Any]]) -> None:
    rows = dedupe_objects(rows, 'github_missing_model_enrichment')
    write_json(GITHUB_MISSING_MODEL_ENRICHMENT_JSON, rows)
    # v5.0: overwrite cache JSONL rather than append duplicate rows every run.
    write_jsonl(GITHUB_MISSING_MODEL_ENRICHMENT_JSONL, rows)


def github_enrichment_success_rows(rows: List[Dict[str, Any]], min_score: float = GITHUB_MEDIUM_CONFIDENCE_SCORE) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in ensure_list(rows):
        if not isinstance(r, dict):
            continue
        url = extract_github_url_from_any(r.get('url') or r.get('html_url') or r.get('repository_url') or r.get('code_repository_url'))
        if not url:
            continue
        try:
            score = float(r.get('match_score') or 0.0)
        except Exception:
            score = 0.0
        label = str(r.get('confidence_label') or '')
        if score >= min_score or label in {'high_confidence_repo', 'medium_confidence_repo'}:
            out.append(r)
    return out


def github_enrichment_has_success(rows: List[Dict[str, Any]], min_score: float = GITHUB_MEDIUM_CONFIDENCE_SCORE) -> bool:
    return bool(github_enrichment_success_rows(rows, min_score=min_score))


def write_github_enrichment_run_report(report: Dict[str, Any]) -> None:
    """Persist a human-readable report for the latest GitHub enrichment planning step.

    v5.1: This exists because the old log line could be misread as "the previous
    80 models were not saved". In reality, successful cached models are skipped,
    and the printed pending count can be the next uncached/failed batch.
    """
    try:
        write_json(GITHUB_ENRICHMENT_RUN_REPORT_JSON, report)
    except Exception:
        pass
    try:
        pending = ensure_list(report.get('pending_model_names'))
        lines = [
            '# GitHub enrichment pending models',
            f'generated_at: {report.get("time")}',
            f'total_unique_model_names: {report.get("total_unique_model_names")}',
            f'cached_success_skipped: {report.get("cached_success_skipped")}',
            f'cached_other_retained: {report.get("cached_other_retained")}',
            f'skipped_already_has_repo: {report.get("skipped_already_has_repo")}',
            f'skipped_non_model_terms: {report.get("skipped_non_model_terms")}',
            f'pending_before_limit: {report.get("pending_before_limit")}',
            f'actual_search_this_run: {report.get("actual_search_this_run")}',
            '',
            '## Pending model names searched in this run',
        ]
        lines.extend([str(x) for x in pending])
        GITHUB_ENRICHMENT_PENDING_MODELS_TXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    except Exception:
        pass


def search_github_for_missing_model_repos(models: List[Dict[str, Any]], max_models: int = 80, repos_per_model: int = 3, force: bool = False, refresh_all: bool = False) -> List[Dict[str, Any]]:
    gh = GitHubClient()
    prior = load_existing_github_enrichment()
    rows: List[Dict[str, Any]] = []
    names: List[Tuple[str, Dict[str, Any]]] = []
    seen: set = set()
    total_unique_model_names = 0
    cached_success_count = 0
    cached_other_count = 0
    skipped_already_has_repo_count = 0
    skipped_non_model_count = 0
    cached_success_names: List[str] = []
    cached_other_names: List[str] = []
    skipped_already_has_repo_names: List[str] = []
    skipped_non_model_names: List[str] = []
    for m in ensure_list(models):
        if not isinstance(m, dict):
            continue
        name = canonicalize_model_name(model_name_from_item(m))
        key = normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        total_unique_model_names += 1
        if should_skip_github_enrichment_model(name, m):
            skipped_non_model_count += 1
            skipped_non_model_names.append(name)
            continue
        prior_rows = prior.get(key, [])
        # v5.1: previously successful enrichment is treated as imported evidence.
        # Even with --force-github-enrichment, high/medium-confidence cached repos are not re-searched
        # unless --refresh-all-github-enrichment is explicitly requested. The pending count below is
        # only the remaining uncached/failed models selected for this run, not the old successful batch.
        if prior_rows and github_enrichment_has_success(prior_rows) and not refresh_all:
            rows.extend(prior_rows)
            cached_success_count += 1
            cached_success_names.append(name)
            continue
        # If there is any prior record and the user did not force, keep it and do not re-query.
        if prior_rows and not force:
            rows.extend(prior_rows)
            cached_other_count += 1
            cached_other_names.append(name)
            continue
        if model_has_github_repo(m) and not force:
            skipped_already_has_repo_count += 1
            skipped_already_has_repo_names.append(name)
            continue
        names.append((name, m))
    pending_before_limit = len(names)
    names = names[:max_models]
    report = {
        'time': now_str(),
        'total_unique_model_names': total_unique_model_names,
        'cached_success_skipped': cached_success_count,
        'cached_success_names': cached_success_names,
        'cached_other_retained': cached_other_count,
        'cached_other_names': cached_other_names,
        'skipped_already_has_repo': skipped_already_has_repo_count,
        'skipped_already_has_repo_names': skipped_already_has_repo_names,
        'skipped_non_model_terms': skipped_non_model_count,
        'skipped_non_model_names': skipped_non_model_names,
        'pending_before_limit': pending_before_limit,
        'actual_search_this_run': len(names),
        'max_models_limit': max_models,
        'force_github_enrichment': force,
        'refresh_all_github_enrichment': refresh_all,
        'pending_model_names': [n for n, _ in names],
    }
    write_github_enrichment_run_report(report)
    print(f'>>> GitHub 补链缓存：候选模型总数 {total_unique_model_names}；已跳过成功缓存 {cached_success_count} 个；保留其他缓存 {cached_other_count} 个；跳过已有 GitHub {skipped_already_has_repo_count} 个；过滤非模型/通用工具 {skipped_non_model_count} 个。', flush=True)
    if pending_before_limit:
        print(f'>>> GitHub 补链：剩余待搜索模型 {pending_before_limit} 个；本轮实际搜索 {len(names)} 个（上限 --github-enrich-max-models={max_models}）。', flush=True)
        print(f'>>> GitHub 补链：本轮待搜名单已写入 {GITHUB_ENRICHMENT_PENDING_MODELS_TXT.relative_to(ROOT)}；运行报告写入 {GITHUB_ENRICHMENT_RUN_REPORT_JSON.relative_to(ROOT)}。', flush=True)
    else:
        print('>>> GitHub 补链：没有新的无 GitHub 模型需要搜索，或已存在缓存。', flush=True)
    for idx, (name, model) in enumerate(names, 1):
        queries = github_search_queries_for_model(name)
        candidate_repos: List[Dict[str, Any]] = []
        seen_urls: set = set()
        for q in queries:
            res = gh.search_repositories(q, rows=max(10, repos_per_model * 5))
            for repo in res:
                url = repo.get('url')
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                score = github_repo_match_score(name, repo)
                if score < GITHUB_MIN_CANDIDATE_SCORE:
                    continue
                rr = dict(repo)
                rr.update({
                    'model_name': name,
                    'matched_model_name': name,
                    'match_score': score,
                    'evidence_level': 'github_search',
                    'repository_type': 'code_candidate',
                    'confidence_label': github_confidence_label(score),
                    'needs_manual_verification': score < GITHUB_HIGH_CONFIDENCE_SCORE,
                    'source': 'github_missing_model_enrichment',
                    'search_time': now_str(),
                    'evidence': f'GitHub repository search result for model name "{name}"; query="{repo.get("query")}"; score={score}; description={trunc(repo.get("description"), 300)}',
                })
                candidate_repos.append(rr)
        candidate_repos.sort(key=lambda x: (float(x.get('match_score') or 0), int(x.get('stars') or 0)), reverse=True)
        selected = candidate_repos[:repos_per_model]
        if not selected:
            # Known/user-confirmed hints are used only as fallback when GitHub search did not return the desired repo.
            hint = None
            for hk, hv in KNOWN_GITHUB_REPO_HINTS.items():
                if hk == normalize_key(name) or hk in normalize_key(name) or normalize_key(name) in hk:
                    hint = dict(hv)
                    break
            if hint:
                score = github_repo_match_score(name, hint)
                hint.update({
                    'model_name': name,
                    'matched_model_name': name,
                    'match_score': score,
                    'confidence_label': github_confidence_label(score),
                    'evidence_level': 'github_known_hint',
                    'repository_type': 'code_candidate',
                    'needs_manual_verification': score < GITHUB_HIGH_CONFIDENCE_SCORE,
                    'source': 'github_missing_model_enrichment',
                    'search_time': now_str(),
                    'evidence': f'Known/user-confirmed GitHub repository hint for model name "{name}"; score={score}; url={hint.get("url")}',
                })
                selected = [hint]
        if selected:
            best = selected[0]
            print(f'    -> [{idx}/{len(names)}] {name}: 找到 {len(selected)} 个候选 GitHub 仓库，best={best.get("url")} score={best.get("match_score")} {best.get("confidence_label")}', flush=True)
            rows.extend(selected)
        else:
            print(f'    -> [{idx}/{len(names)}] {name}: 未找到可保存 GitHub 候选仓库。', flush=True)
            rows.append({'model_name': name, 'matched_model_name': name, 'url': '', 'repository_name': '', 'source': 'github_missing_model_enrichment', 'evidence_level': 'github_search_no_hit', 'match_score': 0.0, 'confidence_label': 'no_candidate', 'needs_manual_verification': True, 'search_time': now_str(), 'queries': queries, 'evidence': f'GitHub search returned no candidate repository above score threshold for model name "{name}".'})
    rows = dedupe_objects(rows, 'github_missing_model_enrichment')
    if rows:
        write_github_enrichment(rows)
    return rows


def github_enrichment_to_evidence_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    repos: List[Dict[str, Any]] = []
    models: List[Dict[str, Any]] = []
    open_questions: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = canonicalize_model_name(r.get('model_name') or r.get('matched_model_name'))
        if not name:
            continue
        url = r.get('url') or r.get('html_url') or ''
        if url:
            repos.append({'name': r.get('name') or r.get('repository_name') or url, 'url': url, 'repository_type': 'code_candidate', 'matched_model_name': name, 'source': 'github_missing_model_enrichment', 'evidence_level': 'github_search', 'match_score': r.get('match_score'), 'stars': r.get('stars'), 'language': r.get('language'), 'description': r.get('description'), 'needs_manual_verification': r.get('needs_manual_verification', True), 'evidence': r.get('evidence')})
            models.append({'model_name': name, 'canonical_name': name, 'code_repository_url': url, 'benchmark_candidate': True, 'candidate_reason': 'GitHub enrichment found a candidate repository by model-name search; requires manual verification before deployment.', 'blocking_issues': ['github_search_candidate_requires_manual_verification'] if r.get('needs_manual_verification', True) else [], 'evidence_level': 'github_search', 'confidence': r.get('match_score'), 'evidence': r.get('evidence')})
        else:
            open_questions.append({'question': f'No confident GitHub repository found for {name}', 'reason': 'github_missing_model_enrichment_no_hit', 'next_action': 'try exact paper title, author name, GitLab, Zenodo, supplementary material, or web server documentation'})
    return {'_batch_no': 'github_missing_model_enrichment', '_stage': 'github_missing_model_enrichment', 'models': models, 'repositories': repos, 'datasets': [], 'metrics': [], 'papers': [], 'important_evidence': ['GitHub enrichment searched model names that lacked repository links before global meeting.', 'Candidate repositories are saved as evidence_level=github_search and must be manually verified before deployment.'], 'open_questions': open_questions}


def add_github_enrichment_to_compact_pool(compact_pool: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return compact_pool
    ev = github_enrichment_to_evidence_batch(rows)
    summary = {'chunk_id': 'github_missing_model_enrichment', 'chunk_type': 'repository_enrichment', 'chunk_name': 'GitHub missing model repository search', 'compression_status': 'deterministic_github_search', 'main_entities': sorted(set([canonicalize_model_name(r.get('model_name') or r.get('matched_model_name')) for r in rows if r.get('model_name') or r.get('matched_model_name')]))[:200], 'models': ev.get('models', []), 'repositories': ev.get('repositories', []), 'datasets': [], 'metrics': [], 'important_evidence': ev.get('important_evidence', []), 'uncertainties': ev.get('open_questions', []), 'source_pmids': [], 'source_dois': [], 'urls': [r.get('url') for r in rows if r.get('url')]}
    summaries = [sm for sm in ensure_list(compact_pool.get('chunk_summaries')) if not (isinstance(sm, dict) and sm.get('chunk_id') == 'github_missing_model_enrichment')]
    summaries.append(summary)
    compact_pool['chunk_summaries'] = summaries
    compact_pool['github_missing_model_enrichment'] = rows
    compact_pool['github_enrichment_enabled'] = True
    compact_pool['github_enrichment_count'] = len(rows)
    compact_pool['chunk_summary_count'] = len(summaries)
    compact_pool['chunk_count'] = len(summaries)
    write_json(COMPACT_EVIDENCE_POOL_JSON, compact_pool)
    append_jsonl(CHUNK_SUMMARIES_JSONL, summary)
    return compact_pool


def enrich_evidence_batches_with_missing_github(evidence_batches: List[Dict[str, Any]], repos: List[Dict[str, Any]], max_models: int = 80, repos_per_model: int = 3, force: bool = False, refresh_all: bool = False) -> Optional[Dict[str, Any]]:
    models = collect_model_rows_for_github_search_from_evidence(evidence_batches)
    rows = search_github_for_missing_model_repos(models, max_models=max_models, repos_per_model=repos_per_model, force=force, refresh_all=refresh_all)
    if not rows:
        return None
    ev = github_enrichment_to_evidence_batch(rows)
    if ev.get('repositories'):
        repos.extend(ensure_list(ev.get('repositories')))
        for r in ensure_list(ev.get('repositories')):
            if isinstance(r, dict):
                append_jsonl(NORMALIZED_REPOS_JSONL, r)
    append_jsonl(FULLTEXT_EVIDENCE_JSONL, ev)
    return ev


def enrich_compact_pool_with_missing_github(compact_pool: Dict[str, Any], max_models: int = 80, repos_per_model: int = 3, force: bool = False, refresh_all: bool = False) -> Dict[str, Any]:
    models = collect_model_rows_for_github_search_from_compact(compact_pool)
    rows = search_github_for_missing_model_repos(models, max_models=max_models, repos_per_model=repos_per_model, force=force, refresh_all=refresh_all)
    if not rows:
        return compact_pool
    return add_github_enrichment_to_compact_pool(compact_pool, rows)


# ------------------------- Qwen-Max web-search enrichment -------------------------
def _qwen_web_candidate_urls(obj: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                urls.extend(URL_RE.findall(v))
            elif isinstance(v, (list, dict)):
                urls.extend(_qwen_web_candidate_urls(v))
    elif isinstance(obj, list):
        for x in obj:
            urls.extend(_qwen_web_candidate_urls(x))
    elif isinstance(obj, str):
        urls.extend(URL_RE.findall(obj))
    out = []
    seen = set()
    for u in urls:
        u = str(u).rstrip('.,;，。；)）]】')
        if u and u not in seen:
            seen.add(u); out.append(u)
    return out


def qwen_web_confidence_label(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if s >= 0.75:
        return 'high_confidence_web_evidence'
    if s >= 0.45:
        return 'medium_confidence_web_evidence'
    if s > 0:
        return 'low_confidence_web_evidence'
    return 'no_candidate'


def qwen_web_model_has_success(rows: List[Dict[str, Any]], min_score: float = 0.45) -> bool:
    for r in ensure_list(rows):
        if not isinstance(r, dict):
            continue
        has_candidate = bool(r.get('repository_candidates') or r.get('dataset_candidates') or r.get('web_server_candidates') or r.get('paper_links') or r.get('weight_candidates'))
        impact = r.get('article_impact') if isinstance(r.get('article_impact'), dict) else {}
        has_impact = bool(_citation_count_from_row(r) > 0 or _impact_factor_from_row(r, None) > 0 or (impact and (_safe_float(impact.get('citation_count'), 0) > 0 or _safe_float(impact.get('journal_impact_factor'), 0) > 0)))
        if has_candidate or has_impact:
            try:
                if float(r.get('confidence') or r.get('match_score') or 0) >= min_score:
                    return True
            except Exception:
                pass
    return False


def load_existing_qwen_web_enrichment() -> Dict[str, List[Dict[str, Any]]]:
    rows = read_jsonl(QWEN_WEB_ENRICHMENT_JSONL)
    if not rows and QWEN_WEB_ENRICHMENT_JSON.exists():
        rows = ensure_list(read_json(QWEN_WEB_ENRICHMENT_JSON, []))
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = canonicalize_model_name(r.get('model_name') or r.get('matched_model_name'))
        if not name:
            continue
        by_model.setdefault(normalize_key(name), []).append(r)
    return by_model


def write_qwen_web_enrichment(rows: List[Dict[str, Any]]) -> None:
    rows = dedupe_objects(rows, 'qwen_web_enrichment')
    write_jsonl(QWEN_WEB_ENRICHMENT_JSONL, rows)
    write_json(QWEN_WEB_ENRICHMENT_JSON, rows)


def write_qwen_web_enrichment_run_report(report: Dict[str, Any]) -> None:
    write_json(QWEN_WEB_ENRICHMENT_RUN_REPORT_JSON, report)
    names = ensure_list(report.get('pending_model_names'))
    QWEN_WEB_ENRICHMENT_PENDING_MODELS_TXT.write_text('\n'.join([str(x) for x in names]) + ('\n' if names else ''), encoding='utf-8')


def _qwen_web_missing_fields(name: str, model: Dict[str, Any]) -> List[str]:
    """Return evidence gaps that Qwen3.7-Max web search should try to fill.

    v5.5 expands web enrichment from "find repo only" to a general
    evidence-completion pass: code, datasets, weights, web server, DOI/PMID,
    citation count, journal and impact factor.  This makes the deployment score
    and final Top 10/20 ranking use fresher web evidence when structured APIs
    did not capture enough metadata.
    """
    missing: List[str] = []
    if should_skip_github_enrichment_model(name, model):
        return missing

    if not model_has_github_repo(model):
        missing.append('code_repository_url')

    ds_text = str(model.get('dataset_source_or_link') or model.get('dataset_url') or model.get('evidence') or '')
    ds_links = extract_links(ds_text).get('dataset_urls')
    if is_missing_value(model.get('dataset_source_or_link')) or not ds_links:
        missing.append('dataset_source_or_link')

    if is_missing_value(model.get('model_weights_url') or model.get('weights_url') or model.get('checkpoint_url')):
        missing.append('model_weights_url')

    if is_missing_value(model.get('web_server_url') or model.get('server_url')):
        missing.append('web_server_url')

    if is_missing_value(model.get('source_doi') or model.get('doi')) and is_missing_value(model.get('source_pmid') or model.get('pmid')):
        missing.append('paper_doi_or_pmid')

    if _citation_count_from_row(model) <= 0:
        missing.append('citation_count')

    if is_missing_value(model.get('source_journal') or model.get('journal') or model.get('venue')):
        missing.append('source_journal')

    # Impact factor can come from the row, local data/journal_impact_factors.*,
    # or Qwen3.7-Max web search.  If we cannot resolve it locally, ask Qwen to look.
    try:
        local_if = _impact_factor_from_row(model, _load_journal_impact_factor_map())
    except Exception:
        local_if = _impact_factor_from_row(model, None)
    if local_if <= 0:
        missing.append('journal_impact_factor')

    issue_text = normalize_key(' '.join([str(x) for x in ensure_list(model.get('blocking_issues'))]))
    if any(k in issue_text for k in ['no code', 'no_code', 'dataset', 'weight', 'not reported', 'unclear', 'needs', 'review only']):
        missing.append('blocking_issue_resolution')

    # Preserve order and de-duplicate.
    out: List[str] = []
    seen = set()
    for x in missing:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def qwen_web_enrichment_should_run_for_model(name: str, model: Dict[str, Any]) -> bool:
    return bool(_qwen_web_missing_fields(name, model))


def qwen_web_prompt_for_model(model_name: str, model: Dict[str, Any]) -> Tuple[str, str]:
    compact = {
        'model_name': model_name,
        'canonical_name': model.get('canonical_name'),
        'task_type': model.get('task_type'),
        'method_family': model.get('method_family'),
        'source_pmid': model.get('source_pmid') or model.get('pmid'),
        'source_doi': model.get('source_doi') or model.get('doi'),
        'known_code_repository_url': model.get('code_repository_url') or model.get('repository_url'),
        'known_web_server_url': model.get('web_server_url'),
        'known_dataset_source_or_link': model.get('dataset_source_or_link'),
        'known_model_weights_url': model.get('model_weights_url') or model.get('weights_url') or model.get('checkpoint_url'),
        'known_source_journal': model.get('source_journal') or model.get('journal') or model.get('venue'),
        'known_citation_count': model.get('citation_count') or model.get('cited_by_count') or model.get('openalex_cited_by_count') or model.get('semantic_scholar_citation_count'),
        'known_journal_impact_factor': model.get('journal_impact_factor') or model.get('impact_factor') or model.get('source_impact_factor'),
        'missing_fields_to_complete': _qwen_web_missing_fields(model_name, model),
        'blocking_issues': model.get('blocking_issues'),
        'evidence': trunc(model.get('evidence') or model.get('candidate_reason') or '', 1200),
    }
    system = """你是 AMP benchmark 项目的联网补漏检索 Agent。你可以使用 Qwen-Max 的联网搜索能力检索公开网页。
目标：为抗菌肽/antimicrobial peptide prediction/classification 模型补充可追溯证据。
只返回 JSON，不要 Markdown，不要解释。不要编造链接；没有把握就返回空数组并写 risk_flags。
优先补全缺失字段：官方 GitHub/GitLab、Zenodo/Figshare/DataCite 数据集、pretrained weights/model weights、web server、论文主页/DOI/PMID、引用量 citation count、期刊名称、期刊影响因子/分区。
引用量和影响因子必须带来源说明：OpenAlex/Semantic Scholar/Crossref/Google Scholar/期刊官网/JCR/Scimago/网页摘要等；找不到就返回 null，不要编造。
必须区分：通用依赖库（LightGBM、SHAP、PyTorch 等）不是 AMP 模型仓库。"""
    user = f"""
请联网搜索并核查下面这个 AMP 模型的公开证据。重点搜索：
1. 官方或高可信代码仓库 GitHub/GitLab
2. 数据集下载链接 Zenodo/Figshare/DataCite/GitHub data
3. pretrained weights / model weights / checkpoint
4. web server / API
5. 原论文 DOI/PMID/标题主页
6. 引用量 citation count / cited_by_count
7. 期刊名称、期刊影响因子或分区信息

模型记录：
{json_dumps(compact, 2)}

请严格返回 JSON 对象，格式如下：
{{
  "model_name": "{model_name}",
  "aliases": ["..."],
  "task_type_guess": "general AMP binary classification / antifungal / anticancer / MIC regression / generation / unclear",
  "repository_candidates": [{{"url":"https://...", "name":"owner/repo", "evidence":"网页摘要或 README 证据", "is_official": true/false/null, "confidence": 0.0}}],
  "dataset_candidates": [{{"url":"https://...", "name":"...", "evidence":"...", "confidence": 0.0}}],
  "weight_candidates": [{{"url":"https://...", "name":"...", "evidence":"...", "confidence": 0.0}}],
  "web_server_candidates": [{{"url":"https://...", "name":"...", "evidence":"...", "confidence": 0.0}}],
  "paper_links": [{{"url":"https://...", "doi":"...", "pmid":"...", "title":"...", "journal":"...", "year":"...", "evidence":"...", "confidence": 0.0}}],
  "article_impact": {{"source_journal":"...", "source_year":"...", "citation_count": null, "citation_source":"OpenAlex/Semantic Scholar/Crossref/Google Scholar/other", "journal_impact_factor": null, "impact_factor_source":"JCR/Scimago/journal page/other", "evidence":"...", "confidence": 0.0}},
  "completed_fields": ["code_repository_url", "dataset_source_or_link", "model_weights_url", "web_server_url", "citation_count", "journal_impact_factor"],
  "risk_flags": ["..."],
  "summary": "一句话总结补到的证据和仍需人工核查的点",
  "confidence": 0.0
}}
"""
    return system, user


def normalize_qwen_web_result(model_name: str, raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {'model_name': model_name, 'raw_output': raw, 'confidence': 0.0}
    row = dict(raw)
    row['model_name'] = canonicalize_model_name(row.get('model_name') or model_name)
    row['matched_model_name'] = row['model_name']
    row['source'] = 'qwen_max_web_enrichment'
    row['evidence_level'] = 'qwen_max_web_search'
    row['search_time'] = now_str()
    # Normalize candidate lists.
    for k in ['repository_candidates', 'dataset_candidates', 'weight_candidates', 'web_server_candidates', 'paper_links', 'aliases', 'risk_flags', 'completed_fields']:
        row[k] = ensure_list(row.get(k))
    if not isinstance(row.get('article_impact'), dict):
        row['article_impact'] = {}

    impact = row.get('article_impact') or {}
    # Promote article-impact fields to top level so final deployment scoring can use them.
    for src, dst in [
        ('source_journal', 'source_journal'),
        ('journal', 'source_journal'),
        ('venue', 'source_journal'),
        ('source_year', 'source_year'),
        ('year', 'source_year'),
        ('citation_count', 'citation_count'),
        ('cited_by_count', 'citation_count'),
        ('journal_impact_factor', 'journal_impact_factor'),
        ('impact_factor', 'journal_impact_factor'),
    ]:
        if is_missing_value(row.get(dst)) and isinstance(impact, dict) and not is_missing_value(impact.get(src)):
            row[dst] = impact.get(src)
    # Paper links may include DOI/PMID/journal/citations too.
    for p in ensure_list(row.get('paper_links')):
        if not isinstance(p, dict):
            continue
        if is_missing_value(row.get('source_doi')) and not is_missing_value(p.get('doi')):
            row['source_doi'] = p.get('doi')
        if is_missing_value(row.get('source_pmid')) and not is_missing_value(p.get('pmid')):
            row['source_pmid'] = p.get('pmid')
        if is_missing_value(row.get('source_journal')) and not is_missing_value(p.get('journal') or p.get('venue')):
            row['source_journal'] = p.get('journal') or p.get('venue')
        if is_missing_value(row.get('citation_count')) and not is_missing_value(p.get('citation_count') or p.get('cited_by_count')):
            row['citation_count'] = p.get('citation_count') or p.get('cited_by_count')

    # Extract any embedded URLs from summary/evidence and store them for traceability.
    row['all_urls'] = _qwen_web_candidate_urls(row)
    confs = []
    for k in ['repository_candidates', 'dataset_candidates', 'weight_candidates', 'web_server_candidates', 'paper_links']:
        for c in ensure_list(row.get(k)):
            if isinstance(c, dict):
                try:
                    confs.append(float(c.get('confidence') or 0))
                except Exception:
                    pass
    if isinstance(impact, dict):
        try:
            confs.append(float(impact.get('confidence') or 0))
        except Exception:
            pass
    try:
        base_conf = float(row.get('confidence') or 0)
    except Exception:
        base_conf = 0.0
    if confs:
        base_conf = max(base_conf, max(confs))
    if not base_conf and row.get('all_urls'):
        base_conf = 0.35
    row['confidence'] = round(min(1.0, max(0.0, base_conf)), 3)
    row['confidence_label'] = qwen_web_confidence_label(row['confidence'])
    row['needs_manual_verification'] = row['confidence'] < 0.75
    return row


def search_qwen_web_for_model_evidence(models: List[Dict[str, Any]], provider_config: Path = Path('llm_providers.json'), provider: str = 'dashscope_qwen37max_search', model_name: str = 'qwen3.7-max', max_models: int = 30, force: bool = False, refresh_all: bool = False) -> List[Dict[str, Any]]:
    prior = load_existing_qwen_web_enrichment()
    rows: List[Dict[str, Any]] = []
    pending: List[Tuple[str, Dict[str, Any]]] = []
    seen: set = set()
    total_unique = 0
    cached_success = 0
    cached_other = 0
    skipped_non_model = 0
    skipped_not_needed = 0
    cached_success_names: List[str] = []
    cached_other_names: List[str] = []
    skipped_non_model_names: List[str] = []
    skipped_not_needed_names: List[str] = []

    for m in ensure_list(models):
        if not isinstance(m, dict):
            continue
        name = canonicalize_model_name(model_name_from_item(m))
        key = normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        total_unique += 1
        if should_skip_github_enrichment_model(name, m):
            skipped_non_model += 1; skipped_non_model_names.append(name); continue
        if not qwen_web_enrichment_should_run_for_model(name, m):
            skipped_not_needed += 1; skipped_not_needed_names.append(name); continue
        prior_rows = prior.get(key, [])
        if prior_rows and qwen_web_model_has_success(prior_rows) and not refresh_all:
            rows.extend(prior_rows); cached_success += 1; cached_success_names.append(name); continue
        if prior_rows and not force:
            rows.extend(prior_rows); cached_other += 1; cached_other_names.append(name); continue
        pending.append((name, m))

    pending_before_limit = len(pending)
    pending = pending[:max_models]
    report = {
        'time': now_str(),
        'provider': provider,
        'model': model_name,
        'total_unique_model_names': total_unique,
        'cached_success_skipped': cached_success,
        'cached_success_names': cached_success_names,
        'cached_other_retained': cached_other,
        'cached_other_names': cached_other_names,
        'skipped_non_model_terms': skipped_non_model,
        'skipped_non_model_names': skipped_non_model_names,
        'skipped_not_needed': skipped_not_needed,
        'skipped_not_needed_names': skipped_not_needed_names,
        'pending_before_limit': pending_before_limit,
        'actual_search_this_run': len(pending),
        'max_models_limit': max_models,
        'force_qwen_web_enrichment': force,
        'refresh_all_qwen_web_enrichment': refresh_all,
        'pending_model_names': [n for n, _ in pending],
    }
    write_qwen_web_enrichment_run_report(report)
    print(f'>>> Qwen3.7-Max 联网补漏：候选模型总数 {total_unique}；已跳过成功缓存 {cached_success} 个；保留其他缓存 {cached_other} 个；过滤非模型/通用工具 {skipped_non_model} 个；跳过证据已较完整 {skipped_not_needed} 个。', flush=True)
    if pending_before_limit:
        print(f'>>> Qwen3.7-Max 联网补漏：剩余待搜索模型 {pending_before_limit} 个；本轮实际搜索 {len(pending)} 个（上限 --qwen-web-max-models={max_models}）。', flush=True)
        print(f'>>> Qwen3.7-Max 联网补漏：本轮名单已写入 {QWEN_WEB_ENRICHMENT_PENDING_MODELS_TXT.relative_to(ROOT)}；运行报告写入 {QWEN_WEB_ENRICHMENT_RUN_REPORT_JSON.relative_to(ROOT)}。', flush=True)
    else:
        print('>>> Qwen3.7-Max 联网补漏：没有新的模型需要联网补漏。', flush=True)

    if not pending:
        if rows:
            write_qwen_web_enrichment(rows)
        return rows

    try:
        qwen = QwenMaxWebSearchLLM(provider=provider, config_path=provider_config, model=model_name)
    except Exception as e:
        print(f'⚠️ Qwen3.7-Max 联网补漏初始化失败：{e}', flush=True)
        if rows:
            write_qwen_web_enrichment(rows)
        return rows

    for idx, (name, model) in enumerate(pending, 1):
        try:
            system, user = qwen_web_prompt_for_model(name, model)
            obj = qwen.chat_json(system, user)
            row = normalize_qwen_web_result(name, obj)
            rows.append(row)
            repo_count = len(ensure_list(row.get('repository_candidates')))
            data_count = len(ensure_list(row.get('dataset_candidates')))
            weight_count = len(ensure_list(row.get('weight_candidates')))
            web_count = len(ensure_list(row.get('web_server_candidates')))
            print(f'    -> [{idx}/{len(pending)}] {name}: Qwen补漏完成 repo={repo_count} dataset={data_count} weights={weight_count} web={web_count} confidence={row.get("confidence")} {row.get("confidence_label")}', flush=True)
        except Exception as e:
            print(f'    ⚠️ [{idx}/{len(pending)}] {name}: Qwen3.7-Max 联网补漏失败：{e}', flush=True)
            rows.append({
                'model_name': name,
                'matched_model_name': name,
                'source': 'qwen_max_web_enrichment',
                'evidence_level': 'qwen_max_web_search_failed',
                'confidence': 0.0,
                'confidence_label': 'no_candidate',
                'needs_manual_verification': True,
                'search_time': now_str(),
                'error': str(e),
                'repository_candidates': [],
                'dataset_candidates': [],
                'weight_candidates': [],
                'web_server_candidates': [],
                'paper_links': [],
                'risk_flags': ['qwen_web_search_failed'],
            })
    rows = dedupe_objects(rows, 'qwen_web_enrichment')
    if rows:
        write_qwen_web_enrichment(rows)
    return rows


def _qwen_best_candidate_url(items: Any) -> str:
    best = ''
    best_score = -1.0
    for c in ensure_list(items):
        if not isinstance(c, dict):
            continue
        u = str(c.get('url') or '').strip()
        if not u:
            continue
        score = _safe_float(c.get('confidence'), 0.0)
        if score > best_score:
            best = u; best_score = score
    return best


def qwen_web_enrichment_to_evidence_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    models: List[Dict[str, Any]] = []
    repos: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    papers: List[Dict[str, Any]] = []
    open_questions: List[Dict[str, Any]] = []
    important: List[str] = ['Qwen-Max web-search enrichment searched and completed missing model evidence: repository, dataset, weights, web server, paper DOI/PMID, citation_count and journal_impact_factor. All returned links are candidate evidence and require verification.']
    for r in ensure_list(rows):
        if not isinstance(r, dict):
            continue
        name = canonicalize_model_name(r.get('model_name') or r.get('matched_model_name'))
        if not name:
            continue
        best_repo = _qwen_best_candidate_url(r.get('repository_candidates'))
        best_dataset = _qwen_best_candidate_url(r.get('dataset_candidates'))
        best_weight = _qwen_best_candidate_url(r.get('weight_candidates'))
        best_web = _qwen_best_candidate_url(r.get('web_server_candidates'))

        for c in ensure_list(r.get('repository_candidates')):
            if isinstance(c, dict) and c.get('url'):
                u = str(c.get('url'))
                repos.append({'name': c.get('name') or u, 'url': u, 'repository_type': 'code_candidate', 'matched_model_name': name, 'source': 'qwen_max_web_enrichment', 'evidence_level': 'qwen_max_web_search', 'confidence': c.get('confidence') or r.get('confidence'), 'description': c.get('evidence'), 'is_official': c.get('is_official'), 'needs_manual_verification': True, 'evidence': c.get('evidence')})
        for c in ensure_list(r.get('dataset_candidates')):
            if isinstance(c, dict) and c.get('url'):
                u = str(c.get('url'))
                datasets.append({'dataset_name': c.get('name') or f'{name} dataset candidate', 'dataset_url': u, 'dataset_source': 'qwen_max_web_enrichment', 'linked_model': name, 'dataset_status': 'candidate_requires_verification', 'dataset_role': 'training_or_benchmark_candidate', 'evidence_level': 'qwen_max_web_search', 'evidence': c.get('evidence'), 'confidence': c.get('confidence') or r.get('confidence')})
        for c in ensure_list(r.get('weight_candidates')):
            if isinstance(c, dict) and c.get('url'):
                u = str(c.get('url'))
                datasets.append({'dataset_name': c.get('name') or f'{name} weights/checkpoint candidate', 'dataset_url': u, 'dataset_source': 'qwen_max_web_enrichment', 'linked_model': name, 'dataset_status': 'candidate_requires_verification', 'dataset_role': 'model_weights_or_checkpoint_candidate', 'evidence_level': 'qwen_max_web_search', 'evidence': c.get('evidence'), 'confidence': c.get('confidence') or r.get('confidence')})

        impact = r.get('article_impact') if isinstance(r.get('article_impact'), dict) else {}
        impact_cites = _citation_count_from_row(r)
        impact_if = _impact_factor_from_row(r, None)
        for c in ensure_list(r.get('paper_links')):
            if isinstance(c, dict) and (c.get('url') or c.get('doi') or c.get('pmid') or c.get('title')):
                paper = {'title': c.get('title'), 'pmid': c.get('pmid'), 'doi': c.get('doi'), 'url': c.get('url'), 'role': 'qwen_web_paper_candidate', 'open_fulltext_status': 'unknown', 'evidence_level': 'qwen_max_web_search', 'linked_model': name, 'evidence': c.get('evidence'), 'confidence': c.get('confidence') or r.get('confidence'), 'source_journal': c.get('journal') or c.get('venue') or r.get('source_journal'), 'year': c.get('year') or r.get('source_year'), 'citation_count': c.get('citation_count') or c.get('cited_by_count') or r.get('citation_count'), 'journal_impact_factor': c.get('journal_impact_factor') or c.get('impact_factor') or r.get('journal_impact_factor'), 'citation_source': impact.get('citation_source'), 'impact_factor_source': impact.get('impact_factor_source')}
                papers.append(clean_row_dict(paper))
        if not ensure_list(r.get('paper_links')) and (r.get('source_doi') or r.get('source_pmid') or r.get('source_journal') or impact_cites > 0 or impact_if > 0):
            papers.append(clean_row_dict({'title': r.get('source_title'), 'pmid': r.get('source_pmid'), 'doi': r.get('source_doi'), 'url': '', 'role': 'qwen_web_article_impact_candidate', 'open_fulltext_status': 'unknown', 'evidence_level': 'qwen_max_web_search', 'linked_model': name, 'evidence': impact.get('evidence') or r.get('summary'), 'confidence': r.get('confidence'), 'source_journal': r.get('source_journal'), 'year': r.get('source_year'), 'citation_count': r.get('citation_count'), 'journal_impact_factor': r.get('journal_impact_factor'), 'citation_source': impact.get('citation_source'), 'impact_factor_source': impact.get('impact_factor_source')}))

        has_any = bool(best_repo or best_dataset or best_weight or best_web or r.get('source_doi') or r.get('source_pmid') or r.get('source_journal') or impact_cites > 0 or impact_if > 0)
        if has_any:
            models.append(clean_row_dict({'model_name': name, 'canonical_name': name, 'code_repository_url': best_repo or 'not_reported_in_available_evidence', 'web_server_url': best_web or 'not_reported_in_available_evidence', 'model_weights_url': best_weight or 'not_reported_in_available_evidence', 'dataset_source_or_link': best_dataset or 'not_reported_in_available_evidence', 'source_doi': r.get('source_doi'), 'source_pmid': r.get('source_pmid'), 'source_journal': r.get('source_journal'), 'source_year': r.get('source_year'), 'citation_count': r.get('citation_count'), 'journal_impact_factor': r.get('journal_impact_factor'), 'article_impact_score': article_impact_score(r, None), 'benchmark_candidate': True, 'candidate_reason': 'Qwen3.7-Max web enrichment completed missing external evidence; verify before deployment.', 'blocking_issues': ['qwen_web_candidate_requires_manual_verification'], 'evidence_level': 'qwen_max_web_search', 'confidence': r.get('confidence'), 'evidence': r.get('summary') or r.get('evidence'), 'completed_fields': r.get('completed_fields'), 'risk_flags': r.get('risk_flags')}))
            if impact_cites > 0 or impact_if > 0:
                important.append(f'Qwen-Max filled article-impact metadata for {name}: citation_count={r.get("citation_count")}, journal_impact_factor={r.get("journal_impact_factor")}, journal={r.get("source_journal")}')
        else:
            open_questions.append({'question': f'Qwen3.7-Max web search found no usable repository/dataset/weights/web/paper-impact evidence for {name}', 'reason': 'qwen_web_enrichment_no_hit', 'next_action': 'try exact paper title, author name, supplementary materials, GitLab, Zenodo, Semantic Scholar, OpenAlex, journal page, or web server documentation'})
    return {'_batch_no': 'qwen_max_web_enrichment', '_stage': 'qwen_max_web_enrichment', 'models': models, 'repositories': repos, 'datasets': datasets, 'metrics': [], 'papers': papers, 'important_evidence': important, 'open_questions': open_questions}


def add_qwen_web_enrichment_to_compact_pool(compact_pool: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return compact_pool
    ev = qwen_web_enrichment_to_evidence_batch(rows)
    urls = []
    for r in rows:
        urls.extend(ensure_list(r.get('all_urls')))
    summary = {
        'chunk_id': 'qwen_max_web_enrichment',
        'chunk_type': 'web_search_enrichment',
        'chunk_name': 'Qwen-Max web-search missing evidence enrichment',
        'compression_status': 'deterministic_qwen_web_search_enrichment',
        'main_entities': sorted(set([canonicalize_model_name(r.get('model_name') or r.get('matched_model_name')) for r in rows if isinstance(r, dict)]))[:200],
        'models': ev.get('models', []),
        'repositories': ev.get('repositories', []),
        'datasets': ev.get('datasets', []),
        'metrics': [],
        'papers': ev.get('papers', []),
        'important_evidence': ev.get('important_evidence', []),
        'uncertainties': ev.get('open_questions', []),
        'source_pmids': [],
        'source_dois': [],
        'urls': sorted(set([u for u in urls if u]))[:500],
    }
    summaries = [sm for sm in ensure_list(compact_pool.get('chunk_summaries')) if not (isinstance(sm, dict) and sm.get('chunk_id') == 'qwen_max_web_enrichment')]
    summaries.append(summary)
    compact_pool['chunk_summaries'] = summaries
    compact_pool['qwen_web_enrichment'] = rows
    compact_pool['qwen_web_enrichment_enabled'] = True
    compact_pool['qwen_web_enrichment_count'] = len(rows)
    compact_pool['chunk_summary_count'] = len(summaries)
    compact_pool['chunk_count'] = len(summaries)
    write_json(COMPACT_EVIDENCE_POOL_JSON, compact_pool)
    append_jsonl(CHUNK_SUMMARIES_JSONL, summary)
    return compact_pool


def enrich_compact_pool_with_qwen_web(compact_pool: Dict[str, Any], provider_config: Path = Path('llm_providers.json'), provider: str = 'dashscope_qwen37max_search', model_name: str = 'qwen3.7-max', max_models: int = 30, force: bool = False, refresh_all: bool = False) -> Dict[str, Any]:
    models = collect_model_rows_for_github_search_from_compact(compact_pool)
    rows = search_qwen_web_for_model_evidence(models, provider_config=provider_config, provider=provider, model_name=model_name, max_models=max_models, force=force, refresh_all=refresh_all)
    if not rows:
        return compact_pool
    return add_qwen_web_enrichment_to_compact_pool(compact_pool, rows)


def enrich_evidence_batches_with_qwen_web(evidence_batches: List[Dict[str, Any]], provider_config: Path = Path('llm_providers.json'), provider: str = 'dashscope_qwen37max_search', model_name: str = 'qwen3.7-max', max_models: int = 30, force: bool = False, refresh_all: bool = False) -> Optional[Dict[str, Any]]:
    models = collect_model_rows_for_github_search_from_evidence(evidence_batches)
    rows = search_qwen_web_for_model_evidence(models, provider_config=provider_config, provider=provider, model_name=model_name, max_models=max_models, force=force, refresh_all=refresh_all)
    if not rows:
        return None
    return qwen_web_enrichment_to_evidence_batch(rows)


class DataCiteClient:
    BASE = 'https://api.datacite.org/dois'

    def search(self, query: str, rows: int = 20) -> List[Dict[str, Any]]:
        try:
            data = HTTP.get_json(self.BASE, {'query': query, 'page[size]': min(rows, 100)})
            cache_raw('datacite', query, data)
        except Exception as e:
            print(f'    ⚠️ DataCite 搜索失败：{e}')
            return []
        out = []
        for item in data.get('data', []) or []:
            attr = item.get('attributes') or {}
            titles = attr.get('titles') or []
            title = titles[0].get('title') if titles else ''
            creators = []
            for c in attr.get('creators') or []:
                creators.append(c.get('name') or ' '.join([c.get('givenName',''), c.get('familyName','')]).strip())
            out.append({
                'name': title, 'title': title, 'url': attr.get('url') or (f'https://doi.org/{attr.get("doi")}' if attr.get('doi') else None),
                'doi': attr.get('doi'), 'resource_type': (attr.get('types') or {}).get('resourceTypeGeneral'),
                'publisher': attr.get('publisher'), 'year': attr.get('publicationYear'), 'creators': [x for x in creators if x],
                'description': ' '.join([d.get('description','') for d in attr.get('descriptions') or []])[:4000],
                'source': 'datacite', 'query': query,
            })
        return out


class ZenodoClient:
    BASE = 'https://zenodo.org/api/records'

    def search(self, query: str, rows: int = 20) -> List[Dict[str, Any]]:
        try:
            data = HTTP.get_json(self.BASE, {'q': query, 'size': min(rows, 100), 'sort': 'bestmatch'})
            cache_raw('zenodo', query, data)
        except Exception as e:
            print(f'    ⚠️ Zenodo 搜索失败：{e}')
            return []
        hits = (data.get('hits') or {}).get('hits') or []
        out = []
        for h in hits:
            meta = h.get('metadata') or {}
            out.append({
                'name': meta.get('title'), 'title': meta.get('title'), 'url': h.get('links', {}).get('html'),
                'doi': meta.get('doi'), 'resource_type': (meta.get('resource_type') or {}).get('type'),
                'creators': [c.get('name') for c in meta.get('creators') or [] if c.get('name')],
                'description': re.sub(r'<[^>]+>', ' ', meta.get('description') or '')[:4000],
                'source': 'zenodo', 'query': query,
            })
        return out


# ------------------------- Full text -------------------------
class FullTextFetcher:
    def __init__(self, pubmed: PubMedClient, epmc: EuropePMCClient):
        self.pubmed = pubmed
        self.epmc = epmc

    def fetch_and_save(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        key = str(rec.get('pmid') or rec.get('pmcid') or rec.get('doi') or stable_hash(rec))
        d = FULLTEXT_CACHE_DIR / safe_name(key, 'paper')
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / 'normalized_paper.json', rec)
        status = {'paper_key': rec.get('candidate_key'), 'pmid': rec.get('pmid'), 'pmcid': rec.get('pmcid'), 'doi': rec.get('doi'), 'cache_dir': str(d.relative_to(ROOT)), 'status': 'not_found', 'source': None, 'text_chars': 0, 'links': {}}
        pmcid = rec.get('pmcid')
        if not pmcid and rec.get('pmid'):
            pmcid = self.pubmed.pmcid_from_pmid(str(rec.get('pmid')))
            if pmcid:
                rec['pmcid'] = pmcid
                status['pmcid'] = pmcid
                write_json(d / 'normalized_paper.json', rec)
        xml_text = None
        xml_source = None
        if pmcid:
            xml_text = self.pubmed.fetch_pmc_xml(str(pmcid))
            xml_source = 'PMC_EFetch_XML' if xml_text else None
            if not xml_text:
                xml_text = self.epmc.fulltext_xml(str(pmcid))
                xml_source = 'EuropePMC_fullTextXML' if xml_text else None
        if xml_text:
            (d / f'{xml_source}.xml').write_text(xml_text, encoding='utf-8', errors='ignore')
            try:
                sections = sections_from_pmc_xml(xml_text)
                text = '\n\n'.join(sections.values()) if sections else text_from_xml(ET.fromstring(xml_text.encode('utf-8', errors='ignore')))
            except Exception:
                sections = {}
                text = re.sub(r'<[^>]+>', ' ', xml_text)
            text = re.sub(r'\s+', ' ', text).strip()
            (d / 'fulltext_text.txt').write_text(text, encoding='utf-8', errors='ignore')
            write_json(d / 'fulltext_sections.json', sections)
            links = extract_links(text + '\n' + json_dumps(rec, 0))
            write_json(d / 'links.json', links)
            status.update({'status': 'open_fulltext_found', 'source': xml_source, 'text_chars': len(text), 'links': links})
        else:
            meta_text = json_dumps(rec, 0)
            links = extract_links(meta_text)
            write_json(d / 'links.json', links)
            status.update({'status': 'no_open_fulltext_found', 'source': None, 'links': links})
        write_json(d / 'fulltext_metadata.json', status)
        return status


# ------------------------- Evidence extraction -------------------------
def compact_record_for_llm(rec: Dict[str, Any], max_text: int = 18000) -> Dict[str, Any]:
    key = str(rec.get('pmid') or rec.get('pmcid') or rec.get('doi') or stable_hash(rec))
    d = FULLTEXT_CACHE_DIR / safe_name(key, 'paper')
    fulltext = ''
    links = {}
    ft_meta = {}
    if (d / 'fulltext_text.txt').exists():
        fulltext = (d / 'fulltext_text.txt').read_text(encoding='utf-8', errors='ignore')
    if (d / 'links.json').exists():
        links = read_json(d / 'links.json', {})
    if (d / 'fulltext_metadata.json').exists():
        ft_meta = read_json(d / 'fulltext_metadata.json', {})
    return {
        'candidate_key': rec.get('candidate_key'),
        'sources': rec.get('sources'),
        'pmid': rec.get('pmid'), 'pmcid': rec.get('pmcid'), 'doi': rec.get('doi'),
        'semantic_scholar_id': rec.get('semantic_scholar_id'), 'openalex_id': rec.get('openalex_id'),
        'title': rec.get('title'), 'abstract': rec.get('abstract'), 'journal': rec.get('journal') or rec.get('venue'),
        'year': rec.get('year'), 'authors': rec.get('authors'), 'urls': rec.get('urls'),
        'fulltext_status': ft_meta.get('status'), 'fulltext_source': ft_meta.get('source'), 'fulltext_cache_dir': ft_meta.get('cache_dir'),
        'extracted_links': links,
        'fulltext_excerpt': trunc(fulltext, max_text),
    }


def regex_evidence_from_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    repos = []
    datasets = []
    for rec in records:
        text = json_dumps(compact_record_for_llm(rec, 8000), 0)
        links = extract_links(text)
        for u in links['github_urls']:
            repos.append({'name': u.rstrip('/').split('/')[-1], 'url': u, 'matched_paper_title': rec.get('title'), 'source_pmid': rec.get('pmid'), 'source_doi': rec.get('doi'), 'evidence_source': 'regex_fulltext_or_metadata', 'confidence': 0.7})
        for u in links['dataset_urls']:
            datasets.append({'dataset_name': u.rstrip('/').split('/')[-1], 'dataset_url': u, 'matched_paper_title': rec.get('title'), 'source_pmid': rec.get('pmid'), 'source_doi': rec.get('doi'), 'evidence_source': 'regex_fulltext_or_metadata', 'confidence': 0.6})
    return {'repositories': repos, 'datasets': datasets}


def extract_info_batch(llm: DeepSeekChatLLM, loader: AgentMDLoader, records: List[Dict[str, Any]], batch_no: int) -> Dict[str, Any]:
    system = loader.load('info_extractor_agent')
    payload = [compact_record_for_llm(r) for r in records]
    user = f"""
下面是第 {batch_no} 批多源文献/全文证据。请提取 AMP 预测模型 benchmark 所需关键信息。

必须返回严格 JSON，schema：
{{
  "batch_summary": "...",
  "models": [{{
    "model_name": "...", "canonical_name":"...", "aliases":[],
    "task_type":"AMP prediction / antimicrobial peptide classification / other",
    "method_family":"ML/DL/feature-engineering/web-server/unknown",
    "architecture_or_algorithm":"...", "input_features":"...",
    "article_source":"paper title", "source_pmid":"", "source_pmcid":"", "source_doi":"", "source_year":"", "source_journal":"",
    "code_repository_url":"", "web_server_url":"", "model_weights_url":"", "code_availability":"available/not_reported/unclear",
    "dataset_source_or_link":"", "training_data_signal":"", "external_test_signal":"",
    "benchmark_candidate": true, "candidate_reason":"", "blocking_issues":[],
    "evidence_level":"fulltext/abstract/review/search_result/repository", "evidence_source":"", "needs_full_text_verification": false,
    "confidence":0.0, "evidence":"short quote or paraphrase"
  }}],
  "repositories": [{{"name":"", "url":"", "repository_type":"code/model/webserver", "matched_model_name":"", "source_pmid":"", "source_doi":"", "evidence_level":"", "evidence":""}}],
  "datasets": [{{"dataset_name":"", "dataset_url":"", "dataset_source":"", "linked_model":"", "source_pmid":"", "source_doi":"", "positive_samples":"", "negative_samples":"", "deduplication_method":"", "evidence_level":"", "evidence":""}}],
  "metrics": [{{"metric_name":"", "usage":"", "source_pmid":"", "source_doi":"", "evidence":""}}],
  "papers": [{{"title":"", "pmid":"", "pmcid":"", "doi":"", "year":"", "role":"model_original/benchmark/dataset/review/unclear", "open_fulltext_status":"", "important_links":[]}}],
  "open_questions": []
}}

规则：
- 不要编造链接；没有明确链接写 not_reported_in_available_evidence。
- 如果全文或摘要明确是 AMP 预测模型，即使代码未知，也要进入 models。
- 证据来自 review 时标记 evidence_level=review，needs_full_text_verification=true。
- 只返回 JSON。

输入证据：
{json_dumps(payload, 2)}
"""
    result = llm.chat_json('info_extractor_agent', system, user)
    if not isinstance(result, dict):
        result = {'raw_result': result}
    regex_ev = regex_evidence_from_records(records)
    result.setdefault('repositories', [])
    result.setdefault('datasets', [])
    if isinstance(result['repositories'], list):
        result['repositories'].extend(regex_ev['repositories'])
    if isinstance(result['datasets'], list):
        result['datasets'].extend(regex_ev['datasets'])
    result['_batch_no'] = batch_no
    result['_record_keys'] = [r.get('candidate_key') for r in records]
    return result


def save_evidence_pool(evidence_batches: List[Dict[str, Any]], records: List[Dict[str, Any]], repos: List[Dict[str, Any]], datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    pool = {
        'created_at': now_str(),
        'paper_count': len(records),
        'evidence_batch_count': len(evidence_batches),
        'source_counts': source_counts(records),
        'papers': records,
        'evidence_batches': evidence_batches,
        'external_repositories': repos,
        'external_datasets': datasets,
    }
    write_json(EVIDENCE_POOL_JSON, pool)
    lines = ['# Evidence Pool', '', f'- Created: {pool["created_at"]}', f'- Papers: {len(records)}', f'- Evidence batches: {len(evidence_batches)}', '', '## Source Counts']
    for k, v in pool['source_counts'].items():
        lines.append(f'- {k}: {v}')
    lines.append('\n## Papers')
    for r in records:
        lines.append(f"- {r.get('year','')} | {r.get('title','')} | PMID={r.get('pmid','')} DOI={r.get('doi','')} PMCID={r.get('pmcid','')} | sources={','.join(ensure_list(r.get('sources')))}")
    lines.append('\n## Evidence Batches')
    for ev in evidence_batches:
        lines.append(f"\n### Batch {ev.get('_batch_no')}\n")
        for m in ensure_list(ev.get('models')):
            if isinstance(m, dict):
                lines.append(f"- MODEL: {m.get('model_name') or m.get('canonical_name')} | {m.get('source_pmid')} | code={m.get('code_repository_url')} | dataset={m.get('dataset_source_or_link')}")
            else:
                lines.append(f'- MODEL: {m}')
    EVIDENCE_POOL_MD.write_text('\n'.join(lines), encoding='utf-8')
    return pool


def source_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for r in records:
        for s in ensure_list(r.get('sources')):
            d[s] = d.get(s, 0) + 1
    return dict(sorted(d.items()))


# ------------------------- Evidence Chunk Compression -------------------------
def compact_record_for_prompt(rec: Dict[str, Any]) -> Dict[str, Any]:
    keys = ['candidate_key','title','year','journal','pmid','pmcid','doi','semantic_scholar_id','openalex_id','sources','urls','abstract','fulltext_status','open_fulltext_status','fulltext_source','cache_dir']
    out = {k: rec.get(k) for k in keys if rec.get(k) not in (None, '', [], {})}
    if out.get('abstract'):
        out['abstract'] = trunc(out['abstract'], 1200)
    return out


def compact_evidence_for_prompt(ev: Dict[str, Any]) -> Dict[str, Any]:
    keep = ['_batch_no','_stage','_record_keys','models','repositories','datasets','dataset_links','metrics','papers','benchmark_implications','open_questions','important_evidence','uncertainties','evidence']
    out = {k: ev.get(k) for k in keep if ev.get(k) not in (None, '', [], {})}
    extra: Dict[str, Any] = {}
    for k, v in ev.items():
        if k.startswith('_') or k in out or k in ['raw_text', 'fulltext', 'full_text']:
            continue
        if isinstance(v, (str, int, float, bool)):
            extra[k] = trunc(v, 800)
        elif isinstance(v, list) and v:
            extra[k] = v[:20]
        elif isinstance(v, dict) and v:
            extra[k] = {kk: vv for kk, vv in list(v.items())[:20]}
    if extra:
        out['other_extracted_fields'] = extra
    return out


def _extract_names_from_item(item: Any) -> List[str]:
    names: List[str] = []
    if isinstance(item, dict):
        for k in ['model_name','canonical_name','name','tool_name','server_name','matched_model_name','linked_model']:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
        for v in ensure_list(item.get('aliases')):
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
    elif isinstance(item, str):
        for m in re.findall(r'\b[A-Za-z][A-Za-z0-9_\-]{2,40}\b', item):
            if any(x in m.lower() for x in ['amp','pred','scanner','bert','deep','svm','cnn','rf','peptide']):
                names.append(m)
    return names


def model_names_from_evidence_batch(ev: Dict[str, Any], limit: int = 12) -> List[str]:
    names: List[str] = []
    for section in ['models','repositories','datasets','dataset_links','papers']:
        for item in ensure_list(ev.get(section)):
            names.extend(_extract_names_from_item(item))
    out: List[str] = []
    seen: set = set()
    for n in names:
        nk = normalize_key(n)
        if not nk or len(nk) < 3 or nk in seen:
            continue
        seen.add(nk); out.append(n)
        if len(out) >= limit:
            break
    return out


def detect_topics_from_evidence(ev: Dict[str, Any]) -> List[str]:
    topics: List[str] = []
    if ensure_list(ev.get('datasets')) or ensure_list(ev.get('dataset_links')):
        topics.append('datasets')
    if ensure_list(ev.get('repositories')):
        topics.append('repositories')
    if ensure_list(ev.get('metrics')):
        topics.append('metrics')
    if ensure_list(ev.get('benchmark_implications')):
        topics.append('benchmark')
    if ensure_list(ev.get('open_questions')) or ensure_list(ev.get('uncertainties')):
        topics.append('uncertainties')
    if not topics:
        topics.append('unknown_models')
    return topics


def split_items(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, int(size or 1))
    return [items[i:i+size] for i in range(0, len(items), size)]


def build_evidence_chunks(evidence_pool: Dict[str, Any], target_items_per_chunk: int = 6, max_chunks: int = 120) -> List[Dict[str, Any]]:
    papers = ensure_list(evidence_pool.get('papers'))
    evidence_batches = ensure_list(evidence_pool.get('evidence_batches'))
    repos = ensure_list(evidence_pool.get('external_repositories'))
    datasets = ensure_list(evidence_pool.get('external_datasets'))
    paper_by_key = {r.get('candidate_key'): r for r in papers if isinstance(r, dict) and r.get('candidate_key')}
    grouped: Dict[str, Dict[str, Any]] = {}

    def add(group_type: str, group_name: str, payload: Dict[str, Any]) -> None:
        gkey = f'{group_type}:{normalize_key(group_name) or safe_name(group_name)}'
        if gkey not in grouped:
            grouped[gkey] = {'chunk_group_type': group_type, 'chunk_group_name': group_name, 'items': [], 'record_keys': set()}
        grouped[gkey]['items'].append(payload)
        for rk in ensure_list(payload.get('record_keys')):
            if rk:
                grouped[gkey]['record_keys'].add(rk)

    for ev in evidence_batches:
        if not isinstance(ev, dict):
            continue
        rec_keys = ensure_list(ev.get('_record_keys'))
        compact_ev = compact_evidence_for_prompt(ev)
        compact_ev['related_papers'] = [compact_record_for_prompt(paper_by_key[k]) for k in rec_keys if k in paper_by_key]
        compact_ev['record_keys'] = rec_keys
        names = model_names_from_evidence_batch(ev)
        if names:
            for name in names[:8]:
                add('model', name, compact_ev)
        else:
            add('topic', 'unknown_models', compact_ev)
        for topic in detect_topics_from_evidence(ev):
            add('topic', topic, compact_ev)

    source_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in papers:
        if not isinstance(r, dict):
            continue
        for src in ensure_list(r.get('sources')) or ['unknown_source']:
            source_groups.setdefault(str(src), []).append(compact_record_for_prompt(r))
    for src, rs in sorted(source_groups.items()):
        for part_no, part in enumerate(split_items(rs, max(10, target_items_per_chunk * 2)), 1):
            add('source', f'{src}_part_{part_no}', {'source': src, 'papers': part, 'record_keys': [p.get('candidate_key') for p in part]})

    for part_no, part in enumerate(split_items(repos, max(8, target_items_per_chunk)), 1):
        if part:
            add('topic', f'external_repositories_part_{part_no}', {'repositories': part, 'record_keys': []})
    for part_no, part in enumerate(split_items(datasets, max(8, target_items_per_chunk)), 1):
        if part:
            add('topic', f'external_datasets_part_{part_no}', {'datasets': part, 'record_keys': []})

    chunks: List[Dict[str, Any]] = []
    order = {'model': 0, 'topic': 1, 'source': 2}
    for _gkey, group in sorted(grouped.items(), key=lambda kv: (order.get(kv[1]['chunk_group_type'], 9), kv[0])):
        items = group['items']
        for part_no, part in enumerate(split_items(items, target_items_per_chunk), 1):
            cid_base = f"{group['chunk_group_type']}_{safe_name(group['chunk_group_name'])}"
            cid = cid_base if len(items) <= target_items_per_chunk else f'{cid_base}_part_{part_no}'
            chunks.append({'chunk_id': cid, 'chunk_type': group['chunk_group_type'], 'chunk_name': group['chunk_group_name'], 'part_no': part_no, 'item_count': len(part), 'record_keys': sorted([x for x in group['record_keys'] if x]), 'items': part})
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def fallback_chunk_summary(chunk: Dict[str, Any], error: str = '') -> Dict[str, Any]:
    text = json_dumps(chunk, 0)
    urls = sorted(set(re.findall(r"https?://[^\s\]})>,\"']+", text)))[:50]
    pmid_pairs = re.findall(r'\bPMID[:= ]?([0-9]{6,10})\b|"pmid"\s*:\s*"?([0-9]{6,10})', text)
    pmids = sorted({a or b for a, b in pmid_pairs if (a or b)})[:50]
    dois = sorted(set(re.findall(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', text)))[:50]
    return {'chunk_id': chunk.get('chunk_id'), 'chunk_type': chunk.get('chunk_type'), 'chunk_name': chunk.get('chunk_name'), 'compression_status': 'fallback_no_llm_or_failed', 'compression_error': error, 'main_entities': [chunk.get('chunk_name')] if chunk.get('chunk_type') == 'model' else [], 'papers': [{'pmid': p} for p in pmids], 'models': [], 'repositories': [{'url': u} for u in urls if 'github.com' in u.lower() or 'gitlab' in u.lower()], 'datasets': [{'url': u} for u in urls if any(x in u.lower() for x in ['zenodo','figshare','dryad','datacite','dataset','supplement'])], 'metrics': [], 'important_evidence': [f'Fallback summary generated for chunk {chunk.get("chunk_id")}. Original chunk is saved in _chunk_index.json.'], 'uncertainties': ['LLM compression failed; inspect raw evidence and chunk index.'], 'source_pmids': pmids, 'source_dois': dois, 'urls': urls}


def compress_evidence_chunks(llm: DeepSeekChatLLM, loader: AgentMDLoader, evidence_pool: Dict[str, Any], target_items_per_chunk: int = 6, max_chunks: int = 120, max_chars_per_chunk: int = 60000) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    CHUNK_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    if CHUNK_SUMMARIES_JSONL.exists():
        CHUNK_SUMMARIES_JSONL.unlink()
    chunks = build_evidence_chunks(evidence_pool, target_items_per_chunk=target_items_per_chunk, max_chunks=max_chunks)
    write_json(CHUNK_INDEX_JSON, {'created_at': now_str(), 'chunk_count': len(chunks), 'chunks': chunks})
    print(f'>>> Evidence chunks built: {len(chunks)} | index={CHUNK_INDEX_JSON.relative_to(ROOT)}')
    system = loader.load('evidence_compressor_agent')
    summaries: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, 1):
        cid = chunk.get('chunk_id') or f'chunk_{idx:03d}'
        print(f'    -> [Compressor {idx}/{len(chunks)}] {cid} ({chunk.get("chunk_type")}: {chunk.get("chunk_name")})')
        chunk_prompt = trunc(json_dumps(chunk, 2), max_chars_per_chunk)
        try:
            summary = llm.chat_json('evidence_compressor_agent', system, f"""
请把下面这个 evidence chunk 压缩成结构化 JSON。不要编造任何论文、链接、数据集或代码仓库。
必须保留来源可追溯字段，例如 PMID/PMCID/DOI/title/url/source。

输入 chunk：
{chunk_prompt}
""")
            if not isinstance(summary, dict):
                summary = {'chunk_id': cid, 'compression_status': 'non_dict_output', 'raw_summary': summary}
            summary.setdefault('chunk_id', cid)
            summary.setdefault('chunk_type', chunk.get('chunk_type'))
            summary.setdefault('chunk_name', chunk.get('chunk_name'))
            summary.setdefault('compression_status', 'ok')
        except Exception as e:
            err = str(e)
            append_jsonl(FAILED_DIR / 'failed_chunk_compression.jsonl', {'chunk_id': cid, 'error': err, 'traceback': traceback.format_exc()})
            print(f'       ⚠️ Chunk 压缩失败，使用 fallback 摘要继续：{err}')
            summary = fallback_chunk_summary(chunk, err)
        fn = CHUNK_SUMMARIES_DIR / f'{idx:03d}_{safe_name(cid)}.json'
        write_json(fn, summary)
        append_jsonl(CHUNK_SUMMARIES_JSONL, summary)
        summaries.append(summary)
    compact_pool = {'created_at': now_str(), 'compression_mode': 'by_model_topic_source', 'paper_count': evidence_pool.get('paper_count'), 'source_counts': evidence_pool.get('source_counts'), 'evidence_batch_count': evidence_pool.get('evidence_batch_count'), 'chunk_count': len(chunks), 'chunk_summary_count': len(summaries), 'chunk_summaries': summaries, 'paper_overview': [compact_record_for_prompt(r) for r in ensure_list(evidence_pool.get('papers'))[:300]]}
    write_json(COMPACT_EVIDENCE_POOL_JSON, compact_pool)
    md_lines = ['# Compact Evidence Pool', '', f'- Created: {compact_pool["created_at"]}', f'- Chunk summaries: {len(summaries)}', '', '## Chunk Summaries']
    for s in summaries:
        ents = ', '.join(map(str, ensure_list(s.get('main_entities'))[:8]))
        md_lines.append(f"- **{s.get('chunk_id')}** | type={s.get('chunk_type')} | name={s.get('chunk_name')} | status={s.get('compression_status')} | entities={ents}")
    COMPACT_EVIDENCE_POOL_MD.write_text('\n'.join(md_lines), encoding='utf-8')
    return compact_pool, summaries


# ------------------------- Global Meeting -------------------------
def global_meeting(llm: DeepSeekChatLLM, loader: AgentMDLoader, compact_evidence_pool: Dict[str, Any], memory_context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meeting_input = {
        'created_at': compact_evidence_pool.get('created_at'),
        'compression_mode': compact_evidence_pool.get('compression_mode', 'none'),
        'paper_count': compact_evidence_pool.get('paper_count'),
        'source_counts': compact_evidence_pool.get('source_counts'),
        'chunk_summary_count': compact_evidence_pool.get('chunk_summary_count'),
        'chunk_summaries': ensure_list(compact_evidence_pool.get('chunk_summaries')),
        'paper_overview': ensure_list(compact_evidence_pool.get('paper_overview'))[:200],
    }
    compact_pool_text = trunc(json_dumps(meeting_input, 2), 120000)

    print('    -> [Global Agent 1] 模型与数据集专家全局会议（读取 chunk summaries）...')
    md_system = loader.load('model_dataset_agent')
    try:
        md_json = llm.chat_json('model_dataset_agent', md_system, f"""
请基于 compact evidence chunk summaries 进行全局整理，合并重复模型和数据集，判断 benchmark 候选状态。
重点使用 chunk_summaries，而不是要求读取全文。
只返回 JSON。

Compact evidence：
{compact_pool_text}

已有记忆摘要：
{trunc(json_dumps(memory_context, 2), 20000)}
""")
    except Exception as e:
        append_jsonl(FAILED_DIR / 'failed_global_agents.jsonl', {'agent': 'model_dataset_agent', 'error': str(e), 'traceback': traceback.format_exc()})
        md_json = {'models': [], 'datasets': [], 'repositories': [], 'open_questions': [f'model_dataset_agent failed: {e}']}

    print('    -> [Global Agent 2] 指标专家全局会议（读取 chunk summaries）...')
    metric_system = loader.load('metric_agent')
    try:
        metric_json = llm.chat_json('metric_agent', metric_system, f"""
请基于 compact evidence chunk summaries 总结 AMP benchmark 的评价指标、数据集切分、外部验证、推荐主指标。
只返回 JSON。

Compact evidence：
{compact_pool_text}
""")
    except Exception as e:
        append_jsonl(FAILED_DIR / 'failed_global_agents.jsonl', {'agent': 'metric_agent', 'error': str(e), 'traceback': traceback.format_exc()})
        metric_json = {'metrics': [], 'benchmark_implications': [], 'open_questions': [f'metric_agent failed: {e}']}

    print('    -> [Global Agent 3] Critic 全局证据审查...')
    critic_system = loader.load('critic_agent')
    try:
        critic_json = llm.chat_json('critic_agent', critic_system, f"""
请审查模型/数据集/指标结论的证据可靠性，指出不能确认的代码仓库、数据集链接、全文缺失和重复模型。
只返回 JSON。

模型数据集专家结果：
{trunc(json_dumps(md_json, 2), 50000)}

指标专家结果：
{trunc(json_dumps(metric_json, 2), 30000)}
""")
    except Exception as e:
        append_jsonl(FAILED_DIR / 'failed_global_agents.jsonl', {'agent': 'critic_agent', 'error': str(e), 'traceback': traceback.format_exc()})
        critic_json = {'open_questions': [f'critic_agent failed: {e}'], 'warnings': []}

    print('    -> [Global Agent 4] Chief 汇总最终全局记忆 JSON...')
    chief_system = loader.load('chief_agent')
    chief_prompt = f"""
请把三位 Agent 的结果汇总成最终严格 JSON，用于长期记忆。

必须返回 schema：
{{
  "all_candidate_models": [],
  "benchmark_ready_models": [],
  "models": [],
  "repositories": [],
  "datasets": [],
  "dataset_links": [],
  "model_dataset_links": [],
  "dataset_followup_tasks": [],
  "metrics": [],
  "papers": [],
  "benchmark_implications": [],
  "open_questions": [],
  "model_classification": [],
  "representative_models_by_category": [],
  "agent_discussion": []
}}

要求：
- 不允许因为代码缺失、数据集缺失、review_only 或低置信度而删除候选模型；必须至少放入 all_candidate_models，并用 blocking_issues/证据等级分类。
- models 可以放更适合 benchmark 的精选模型；benchmark_ready_models 放可优先复现/benchmark 的模型。
- 对每个模型尽量给出 model_dataset_links；没有数据集链接时也要写 dataset_status=not_reported/source_database_named/described_no_link。
- dataset_followup_tasks 专门记录需要继续搜索的数据集/补充材料/仓库 README/数据 DOI。
- 所有链接必须来自 evidence/chunk summaries，不要编造。
- benchmark_implications 必须是对象列表，每个对象包含 topic/decision/reason/evidence。
- model_classification 要按模型名称/主题/来源综合分类，覆盖：纯二分类、深度学习、传统 ML、Web server/tool、MIC/活性回归、生成式/设计、跨界/非核心、review/低置信。
- representative_models_by_category 每类选 1-2 个代表模型，并说明为什么代表该类。
- agent_discussion 需要能渲染成接近 meeting_trace.md 的会议记录：Scout 初版、Metrics 初版、Critic 质疑、Scout 辩护、Metrics 辩护、Critic 终审、Final Consensus。
- 尽量输出紧凑 JSON，不要复制长 evidence。
- 只返回 JSON。

模型数据集专家：
{trunc(json_dumps(md_json, 2), 60000)}

指标专家：
{trunc(json_dumps(metric_json, 2), 30000)}

Critic：
{trunc(json_dumps(critic_json, 2), 30000)}
"""
    try:
        final_json = llm.chat_json('chief_agent', chief_system, chief_prompt)
    except Exception as e:
        append_jsonl(FAILED_DIR / 'failed_global_agents.jsonl', {'agent': 'chief_agent', 'error': str(e), 'traceback': traceback.format_exc()})
        final_json = fallback_final_from_agents(md_json, metric_json, critic_json, error=str(e))

    if not isinstance(final_json, dict):
        final_json = {'models': [], 'repositories': [], 'datasets': [], 'dataset_links': [], 'metrics': [], 'papers': [], 'benchmark_implications': [], 'open_questions': [final_json]}

    # v3: Chief 的输出不能作为唯一来源。这里从 chunk summaries 自动补回全量候选模型、数据集关系和缺失数据集 follow-up。
    final_json = enrich_final_from_chunks(final_json, compact_evidence_pool, md_json, metric_json, critic_json)

    raw = {'time': now_str(), 'meeting_input_type': 'compact_chunk_summaries', 'model_dataset_agent': md_json, 'metric_agent': metric_json, 'critic_agent': critic_json, 'chief_agent': final_json, 'agent_discussion': final_json.get('agent_discussion', [])}
    append_jsonl(GLOBAL_MEETING_RAW_JSONL, raw)
    return final_json, raw


def fallback_final_from_agents(md_json: Any, metric_json: Any, critic_json: Any, error: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'all_candidate_models': [],
        'benchmark_ready_models': [],
        'models': [],
        'repositories': [],
        'datasets': [],
        'dataset_links': [],
        'model_dataset_links': [],
        'dataset_followup_tasks': [],
        'metrics': [],
        'papers': [],
        'benchmark_implications': [],
        'open_questions': [],
        'agent_discussion': [],
    }
    for src in [md_json, metric_json, critic_json]:
        if not isinstance(src, dict):
            out['open_questions'].append({'question': 'non_dict_agent_output', 'reason': str(src)[:500], 'next_action': 'inspect global raw meeting output'})
            continue
        for k in out:
            if k in src:
                out[k].extend(ensure_list(src.get(k)))
    if error:
        out['open_questions'].append({'question': 'chief_agent_failed_or_timed_out', 'reason': error, 'next_action': 'rerun with smaller --chunk-target-size or lower --max-results'})
    for k in out:
        out[k] = ensure_list(out[k])
    return out


# ------------------------- Final-data enrichment -------------------------
BAD_VALUES = {'', 'none', 'null', 'nan', 'n/a', 'na', 'not reported', 'not_reported', 'not_reported_in_available_evidence', 'not available', 'unknown'}


def is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) == 0
    return normalize_key(v) in BAD_VALUES


def first_nonempty(*vals: Any) -> Any:
    for v in vals:
        if not is_missing_value(v):
            return v
    return ''


def clean_row_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        if isinstance(v, str):
            v = v.strip()
        if v in ([], {}, None):
            continue
        out[k] = v
    return out


def model_name_from_item(item: Dict[str, Any]) -> str:
    return str(first_nonempty(item.get('model_name'), item.get('canonical_name'), item.get('name'), item.get('matched_model_name'), item.get('linked_model')) or '').strip()


def canonical_model_from_summary(item: Dict[str, Any], summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    name = model_name_from_item(item)
    if not name:
        return None
    out = clean_row_dict({
        'model_name': name,
        'canonical_name': first_nonempty(item.get('canonical_name'), name),
        'aliases': ensure_list(item.get('aliases')),
        'task_type': first_nonempty(item.get('task_type'), item.get('task')),
        'method_family': first_nonempty(item.get('method_family'), item.get('model_type'), item.get('algorithm_family')),
        'architecture_or_algorithm': first_nonempty(item.get('architecture_or_algorithm'), item.get('architecture'), item.get('algorithm')),
        'input_features': first_nonempty(item.get('input_features'), item.get('features')),
        'source_pmid': first_nonempty(item.get('source_pmid'), item.get('pmid'), (ensure_list(summary.get('source_pmids')) or [''])[0]),
        'source_pmcid': first_nonempty(item.get('source_pmcid'), item.get('pmcid')),
        'source_doi': first_nonempty(item.get('source_doi'), item.get('doi'), (ensure_list(summary.get('source_dois')) or [''])[0]),
        'code_repository_url': first_nonempty(item.get('code_repository_url'), item.get('repository_url'), item.get('code_url')),
        'web_server_url': first_nonempty(item.get('web_server_url'), item.get('server_url'), item.get('web_url')),
        'model_weights_url': first_nonempty(item.get('model_weights_url'), item.get('weights_url')),
        'dataset_source_or_link': first_nonempty(item.get('dataset_source_or_link'), item.get('dataset_source'), item.get('dataset_url')),
        'benchmark_candidate': item.get('benchmark_candidate') if isinstance(item.get('benchmark_candidate'), bool) else first_nonempty(item.get('benchmark_candidate'), ''),
        'candidate_reason': first_nonempty(item.get('candidate_reason'), item.get('reason')),
        'blocking_issues': ensure_list(item.get('blocking_issues')),
        'evidence_level': first_nonempty(item.get('evidence_level'), item.get('evidence_source'), summary.get('chunk_type'), 'chunk_summary'),
        'confidence': first_nonempty(item.get('confidence'), ''),
        'chunk_id': summary.get('chunk_id'),
        'chunk_type': summary.get('chunk_type'),
        'chunk_name': summary.get('chunk_name'),
    })
    if is_missing_value(out.get('code_repository_url')):
        out.pop('code_repository_url', None)
    if is_missing_value(out.get('web_server_url')):
        out.pop('web_server_url', None)
    if is_missing_value(out.get('dataset_source_or_link')):
        out['dataset_source_or_link'] = 'not_reported_in_available_evidence'
    return out


def repository_from_summary(item: Dict[str, Any], summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    url = first_nonempty(item.get('url'), item.get('repository_url'), item.get('code_repository_url'), item.get('web_server_url'))
    name = first_nonempty(item.get('name'), item.get('repository_name'), item.get('matched_model_name'), item.get('model_name'))
    if is_missing_value(url) and is_missing_value(name):
        return None
    return clean_row_dict({
        'name': name,
        'url': url,
        'repository_type': first_nonempty(item.get('repository_type'), item.get('type'), 'code_or_web'),
        'matched_model_name': first_nonempty(item.get('matched_model_name'), item.get('model_name'), item.get('linked_model')),
        'source_pmid': first_nonempty(item.get('source_pmid'), item.get('pmid'), (ensure_list(summary.get('source_pmids')) or [''])[0]),
        'source_doi': first_nonempty(item.get('source_doi'), item.get('doi'), (ensure_list(summary.get('source_dois')) or [''])[0]),
        'evidence_level': first_nonempty(item.get('evidence_level'), item.get('evidence_source'), 'chunk_summary'),
        'chunk_id': summary.get('chunk_id'),
    })


def infer_dataset_status(source_or_url: Any, dataset_url: Any = '') -> str:
    text = f'{source_or_url or ""} {dataset_url or ""}'
    low = text.lower()
    if not is_missing_value(dataset_url) or re.search(r'https?://|doi\.org|zenodo|figshare|dryad|dataverse|github\.com|kaggle|huggingface', low):
        return 'direct_url_found'
    if re.search(r'(apd3?|dramp|dbaasp|camp|dbamp|uniprot|swiss-prot|satpdb|yadamp|ncbi|genbank|ensembl)', low, re.I):
        return 'source_database_named'
    if re.search(r'supplement|additional file|supporting information|appendix', low, re.I):
        return 'supplementary_material_mentioned'
    if not is_missing_value(source_or_url):
        return 'described_no_link'
    return 'not_reported'


def dataset_from_summary(item: Dict[str, Any], summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    name = first_nonempty(item.get('dataset_name'), item.get('name'), item.get('dataset'), item.get('source'))
    src = first_nonempty(item.get('dataset_source'), item.get('source'), item.get('dataset_source_or_link'))
    url = first_nonempty(item.get('dataset_url'), item.get('url'), item.get('link'))
    linked_model = first_nonempty(item.get('linked_model'), item.get('model_name'), item.get('matched_model_name'))
    if is_missing_value(name) and is_missing_value(src) and is_missing_value(url) and is_missing_value(linked_model):
        return None
    status = first_nonempty(item.get('dataset_status'), infer_dataset_status(src, url))
    return clean_row_dict({
        'dataset_name': name or src or f'dataset_for_{linked_model}',
        'dataset_url': '' if is_missing_value(url) else url,
        'dataset_source': src,
        'linked_model': linked_model,
        'dataset_status': status,
        'dataset_role': first_nonempty(item.get('dataset_role'), item.get('role')),
        'source_pmid': first_nonempty(item.get('source_pmid'), item.get('pmid'), (ensure_list(summary.get('source_pmids')) or [''])[0]),
        'source_doi': first_nonempty(item.get('source_doi'), item.get('doi'), (ensure_list(summary.get('source_dois')) or [''])[0]),
        'positive_samples': first_nonempty(item.get('positive_samples'), item.get('positives')),
        'negative_samples': first_nonempty(item.get('negative_samples'), item.get('negatives')),
        'deduplication_method': first_nonempty(item.get('deduplication_method'), item.get('deduplication')),
        'split_method': first_nonempty(item.get('split_method'), item.get('split')),
        'evidence_level': first_nonempty(item.get('evidence_level'), item.get('evidence_source'), 'chunk_summary'),
        'chunk_id': summary.get('chunk_id'),
    })


def model_dataset_link_from_model(model: Dict[str, Any]) -> Dict[str, Any]:
    model_name = model.get('model_name') or model.get('canonical_name') or ''
    src = model.get('dataset_source_or_link') or ''
    status = infer_dataset_status(src, src if isinstance(src, str) and src.startswith('http') else '')
    return clean_row_dict({
        'model_name': model_name,
        'dataset_name': src if status != 'not_reported' else 'not_reported_in_available_evidence',
        'dataset_role': 'training_or_benchmark_unspecified',
        'dataset_source': src if status != 'not_reported' else '',
        'dataset_url': src if isinstance(src, str) and src.startswith('http') else '',
        'dataset_status': status,
        'source_pmid': model.get('source_pmid'),
        'source_doi': model.get('source_doi'),
        'evidence_level': model.get('evidence_level'),
        'needs_followup': status != 'direct_url_found',
        'followup_reason': 'dataset link/source missing or incomplete' if status != 'direct_url_found' else '',
    })


def model_dataset_link_from_dataset(dataset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(dataset, dict):
        return None
    model = first_nonempty(dataset.get('linked_model'), dataset.get('model_name'), dataset.get('matched_model_name'))
    if is_missing_value(model):
        return None
    return clean_row_dict({
        'model_name': model,
        'dataset_name': first_nonempty(dataset.get('dataset_name'), dataset.get('dataset_source'), 'unnamed_dataset'),
        'dataset_role': first_nonempty(dataset.get('dataset_role'), 'training_or_benchmark_unspecified'),
        'dataset_source': dataset.get('dataset_source'),
        'dataset_url': dataset.get('dataset_url'),
        'dataset_status': dataset.get('dataset_status') or infer_dataset_status(dataset.get('dataset_source'), dataset.get('dataset_url')),
        'source_pmid': dataset.get('source_pmid'),
        'source_doi': dataset.get('source_doi'),
        'positive_samples': dataset.get('positive_samples'),
        'negative_samples': dataset.get('negative_samples'),
        'deduplication_method': dataset.get('deduplication_method'),
        'split_method': dataset.get('split_method'),
        'evidence_level': dataset.get('evidence_level'),
        'needs_followup': (dataset.get('dataset_status') or infer_dataset_status(dataset.get('dataset_source'), dataset.get('dataset_url'))) != 'direct_url_found',
    })


def dedupe_objects(items: Any, section: str) -> List[Any]:
    # For model-like sections, dedupe by canonical model name instead of full dict.
    # This prevents duplicate rows such as Co-AMPpred / AMP-BERT variants and empty duplicate rows.
    if section in {'all_candidate_models', 'benchmark_ready_models', 'models'}:
        try:
            return dedupe_models_by_name(items)
        except NameError:
            pass
    out: List[Any] = []
    seen: set = set()
    for item in ensure_list(items):
        if isinstance(item, dict):
            item = clean_row_dict(item)
            if not item:
                continue
            # skip fully empty rows even if they have only generic fields
            if not any(not is_missing_value(item.get(k)) for k in ['model_name','canonical_name','name','url','dataset_name','dataset_source','dataset_url','metric_name','title','topic','question','category','category_title']):
                continue
        key_obj = item if not isinstance(item, dict) else {k: item.get(k) for k in sorted(item) if k not in ['evidence','important_evidence']}
        key = stable_hash(key_obj)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_chunk_derived_final(compact_evidence_pool: Dict[str, Any]) -> Dict[str, Any]:
    summaries = ensure_list(compact_evidence_pool.get('chunk_summaries'))
    out: Dict[str, Any] = {
        'all_candidate_models': [],
        'benchmark_ready_models': [],
        'repositories': [],
        'datasets': [],
        'dataset_links': [],
        'model_dataset_links': [],
        'dataset_followup_tasks': [],
        'metrics': [],
        'papers': [],
        'benchmark_implications': [],
        'open_questions': [],
    }
    for s in summaries:
        if not isinstance(s, dict):
            continue
        for m in ensure_list(s.get('models')):
            mm = canonical_model_from_summary(m, s) if isinstance(m, dict) else None
            if mm:
                out['all_candidate_models'].append(mm)
                if mm.get('benchmark_candidate') is True or normalize_key(mm.get('benchmark_candidate')) in ['true', 'yes', '1']:
                    out['benchmark_ready_models'].append(mm)
                out['model_dataset_links'].append(model_dataset_link_from_model(mm))
        for r in ensure_list(s.get('repositories')):
            rr = repository_from_summary(r, s) if isinstance(r, dict) else None
            if rr:
                out['repositories'].append(rr)
        for d in ensure_list(s.get('datasets')) + ensure_list(s.get('dataset_links')):
            dd = dataset_from_summary(d, s) if isinstance(d, dict) else None
            if dd:
                out['datasets'].append(dd)
                out['dataset_links'].append({
                    'dataset_name': dd.get('dataset_name'),
                    'url': dd.get('dataset_url'),
                    'source': dd.get('dataset_source'),
                    'linked_model': dd.get('linked_model'),
                    'dataset_status': dd.get('dataset_status'),
                    'evidence': dd.get('evidence_level'),
                    'source_pmid': dd.get('source_pmid'),
                    'source_doi': dd.get('source_doi'),
                })
                link = model_dataset_link_from_dataset(dd)
                if link:
                    out['model_dataset_links'].append(link)
        for p in ensure_list(s.get('papers')):
            if isinstance(p, dict):
                out['papers'].append(clean_row_dict(p))
        for mt in ensure_list(s.get('metrics')):
            if isinstance(mt, dict):
                out['metrics'].append(clean_row_dict(mt))
            elif isinstance(mt, str) and mt.strip():
                out['metrics'].append({'metric_name': mt.strip(), 'usage': 'reported_or_recommended_in_chunk_summary'})
        for bi in ensure_list(s.get('benchmark_implications')):
            if isinstance(bi, dict):
                out['benchmark_implications'].append(clean_row_dict(bi))
            elif isinstance(bi, str) and bi.strip():
                out['benchmark_implications'].append({'topic': bi.strip(), 'decision': '', 'reason': '', 'evidence': 'chunk_summary'})
        for q in ensure_list(s.get('uncertainties')) + ensure_list(s.get('open_questions')):
            if isinstance(q, dict):
                out['open_questions'].append(clean_row_dict(q))
            elif isinstance(q, str) and q.strip():
                out['open_questions'].append({'question': q.strip(), 'reason': 'chunk_summary_uncertainty', 'next_action': 'manual_or_followup_search'})

    # Dataset follow-up tasks: one per missing/incomplete model-dataset relation
    for link in ensure_list(out.get('model_dataset_links')):
        if isinstance(link, dict) and link.get('needs_followup'):
            out['dataset_followup_tasks'].append({
                'model_name': link.get('model_name'),
                'dataset_status': link.get('dataset_status'),
                'reason': link.get('followup_reason') or 'dataset source/link incomplete',
                'next_action': 'search full text data availability, supplementary materials, GitHub README, Zenodo/Figshare/Dryad/DataCite, and original paper',
                'source_pmid': link.get('source_pmid'),
                'source_doi': link.get('source_doi'),
            })

    for k in list(out):
        out[k] = dedupe_objects(out[k], k)
    return out


def summarize_agent_output_for_discussion(agent_name: str, role: str, output: Any) -> Dict[str, Any]:
    if not isinstance(output, dict):
        return {'agent': agent_name, 'role': role, 'status': 'non_dict_output', 'summary': trunc(output, 800)}
    counts = {k: len(ensure_list(output.get(k))) for k in ['models','all_candidate_models','benchmark_ready_models','repositories','datasets','dataset_links','model_dataset_links','dataset_followup_tasks','model_classification','representative_models_by_category','metrics','papers','benchmark_implications','open_questions'] if output.get(k) not in (None, [], {})}
    sample_decisions: List[Any] = []
    for k in ['benchmark_implications','open_questions','warnings','uncertainties']:
        for item in ensure_list(output.get(k))[:5]:
            sample_decisions.append(item)
    return {
        'agent': agent_name,
        'role': role,
        'status': 'ok',
        'counts': counts,
        'key_points': sample_decisions[:8],
    }


def build_agent_discussion(md_json: Any, metric_json: Any, critic_json: Any, chief_json: Any, compact_evidence_pool: Dict[str, Any]) -> List[Dict[str, Any]]:
    discussion = [
        {
            'agent': 'evidence_compressor_agent',
            'role': '按模型名称 / 主题 / 来源分块压缩 evidence',
            'status': 'ok',
            'counts': {
                'chunk_summary_count': len(ensure_list(compact_evidence_pool.get('chunk_summaries'))),
                'paper_count': compact_evidence_pool.get('paper_count'),
                'source_counts': compact_evidence_pool.get('source_counts'),
            },
            'key_points': [
                '每个 chunk 保留 PMID/PMCID/DOI/title/url/source 等可追溯证据。',
                '未发现数据集链接时不删除模型，而是记录 dataset_status 与 followup task。',
            ],
        },
        summarize_agent_output_for_discussion('model_dataset_agent', '全局合并模型、数据集、代码仓库，并判断 benchmark 候选状态', md_json),
        summarize_agent_output_for_discussion('metric_agent', '全局整理评价指标、外部验证、推荐 benchmark 指标', metric_json),
        summarize_agent_output_for_discussion('critic_agent', '审查重复模型、证据不足、链接不确定、数据集缺失', critic_json),
        summarize_agent_output_for_discussion('chief_agent', '合并三位 Agent 输出为长期记忆 JSON；不得删除候选模型，只能分类', chief_json),
    ]
    return discussion


def enrich_final_from_chunks(final_data: Dict[str, Any], compact_evidence_pool: Dict[str, Any], md_json: Any = None, metric_json: Any = None, critic_json: Any = None) -> Dict[str, Any]:
    if not isinstance(final_data, dict):
        final_data = {}
    derived = build_chunk_derived_final(compact_evidence_pool)
    # Keep Chief-selected models in models, but never lose derived candidates.
    final_data.setdefault('all_candidate_models', [])
    final_data['all_candidate_models'] = merge_items(derived.get('all_candidate_models', []), ensure_list(final_data.get('all_candidate_models')) + ensure_list(final_data.get('models')), 'all_candidate_models')

    # models remains a practical selected table, but is enriched from benchmark_ready when Chief returns too few.
    if len(ensure_list(final_data.get('models'))) < 10:
        final_data['models'] = merge_items(ensure_list(final_data.get('models')), derived.get('benchmark_ready_models', [])[:80], 'models')

    for section in ['benchmark_ready_models','repositories','datasets','dataset_links','model_dataset_links','dataset_followup_tasks','model_classification','representative_models_by_category','metrics','papers','benchmark_implications','open_questions']:
        final_data[section] = merge_items(derived.get(section, []), ensure_list(final_data.get(section)), section)
        final_data[section] = dedupe_objects(final_data[section], section)

    # Carry deterministic GitHub enrichment evidence into final memory so it is visible and discussable.
    github_rows = ensure_list(compact_evidence_pool.get('github_missing_model_enrichment'))
    if github_rows:
        final_data['github_missing_model_enrichment'] = dedupe_objects(github_rows, 'github_missing_model_enrichment')
        final_data.setdefault('benchmark_implications', [])
        final_data['benchmark_implications'].append({
            'topic': 'GitHub missing-link enrichment',
            'decision': 'Models without repository links were searched by exact model name on GitHub before the global meeting; candidate repositories were saved as evidence_level=github_search.',
            'reason': 'Some models lacked GitHub links in literature evidence; repository evidence should be added to the evidence pool before deployment decisions.',
            'evidence': f'{len(github_rows)} GitHub enrichment records saved to {GITHUB_MISSING_MODEL_ENRICHMENT_JSON.relative_to(ROOT)}',
        })

    # Carry Qwen-Max web-search enrichment evidence into final memory.
    qwen_rows = ensure_list(compact_evidence_pool.get('qwen_web_enrichment'))
    if qwen_rows:
        final_data['qwen_web_enrichment'] = dedupe_objects(qwen_rows, 'qwen_web_enrichment')
        final_data.setdefault('benchmark_implications', [])
        final_data['benchmark_implications'].append({
            'topic': 'Qwen-Max web-search enrichment',
            'decision': 'Qwen3.7-Max web search was used as a supplemental missing-evidence layer for repositories, datasets, weights, web servers, and paper pages.',
            'reason': 'Structured databases and GitHub API may miss aliases, author pages, supplementary links, and new web evidence.',
            'evidence': f'{len(qwen_rows)} Qwen web enrichment records saved to {QWEN_WEB_ENRICHMENT_JSON.relative_to(ROOT)}',
        })

    # Build taxonomy and discussion after final enrichment so counts reflect final output.
    final_data = enrich_model_taxonomy_and_representatives(final_data)
    base_discussion = build_agent_discussion(md_json, metric_json, critic_json, final_data, compact_evidence_pool)
    meeting_trace_md = build_meeting_trace_markdown(final_data, md_json, metric_json, critic_json, compact_evidence_pool)
    final_data['agent_discussion'] = [{'agent': 'meeting_trace', 'role': '多 Agent 全局会议记录', 'status': 'ok', 'markdown': meeting_trace_md}] + base_discussion
    return final_data





# ------------------------- Meeting trace + taxonomy helpers -------------------------
# v4.5: 用户指定的两套模型分类体系：
#   1) 数据/输入表示（Representation）
#   2) 模型架构（Architecture）
# 分类和代表模型选择采用确定性规则，并按规范模型名去重，避免 Co-AMPpred/AMP-BERT 等重复出现。

MODEL_ALIAS_MAP: Dict[str, str] = {
    'amp scanner v2': 'AMP Scanner v2',
    'ampscanner v2': 'AMP Scanner v2',
    'ampscanner vr 2': 'AMP Scanner v2',
    'antimicrobial peptide scanner vr 2': 'AMP Scanner v2',
    'amp scanner vr 2': 'AMP Scanner v2',
    'ampscannerv2': 'AMP Scanner v2',
    'amp scanner': 'AMP Scanner',
    'ampscanner': 'AMP Scanner',
    'amp bert': 'AMP-BERT',
    'ampbert': 'AMP-BERT',
    'samp pred gat': 'sAMPpred-GAT',
    'samp pred gat': 'sAMPpred-GAT',
    'sampred gat': 'sAMPpred-GAT',
    'samp-pred-gat': 'sAMPpred-GAT',
    'samppred gat': 'sAMPpred-GAT',
    'samppred-gat': 'sAMPpred-GAT',
    'ampir': 'Ampir',
    'ampeppy': 'amPEPpy',
    'ampep': 'AmPEP',
    'amplify': 'AMPlify',
    'ai4amp predictor': 'AI4AMP',
    'af qsam ampdiscover': 'AMPDiscover',
    'ampdiscover': 'AMPDiscover',
    'apex 1 1': 'APEX 1.1',
    'apex 11': 'APEX 1.1',
    'broadamp gpt amp prediction model': 'BroadAMP-GPT AMP prediction model',
    'broadamp gpt generation model': 'BroadAMP-GPT generation model',
    'broadamp gpt mic prediction models': 'BroadAMP-GPT MIC prediction models',
    'proteogpt ampsorter': 'ProteoGPT (AMPSorter)',
    'ampsorter': 'ProteoGPT (AMPSorter)',
    'proteogpt': 'ProteoGPT (AMPSorter)',
    'c amps predict': 'C_AMPs-predict',
    'c-amps-predict': 'C_AMPs-predict',
    'unidl4biopep': 'UniDL4BioPep',
    'unidl4biopep': 'UniDL4BioPep',
    'samp pfpdeep': 'sAMP-PFPDeep',
    'iamp ca2l': 'iAMP-CA2L',
    'iamp ca 2l': 'iAMP-CA2L',
    'labampsgcn': 'LABAMPsGCN',
    'labamps gcn': 'LABAMPsGCN',
    'esm axp gdl': 'esm-AxP-GDL',
    'esm-axp-gdl': 'esm-AxP-GDL',
}


def canonicalize_model_name(name: Any) -> str:
    raw = str(name or '').strip()
    if not raw:
        return ''
    n = normalize_key(raw.replace('_', ' '))
    n = n.replace(' vr ', ' vr ').strip()
    if n in MODEL_ALIAS_MAP:
        return MODEL_ALIAS_MAP[n]
    # normalize common punctuation-only variants while preserving visible name.
    return raw.strip().rstrip('.')


def model_key(item: Any) -> str:
    if isinstance(item, dict):
        raw = item.get('canonical_name') or item.get('model_name') or item.get('name') or ''
    else:
        raw = str(item or '')
    can = canonicalize_model_name(raw)
    return normalize_key(can)


def dedupe_models_by_name(items: Any) -> List[Dict[str, Any]]:
    """Deduplicate model-like rows by canonical model name.
    Prefer rows with stronger evidence/code/dataset fields, then merge missing fields.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for item in ensure_list(items):
        if not isinstance(item, dict):
            continue
        row = clean_row_dict(item)
        if not row:
            continue
        name = row.get('model_name') or row.get('canonical_name') or row.get('name')
        if is_missing_value(name):
            continue
        canonical = canonicalize_model_name(name)
        key = normalize_key(canonical)
        if not key:
            continue
        row['model_name'] = canonical
        row['canonical_name'] = row.get('canonical_name') or canonical
        if key not in buckets:
            buckets[key] = row
        else:
            old = buckets[key]
            # Keep the better row as base, merge the other into it.
            if model_quality_score(row) > model_quality_score(old):
                buckets[key] = merge_candidate(row, old)
            else:
                buckets[key] = merge_candidate(old, row)
    return list(buckets.values())


REPRESENTATION_CATEGORY_DEFS: List[Dict[str, Any]] = [
    {
        'taxonomy': 'representation',
        'category': 'traditional_physicochemical_statistical_features',
        'title': '传统理化/统计特征为主',
        'description': '以全局理化性质、氨基酸组成/伪氨基酸组成、k-mer 统计等手工特征为主，不显式保留序列位置，只做整体向量化。',
        'model_examples': ['Macrel', 'Ampir', 'amPEPpy', 'AMPpred-EL'],
        'preferred_representatives': ['Macrel', 'amPEPpy'],
        'include_keywords': ['random forest', 'svm', 'support vector', 'xgboost', 'lightgbm', 'composition', 'physicochemical', 'pse', 'feature', 'k-mer', 'kmer', 'statistical', 'ml'],
        'exclude_keywords': ['bert', 'transformer', 'prot', 'esm', 'gpt', 'graph', 'gat', 'gcn'],
    },
    {
        'taxonomy': 'representation',
        'category': 'sequence_encoding_representation',
        'title': '纯序列/编码表示',
        'description': '直接对氨基酸序列做编码，如 one-hot、索引 embedding、PC6 理化编码、PseKRAAC 降维编码，或把序列转成小图像；不依赖大型 PLM，也不显式用 3D 结构。',
        'model_examples': ['APIN', 'AMP Scanner v2', 'AI4AMP', 'AMPlify', 'APEX', 'APEX 1.1', 'iAMPCN', 'Deep-AmPEP30', 'sAMP-PFPDeep', 'iAMP-CA2L'],
        'preferred_representatives': ['AI4AMP', 'AMPlify'],
        'include_keywords': ['one-hot', 'one hot', 'embedding', 'pc6', 'psekraac', 'sequence encoding', 'sequence', 'lstm', 'cnn', 'rnn', 'capsnet'],
        'exclude_keywords': ['bert', 'protbert', 'prott5', 'esm', 'gpt', 'graph', 'gat', 'gcn', 'structure'],
    },
    {
        'taxonomy': 'representation',
        'category': 'protein_language_model_representation',
        'title': '蛋白语言模型（PLM）表示',
        'description': '使用预训练蛋白语言模型（BERT/T5/ESM/ProtT5/GPT-2 等）从序列生成高维 embedding，再接 CNN/MLP 等下游分类器。',
        'model_examples': ['C_AMPs-predict', 'LMPred', 'UniDL4BioPep', 'ProteoGPT (AMPSorter)', 'PepNet'],
        'preferred_representatives': ['LMPred', 'PepNet'],
        'include_keywords': ['bert', 'protein language model', 'language model', 'plm', 'esm', 'prott5', 'prot t5', 't5', 'gpt', 'prottrans', 'transformer embedding'],
        'exclude_keywords': [],
    },
    {
        'taxonomy': 'representation',
        'category': 'structure_graph_representation',
        'title': '结构/图表示',
        'description': '将肽构造成图：节点可以是原子、残基或 k-mer；边来自共价键、空间距离或共现关系；有的结合预测 3D 结构，节点特征可叠加 ESM embedding。',
        'model_examples': ['sAMPpred-GAT', 'LABAMPsGCN', 'AMPs-Net', 'esm-AxP-GDL'],
        'preferred_representatives': ['sAMPpred-GAT', 'AMPs-Net'],
        'include_keywords': ['graph', 'gat', 'gcn', 'gnn', 'message passing', 'node', 'edge', '3d', 'structure', 'esmfold', 'trrosetta'],
        'exclude_keywords': [],
    },
    {
        'taxonomy': 'representation',
        'category': 'multimodal_hybrid_representation',
        'title': '多模态 / 混合表示',
        'description': '同时使用两种及以上类型的输入，如 one-hot 序列 + 大量理化特征，或 PLM embedding + 手工 PD 特征等。',
        'model_examples': ['AMPidentifier', 'SMEP', 'SenseXAMP'],
        'preferred_representatives': ['SenseXAMP', 'AMPidentifier'],
        'include_keywords': ['multi-modal', 'multimodal', 'hybrid', 'fusion', 'combined', 'cross-attention', 'cross attention', 'multi feature', 'multi-feature'],
        'exclude_keywords': [],
    },
]


ARCHITECTURE_CATEGORY_DEFS: List[Dict[str, Any]] = [
    {
        'taxonomy': 'architecture',
        'category': 'machine_learning_models',
        'title': '机器学习模型',
        'description': '基于特征工程 + 传统分类器/回归器，如 Random Forest、SVM、LightGBM、逻辑回归等。',
        'model_examples': ['Macrel', 'Ampir', 'amPEPpy', 'AMPpred-EL'],
        'preferred_representatives': ['Macrel', 'amPEPpy'],
        'include_keywords': ['random forest', 'svm', 'support vector', 'lightgbm', 'xgboost', 'logistic', 'knn', 'machine learning', 'ml'],
        'exclude_keywords': ['cnn', 'lstm', 'rnn', 'bert', 'transformer', 'gpt', 'gnn', 'gat', 'gcn'],
    },
    {
        'taxonomy': 'architecture',
        'category': 'cnn_dominant_models',
        'title': 'CNN 主导模型',
        'description': '一维/二维卷积是主干，负责自动学习局部 motif 与局部模式，有时堆叠多层 CNN（DenseNet、VGG、ResNet 等）。',
        'model_examples': ['APIN', 'AMPidentifier', 'LMPred', 'UniDL4BioPep', 'Deep-AmPEP30', 'sAMP-PFPDeep', 'iAMPCN'],
        'preferred_representatives': ['APIN', 'Deep-AmPEP30'],
        'include_keywords': ['cnn', 'convolution', 'convolutional', 'resnet', 'vgg', 'densenet', 'capsnet'],
        'exclude_keywords': ['graph', 'gat', 'gcn'],
    },
    {
        'taxonomy': 'architecture',
        'category': 'rnn_lstm_dominant_models',
        'title': 'RNN/LSTM 主导模型',
        'description': '以（双向）LSTM/GRU 为主干，建模序列顺序依赖；注意力层通常作为辅助模块。',
        'model_examples': ['C_AMPs-predict', 'AMPlify'],
        'preferred_representatives': ['AMPlify', 'C_AMPs-predict'],
        'include_keywords': ['lstm', 'gru', 'rnn', 'bi-lstm', 'bilstm', 'bidirectional lstm'],
        'exclude_keywords': [],
    },
    {
        'taxonomy': 'architecture',
        'category': 'cnn_rnn_hybrid_models',
        'title': 'CNN + RNN 混合模型',
        'description': '先用 CNN 抽局部 motif，再用 LSTM/BiLSTM 建模长程依赖，最后接全连接/分类器。',
        'model_examples': ['AMP Scanner v2', 'AI4AMP', 'iAMP-CA2L'],
        'preferred_representatives': ['AMP Scanner v2', 'AI4AMP'],
        'include_keywords': ['cnn+lstm', 'cnn lstm', 'cnn-bilstm', 'cnn bilstm', 'convolution lstm', 'hybrid cnn'],
        'exclude_keywords': [],
    },
    {
        'taxonomy': 'architecture',
        'category': 'transformer_llm_dominant_models',
        'title': 'Transformer / LLM 主导模型',
        'description': '主干是多头自注意力/Transformer 模块（包括 GPT-2、BERT、ProtT5 等大模型），或在下游显式堆叠 Transformer block、cross-attention 作为核心特征提取器。',
        'model_examples': ['ProteoGPT (AMPSorter/AMPGenix)', 'SenseXAMP', 'PepNet'],
        'preferred_representatives': ['PepNet', 'SenseXAMP'],
        'include_keywords': ['transformer', 'bert', 'prott5', 'prot t5', 'gpt', 'llm', 'self-attention', 'self attention', 'cross-attention', 'cross attention'],
        'exclude_keywords': [],
    },
    {
        'taxonomy': 'architecture',
        'category': 'gnn_models',
        'title': '图神经网络（GNN）模型',
        'description': '将肽建模为图结构（原子/残基/k-mer 为节点，键/空间距离/共现为边），使用 GCN、GAT 等进行 message passing。',
        'model_examples': ['sAMPpred-GAT', 'LABAMPsGCN', 'AMPs-Net', 'esm-AxP-GDL'],
        'preferred_representatives': ['sAMPpred-GAT', 'AMPs-Net'],
        'include_keywords': ['gnn', 'gcn', 'gat', 'graph neural', 'graph attention', 'message passing'],
        'exclude_keywords': [],
    },
    {
        'taxonomy': 'architecture',
        'category': 'pipeline_or_ensemble_frameworks',
        'title': '其他（多阶段流水线 / 集成框架）',
        'description': '用多个模型串联或集成，或多模型 + 集成/堆叠。',
        'model_examples': ['APEX', 'APEX 1.1', 'SMEP'],
        'preferred_representatives': ['APEX', 'APEX 1.1'],
        'include_keywords': ['pipeline', 'ensemble', 'stacking', 'stacked', 'multi-stage', 'framework', 'workflow', 'apex'],
        'exclude_keywords': [],
    },
]

MODEL_TAXONOMY_DEFS: List[Dict[str, Any]] = REPRESENTATION_CATEGORY_DEFS + ARCHITECTURE_CATEGORY_DEFS


def _model_text(m: Dict[str, Any]) -> str:
    vals = []
    for k in ['model_name','canonical_name','task_type','method_family','architecture_or_algorithm','input_features','model_category_hint','candidate_reason','dataset_source_or_link','evidence_level']:
        vals.append(str(m.get(k, '') or ''))
    vals.extend(map(str, ensure_list(m.get('blocking_issues'))))
    return normalize_key(' '.join(vals))


def _has_any_keyword(text: str, keywords: List[str]) -> bool:
    return any(normalize_key(k) in text for k in keywords if k)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _pick_numeric_field(obj: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if not isinstance(obj, dict):
            continue
        v = obj.get(k)
        if v is None or is_missing_value(v):
            continue
        try:
            f = float(v)
            if f >= 0:
                return f
        except Exception:
            continue
    return default


def _pick_text_field(obj: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if isinstance(obj, dict):
            v = obj.get(k)
            if v is not None and not is_missing_value(v):
                return str(v)
    return ''


def _load_journal_impact_factor_map() -> Dict[str, float]:
    """Load optional user-maintained journal impact factors.

    The project should not hallucinate impact factors.  If the evidence pool does
    not contain an IF value, users may provide one of these files:
      data/journal_impact_factors.json
      data/journal_impact_factors.csv

    JSON accepted shapes:
      {"Briefings in Bioinformatics": 9.5, ...}
      [{"journal": "...", "impact_factor": 9.5}, ...]
    CSV columns: journal, impact_factor.
    """
    out: Dict[str, float] = {}
    base = Path('data')
    json_path = base / 'journal_impact_factors.json'
    csv_path = base / 'journal_impact_factors.csv'
    def add(journal: Any, value: Any) -> None:
        j = normalize_key(journal)
        f = _safe_float(value, 0.0)
        if j and f > 0:
            out[j] = f
    try:
        if json_path.exists():
            data = json_load(json_path)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        add(k or v.get('journal') or v.get('source_journal'), v.get('impact_factor') or v.get('journal_impact_factor') or v.get('if'))
                    else:
                        add(k, v)
            elif isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        add(row.get('journal') or row.get('source_journal') or row.get('venue'), row.get('impact_factor') or row.get('journal_impact_factor') or row.get('if'))
    except Exception as e:
        print(f'    ⚠️ journal_impact_factors.json 读取失败：{e}')
    try:
        if csv_path.exists():
            import csv
            with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    add(row.get('journal') or row.get('source_journal') or row.get('venue'), row.get('impact_factor') or row.get('journal_impact_factor') or row.get('if'))
    except Exception as e:
        print(f'    ⚠️ journal_impact_factors.csv 读取失败：{e}')
    return out


def _citation_count_from_row(row: Dict[str, Any]) -> float:
    return _pick_numeric_field(row, [
        'citation_count', 'citations', 'cited_by_count', 'openalex_cited_by_count',
        'semantic_scholar_citation_count', 's2_citation_count', 'source_citation_count',
        'article_citation_count', 'paper_citation_count', 'influential_citation_count',
    ], 0.0)


def _impact_factor_from_row(row: Dict[str, Any], journal_if_map: Optional[Dict[str, float]] = None) -> float:
    f = _pick_numeric_field(row, [
        'journal_impact_factor', 'impact_factor', 'source_impact_factor', 'if', 'jif',
    ], 0.0)
    if f > 0:
        return f
    journal = normalize_key(_pick_text_field(row, ['source_journal', 'journal', 'venue', 'container_title']))
    if journal and journal_if_map:
        return float(journal_if_map.get(journal, 0.0))
    return 0.0


def article_impact_score(row: Dict[str, Any], journal_if_map: Optional[Dict[str, float]] = None) -> float:
    """Score paper influence without overpowering deployment readiness.

    Citation counts and journal IF are helpful for choosing which deployable AMP
    models to audit first, but they must not override core reproducibility:
    code/data/task match still matter more.  The score is capped at 5.0.
    """
    import math
    cites = max(0.0, _citation_count_from_row(row))
    jif = max(0.0, _impact_factor_from_row(row, journal_if_map))
    citation_component = min(math.log10(cites + 1.0), 3.0)        # 0..3
    impact_factor_component = min(math.log10(jif + 1.0) * 1.5, 2.0)  # 0..2
    return round(citation_component + impact_factor_component, 4)


def model_quality_score(m: Dict[str, Any]) -> float:
    score = 0.0
    if m.get('code_repository_url') and not is_missing_value(m.get('code_repository_url')): score += 3.0
    if m.get('model_weights_url') and not is_missing_value(m.get('model_weights_url')): score += 1.5
    if m.get('web_server_url') and not is_missing_value(m.get('web_server_url')): score += 1.0
    ds = m.get('dataset_source_or_link') or ''
    if not is_missing_value(ds) and normalize_key(ds) != 'not reported in available evidence': score += 1.0
    ev = normalize_key(m.get('evidence_level'))
    if 'fulltext' in ev: score += 2.0
    elif 'repository' in ev: score += 1.5
    elif 'abstract' in ev: score += 0.6
    elif 'review' in ev: score += 0.2
    score += min(_safe_float(m.get('confidence'), 0.0), 1.0)
    issue_text = normalize_key(' '.join(map(str, ensure_list(m.get('blocking_issues')))))
    if 'no code' in issue_text or 'no_code' in issue_text: score -= 1.0
    if 'review only' in issue_text or 'review_only' in issue_text: score -= 0.8
    if 'not amp' in issue_text or 'task mismatch' in issue_text: score -= 2.0
    if 'no dataset' in issue_text or 'dataset not' in issue_text: score -= 0.3
    return score


def _name_matches_example(m: Dict[str, Any], names: List[str]) -> bool:
    key = model_key(m)
    for name in names:
        nk = normalize_key(canonicalize_model_name(name))
        if key == nk or (nk and nk in key) or (key and key in nk):
            return True
    return False


def _model_matches_taxonomy(m: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    text = _model_text(m)
    if _name_matches_example(m, ensure_list(spec.get('model_examples'))):
        return True
    if _has_any_keyword(text, ensure_list(spec.get('exclude_keywords'))):
        return False
    return _has_any_keyword(text, ensure_list(spec.get('include_keywords')))


def classify_representation_item(m: Dict[str, Any]) -> str:
    for spec in REPRESENTATION_CATEGORY_DEFS:
        if _model_matches_taxonomy(m, spec):
            return spec['category']
    return 'sequence_encoding_representation'


def classify_architecture_item(m: Dict[str, Any]) -> str:
    for spec in ARCHITECTURE_CATEGORY_DEFS:
        if _model_matches_taxonomy(m, spec):
            return spec['category']
    text = _model_text(m)
    if _has_any_keyword(text, ['dl', 'deep learning', 'neural']):
        return 'cnn_dominant_models'
    if _has_any_keyword(text, ['ml', 'machine learning']):
        return 'machine_learning_models'
    return 'pipeline_or_ensemble_frameworks'


def classify_model_item(m: Dict[str, Any]) -> str:
    """Backward-compatible single category used in older tables.
    v4.5 returns the representation category as the primary visible category.
    """
    return classify_representation_item(m)


def _find_model_by_name(models: List[Dict[str, Any]], target: str) -> Optional[Dict[str, Any]]:
    t = normalize_key(canonicalize_model_name(target))
    for m in models:
        k = model_key(m)
        if k == t or (t and t in k) or (k and k in t):
            return m
    return None


def model_has_real_evidence(m: Any) -> bool:
    """True only when a model row is backed by the current evidence/memory.
    Taxonomy template names such as APIN/PepNet/SenseXAMP must not be rendered
    unless they actually appear in the evidence-derived model rows.
    """
    if not isinstance(m, dict):
        return False
    name = m.get('model_name') or m.get('canonical_name') or m.get('name')
    if is_missing_value(name):
        return False
    evidence_level = normalize_key(m.get('evidence_level'))
    if evidence_level in {'taxonomy_defined_pending_evidence', 'pending_evidence'}:
        return False
    issue_text = normalize_key(' '.join(map(str, ensure_list(m.get('blocking_issues')))))
    if 'not found in current evidence pool' in issue_text or 'not_found_in_current_evidence_pool' in issue_text:
        return False
    evidence_fields = [
        'source_pmid', 'pmid', 'source_pmcid', 'pmcid', 'source_doi', 'doi',
        'source_title', 'title', 'code_repository_url', 'web_server_url',
        'dataset_source_or_link', 'model_weights_url', 'evidence_level', 'chunk_id',
    ]
    for f in evidence_fields:
        v = m.get(f)
        if not is_missing_value(v) and normalize_key(v) not in {'taxonomy defined pending evidence'}:
            return True
    # A row with task/method only but no source/link is likely produced by an LLM memory merge, not evidence.
    return False


def filter_models_with_evidence(items: Any) -> List[Dict[str, Any]]:
    return [m for m in dedupe_models_by_name(items) if model_has_real_evidence(m)]


def _select_representatives(models: List[Dict[str, Any]], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set = set()
    evidence_models = [m for m in dedupe_models_by_name(models) if model_has_real_evidence(m)]
    # First follow user-specified representatives/examples, but ONLY if the model exists in evidence.
    # Do not fabricate taxonomy-only placeholder rows.
    for name in ensure_list(spec.get('preferred_representatives')) + ensure_list(spec.get('model_examples')):
        m = _find_model_by_name(evidence_models, name)
        if m is None:
            continue
        k = model_key(m)
        if not k or k in seen:
            continue
        selected.append(m)
        seen.add(k)
        if len(selected) >= 2:
            break
    # Then fill from evidence rows if needed.
    if len(selected) < 2:
        candidates = [m for m in evidence_models if _model_matches_taxonomy(m, spec) and model_key(m) not in seen]
        candidates = sorted(candidates, key=model_quality_score, reverse=True)
        for m in candidates:
            k = model_key(m)
            if not k or k in seen:
                continue
            selected.append(m)
            seen.add(k)
            if len(selected) >= 2:
                break
    return selected[:2]


def build_model_classification(final_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build user-requested Representation + Architecture taxonomy and 1-2 representatives per category.
    Deterministic and deduplicated by canonical model name.
    """
    raw_models: List[Any] = []
    for section in ['all_candidate_models', 'benchmark_ready_models', 'models']:
        raw_models.extend(ensure_list(final_data.get(section)))
    # Only evidence-backed rows are allowed to participate in classification output.
    # User-provided taxonomy model names are examples, not evidence.
    models = filter_models_with_evidence(raw_models)

    # Annotate categories once, then use the same deduped row in all tables.
    annotated: List[Dict[str, Any]] = []
    for m in models:
        row = dict(m)
        row['representation_category'] = classify_representation_item(row)
        row['architecture_category'] = classify_architecture_item(row)
        # model_category kept for backward compatibility; it mirrors representation_category.
        row['model_category'] = row.get('model_category') or row['representation_category']
        annotated.append(row)

    classification: List[Dict[str, Any]] = []
    representatives: List[Dict[str, Any]] = []
    for spec in MODEL_TAXONOMY_DEFS:
        taxonomy = spec['taxonomy']
        cat = spec['category']
        if taxonomy == 'representation':
            rows = [m for m in annotated if m.get('representation_category') == cat]
        else:
            rows = [m for m in annotated if m.get('architecture_category') == cat]
        rows = dedupe_models_by_name(rows)
        rep_rows = _select_representatives(rows, spec)
        rep_names = [r.get('model_name') or r.get('canonical_name') for r in rep_rows if r.get('model_name') or r.get('canonical_name')]
        concrete_models = []
        # List only models actually found in current evidence rows for this category.
        # The user's taxonomy examples are not rendered when absent from evidence.
        for m in rows:
            cname = canonicalize_model_name(m.get('model_name') or m.get('canonical_name'))
            if cname and cname not in concrete_models:
                concrete_models.append(cname)
        concrete_models = concrete_models[:30]
        classification.append({
            'taxonomy': taxonomy,
            'category': cat,
            'title': spec['title'],
            'description': spec['description'],
            'model_count': len(rows),
            'concrete_models': concrete_models,
            'representative_models': rep_names,
            'selection_rule': 'v4.5 用户指定分类体系 + 确定性去重：按 canonical model name 去重；优先选择用户给定示例中已在 evidence 中出现且证据/代码/数据集线索较强的 1-2 个模型。',
            'models': [r.get('model_name') or r.get('canonical_name') for r in rows if r.get('model_name') or r.get('canonical_name')],
        })
        for r in rep_rows:
            representatives.append(clean_row_dict({
                'taxonomy': taxonomy,
                'category': cat,
                'category_title': spec['title'],
                'model_name': r.get('model_name') or r.get('canonical_name'),
                'task_type': r.get('task_type'),
                'method_family': r.get('method_family'),
                'code_repository_url': r.get('code_repository_url'),
                'web_server_url': r.get('web_server_url'),
                'dataset_source_or_link': r.get('dataset_source_or_link'),
                'source_pmid': r.get('source_pmid'),
                'source_doi': r.get('source_doi'),
                'evidence_level': r.get('evidence_level'),
                'confidence': r.get('confidence'),
                'why_representative': f"按用户指定的 {spec['title']} 类别选择；优先代表该类输入表示/架构路线，后续先核查代码、权重、数据集和批量推理可行性。",
            }))
    return classification, dedupe_objects(representatives, 'representative_models_by_category')


def _is_main_amp_binary_candidate(m: Dict[str, Any]) -> bool:
    text = normalize_key(' '.join([
        str(m.get('model_name') or ''), str(m.get('canonical_name') or ''),
        str(m.get('task_type') or ''), str(m.get('method_family') or ''),
        str(m.get('candidate_reason') or ''), ' '.join(map(str, ensure_list(m.get('blocking_issues')))),
    ]))
    hard_exclude = [
        'anticancer', 'anti cancer', 'antifungal', 'anti fungal', 'antiviral', 'anti viral',
        'anti-inflammatory', 'hemolysis', 'toxicity', 'toxic', 'mic prediction', 'mic regression',
        'minimum inhibitory concentration', 'regression', 'generation', 'generator', 'generative',
        'design', 'database', 'smorf', 'small open reading frame', 'cleavage site', 'non amp',
    ]
    if any(x in text for x in hard_exclude):
        return False
    return ('amp' in text or 'antimicrobial peptide' in text or 'antibacterial peptide' in text) and any(x in text for x in ['classification','classifier','prediction','predictor','recognition','identification'])


def deployment_score(m: Dict[str, Any]) -> float:
    score = model_quality_score(m)
    # v5.4: 文章影响力参与部署排序，但不覆盖可部署性。
    # article_impact_score 通常由 citation_count + journal_impact_factor 计算得出，最大约 5。
    score += min(_safe_float(m.get('article_impact_score'), 0.0), 5.0) * 0.9
    if _is_main_amp_binary_candidate(m):
        score += 4.0
    if m.get('code_repository_url') and not is_missing_value(m.get('code_repository_url')):
        score += 4.0
    if m.get('dataset_source_or_link') and not is_missing_value(m.get('dataset_source_or_link')):
        score += 1.5
    if m.get('web_server_url') and not is_missing_value(m.get('web_server_url')):
        score += 0.7
    text = normalize_key(' '.join(map(str, ensure_list(m.get('blocking_issues')))))
    if 'no code' in text or 'no_code' in text or 'code not' in text:
        score -= 4.0
    if 'webserver_only' in text or 'web server only' in text:
        score -= 1.5
    if 'review_only' in text or 'review only' in text:
        score -= 2.0
    if not _is_main_amp_binary_candidate(m):
        score -= 6.0
    return score


def _deployment_status(m: Dict[str, Any]) -> str:
    has_code = bool(m.get('code_repository_url') and not is_missing_value(m.get('code_repository_url')))
    has_ds = bool(m.get('dataset_source_or_link') and not is_missing_value(m.get('dataset_source_or_link')))
    if has_code and has_ds and _is_main_amp_binary_candidate(m):
        return 'deploy_first_after_weight_check'
    if has_code and _is_main_amp_binary_candidate(m):
        return 'deploy_after_dataset_and_weight_check'
    if m.get('web_server_url') and not is_missing_value(m.get('web_server_url')) and _is_main_amp_binary_candidate(m):
        return 'web_or_api_wrapper_only'
    return 'not_in_final_deployment'



# v5.4 最终执行决策：核心主榜不少于 10 个，扩展部署池最多 20 个，并把论文引用量/影响因子纳入排序。
# 注意：最终部署榜按“任务匹配 + 代码证据 + 可复现潜力 + 论文影响力”排序。
FINAL_DEPLOYMENT_CORE_MIN = 10
FINAL_DEPLOYMENT_MAX = 20

# 先按明确证据和方法多样性给出优先顺序；如果某个模型在当前证据池中缺失或被过滤，后续会用高分候选自动补足。
FINAL_MAIN_DEPLOYMENT_ORDER = [
    # Core 10 / 主榜核心模型
    'AMP-BERT',
    'AMPlify',
    'Co-AMPpred',
    'iAMP-SeE',
    'LMPred',
    'MAPLE',
    'CalcAMP',
    'sAMPpred-GAT',
    'SAMP',
    'AMP Ensemble Model',
    # Extended pool / 扩展部署池候选，最多补到 20 个
    'DDM',
    'iAMPCN',
    'ACEP',
    'AI4AMP',
    'AntiBP3',
    'AMPSpeciesSpecific',
    'PyAMPA',
    'BBATProt',
    'BPFun',
    'E-CLEAP',
    'SGAC',
    'AMPBAN',
    'UniproLcad',
    'TriStack',
    'iAMP-DL',
]

# 精确部署别名：解决 CalcAMP 被误匹配成 CAMP、AMP Scanner v2 写法不一致等问题。
# 这里不使用短字符串包含匹配，只允许 exact compact-key / 明确别名匹配。
DEPLOYMENT_NAME_ALIASES: Dict[str, List[str]] = {
    'AMP-BERT': ['AMP-BERT', 'AMPBERT', 'GIST-CSBL/AMP-BERT'],
    'AMPlify': ['AMPlify', 'bcgsc/AMPlify'],
    'Co-AMPpred': ['Co-AMPpred', 'CoAMPpred', 'onkarS23/CoAMPpred'],
    'iAMP-SeE': ['iAMP-SeE', 'iAMPSeE', 'iamp-see', 'cqw0715/iAMP-SeE'],
    'LMPred': ['LMPred', 'LMPred_AMP_Prediction', 'williamdee1/LMPred_AMP_Prediction'],
    'MAPLE': ['MAPLE', 'Harkool/MAPLE'],
    'CalcAMP': ['CalcAMP', 'calc-amp', 'CDDLeiden/CalcAMP'],
    'sAMPpred-GAT': ['sAMPpred-GAT', 'sAMP-pred-GAT', 'samppred-gat', 'HongWuL/sAMPpred-GAT'],
    'SAMP': ['SAMP', 'wan-mlab/SAMP'],
    'AMP Ensemble Model': ['AMP Ensemble Model', 'researchprotein/amp'],
    'DDM': ['DDM', 'kww567upup/DDM'],
    'iAMPCN': ['iAMPCN', 'joy50706/iAMPCN'],
    'ACEP': ['ACEP', 'Fuhaoyi/ACEP'],
    'AI4AMP': ['AI4AMP', 'AI4AMP_predictor', 'LinTzuTang/AI4AMP_predictor'],
    'AntiBP3': ['AntiBP3', 'Anti-BP3', 'raghavalab/antibp3'],
    'AMPSpeciesSpecific': ['AMPSpeciesSpecific', 'AMP-Species-Specific', 'bzlee-bio/AMPSpeciesSpecific'],
    'PyAMPA': ['PyAMPA', 'SysBioUAB/PyAMPA'],
    'BBATProt': ['BBATProt', 'Xukai-YE/BBATProt'],
    'BPFun': ['BPFun', '291357657/BPFun'],
    'E-CLEAP': ['E-CLEAP', 'ECLEAP', 'Wangsicheng52/E-CLEAP'],
    'SGAC': ['SGAC', 'wyxwyx46941930/SGAC'],
    'AMPBAN': ['AMPBAN', 'AMP-BAN', 'baiwenhuim/ampban'],
    'UniproLcad': ['UniproLcad', 'harkic/UniproLcad'],
    'TriStack': ['TriStack', 'hjy23/TriStack'],
    'iAMP-DL': ['iAMP-DL', 'iAMPDL', 'mldlproject/2022-iAMP-DL'],
    'AMP Scanner v2': ['AMP Scanner v2', 'AMPScanner V2', 'Antimicrobial Peptide Scanner vr.2', 'dan-veltri/amp-scanner-v2'],
}

FINAL_DATASET_RECOMMENDATIONS = [
    {
        'dataset_name': 'iAMP-SeE Dataset / Zenodo',
        'linked_model': 'iAMP-SeE',
        'recommended_role': 'primary benchmark candidate / 主测试集候选',
        'why_selected': '当前证据中来源最清楚，包含 DRAMP、dbAMP、CAMPr-4、AMPfun、ADAPTABLE、UniProt 负样本，并有 Zenodo 线索；最适合作为第一版 benchmark 的主数据集候选。',
        'required_cleaning': '核查 Zenodo 文件；统一 FASTA/CSV 格式；确认正负标签；用 CD-HIT/MMseqs2 去冗余；过滤与模型训练集高度同源序列。',
    },
    {
        'dataset_name': 'Co-AMPpred / DEEP-AmPEP30 derived dataset',
        'linked_model': 'Co-AMPpred',
        'recommended_role': 'classic comparison set / 经典对照测试集',
        'why_selected': 'Co-AMPpred 有 GitHub 和 DEEP-AmPEP30 衍生数据线索，适合做传统 ML 与经典 AMP benchmark 对照。',
        'required_cleaning': '核查 GitHub 中正负样本数量、负样本来源和去重方式；排除训练集重叠；补充数据集版本记录。',
    },
    {
        'dataset_name': 'AMP-BERT GitHub dataset',
        'linked_model': 'AMP-BERT',
        'recommended_role': 'PLM reproduction set / PLM 模型复现测试集',
        'why_selected': 'AMP-BERT 与 PLM 模型直接配套，GitHub 中有代码和数据线索，适合验证 PLM 路线和复现 AMP-BERT。',
        'required_cleaning': '核查数据文件、标签列和训练/测试划分；拆分出外部测试集；做低同源过滤，避免 PLM 模型过拟合历史划分。',
    },
]

FINAL_METRICS_PLAN = {
    'primary_weighted_metrics': [
        {'metric_name': 'AUPRC', 'weight': 0.35, 'reason': '主指标；适合不平衡二分类，优于只看 AUROC。'},
        {'metric_name': 'MCC', 'weight': 0.30, 'reason': '综合 TP/TN/FP/FN，对类别不平衡更稳健。'},
        {'metric_name': 'Recall / Sensitivity', 'weight': 0.20, 'reason': '控制 AMP 漏检，适合发现任务。'},
        {'metric_name': 'Precision', 'weight': 0.15, 'reason': '控制假阳性，避免大量错误候选进入后续实验。'},
    ],
    'mandatory_report_metrics': ['Accuracy', 'Specificity', 'AUROC', 'F1-score', 'Confusion Matrix'],
    'statistical_reporting': ['95% bootstrap confidence interval', 'paired bootstrap or McNemar test for model comparison'],
    'threshold_policy': '在验证集上用 Max MCC 或 Max Youden Index 确定阈值；测试集禁止后验调阈值，禁止默认固定 0.5。',
    'test_matrix': ['1:1 balanced test', '1:10 mild imbalance test', '1:100 severe imbalance test', 'low-homology independent test'],
}



def _paper_key_values(row: Dict[str, Any]) -> List[str]:
    vals: List[str] = []
    for k in ['source_pmid', 'pmid', 'PMID']:
        v = row.get(k) if isinstance(row, dict) else None
        if v and not is_missing_value(v):
            vals.append('pmid:' + str(v).strip())
    for k in ['source_doi', 'doi', 'DOI']:
        v = row.get(k) if isinstance(row, dict) else None
        if v and not is_missing_value(v):
            vals.append('doi:' + normalize_key(str(v).strip()))
    title = _pick_text_field(row, ['article_source', 'source_title', 'title'])
    if title:
        vals.append('title:' + normalize_key(title)[:180])
    return vals


def _build_paper_impact_index(final_data: Dict[str, Any], journal_if_map: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Index papers by PMID/DOI/title so model rows can inherit citation/IF evidence."""
    index: Dict[str, Dict[str, Any]] = {}
    for section in ['papers', 'records', 'evidence_records', 'compact_evidence_pool']:
        for p in ensure_list(final_data.get(section)):
            if not isinstance(p, dict):
                continue
            impact_row = dict(p)
            cites = _citation_count_from_row(impact_row)
            jif = _impact_factor_from_row(impact_row, journal_if_map)
            if cites <= 0 and jif <= 0:
                # Still keep journal metadata if present; it can be helpful in memory.
                pass
            impact_row['citation_count'] = cites
            impact_row['journal_impact_factor'] = jif
            impact_row['article_impact_score'] = article_impact_score(impact_row, journal_if_map)
            for key in _paper_key_values(impact_row):
                old = index.get(key)
                if old is None or _safe_float(impact_row.get('article_impact_score'), 0) > _safe_float(old.get('article_impact_score'), 0):
                    index[key] = impact_row
    return index


def _attach_article_impact(row: Dict[str, Any], paper_index: Dict[str, Dict[str, Any]], journal_if_map: Dict[str, float]) -> Dict[str, Any]:
    """Attach citation count / journal IF / impact score to a model row.

    Priority:
      1. Values already present in model row.
      2. Matching paper record by PMID/DOI/title.
      3. Optional journal_impact_factors.* file.
    """
    out = dict(row)
    matched: Optional[Dict[str, Any]] = None
    for key in _paper_key_values(out):
        if key in paper_index:
            matched = paper_index[key]
            break
    if matched:
        for src, dst in [
            ('citation_count', 'citation_count'),
            ('cited_by_count', 'citation_count'),
            ('openalex_cited_by_count', 'citation_count'),
            ('semantic_scholar_citation_count', 'citation_count'),
            ('journal', 'source_journal'),
            ('venue', 'source_journal'),
            ('source_journal', 'source_journal'),
            ('year', 'source_year'),
        ]:
            if is_missing_value(out.get(dst)) and not is_missing_value(matched.get(src)):
                out[dst] = matched.get(src)
    # Direct row values override copied values when present.
    cites = _citation_count_from_row(out)
    jif = _impact_factor_from_row(out, journal_if_map)
    out['citation_count'] = cites
    out['journal_impact_factor'] = jif
    out['article_impact_score'] = article_impact_score(out, journal_if_map)
    journal = _pick_text_field(out, ['source_journal', 'journal', 'venue'])
    if journal:
        out['source_journal'] = journal
    return out


def _deployment_priority_bonus(row: Dict[str, Any]) -> float:
    """Small method-diversity / curated-priority bonus.

    v5.4 keeps the curated list as a weak prior only.  Strong citation/IF evidence
    can reorder candidates, but weak/non-deployable models still cannot enter.
    """
    for i, wanted in enumerate(FINAL_MAIN_DEPLOYMENT_ORDER):
        if _row_matches_deployment_target(row, wanted):
            return max(0.2, 2.0 - i * 0.06)
    return 0.0


def final_deployment_selection_score(row: Dict[str, Any]) -> float:
    return round(deployment_score(row) + _deployment_priority_bonus(row), 4)

def _deployment_compact_name(name: Any) -> str:
    return github_compact_key(canonicalize_model_name(name))


def _deployment_alias_keys(target: str) -> set:
    names = [target] + DEPLOYMENT_NAME_ALIASES.get(target, [])
    return {_deployment_compact_name(x) for x in names if _deployment_compact_name(x)}


def _row_name_keys(row: Dict[str, Any]) -> set:
    vals = [row.get('model_name'), row.get('canonical_name'), row.get('name'), row.get('matched_model_name')]
    # 仓库 URL 中的 owner/repo 也用于精确别名匹配，但不做泛化包含匹配。
    for k in ['code_repository_url', 'url', 'repository_url']:
        v = str(row.get(k) or '')
        if 'github.com/' in v.lower() or 'gitlab.com/' in v.lower():
            vals.append(v.rstrip('/').split('/')[-1])
            parts = v.rstrip('/').split('/')
            if len(parts) >= 2:
                vals.append('/'.join(parts[-2:]))
    return {_deployment_compact_name(x) for x in vals if _deployment_compact_name(x)}


def _row_matches_deployment_target(row: Dict[str, Any], target: str) -> bool:
    return bool(_row_name_keys(row) & _deployment_alias_keys(target))


def _strict_main_deployment_candidate(m: Dict[str, Any]) -> bool:
    """只允许进入最终部署榜的模型。

    这里比 all_candidate_models 严格：必须是 AMP 二分类/识别/预测方向，且至少有代码仓库证据。
    但不会因为 dataset_source 里出现 “database” 就误杀，因为很多合格数据集来自 APD/DRAMP/dbAMP 等数据库。
    """
    if not isinstance(m, dict):
        return False
    if not model_has_real_evidence(m):
        return False
    if not _is_main_amp_binary_candidate(m):
        return False
    if is_missing_value(m.get('code_repository_url')):
        return False
    text = normalize_key(' '.join([
        str(m.get('model_name') or ''),
        str(m.get('canonical_name') or ''),
        str(m.get('task_type') or ''),
        str(m.get('method_family') or ''),
        str(m.get('candidate_reason') or ''),
        ' '.join(map(str, ensure_list(m.get('blocking_issues')))),
    ]))
    hard_exclude = [
        'antifungal', 'anti fungal', 'afp ', ' afp',
        'anticancer', 'anti cancer', 'acp ', ' acp',
        'antiviral', 'anti viral', 'avp ', ' avp',
        'antimalarial', 'anti malarial', 'malaria',
        'mic regression', 'mic prediction', 'regression task', 'toxicity', 'hemolysis',
        'generation', 'generator', 'generative', 'design', 'webserver only', 'web-server only',
    ]
    # DBAASP / CAMP 这类数据库或平台不能作为本地部署模型；但 APD/DRAMP/dbAMP 作为数据来源不应误杀模型。
    database_like_names = {'dbaasp', 'camp', 'campr3', 'apd3', 'dramp', 'dbamp', 'ampsphere'}
    if _deployment_compact_name(m.get('model_name') or m.get('canonical_name')) in database_like_names:
        return False
    return not any(x in text for x in hard_exclude)


def _best_model_row_by_name(models: List[Dict[str, Any]], wanted_name: str) -> Optional[Dict[str, Any]]:
    """Find model by exact normalized aliases only.

    v5.3 修复：不能再用 “k in target / target in k” 的短字符串包含匹配，
    否则 CalcAMP 会被 CAMP 命中，AMP 也会误命中大量非目标模型。
    """
    matches = [m for m in models if isinstance(m, dict) and _row_matches_deployment_target(m, wanted_name)]
    if not matches:
        return None
    return sorted(matches, key=deployment_score, reverse=True)[0]


def _deployment_readiness(m: Dict[str, Any], rank: Optional[int] = None) -> str:
    has_ds = bool(m.get('dataset_source_or_link') and not is_missing_value(m.get('dataset_source_or_link')))
    prefix = 'core_main_' if rank and rank <= FINAL_DEPLOYMENT_CORE_MIN else 'extended_pool_'
    if has_ds:
        return prefix + 'deploy_after_weight_and_data_check'
    return prefix + 'deploy_after_dataset_mapping_and_weight_check'


def _deployment_next_action(m: Dict[str, Any]) -> str:
    actions = ['检查仓库是否有可批量推理脚本', '确认预训练权重或训练复现脚本']
    if is_missing_value(m.get('dataset_source_or_link')):
        actions.append('补充模型对应训练/测试数据集链接')
    else:
        actions.append('下载并标准化对应数据集')
    actions.append('封装为统一 predict_proba(input_fasta) 接口')
    return '；'.join(actions)


def _deployment_tier(rank: int) -> str:
    return 'core_main_benchmark_top10' if rank <= FINAL_DEPLOYMENT_CORE_MIN else 'extended_deployment_pool_11_20'


def _make_deployment_row(row: Dict[str, Any], rank: int, reason_name: Optional[str] = None) -> Dict[str, Any]:
    reason_name = reason_name or (row.get('model_name') or row.get('canonical_name') or '')
    return clean_row_dict({
        'deployment_rank': rank,
        'deployment_tier': _deployment_tier(rank),
        'model_name': row.get('model_name') or row.get('canonical_name') or reason_name,
        'canonical_name': row.get('canonical_name') or canonicalize_model_name(row.get('model_name') or reason_name),
        'representation_category': row.get('representation_category'),
        'architecture_category': row.get('architecture_category'),
        'task_type': row.get('task_type'),
        'method_family': row.get('method_family'),
        'code_repository_url': row.get('code_repository_url'),
        'web_server_url': row.get('web_server_url'),
        'dataset_source_or_link': row.get('dataset_source_or_link'),
        'deployment_status': _deployment_readiness(row, rank),
        'deployment_reason': _final_model_reason(reason_name),
        'first_next_action': _deployment_next_action(row),
        'blocking_issues': row.get('blocking_issues'),
        'evidence_level': row.get('evidence_level'),
        'confidence': row.get('confidence'),
        'source_journal': row.get('source_journal') or row.get('journal') or row.get('venue'),
        'citation_count': row.get('citation_count'),
        'journal_impact_factor': row.get('journal_impact_factor'),
        'article_impact_score': row.get('article_impact_score'),
        'deployment_selection_score': row.get('deployment_selection_score'),
        'source_pmid': row.get('source_pmid'),
        'source_doi': row.get('source_doi'),
    })


def build_final_deployment_models(final_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v5.4: 生成 10-20 个最终部署模型，并纳入文章影响力。

    规则：
    1. 先过滤：必须是 AMP prediction / classification 主任务，并有代码证据。
    2. 排序：deployment readiness + evidence quality + citation_count + journal_impact_factor + weak curated-priority prior。
    3. 引用量来自 evidence/model/paper 记录中的 citation_count / cited_by_count / OpenAlex / Semantic Scholar 字段。
    4. 影响因子不凭空编；优先使用记录字段，或用户维护的 data/journal_impact_factors.json/csv。
    5. 输出 Top 10 core main benchmark + 11-20 extended deployment pool。
    """
    raw: List[Any] = []
    for section in ['benchmark_ready_models', 'models', 'all_candidate_models']:
        raw.extend(ensure_list(final_data.get(section)))

    journal_if_map = _load_journal_impact_factor_map()
    paper_index = _build_paper_impact_index(final_data, journal_if_map)

    models = []
    for item in filter_models_with_evidence(raw):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row['representation_category'] = row.get('representation_category') or classify_representation_item(row)
        row['architecture_category'] = row.get('architecture_category') or classify_architecture_item(row)
        row = _attach_article_impact(row, paper_index, journal_if_map)
        row['deployment_selection_score'] = final_deployment_selection_score(row)
        models.append(row)
    models = dedupe_models_by_name(models)

    eligible = [m for m in models if _strict_main_deployment_candidate(m)]

    # If a curated preferred model has multiple aliases/rows, keep the best row by the new composite score.
    curated_rows: List[Dict[str, Any]] = []
    curated_seen: set = set()
    for wanted in FINAL_MAIN_DEPLOYMENT_ORDER:
        matches = [m for m in eligible if _row_matches_deployment_target(m, wanted)]
        if not matches:
            continue
        row = sorted(matches, key=final_deployment_selection_score, reverse=True)[0]
        k = model_key(row)
        if k and k not in curated_seen:
            curated_seen.add(k)
            row = dict(row)
            row['_curated_reason_name'] = wanted
            curated_rows.append(row)

    # Add all other eligible rows, then rank globally.  This makes citation/IF matter
    # rather than blindly following the hand-written priority list.
    all_rows: List[Dict[str, Any]] = []
    seen: set = set()
    for row in curated_rows + eligible:
        k = model_key(row)
        if not k or k in seen:
            continue
        seen.add(k)
        row = dict(row)
        row['deployment_selection_score'] = final_deployment_selection_score(row)
        all_rows.append(row)

    all_rows = sorted(all_rows, key=final_deployment_selection_score, reverse=True)

    selected: List[Dict[str, Any]] = []
    for row in all_rows:
        if len(selected) >= FINAL_DEPLOYMENT_MAX:
            break
        rank = len(selected) + 1
        reason_name = row.get('_curated_reason_name') or row.get('model_name') or row.get('canonical_name') or 'evidence-backed AMP model'
        selected.append(_make_deployment_row(row, rank, reason_name))

    return selected


def _final_model_reason(name: str) -> str:
    reasons = {
        'AMP-BERT': 'PLM/Transformer 路线代表；有代码和数据线索，适合作为现代 PLM 基线。',
        'AMPlify': '经典 AMP 分类深度学习模型；有 GitHub，适合作为纯序列/RNN-Attention 基线。',
        'Co-AMPpred': '传统特征/ML 路线代表；有 GitHub 和 DEEP-AmPEP30 衍生数据线索。',
        'iAMP-SeE': '有 GitHub 和 Zenodo 数据线索，适合作为强基准模型和主测试集配套模型。',
        'LMPred': 'PLM embedding + CNN 路线代表；有 GitHub 和数据线索。',
        'MAPLE': '较新的 PLM/Transformer AMP 模型；数据规模大，有 GitHub，适合现代路线对比。',
        'CalcAMP': '轻量、可解释、数据有 Zenodo 线索，适合作为传统/结构特征 baseline；注意不能误匹配为 CAMP 数据库。',
        'sAMPpred-GAT': '结构/图/GNN 路线代表；有 GitHub/web server，适合覆盖图神经网络类别。',
        'SAMP': '传统特征/集成 ML 路线代表；有 GitHub，适合作为轻量可复现 baseline。',
        'AMP Ensemble Model': '公开代码与 web server 线索较完整，适合作为 ensemble/工具化模型对照。',
        'DDM': '有 GitHub 和数据线索的 AMP 分类模型，适合作为扩展部署池候选。',
        'iAMPCN': 'CNN 路线代表，有 GitHub，适合覆盖深度序列模型。',
        'ACEP': 'CNN/特征混合路线代表，有代码线索，适合作为早期深度学习 AMP 识别对照。',
        'AI4AMP': 'CNN+RNN/PC6 编码路线代表，有代码和 web server 线索，适合扩展对比。',
        'AntiBP3': '传统 ML 抗菌肽预测路线，有代码和 web server 线索，可作为 antibacterial/AMP 边界任务对照。',
        'AMPSpeciesSpecific': '物种特异 AMP 分类模型，有 GitHub，适合扩展任务或分物种 benchmark。',
        'PyAMPA': 'Python 工具化 AMP predictor，有 GitHub，适合工程 baseline 和批量化封装。',
        'BBATProt': 'PLM/深度学习路线，有 GitHub，适合补充现代序列模型。',
        'BPFun': 'Bioactive peptide function prediction 工具，有 GitHub 和数据线索，可作为多功能肽扩展对照。',
        'E-CLEAP': '传统/解释性 ML 方向，有 GitHub 和数据线索，适合作为可解释模型扩展对照。',
        'SGAC': '结构/图或注意力路线，有 GitHub，可作为 GNN/结构扩展部署候选。',
        'AMPBAN': 'GNN/PLM 融合路线，有 GitHub 数据线索，适合作为扩展池候选。',
        'UniproLcad': 'CNN 路线，有 GitHub，适合作为扩展序列模型。',
        'TriStack': 'CNN/stacking 路线，有 GitHub，适合作为 ensemble/stacking 扩展模型。',
        'iAMP-DL': '深度学习 AMP prediction 模型，有 GitHub，适合作为扩展部署候选。',
    }
    return reasons.get(name, '证据池命中、任务属于 AMP prediction/classification 且有代码线索，适合作为部署候选。')


def build_final_recommended_datasets(final_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw: List[Any] = []
    for section in ['all_candidate_models', 'benchmark_ready_models', 'models']:
        raw.extend(ensure_list(final_data.get(section)))
    models = [m for m in filter_models_with_evidence(raw) if isinstance(m, dict)]
    out: List[Dict[str, Any]] = []
    for rank, spec in enumerate(FINAL_DATASET_RECOMMENDATIONS, start=1):
        m = _best_model_row_by_name(models, spec['linked_model']) or {}
        ds = first_nonempty(m.get('dataset_source_or_link'), spec.get('dataset_source_or_link'), '需要从仓库/Zenodo/补充材料中确认')
        out.append(clean_row_dict({
            'dataset_rank': rank,
            'dataset_name': spec['dataset_name'],
            'linked_model': spec['linked_model'],
            'recommended_role': spec['recommended_role'],
            'dataset_source_or_link': ds,
            'why_selected': spec['why_selected'],
            'required_cleaning': spec['required_cleaning'],
            'source_pmid': m.get('source_pmid'),
            'source_doi': m.get('source_doi'),
            'evidence_level': m.get('evidence_level'),
            'status': 'recommended_top3_dataset_needs_cleaning_and_version_lock',
        }))
    return out


def build_final_metrics_plan(final_data: Dict[str, Any]) -> Dict[str, Any]:
    return dict(FINAL_METRICS_PLAN)


def build_final_execution_decision(final_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'summary': '核心主榜不少于 10 个模型，扩展部署池最多 20 个模型；部署名单按任务匹配、代码/数据可复现性、文章引用量和期刊影响因子综合排序；推荐 3 个数据集；采用 AUPRC/MCC/Recall/Precision 加权主指标。候选模型和 Agent 讨论放在后文附录。',
        'final_models': final_data.get('final_deployment_models', []),
        'final_datasets': final_data.get('final_recommended_datasets', []),
        'final_metrics_plan': final_data.get('final_metrics_plan', {}),
        'excluded_from_main_benchmark': [
            {'group': '跨界模型', 'examples': ['AI4AFP', 'ESM2-AFPpred', 'ACP-DL', 'CTCM-Neo & ConformaX-PEP'], 'reason': '抗真菌/抗癌/抗疟疾等任务不是通用 AMP 二分类主榜。'},
            {'group': '回归或 MIC 模型', 'examples': ['ANIA', 'STAMP', 'LLAMP'], 'reason': '回归任务指标体系不同，不进入二分类主榜。'},
            {'group': '生成式/设计模型', 'examples': ['AMP-GPT', 'AmpGPT2', 'AMPGAN', 'PepGen', 'AMP-Designer'], 'reason': '生成模型应放入扩展设计任务，不进入判别式二分类主榜。'},
            {'group': '仅 webserver 或数据库', 'examples': ['ADAM', 'CAMPR3', 'DBAASP', 'AMPDiscover'], 'reason': '无法稳定本地批量推理，先不作为主部署模型。'},
        ],
    }

def enrich_model_taxonomy_and_representatives(final_data: Dict[str, Any]) -> Dict[str, Any]:
    classification, reps = build_model_classification(final_data)
    final_data['model_classification'] = classification
    final_data['representative_models_by_category'] = reps
    for section in ['all_candidate_models', 'benchmark_ready_models', 'models']:
        rows = []
        for item in filter_models_with_evidence(final_data.get(section)):
            row = dict(item)
            row['representation_category'] = classify_representation_item(row)
            row['architecture_category'] = classify_architecture_item(row)
            row['model_category'] = row.get('model_category') or row['representation_category']
            rows.append(row)
        final_data[section] = rows
    final_data['final_deployment_models'] = build_final_deployment_models(final_data)
    final_data['final_recommended_datasets'] = build_final_recommended_datasets(final_data)
    final_data['final_metrics_plan'] = build_final_metrics_plan(final_data)
    final_data['final_execution_decision'] = build_final_execution_decision(final_data)
    return final_data


def compact_agent_decisions(output: Any, max_items: int = 8) -> List[str]:
    if not isinstance(output, dict):
        return [trunc(str(output), 700)] if output else []
    points: List[str] = []
    for key in ['final_decision','summary','reasoning','decision','conclusion','scout_report_markdown','metrics_report_markdown','critic_report_markdown']:
        if output.get(key):
            points.append(str(output.get(key)))
    for key in ['benchmark_implications','open_questions','warnings','critical_warnings','dataset_followup_tasks','blocking_or_filter_records']:
        for item in ensure_list(output.get(key))[:max_items]:
            if isinstance(item, dict):
                points.append(trunc(json_dumps(item, 0), 600))
            elif item:
                points.append(trunc(str(item), 600))
    for key, label in [('all_candidate_models','候选模型'),('benchmark_ready_models','优先模型'),('datasets','数据集'),('model_dataset_links','模型-数据集关系')]:
        n = len(ensure_list(output.get(key)))
        if n:
            points.insert(0, f'{label}数量：{n}')
    return points[:max_items]


def render_bullet_list(items: List[str], fallback: str = '暂无明确结论。') -> str:
    if not items:
        return f'- {fallback}'
    return '\n'.join([f'- {x}' for x in items])


def _taxonomy_label(taxonomy: str) -> str:
    return '数据/输入表示（Representation）' if taxonomy == 'representation' else '模型架构（Architecture）'


def classification_lines_by_taxonomy(classification: List[Dict[str, Any]], taxonomy: str) -> List[str]:
    lines: List[str] = []
    for c in classification:
        if not isinstance(c, dict) or c.get('taxonomy') != taxonomy:
            continue
        examples = ', '.join(map(str, ensure_list(c.get('concrete_models'))))
        reps = ', '.join(map(str, ensure_list(c.get('representative_models'))[:2])) or '待定'
        lines.append(f"| {c.get('title')} | {c.get('description')} | {examples} | {reps} |")
    if not lines:
        lines.append('| 暂无分类 | 暂无 | 暂无 | 待定 |')
    return lines


def representative_lines_by_taxonomy(reps: List[Dict[str, Any]], taxonomy: str) -> List[str]:
    lines: List[str] = []
    seen = set()
    for r in reps:
        if not isinstance(r, dict) or r.get('taxonomy') != taxonomy:
            continue
        key = (r.get('category_title'), normalize_key(r.get('model_name')))
        if key in seen:
            continue
        seen.add(key)
        link = first_nonempty(r.get('code_repository_url'), r.get('web_server_url'), 'not_reported')
        lines.append(f"| {r.get('category_title')} | {r.get('model_name')} | {r.get('method_family','')} | {link} | {r.get('dataset_source_or_link','')} | {r.get('why_representative','')} |")
    if not lines:
        lines.append('| 暂无 | 待定 |  |  |  |  |')
    return lines


def build_meeting_trace_markdown(final_data: Dict[str, Any], md_json: Any, metric_json: Any, critic_json: Any, compact_evidence_pool: Dict[str, Any]) -> str:
    classification = ensure_list(final_data.get('model_classification'))
    reps = ensure_list(final_data.get('representative_models_by_category'))
    source_counts = compact_evidence_pool.get('source_counts') or {}
    baseline_names = ', '.join([str(x.get('model_name') or x.get('canonical_name')) for x in ensure_list(final_data.get('models'))[:12] if isinstance(x, dict)]) or '待从 memory 中确认'
    scout_points = compact_agent_decisions(md_json, 8)
    metrics_points = compact_agent_decisions(metric_json, 8)
    critic_points = compact_agent_decisions(critic_json, 10)

    rep_class_lines = classification_lines_by_taxonomy(classification, 'representation')
    arch_class_lines = classification_lines_by_taxonomy(classification, 'architecture')
    rep_model_lines = representative_lines_by_taxonomy(reps, 'representation')
    arch_model_lines = representative_lines_by_taxonomy(reps, 'architecture')

    return f"""# 🧠 AMP 文献证据全局会议记录

## 📚 历史共识基线
```text
【现有记忆/精选模型摘要】:
{baseline_names}
```

## 🕵️ Agent 1 (Scout / Model-Dataset) 增量提案

### 一、证据池与召回概况
- Chunk summaries: {compact_evidence_pool.get('chunk_summary_count')}
- Paper count: {compact_evidence_pool.get('paper_count')}
- Source counts: {source_counts}

### 二、模型与数据集初步提案
{render_bullet_list(scout_points)}

### 三、模型分类梳理：数据/输入表示（Representation）
| 类别 | 类别特点 | 具体模型 | 每类代表模型 1-2 个 |
|:---|:---|:---|:---|
{chr(10).join(rep_class_lines)}

### 四、Representation 每类代表模型选择依据
| 类别 | 代表模型 | 方法族 | 代码/工具链接 | 数据集线索 | 代表性理由 |
|:---|:---|:---|:---|:---|:---|
{chr(10).join(rep_model_lines)}

### 五、模型分类梳理：模型架构（Architecture）
| 类别 | 类别特点 | 具体模型 | 每类代表模型 1-2 个 |
|:---|:---|:---|:---|
{chr(10).join(arch_class_lines)}

### 六、Architecture 每类代表模型选择依据
| 类别 | 代表模型 | 方法族 | 代码/工具链接 | 数据集线索 | 代表性理由 |
|:---|:---|:---|:---|:---|:---|
{chr(10).join(arch_model_lines)}

## 📐 Agent 2 (Metrics) 指标与测试集提案
{render_bullet_list(metrics_points)}

## ⚖️ Agent 3 (Critic) 深度质疑
{render_bullet_list(critic_points)}

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
""".strip()


# ------------------------- Memory -------------------------
class MemoryManager:
    def __init__(self):
        self.memory = read_json(MEMORY_JSON, {'all_candidate_models': [], 'benchmark_ready_models': [], 'models': [], 'repositories': [], 'datasets': [], 'dataset_links': [], 'model_dataset_links': [], 'dataset_followup_tasks': [], 'model_classification': [], 'representative_models_by_category': [], 'final_deployment_models': [], 'final_recommended_datasets': [], 'final_metrics_plan': {}, 'final_execution_decision': {}, 'github_missing_model_enrichment': [], 'qwen_web_enrichment': [], 'metrics': [], 'papers': [], 'benchmark_implications': [], 'open_questions': [], 'agent_discussion': [], 'runs': []})
        self.index = read_json(INDEX_JSON, {'processed_keys': [], 'processed_pmids': [], 'processed_dois': [], 'processed_titles': []})

    def has_processed(self, rec: Dict[str, Any]) -> bool:
        # 历史 index 文件可能混入 dict/list。这里先清洗再 set，避免 unhashable type。
        keys = set(clean_index_list(self.index.get('processed_keys', [])))
        pmids = set(clean_index_list(self.index.get('processed_pmids', [])))
        dois = set(clean_index_list(self.index.get('processed_dois', [])))
        titles = set(clean_index_list(self.index.get('processed_titles', [])))
        return bool((rec.get('candidate_key') and str(rec.get('candidate_key')) in keys) or (rec.get('pmid') and str(rec.get('pmid')) in pmids) or (rec.get('doi') and normalize_key(rec.get('doi')) in dois) or (rec.get('title') and normalize_key(rec.get('title')) in titles))

    def context(self) -> Dict[str, Any]:
        return {k: self.memory.get(k, [])[:80] for k in ['all_candidate_models','benchmark_ready_models','models','repositories','datasets','dataset_links','model_dataset_links','dataset_followup_tasks','model_classification','representative_models_by_category','metrics','papers','benchmark_implications','open_questions']}

    def merge_final(self, final_data: Dict[str, Any], records: List[Dict[str, Any]], run_info: Dict[str, Any]) -> None:
        replace_sections = {'model_classification', 'representative_models_by_category', 'final_deployment_models', 'final_recommended_datasets', 'final_metrics_plan', 'final_execution_decision', 'agent_discussion'}
        for section in ['all_candidate_models', 'benchmark_ready_models', 'models', 'repositories', 'datasets', 'dataset_links', 'model_dataset_links', 'dataset_followup_tasks', 'model_classification', 'representative_models_by_category', 'final_deployment_models', 'final_recommended_datasets', 'final_metrics_plan', 'final_execution_decision', 'github_missing_model_enrichment', 'qwen_web_enrichment', 'metrics', 'papers', 'benchmark_implications', 'open_questions', 'agent_discussion']:
            self.memory.setdefault(section, [])
            incoming = ensure_list(final_data.get(section))
            if section in replace_sections and incoming:
                # v4.2: these sections describe the latest meeting/rendering state. Do not keep stale v3 summaries.
                self.memory[section] = incoming
            else:
                self.memory[section] = merge_items(self.memory.get(section, []), incoming, section)
        self.memory.setdefault('runs', []).append(run_info)
        # update index
        for r in records:
            if r.get('candidate_key'):
                self.index.setdefault('processed_keys', []).append(r.get('candidate_key'))
            if r.get('pmid'):
                self.index.setdefault('processed_pmids', []).append(str(r.get('pmid')))
            if r.get('doi'):
                self.index.setdefault('processed_dois', []).append(normalize_key(r.get('doi')))
            if r.get('title'):
                self.index.setdefault('processed_titles', []).append(normalize_key(r.get('title')))
        for k in list(self.index):
            self.index[k] = clean_index_list(self.index.get(k, []))
        write_json(MEMORY_JSON, self.memory)
        write_json(INDEX_JSON, self.index)
        MEMORY_MD.write_text(render_memory_md(self.memory, run_info), encoding='utf-8')


def merge_items(existing: List[Any], incoming: List[Any], section: str) -> List[Any]:
    result = list(existing or [])
    seen: Dict[str, int] = {}
    def key_for(item: Any) -> str:
        if isinstance(item, dict):
            if section in {'all_candidate_models', 'benchmark_ready_models', 'models'}:
                mk = model_key(item)
                if mk:
                    return section + ':' + mk
            for k in ['canonical_name','model_name','name','url','dataset_url','dataset_name','metric_name','doi','pmid','title','topic','question','agent','category','category_title']:
                if item.get(k):
                    return section + ':' + normalize_key(item.get(k))
        return section + ':' + stable_hash(item)
    for i, item in enumerate(result):
        seen[key_for(item)] = i
    for item in incoming:
        key = key_for(item)
        if key in seen:
            idx = seen[key]
            if isinstance(result[idx], dict) and isinstance(item, dict):
                result[idx] = merge_candidate(result[idx], item)
            elif result[idx] == item:
                pass
            else:
                result.append(item)
        else:
            seen[key] = len(result)
            result.append(item)
    return result


def item_has_any_value(item: Any, cols: Optional[List[str]] = None) -> bool:
    if isinstance(item, dict):
        check_cols = cols or list(item.keys())
        return any(not is_missing_value(item.get(c)) for c in check_cols)
    return not is_missing_value(item)


def render_table(items: Any, cols: List[str]) -> str:
    rows = [r for r in ensure_list(items) if item_has_any_value(r, cols)]
    if not rows:
        return '_None yet._'
    out = ['|' + '|'.join(cols) + '|', '|' + '|'.join(['---'] * len(cols)) + '|']
    for it in rows:
        if isinstance(it, dict):
            vals = []
            for c in cols:
                v = it.get(c, '')
                if isinstance(v, (list, dict)):
                    v = json_dumps(v, 0)
                vals.append(str(v).replace('\n', '<br>').replace('|', '\\|')[:900])
        else:
            vals = [str(it).replace('\n', '<br>').replace('|', '\\|')[:900]] + [''] * (len(cols)-1)
        out.append('|' + '|'.join(vals) + '|')
    return '\n'.join(out)



def render_agent_discussion_md(items: Any) -> str:
    rows = ensure_list(items)
    if not rows:
        return '_No agent discussion saved yet._'
    lines: List[str] = []
    for item in rows:
        if isinstance(item, dict) and item.get('markdown'):
            lines.append(str(item.get('markdown')).strip())
            lines.append('')
            continue
        if not isinstance(item, dict):
            lines.append(f'- {str(item)}')
            continue
        agent = item.get('agent', 'agent')
        role = item.get('role', '')
        status = item.get('status', '')
        lines.append(f"### {agent}")
        lines.append('')
        lines.append(f"- **Role**: {role}")
        lines.append(f"- **Status**: {status}")
        counts = item.get('counts')
        if isinstance(counts, dict) and counts:
            count_str = ', '.join([f'{k}={v}' for k, v in counts.items()])
            lines.append(f"- **Counts**: {count_str}")
        key_points = ensure_list(item.get('key_points'))
        if key_points:
            lines.append('- **Discussion / key decisions**:')
            for kp in key_points[:12]:
                if isinstance(kp, dict):
                    lines.append(f"  - `{trunc(json_dumps(kp, 0), 700)}`")
                else:
                    lines.append(f"  - {trunc(kp, 700)}")
        lines.append('')
    return '\n'.join(lines).strip()


def render_model_classification_md(items: Any) -> str:
    rows = [r for r in ensure_list(items) if isinstance(r, dict)]
    if not rows:
        return '_No model classification saved yet._'
    parts: List[str] = []
    for taxonomy in ['representation', 'architecture']:
        sub = [r for r in rows if r.get('taxonomy') == taxonomy]
        if not sub:
            continue
        parts.append(f"### {_taxonomy_label(taxonomy)}")
        parts.append('')
        parts.append('|类别|类别特点|具体模型|每类代表模型 1-2 个|当前证据池命中数|')
        parts.append('|---|---|---|---|---:|')
        for r in sub:
            examples = ', '.join(map(str, ensure_list(r.get('concrete_models'))))
            reps = ', '.join(map(str, ensure_list(r.get('representative_models'))[:2]))
            parts.append('|' + '|'.join([
                str(r.get('title','')).replace('|','\\|'),
                str(r.get('description','')).replace('|','\\|')[:700],
                examples.replace('|','\\|'),
                reps.replace('|','\\|'),
                str(r.get('model_count','')).replace('|','\\|'),
            ]) + '|')
        parts.append('')
    return '\n'.join(parts).strip()


def render_representative_models_md(items: Any) -> str:
    rows = [r for r in ensure_list(items) if isinstance(r, dict)]
    if not rows:
        return '_None yet._'
    parts: List[str] = []
    for taxonomy in ['representation', 'architecture']:
        sub = []
        seen = set()
        for r in rows:
            if r.get('taxonomy') != taxonomy:
                continue
            key = (r.get('category_title'), normalize_key(r.get('model_name')))
            if key in seen:
                continue
            seen.add(key)
            sub.append(r)
        if not sub:
            continue
        parts.append(f"### {_taxonomy_label(taxonomy)}")
        parts.append('')
        parts.append(render_table(sub, ['category_title','model_name','task_type','method_family','code_repository_url','web_server_url','dataset_source_or_link','source_pmid','source_doi','evidence_level','why_representative']))
        parts.append('')
    return '\n'.join(parts).strip()



def render_final_metrics_plan_md(plan: Any) -> str:
    if not isinstance(plan, dict) or not plan:
        plan = FINAL_METRICS_PLAN
    lines: List[str] = []
    lines.append('### 主排名指标')
    lines.append('')
    lines.append('|指标|权重|用途|')
    lines.append('|---|---:|---|')
    for m in ensure_list(plan.get('primary_weighted_metrics')):
        if not isinstance(m, dict):
            continue
        lines.append(f"|{m.get('metric_name','')}|{m.get('weight','')}|{str(m.get('reason','')).replace('|','\\|')}|")
    lines.append('')
    lines.append('### 强制报告指标')
    lines.append('')
    lines.append(', '.join(map(str, ensure_list(plan.get('mandatory_report_metrics')))) or '_None_')
    lines.append('')
    lines.append('### 阈值与测试矩阵')
    lines.append('')
    lines.append(f"- 阈值策略：{plan.get('threshold_policy','')}")
    matrix = ', '.join(map(str, ensure_list(plan.get('test_matrix'))))
    if matrix:
        lines.append(f"- 测试矩阵：{matrix}")
    stats = ', '.join(map(str, ensure_list(plan.get('statistical_reporting'))))
    if stats:
        lines.append(f"- 统计报告：{stats}")
    return '\n'.join(lines).strip()


def render_final_execution_decision_md(mem: Dict[str, Any]) -> str:
    decision = mem.get('final_execution_decision') if isinstance(mem, dict) else {}
    if not isinstance(decision, dict):
        decision = {}
    models = ensure_list(decision.get('final_models') or mem.get('final_deployment_models'))
    datasets = ensure_list(decision.get('final_datasets') or mem.get('final_recommended_datasets'))
    metrics_plan = decision.get('final_metrics_plan') or mem.get('final_metrics_plan') or FINAL_METRICS_PLAN
    excluded = ensure_list(decision.get('excluded_from_main_benchmark'))
    parts: List[str] = []
    parts.append('## Final Execution Decision / 最终执行决策')
    parts.append('')
    parts.append(decision.get('summary') or '主榜先部署模型、推荐数据集和指标如下；候选模型与 Agent 讨论放在后文附录。')
    parts.append('')
    parts.append('### 1. 最终先部署模型')
    parts.append('')
    parts.append(render_table(models, ['deployment_rank','deployment_tier','model_name','representation_category','architecture_category','task_type','code_repository_url','dataset_source_or_link','source_journal','citation_count','journal_impact_factor','article_impact_score','deployment_selection_score','deployment_status','deployment_reason','first_next_action','blocking_issues']))
    parts.append('')
    parts.append('### 2. 推荐最合适的 3 个数据集')
    parts.append('')
    parts.append(render_table(datasets, ['dataset_rank','dataset_name','linked_model','recommended_role','dataset_source_or_link','why_selected','required_cleaning','status']))
    parts.append('')
    parts.append('### 3. 最终指标体系')
    parts.append('')
    parts.append(render_final_metrics_plan_md(metrics_plan))
    if excluded:
        parts.append('')
        parts.append('### 4. 暂不进入主榜的模型类型')
        parts.append('')
        parts.append(render_table(excluded, ['group','examples','reason']))
    return '\n'.join(parts).strip()


def render_final_deployment_models_md(items: Any) -> str:
    rows = [r for r in ensure_list(items) if isinstance(r, dict) and item_has_any_value(r)]
    if not rows:
        return '_No evidence-backed deployment candidates yet._'
    return render_table(rows, ['deployment_rank','deployment_tier','model_name','canonical_name','representation_category','architecture_category','task_type','method_family','code_repository_url','web_server_url','dataset_source_or_link','source_journal','citation_count','journal_impact_factor','article_impact_score','deployment_selection_score','deployment_status','deployment_reason','first_next_action','blocking_issues','evidence_level','confidence','source_pmid','source_doi'])


def render_final_recommended_datasets_md(items: Any) -> str:
    rows = [r for r in ensure_list(items) if isinstance(r, dict) and item_has_any_value(r)]
    if not rows:
        return '_No recommended datasets yet._'
    return render_table(rows, ['dataset_rank','dataset_name','linked_model','recommended_role','dataset_source_or_link','why_selected','required_cleaning','source_pmid','source_doi','evidence_level','status'])


def render_github_enrichment_md(items: Any) -> str:
    rows = [r for r in ensure_list(items) if isinstance(r, dict) and item_has_any_value(r)]
    if not rows:
        return '_No GitHub missing-link enrichment evidence yet._'
    return render_table(rows, ['model_name','matched_model_name','name','url','description','stars','language','match_score','confidence_label','needs_manual_verification','evidence_level','query'])


def _first_candidate_url(items: Any) -> str:
    for x in ensure_list(items):
        if isinstance(x, dict) and x.get('url'):
            return str(x.get('url'))
        if isinstance(x, str) and URL_RE.search(x):
            return URL_RE.search(x).group(0)
    return ''


def render_qwen_web_enrichment_md(items: Any) -> str:
    rows = []
    for r in ensure_list(items):
        if not isinstance(r, dict) or not item_has_any_value(r):
            continue
        rows.append({
            'model_name': r.get('model_name'),
            'task_type_guess': r.get('task_type_guess'),
            'repo_url': _first_candidate_url(r.get('repository_candidates')),
            'dataset_url': _first_candidate_url(r.get('dataset_candidates')),
            'weights_url': _first_candidate_url(r.get('weight_candidates')),
            'web_server_url': _first_candidate_url(r.get('web_server_candidates')),
            'paper_url': _first_candidate_url(r.get('paper_links')),
            'source_journal': r.get('source_journal'),
            'citation_count': r.get('citation_count'),
            'journal_impact_factor': r.get('journal_impact_factor'),
            'impact_evidence': (r.get('article_impact') or {}).get('evidence') if isinstance(r.get('article_impact'), dict) else '',
            'completed_fields': r.get('completed_fields'),
            'confidence': r.get('confidence'),
            'confidence_label': r.get('confidence_label'),
            'needs_manual_verification': r.get('needs_manual_verification'),
            'summary': r.get('summary'),
            'risk_flags': r.get('risk_flags'),
        })
    if not rows:
        return '_No Qwen-Max web-search enrichment evidence yet._'
    return render_table(rows, ['model_name','task_type_guess','repo_url','dataset_url','weights_url','web_server_url','paper_url','source_journal','citation_count','journal_impact_factor','impact_evidence','completed_fields','confidence','confidence_label','needs_manual_verification','summary','risk_flags'])


def render_memory_md(mem: Dict[str, Any], run_info: Dict[str, Any]) -> str:
    return f"""# AMP Benchmark Literature Memory

Updated: {now_str()}

## Latest Run

```json
{json_dumps(run_info, 2)}
```

{render_final_execution_decision_md(mem)}

## Agent Discussion Process

{render_agent_discussion_md(mem.get('agent_discussion', []))}

## Model Classification Overview

{render_model_classification_md(mem.get('model_classification', []))}

## Representative Models by Category

{render_representative_models_md(mem.get('representative_models_by_category', []))}

## Final Deployment Model List

{render_final_deployment_models_md(mem.get('final_deployment_models', []))}

## Final Recommended Datasets

{render_final_recommended_datasets_md(mem.get('final_recommended_datasets', []))}

## Final Metrics Plan

{render_final_metrics_plan_md(mem.get('final_metrics_plan', {}))}

## All Candidate Models

{render_table(mem.get('all_candidate_models', []), ['model_name','canonical_name','representation_category','architecture_category','task_type','method_family','source_pmid','source_doi','code_repository_url','web_server_url','dataset_source_or_link','benchmark_candidate','blocking_issues','evidence_level','confidence','chunk_id'])}

## Benchmark Ready Models

{render_table(mem.get('benchmark_ready_models', []), ['model_name','canonical_name','representation_category','architecture_category','task_type','method_family','source_pmid','source_doi','code_repository_url','web_server_url','dataset_source_or_link','benchmark_candidate','candidate_reason','blocking_issues','evidence_level','confidence'])}

## Selected Models

{render_table(mem.get('models', []), ['model_name','canonical_name','representation_category','architecture_category','task_type','method_family','source_pmid','source_doi','code_repository_url','web_server_url','dataset_source_or_link','benchmark_candidate','blocking_issues','evidence_level','confidence'])}

## Repositories

{render_table(mem.get('repositories', []), ['name','url','repository_type','matched_model_name','source_pmid','source_doi','evidence_level'])}

## GitHub Missing-Link Enrichment Evidence

{render_github_enrichment_md(mem.get('github_missing_model_enrichment', []))}

## Qwen-Max Web-Search Enrichment Evidence

{render_qwen_web_enrichment_md(mem.get('qwen_web_enrichment', []))}

## Datasets

{render_table(mem.get('datasets', []), ['dataset_name','dataset_url','dataset_source','linked_model','dataset_status','dataset_role','source_pmid','source_doi','positive_samples','negative_samples','deduplication_method','split_method','evidence_level'])}

## Model-Dataset Links

{render_table(mem.get('model_dataset_links', []), ['model_name','dataset_name','dataset_role','dataset_source','dataset_url','dataset_status','source_pmid','source_doi','positive_samples','negative_samples','deduplication_method','split_method','needs_followup','evidence_level'])}

## Dataset Links

{render_table(mem.get('dataset_links', []), ['dataset_name','url','source','linked_model','dataset_status','evidence','source_pmid','source_doi'])}

## Dataset Follow-up Tasks

{render_table(mem.get('dataset_followup_tasks', []), ['model_name','dataset_status','reason','next_action','source_pmid','source_doi'])}

## Metrics

{render_table(mem.get('metrics', []), ['metric_name','usage','source_pmid','source_doi','evidence'])}

## Papers

{render_table(mem.get('papers', []), ['title','pmid','pmcid','doi','year','role','open_fulltext_status'])}

## Benchmark Implications

{render_table(mem.get('benchmark_implications', []), ['topic','decision','reason','evidence'])}

## Open Questions

{render_table(mem.get('open_questions', []), ['question','reason','next_action'])}
"""



def _load_json_files_from_dir(directory: Path) -> List[Dict[str, Any]]:
    """读取目录下的 JSON 文件。用于续跑时从 data/chunk_summaries/ 恢复 chunk summaries。"""
    out: List[Dict[str, Any]] = []
    if not directory.exists():
        return out
    for fp in sorted(directory.glob('*.json')):
        if fp.name == '_chunk_index.json':
            continue
        obj = read_json(fp, None)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_compact_evidence_pool_from_disk() -> Dict[str, Any]:
    """从磁盘恢复 compact evidence pool。

    优先读取 data/compact_evidence_pool.json；如果不存在或缺少 chunk_summaries，
    自动从 data/chunk_summaries/*.json 和 data/normalized_papers.jsonl 重建一个最小可用版本。
    这样 --resume-global-only / --use-existing-meeting 不需要重新搜索、抓全文或压缩 chunk。
    """
    pool = read_json(COMPACT_EVIDENCE_POOL_JSON, {})
    if not isinstance(pool, dict):
        pool = {}

    summaries = ensure_list(pool.get('chunk_summaries'))
    if not summaries:
        summaries = _load_json_files_from_dir(CHUNK_SUMMARIES_DIR)
        if summaries:
            pool['chunk_summaries'] = summaries

    # 兼容旧字段：chunk_count 与 chunk_summary_count 都补齐。
    pool.setdefault('created_at', now_str())
    pool.setdefault('compression_mode', 'by_model_topic_source_recovered_from_disk')
    pool['chunk_summary_count'] = len(ensure_list(pool.get('chunk_summaries')))
    pool.setdefault('chunk_count', pool['chunk_summary_count'])

    papers = read_jsonl(NORMALIZED_PAPERS_JSONL)
    if papers and not ensure_list(pool.get('paper_overview')):
        pool['paper_overview'] = [compact_record_for_prompt(r) for r in papers[:300] if isinstance(r, dict)]
    if papers and not pool.get('paper_count'):
        pool['paper_count'] = len(papers)

    if not ensure_list(pool.get('chunk_summaries')):
        raise FileNotFoundError(
            '没有找到 compact evidence pool 或 chunk summaries。请确认存在 data/compact_evidence_pool.json 或 data/chunk_summaries/*.json。'
        )

    return pool


def load_records_for_index_from_disk() -> List[Dict[str, Any]]:
    """续跑时用于 literature_deep_research_index.json 的记录来源。

    不重新搜索，直接尽量从已有磁盘文件恢复：
    1. normalized_papers.jsonl
    2. raw_candidates.jsonl
    3. evidence_pool.json 的 papers 字段
    4. compact_evidence_pool.json 的 paper_overview 字段
    """
    records: List[Dict[str, Any]] = []
    for fp in [NORMALIZED_PAPERS_JSONL, RAW_CANDIDATES_JSONL]:
        for row in read_jsonl(fp):
            if isinstance(row, dict):
                records.append(row)

    if not records:
        ep = read_json(EVIDENCE_POOL_JSON, {})
        if isinstance(ep, dict):
            for row in ensure_list(ep.get('papers')):
                if isinstance(row, dict):
                    records.append(row)

    if not records:
        cp = read_json(COMPACT_EVIDENCE_POOL_JSON, {})
        if isinstance(cp, dict):
            for row in ensure_list(cp.get('paper_overview')):
                if isinstance(row, dict):
                    records.append(row)

    # 去重，避免 index 膨胀。
    dedup: List[Dict[str, Any]] = []
    seen: set = set()
    for r in records:
        key = r.get('candidate_key') or r.get('pmid') or r.get('doi') or r.get('title') or stable_hash(json_dumps(r, 0, sort_keys=True))
        key = str(key)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup

def latest_global_meeting_raw() -> Optional[Dict[str, Any]]:
    """读取最近一次已完成的全局会议结果，适用于上次只在 Write Memory 阶段失败的情况。"""
    for fp in [GLOBAL_MEETING_RAW_JSONL, DATA_DIR / 'global_meeting_raw.jsonl']:
        rows = read_jsonl(fp)
        for row in reversed(rows):
            if isinstance(row, dict) and ('chief_agent' in row or 'final_data' in row):
                return row
    return None


def run_resume_global_meeting_only(
    provider: str,
    provider_config: Path,
    meeting_agent_dir: Path,
    use_existing_meeting: bool = False,
    run_note: str = 'main_resume_global_meeting_only',
    github_enrichment: bool = True,
    github_enrich_max_models: int = 80,
    github_enrich_repos_per_model: int = 3,
    force_github_enrichment: bool = False,
    refresh_all_github_enrichment: bool = False,
    qwen_web_enrichment: bool = False,
    qwen_web_provider: str = 'dashscope_qwen37max_search',
    qwen_web_model: str = 'qwen3.7-max',
    qwen_web_max_models: int = 30,
    force_qwen_web_enrichment: bool = False,
    refresh_all_qwen_web_enrichment: bool = False,
) -> Dict[str, Any]:
    """主流程内置续跑：不重新搜索、不重新抓全文、不重新压缩 chunk，只跑/复用全局会议并写 memory。"""
    print('========== [Resume Global Meeting Only - Main Flow] ==========', flush=True)
    compact_pool = load_compact_evidence_pool_from_disk()
    records = load_records_for_index_from_disk()
    memory = MemoryManager()
    print(f'>>> Chunk summaries: {len(ensure_list(compact_pool.get("chunk_summaries")))}', flush=True)
    print('>>> 不重新搜索、不重新抓全文、不重新压缩 chunk。', flush=True)
    if github_enrichment:
        print('>>> GitHub 补链：将对缺少 GitHub 链接的模型做模型名搜索，并写回 compact evidence pool。', flush=True)
        compact_pool = enrich_compact_pool_with_missing_github(compact_pool, max_models=github_enrich_max_models, repos_per_model=github_enrich_repos_per_model, force=force_github_enrichment, refresh_all=refresh_all_github_enrichment)
    else:
        print('>>> GitHub 补链：已关闭。', flush=True)

    if qwen_web_enrichment:
        print('>>> Qwen3.7-Max 联网补漏：将补充 GitHub / 数据集 / 权重 / web server / 论文主页证据，并写回 compact evidence pool。', flush=True)
        compact_pool = enrich_compact_pool_with_qwen_web(compact_pool, provider_config=provider_config, provider=qwen_web_provider, model_name=qwen_web_model, max_models=qwen_web_max_models, force=force_qwen_web_enrichment, refresh_all=refresh_all_qwen_web_enrichment)
    else:
        print('>>> Qwen3.7-Max 联网补漏：未启用。', flush=True)

    if use_existing_meeting:
        raw = latest_global_meeting_raw()
        if raw:
            final_data = raw.get('chief_agent') or raw.get('final_data')
            if not isinstance(final_data, dict):
                print('⚠️ 找到了 global meeting raw，但里面没有可用 chief_agent/final_data；改为从 chunk summaries 直接重建 memory。', flush=True)
                final_data = build_chunk_derived_final(compact_pool)
        else:
            # v4.5 fallback: --use-existing-meeting 原意是不再调用 DeepSeek。
            # 如果用户机器里没有 data/deepseek_meeting_raw.jsonl，不应直接报错；
            # 直接从 data/compact_evidence_pool.json + data/chunk_summaries/*.json 确定性重建最终 memory。
            print('⚠️ 没有找到已完成的 global meeting raw；将不调用 DeepSeek，直接从 chunk summaries 重建 memory。', flush=True)
            final_data = build_chunk_derived_final(compact_pool)
            raw = {
                'fallback_from_chunk_summaries': True,
                'reason': 'global_meeting_raw_not_found',
                'created_at': now_str(),
            }
        # 即使复用旧 meeting 或 fallback，也从 chunk summaries 补回全量候选、数据集关系、分类和会议记录。
        final_data = enrich_final_from_chunks(
            final_data,
            compact_pool,
            raw.get('model_dataset_agent') if isinstance(raw, dict) else None,
            raw.get('metric_agent') if isinstance(raw, dict) else None,
            raw.get('critic_agent') if isinstance(raw, dict) else None,
        )
        raw_meeting = raw
        print('✅ 已使用已有 chunk summaries 生成 memory；未重新搜索、未重新抓全文、未重新压缩 chunk、未调用 DeepSeek。', flush=True)
    else:
        llm = DeepSeekChatLLM(provider=provider, config_path=provider_config)
        loader = AgentMDLoader(meeting_agent_dir)
        final_data, raw_meeting = global_meeting(llm, loader, compact_pool, memory.context())

    run_info = {
        'time': now_str(),
        'mode': 'resume_global_meeting_only_from_main_flow',
        'note': run_note,
        'use_existing_meeting': bool(use_existing_meeting),
        'chunk_summary_count': len(ensure_list(compact_pool.get('chunk_summaries'))),
        'compact_evidence_pool': str(COMPACT_EVIDENCE_POOL_JSON.relative_to(ROOT)) if COMPACT_EVIDENCE_POOL_JSON.exists() else None,
        'record_count_for_index': len(records),
        'github_enrichment': bool(github_enrichment),
        'github_enrichment_file': str(GITHUB_MISSING_MODEL_ENRICHMENT_JSON.relative_to(ROOT)) if GITHUB_MISSING_MODEL_ENRICHMENT_JSON.exists() else None,
        'github_enrichment_count': len(ensure_list(compact_pool.get('github_missing_model_enrichment'))),
        'qwen_web_enrichment': bool(qwen_web_enrichment),
        'qwen_web_model': qwen_web_model if qwen_web_enrichment else None,
        'qwen_web_enrichment_file': str(QWEN_WEB_ENRICHMENT_JSON.relative_to(ROOT)) if QWEN_WEB_ENRICHMENT_JSON.exists() else None,
        'qwen_web_enrichment_count': len(ensure_list(compact_pool.get('qwen_web_enrichment'))),
    }
    print('    -> [Safe Write Memory] 写入 MD + JSON 长期记忆，并安全清理 index 中的 dict/list...', flush=True)
    memory.merge_final(final_data, records, run_info)
    print('✅ 续跑完成。', flush=True)
    print(f'   Memory MD: {MEMORY_MD.relative_to(ROOT)}')
    print(f'   Memory JSON: {MEMORY_JSON.relative_to(ROOT)}')
    return {'run_info': run_info, 'final_data': final_data, 'raw_meeting': raw_meeting}


# ------------------------- Search orchestration -------------------------
def run_source_searches(plan: Dict[str, List[Dict[str, str]]], max_results: int, year_from: Optional[int], year_to: Optional[int], enabled_sources: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    pubmed = PubMedClient(); epmc = EuropePMCClient(); crossref = CrossrefClient(); openalex = OpenAlexClient(); s2 = SemanticScholarClient(); gh = GitHubClient(); dc = DataCiteClient(); zen = ZenodoClient()
    candidates: List[Dict[str, Any]] = []
    repos: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    summary = {'time': now_str(), 'queries': {}, 'errors': []}

    def enabled(s: str) -> bool:
        return 'all' in enabled_sources or s in enabled_sources

    if enabled('pubmed'):
        summary['queries']['pubmed'] = []
        for q in plan.get('pubmed', []):
            try:
                pmids = pubmed.esearch(q['query'], retmax=max_results, year_from=year_from, year_to=year_to)
                print(f"    -> PubMed[{q['name']}] 返回 PMID 数量: {len(pmids)}")
                records = pubmed.efetch_records(pmids)
                for r in records:
                    r.setdefault('sources', []).append('pubmed') if 'pubmed' not in ensure_list(r.get('sources')) else None
                    r['query_name'] = q['name']; r['query_text'] = q['query']
                candidates.extend(records)
                summary['queries']['pubmed'].append({'name': q['name'], 'query': q['query'], 'pmids': pmids})
                time.sleep(0.35)
            except Exception as e:
                summary['errors'].append({'source': 'pubmed', 'query': q, 'error': str(e)})
                print(f"    ⚠️ PubMed query 失败 {q['name']}: {e}")

    if enabled('europe_pmc'):
        summary['queries']['europe_pmc'] = []
        for q in plan.get('europe_pmc', []) + plan.get('preprint', []):
            try:
                records = epmc.search(q['query'], page_size=max_results)
                print(f"    -> Europe PMC[{q['name']}] 返回数量: {len(records)}")
                for r in records:
                    r['query_name'] = q['name']; r['query_text'] = q['query']
                candidates.extend(records)
                summary['queries']['europe_pmc'].append({'name': q['name'], 'query': q['query'], 'count': len(records)})
            except Exception as e:
                summary['errors'].append({'source': 'europe_pmc', 'query': q, 'error': str(e)})
                print(f"    ⚠️ Europe PMC query 失败 {q['name']}: {e}")

    if enabled('crossref'):
        summary['queries']['crossref'] = []
        for q in plan.get('crossref', []):
            try:
                records = crossref.search(q['query'], rows=max_results)
                print(f"    -> Crossref[{q['name']}] 返回数量: {len(records)}")
                candidates.extend(records)
                summary['queries']['crossref'].append({'name': q['name'], 'count': len(records)})
            except Exception as e:
                summary['errors'].append({'source': 'crossref', 'query': q, 'error': str(e)})

    if enabled('openalex'):
        summary['queries']['openalex'] = []
        for q in plan.get('openalex', []):
            try:
                records = openalex.search(q['query'], rows=max_results)
                print(f"    -> OpenAlex[{q['name']}] 返回数量: {len(records)}")
                candidates.extend(records)
                summary['queries']['openalex'].append({'name': q['name'], 'count': len(records)})
            except Exception as e:
                summary['errors'].append({'source': 'openalex', 'query': q, 'error': str(e)})

    if enabled('semantic_scholar'):
        summary['queries']['semantic_scholar'] = []
        for q in plan.get('semantic_scholar', []):
            try:
                records = s2.search(q['query'], rows=min(max_results, 100))
                print(f"    -> Semantic Scholar[{q['name']}] 返回数量: {len(records)}")
                candidates.extend(records)
                summary['queries']['semantic_scholar'].append({'name': q['name'], 'count': len(records)})
            except Exception as e:
                summary['errors'].append({'source': 'semantic_scholar', 'query': q, 'error': str(e)})
                print(f"    ⚠️ Semantic Scholar query 失败 {q['name']}: {e}")

    if enabled('github'):
        summary['queries']['github'] = []
        for q in plan.get('github', []):
            res = gh.search_repositories(q['query'], rows=min(max_results, 100))
            print(f"    -> GitHub[{q['name']}] 返回仓库数量: {len(res)}")
            repos.extend(res)
            summary['queries']['github'].append({'name': q['name'], 'count': len(res)})

    if enabled('datacite'):
        summary['queries']['datacite'] = []
        for q in plan.get('datacite', []):
            res = dc.search(q['query'], rows=max_results)
            print(f"    -> DataCite[{q['name']}] 返回数据/软件数量: {len(res)}")
            datasets.extend(res)
            summary['queries']['datacite'].append({'name': q['name'], 'count': len(res)})

    if enabled('zenodo'):
        summary['queries']['zenodo'] = []
        for q in plan.get('zenodo', []):
            res = zen.search(q['query'], rows=max_results)
            print(f"    -> Zenodo[{q['name']}] 返回数量: {len(res)}")
            datasets.extend(res)
            summary['queries']['zenodo'].append({'name': q['name'], 'count': len(res)})

    return candidates, repos, datasets, summary


def extract_model_names_from_evidence(evidence_batches: List[Dict[str, Any]], limit: int = 50) -> List[str]:
    names = []
    for ev in evidence_batches:
        for m in ensure_list(ev.get('models')):
            if isinstance(m, dict):
                for k in ['canonical_name', 'model_name']:
                    v = m.get(k)
                    if v and normalize_key(v) not in ['unknown', 'not reported', 'not reported in available evidence']:
                        names.append(str(v).strip())
                for a in ensure_list(m.get('aliases')):
                    if a:
                        names.append(str(a).strip())
    out = []
    seen = set()
    for n in names:
        key = normalize_key(n)
        if key and key not in seen and len(n) >= 3:
            seen.add(key); out.append(n)
    return out[:limit]


def build_model_backsearch_plan(model_names: List[str]) -> Dict[str, List[Dict[str, str]]]:
    plan = {k: [] for k in DEFAULT_QUERY_PLAN}
    for name in model_names:
        qname = safe_name(name, 'model')
        plan['pubmed'].append({'name': f'model_pubmed_{qname}', 'query': f'"{name}"'})
        plan['europe_pmc'].append({'name': f'model_epmc_{qname}', 'query': f'"{name}"'})
        plan['crossref'].append({'name': f'model_crossref_{qname}', 'query': name})
        plan['openalex'].append({'name': f'model_openalex_{qname}', 'query': name})
        plan['semantic_scholar'].append({'name': f'model_s2_{qname}', 'query': name})
        plan['github'].append({'name': f'model_github_{qname}', 'query': f'"{name}" antimicrobial peptide'})
        plan['datacite'].append({'name': f'model_datacite_{qname}', 'query': f'{name} antimicrobial peptide dataset'})
        plan['zenodo'].append({'name': f'model_zenodo_{qname}', 'query': f'{name} antimicrobial peptide'})
    return plan


def expand_similar_and_citations(records: List[Dict[str, Any]], per_seed: int = 10, max_seeds: int = 20) -> List[Dict[str, Any]]:
    pubmed = PubMedClient(); s2 = SemanticScholarClient()
    out: List[Dict[str, Any]] = []
    seeds = [r for r in records if r.get('pmid') or r.get('semantic_scholar_id') or r.get('doi')][:max_seeds]
    for r in seeds:
        if r.get('pmid'):
            pmids = pubmed.similar_pmids(str(r.get('pmid')), retmax=per_seed)
            if pmids:
                fetched = pubmed.efetch_records(pmids)
                for f in fetched:
                    f.setdefault('sources', []).append('pubmed_similar')
                    f['expanded_from'] = r.get('candidate_key')
                print(f"    -> PubMed Similar PMID={r.get('pmid')} 扩展: {len(fetched)}")
                out.extend(fetched)
        pid = r.get('semantic_scholar_id') or (f'DOI:{r.get("doi")}' if r.get('doi') else None)
        if pid:
            try:
                cr = s2.citations_and_references(str(pid), rows=per_seed)
                print(f"    -> Semantic Scholar citations/references {pid}: {len(cr)}")
                out.extend(cr)
            except Exception as e:
                print(f'    ⚠️ S2 citation expansion failed {pid}: {e}')
    return out


# ------------------------- Main pipeline -------------------------
def run_pipeline(
    max_results: int = 30,
    batch_size: int = 4,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    provider: str = 'dashscope',
    provider_config: Path = Path('llm_providers.json'),
    planner_agent_dir: Path = Path('agents/pubmed_planner'),
    meeting_agent_dir: Path = Path('agents/deepseek_meeting'),
    fetch_fulltext: bool = True,
    no_planner: bool = False,
    reprocess: bool = False,
    max_queries: int = 20,
    sources: str = 'all',
    no_light_filter: bool = False,
    backsearch_models: bool = True,
    expand_citations: bool = True,
    citation_seed_limit: int = 12,
    evidence_compression: bool = True,
    chunk_target_size: int = 6,
    max_chunks: int = 120,
    max_chars_per_chunk: int = 60000,
    github_enrichment: bool = True,
    github_enrich_max_models: int = 80,
    github_enrich_repos_per_model: int = 3,
    force_github_enrichment: bool = False,
    refresh_all_github_enrichment: bool = False,
    qwen_web_enrichment: bool = False,
    qwen_web_provider: str = 'dashscope_qwen37max_search',
    qwen_web_model: str = 'qwen3.7-max',
    qwen_web_max_models: int = 30,
    force_qwen_web_enrichment: bool = False,
    refresh_all_qwen_web_enrichment: bool = False,
) -> Dict[str, Any]:
    print('\n========== [DeepSeek Multi-Source AMP Literature + GLOBAL Meeting] ==========')
    print(f'>>> Provider: {provider}')
    print(f'>>> Sources: {sources}')
    print(f'>>> Max results per query: {max_results}')
    print(f'>>> Fetch open full text: {fetch_fulltext}')
    print('>>> Meeting mode: CHUNKED_GLOBAL_MEETING_AFTER_ALL_EVIDENCE')
    print(f'>>> Evidence compression: {evidence_compression} | chunk_target_size={chunk_target_size} | max_chunks={max_chunks}')
    print(f'>>> GitHub missing-link enrichment: {github_enrichment} | max_models={github_enrich_max_models} | repos_per_model={github_enrich_repos_per_model} | force={force_github_enrichment} | refresh_all={refresh_all_github_enrichment}')
    print(f'>>> Qwen3.7-Max web enrichment: {qwen_web_enrichment} | provider={qwen_web_provider} | model={qwen_web_model} | max_models={qwen_web_max_models} | force={force_qwen_web_enrichment} | refresh_all={refresh_all_qwen_web_enrichment}')

    enabled_sources = [s.strip() for s in sources.split(',') if s.strip()] or ['all']
    llm = DeepSeekChatLLM(provider, provider_config)
    planner_loader = AgentMDLoader(planner_agent_dir)
    meeting_loader = AgentMDLoader(meeting_agent_dir)
    memory = MemoryManager()

    if no_planner:
        plan = DEFAULT_QUERY_PLAN
    else:
        print('\n>>> DeepSeek 正在规划多源搜索 queries...')
        plan = llm_plan_queries(llm, planner_loader, max_queries=max_queries)
        # Merge with defaults to avoid LLM under-searching.
        for src, items in DEFAULT_QUERY_PLAN.items():
            plan.setdefault(src, [])
            existing = {normalize_key(x.get('query')) for x in plan[src]}
            for it in items:
                if normalize_key(it['query']) not in existing:
                    plan[src].append(it)

    # limit total queries from planner/default, but do not cut too aggressively per source.
    print('\n>>> Multi-source Queries:')
    for src, items in plan.items():
        if sources != 'all' and src not in enabled_sources and not (src == 'preprint' and 'europe_pmc' in enabled_sources):
            continue
        print(f'  [{src}]')
        for it in items:
            print(f"    - {it['name']}: {it['query'][:220]}")

    print('\n========== [Stage 1] 多源搜索 ==========')
    raw_candidates, repos, datasets, search_summary = run_source_searches(plan, max_results, year_from, year_to, enabled_sources)
    print(f'>>> 多源原始候选 paper 数量: {len(raw_candidates)}')
    for c in raw_candidates:
        append_jsonl(RAW_CANDIDATES_JSONL, c)

    candidates = dedupe_candidates(raw_candidates)
    print(f'>>> 去重后 paper 数量: {len(candidates)}')
    if not no_light_filter:
        before = len(candidates)
        candidates = [c for c in candidates if looks_relevant(c)]
        print(f'>>> AMP/模型/benchmark 轻量过滤: {before} -> {len(candidates)}')
    for c in candidates:
        append_jsonl(NORMALIZED_PAPERS_JSONL, c)
    for r in repos:
        append_jsonl(NORMALIZED_REPOS_JSONL, r)
    for d in datasets:
        append_jsonl(NORMALIZED_DATASETS_JSONL, d)
    write_json(SEARCH_SUMMARY_JSON, search_summary)

    if expand_citations:
        print('\n========== [Stage 2] PubMed Similar / Semantic Scholar Citation 扩展 ==========')
        expanded = expand_similar_and_citations(candidates, per_seed=min(10, max_results), max_seeds=citation_seed_limit)
        print(f'>>> 引用/相似扩展新增候选: {len(expanded)}')
        candidates = dedupe_candidates(candidates + expanded)
        if not no_light_filter:
            before = len(candidates)
            candidates = [c for c in candidates if looks_relevant(c)]
            print(f'>>> 扩展后轻量过滤: {before} -> {len(candidates)}')

    if reprocess:
        new_records = candidates
        print('>>> Reprocess=True：本轮不会根据历史记忆跳过文献')
    else:
        skipped = [c for c in candidates if memory.has_processed(c)]
        new_records = [c for c in candidates if not memory.has_processed(c)]
        print(f'>>> 记忆跳过已处理文献: {len(skipped)}')
    print(f'>>> 本轮待处理 paper 数量: {len(new_records)}')

    pubmed_pmids = [r.get('pmid') for r in candidates if r.get('pmid')]
    write_json(PUBMED_SEARCH_JSON, {'time': now_str(), 'unique_pmids': sorted(set(map(str, pubmed_pmids))), 'source_counts': source_counts(candidates)})

    pubmed = PubMedClient(); epmc = EuropePMCClient(); ft_fetcher = FullTextFetcher(pubmed, epmc)
    evidence_batches: List[Dict[str, Any]] = []
    failed_batches: List[Dict[str, Any]] = []

    print('\n========== [Stage 3] 全文获取 + Evidence 提取，暂不开会 ==========', flush=True)
    for i in range(0, len(new_records), batch_size):
        batch_no = i // batch_size + 1
        batch = new_records[i:i+batch_size]
        print(f'\n========== [Evidence Collection Batch {batch_no}] 处理 {len(batch)} 篇 ==========', flush=True)
        if fetch_fulltext:
            print('    -> [Step 1] 获取并保存开放全文证据（PMC / Europe PMC）...', flush=True)
            for r in batch:
                st = ft_fetcher.fetch_and_save(r)
                print(f"       PMID {st.get('pmid') or '-'} DOI {st.get('doi') or '-'}: {st.get('status')} | {st.get('source')} | PMCID={st.get('pmcid')} | cache={st.get('cache_dir')}", flush=True)
        else:
            print('    -> [Step 1] 跳过全文获取，只使用 metadata/abstract。')

        print('    -> [Step 2] DeepSeek 提取关键信息 evidence...', flush=True)
        try:
            ev = extract_info_batch(llm, meeting_loader, batch, batch_no)
            evidence_batches.append(ev)
            append_jsonl(FULLTEXT_EVIDENCE_JSONL, ev)
            print('    -> [Step 3] 保存 evidence，暂不开会。')
            print('       ✅ 本批证据已保存，不进行多 Agent 会议。')
        except Exception as e:
            err = {'batch_no': batch_no, 'error': str(e), 'traceback': traceback.format_exc(), 'record_keys': [r.get('candidate_key') for r in batch]}
            failed_batches.append(err)
            append_jsonl(FAILED_DIR / 'failed_evidence_batches.jsonl', err)
            print(f'       ❌ 本批 evidence 提取失败，继续后续 batch：{e}')

    # Model name back-search after initial evidence.
    if backsearch_models:
        model_names = extract_model_names_from_evidence(evidence_batches)
        print('\n========== [Stage 4] 模型名称回搜 ==========', flush=True)
        print(f'>>> 从 evidence 提取到模型名数量: {len(model_names)}')
        if model_names:
            print('>>> 模型名示例:', ', '.join(model_names[:20]))
            back_plan = build_model_backsearch_plan(model_names[:30])
            back_candidates, back_repos, back_datasets, back_summary = run_source_searches(back_plan, max_results=min(max_results, 20), year_from=year_from, year_to=year_to, enabled_sources=enabled_sources)
            repos.extend(back_repos); datasets.extend(back_datasets)
            combined = dedupe_candidates(candidates + back_candidates)
            old_keys = {c.get('candidate_key') for c in candidates}
            new_back_records = [c for c in combined if c.get('candidate_key') not in old_keys]
            if not no_light_filter:
                new_back_records = [c for c in new_back_records if looks_relevant(c) or normalize_key(c.get('title')) in [normalize_key(n) for n in model_names]]
            if not reprocess:
                new_back_records = [c for c in new_back_records if not memory.has_processed(c)]
            print(f'>>> 模型名回搜新增待处理 paper: {len(new_back_records)}')
            candidates = combined
            for c in new_back_records:
                append_jsonl(NORMALIZED_PAPERS_JSONL, c)
            for r in back_repos:
                append_jsonl(NORMALIZED_REPOS_JSONL, r)
            for d in back_datasets:
                append_jsonl(NORMALIZED_DATASETS_JSONL, d)
            for i in range(0, len(new_back_records), batch_size):
                batch_no = len(evidence_batches) + 1
                batch = new_back_records[i:i+batch_size]
                print(f'\n========== [Model Backsearch Evidence Batch {batch_no}] 处理 {len(batch)} 篇 ==========', flush=True)
                if fetch_fulltext:
                    for r in batch:
                        st = ft_fetcher.fetch_and_save(r)
                        print(f"       PMID {st.get('pmid') or '-'} DOI {st.get('doi') or '-'}: {st.get('status')} | {st.get('source')} | cache={st.get('cache_dir')}", flush=True)
                try:
                    ev = extract_info_batch(llm, meeting_loader, batch, batch_no)
                    ev['_stage'] = 'model_name_backsearch'
                    evidence_batches.append(ev)
                    append_jsonl(FULLTEXT_EVIDENCE_JSONL, ev)
                    print('       ✅ 回搜 evidence 已保存，不开会。')
                except Exception as e:
                    err = {'batch_no': batch_no, 'stage': 'model_name_backsearch', 'error': str(e), 'traceback': traceback.format_exc()}
                    failed_batches.append(err)
                    append_jsonl(FAILED_DIR / 'failed_evidence_batches.jsonl', err)
        else:
            print('>>> 未提取到可用于回搜的模型名，跳过。')

    if github_enrichment:
        print('\n========== [Stage 4.5] GitHub 缺失链接补链 ==========', flush=True)
        gh_ev = enrich_evidence_batches_with_missing_github(evidence_batches, repos, max_models=github_enrich_max_models, repos_per_model=github_enrich_repos_per_model, force=force_github_enrichment, refresh_all=refresh_all_github_enrichment)
        if gh_ev:
            evidence_batches.append(gh_ev)
            print(f'>>> GitHub 补链 evidence 已保存：repositories={len(ensure_list(gh_ev.get("repositories")))} models={len(ensure_list(gh_ev.get("models")))}', flush=True)
        else:
            print('>>> GitHub 补链没有产生新 evidence。', flush=True)
    else:
        print('\n========== [Stage 4.5] GitHub 缺失链接补链已关闭 ==========', flush=True)

    if qwen_web_enrichment:
        print('\n========== [Stage 4.6] Qwen3.7-Max 联网补漏 ==========', flush=True)
        qwen_ev = enrich_evidence_batches_with_qwen_web(evidence_batches, provider_config=provider_config, provider=qwen_web_provider, model_name=qwen_web_model, max_models=qwen_web_max_models, force=force_qwen_web_enrichment, refresh_all=refresh_all_qwen_web_enrichment)
        if qwen_ev:
            evidence_batches.append(qwen_ev)
            for r in ensure_list(qwen_ev.get('repositories')):
                if isinstance(r, dict):
                    append_jsonl(NORMALIZED_REPOS_JSONL, r)
            for d in ensure_list(qwen_ev.get('datasets')):
                if isinstance(d, dict):
                    append_jsonl(NORMALIZED_DATASETS_JSONL, d)
            print(f'>>> Qwen3.7-Max 联网补漏 evidence 已保存：repositories={len(ensure_list(qwen_ev.get("repositories")))} datasets={len(ensure_list(qwen_ev.get("datasets")))} papers={len(ensure_list(qwen_ev.get("papers")))} models={len(ensure_list(qwen_ev.get("models")))}', flush=True)
        else:
            print('>>> Qwen3.7-Max 联网补漏没有产生新 evidence。', flush=True)
    else:
        print('\n========== [Stage 4.6] Qwen3.7-Max 联网补漏未启用 ==========', flush=True)

    print('\n========== [Evidence Collection Finished] ==========')
    print(f'>>> Evidence batches collected: {len(evidence_batches)}')
    print(f'>>> Failed evidence batches: {len(failed_batches)}')
    evidence_pool = save_evidence_pool(evidence_batches, candidates, repos, datasets)
    print(f'>>> Evidence pool saved: {EVIDENCE_POOL_JSON.relative_to(ROOT)} / {EVIDENCE_POOL_MD.relative_to(ROOT)}')

    if not evidence_batches:
        print('>>> 没有成功 evidence batch，停止全局会议。')
        return {'evidence_batches': 0, 'failed_batches': failed_batches}

    print('\n========== [Stage 5] 准备 Evidence 分块压缩 ==========', flush=True)
    if evidence_compression:
        print('\n========== [Stage 5] Evidence 分块压缩：按模型名称 / 主题 / 来源 ==========', flush=True)
        compact_pool, chunk_summaries = compress_evidence_chunks(
            llm, meeting_loader, evidence_pool,
            target_items_per_chunk=chunk_target_size,
            max_chunks=max_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        print(f'>>> Chunk summaries saved: {CHUNK_SUMMARIES_DIR.relative_to(ROOT)}')
        print(f'>>> Compact evidence pool saved: {COMPACT_EVIDENCE_POOL_JSON.relative_to(ROOT)} / {COMPACT_EVIDENCE_POOL_MD.relative_to(ROOT)}')
    else:
        print('>>> 跳过 evidence 压缩，将使用轻量化 evidence_pool 进入全局会议。')
        compact_pool = dict(evidence_pool)
        compact_pool['chunk_summaries'] = []
        compact_pool['paper_overview'] = [compact_record_for_prompt(r) for r in ensure_list(evidence_pool.get('papers'))[:300]]
        compact_pool['compression_mode'] = 'disabled'
        compact_pool['chunk_summary_count'] = 0

    # Ensure deterministic enrichment rows are visible as standalone compact-pool sections,
    # even when the general compressor already summarized them in another chunk.
    if github_enrichment and not ensure_list(compact_pool.get('github_missing_model_enrichment')) and GITHUB_MISSING_MODEL_ENRICHMENT_JSON.exists():
        gh_rows_for_compact = ensure_list(read_json(GITHUB_MISSING_MODEL_ENRICHMENT_JSON, []))
        if gh_rows_for_compact:
            compact_pool = add_github_enrichment_to_compact_pool(compact_pool, gh_rows_for_compact)
    if qwen_web_enrichment and not ensure_list(compact_pool.get('qwen_web_enrichment')) and QWEN_WEB_ENRICHMENT_JSON.exists():
        qwen_rows_for_compact = ensure_list(read_json(QWEN_WEB_ENRICHMENT_JSON, []))
        if qwen_rows_for_compact:
            compact_pool = add_qwen_web_enrichment_to_compact_pool(compact_pool, qwen_rows_for_compact)

    print('\n========== [Global Meeting] 所有 chunk summaries 收集完成，开始一次性全局多 Agent 会议 ==========')
    final_data, raw_meeting = global_meeting(llm, meeting_loader, compact_pool, memory.context())

    run_info = {
        'time': now_str(), 'mode': 'multi_source_global_meeting', 'sources': enabled_sources,
        'max_results': max_results, 'batch_size': batch_size, 'paper_count': len(candidates),
        'processed_this_run': len(new_records), 'evidence_batches': len(evidence_batches), 'failed_evidence_batches': len(failed_batches),
        'source_counts': source_counts(candidates), 'fetch_fulltext': fetch_fulltext,
        'backsearch_models': backsearch_models, 'expand_citations': expand_citations,
        'evidence_compression': evidence_compression, 'chunk_target_size': chunk_target_size,
        'max_chunks': max_chunks, 'max_chars_per_chunk': max_chars_per_chunk,
        'compact_evidence_pool': str(COMPACT_EVIDENCE_POOL_JSON.relative_to(ROOT)) if evidence_compression else None,
        'github_enrichment': bool(github_enrichment),
        'github_enrichment_file': str(GITHUB_MISSING_MODEL_ENRICHMENT_JSON.relative_to(ROOT)) if GITHUB_MISSING_MODEL_ENRICHMENT_JSON.exists() else None,
        'qwen_web_enrichment': bool(qwen_web_enrichment),
        'qwen_web_model': qwen_web_model if qwen_web_enrichment else None,
        'qwen_web_enrichment_file': str(QWEN_WEB_ENRICHMENT_JSON.relative_to(ROOT)) if QWEN_WEB_ENRICHMENT_JSON.exists() else None,
    }
    print('    -> [Write Memory] 写入 MD + JSON 长期记忆...')
    memory.merge_final(final_data, candidates, run_info)
    print('✅ 完成。')
    print(f'   Memory MD: {MEMORY_MD.relative_to(ROOT)}')
    print(f'   Memory JSON: {MEMORY_JSON.relative_to(ROOT)}')
    return {'run_info': run_info, 'final_data': final_data, 'raw_meeting': raw_meeting}


# ------------------------- CLI -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Multi-source AMP literature evidence collector + DeepSeek global meeting')
    p.add_argument('--max-results', type=int, default=30, help='每个 query 每个来源最多返回多少条。更全面建议 50~100。')
    p.add_argument('--batch-size', type=int, default=4, help='DeepSeek evidence 提取批大小。全文较长时建议 2~5。')
    p.add_argument('--year-from', type=int, default=None)
    p.add_argument('--year-to', type=int, default=None)
    p.add_argument('--provider', default='dashscope')
    p.add_argument('--provider-config', default='llm_providers.json')
    p.add_argument('--planner-agent-dir', default='agents/pubmed_planner')
    p.add_argument('--meeting-agent-dir', default='agents/deepseek_meeting')
    p.add_argument('--no-fulltext', action='store_true', help='不抓开放全文，只用 metadata/abstract。')
    p.add_argument('--no-planner', action='store_true', help='不用 DeepSeek 规划 query，只使用内置 query 词库。')
    p.add_argument('--reprocess', action='store_true', help='忽略历史记忆，重新处理已处理文章。')
    p.add_argument('--max-queries', type=int, default=20)
    p.add_argument('--sources', default='all', help='逗号分隔来源：all 或 pubmed,europe_pmc,crossref,openalex,semantic_scholar,github,datacite,zenodo')
    p.add_argument('--no-light-filter', action='store_true', help='关闭 AMP/模型轻量过滤，保留更多候选。')
    p.add_argument('--no-backsearch-models', action='store_true', help='关闭模型名称回搜。')
    p.add_argument('--no-citation-expansion', action='store_true', help='关闭 PubMed Similar / Semantic Scholar 引用扩展。')
    p.add_argument('--citation-seed-limit', type=int, default=12, help='用于相似/引用扩展的 seed 文章数量。')
    p.add_argument('--no-evidence-compression', action='store_true', help='关闭分块压缩，直接用轻量 evidence 进入全局会议；不推荐。')
    p.add_argument('--chunk-target-size', type=int, default=6, help='每个 evidence chunk 里包含的条目数，越小越稳但调用次数越多。')
    p.add_argument('--max-chunks', type=int, default=120, help='最多压缩多少个 chunk，防止超大搜索导致调用过多。')
    p.add_argument('--max-chars-per-chunk', type=int, default=60000, help='每个 chunk 送入 compressor 的最大字符数。')
    p.add_argument('--resume-global-only', action='store_true', help='不重新搜索/全文/压缩，只读取已有 compact_evidence_pool/chunk_summaries 后跑最后全局会议并写 memory。')
    p.add_argument('--use-existing-meeting', action='store_true', help='和 --resume-global-only 配合；不重新调用 DeepSeek，直接复用最近一次 data/deepseek_meeting_raw.jsonl 的 chief_agent 结果写 memory。')
    p.add_argument('--no-github-enrichment', action='store_true', help='关闭全局会议前的 GitHub 缺失链接补链搜索。')
    p.add_argument('--github-enrich-max-models', type=int, default=80, help='GitHub 补链最多搜索多少个缺 GitHub 的模型。')
    p.add_argument('--github-enrich-repos-per-model', type=int, default=3, help='每个模型最多保留多少个候选 GitHub 仓库证据。')
    p.add_argument('--force-github-enrichment', action='store_true', help='重新搜索失败/低置信 GitHub 补链；v5.0 起默认不会重复搜索已成功的 high/medium 缓存。')
    p.add_argument('--refresh-all-github-enrichment', action='store_true', help='真正忽略所有 GitHub 补链缓存，连已成功的 high/medium 结果也全部重搜。')
    p.add_argument('--qwen-web-enrichment', action='store_true', help='启用 Qwen3.7-Max 联网补漏：搜索缺失的 GitHub/数据集/权重/web server/论文主页证据。')
    p.add_argument('--qwen-web-provider', default='dashscope_qwen37max_search', help='llm_providers.json 中用于 Qwen 联网搜索的 provider 名称，默认 dashscope_qwen37max_search。')
    p.add_argument('--qwen-web-model', default='qwen3.7-max', help='用于联网补漏的 Qwen 模型，默认 qwen3.7-max（Responses API）。')
    p.add_argument('--qwen-web-max-models', type=int, default=30, help='Qwen3.7-Max 联网补漏本轮最多处理多少个模型，避免费用过高。')
    p.add_argument('--force-qwen-web-enrichment', action='store_true', help='重新搜索失败/低置信 Qwen 联网补漏缓存；默认不重复搜索已成功缓存。')
    p.add_argument('--refresh-all-qwen-web-enrichment', action='store_true', help='真正忽略所有 Qwen 联网补漏缓存，全部重搜。')
    p.add_argument('--resume-note', default='main_resume_global_meeting_only', help='续跑写入 runs 的备注。')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_global_only or args.use_existing_meeting:
        run_resume_global_meeting_only(
            provider=args.provider,
            provider_config=Path(args.provider_config),
            meeting_agent_dir=Path(args.meeting_agent_dir),
            use_existing_meeting=args.use_existing_meeting,
            run_note=args.resume_note,
            github_enrichment=not args.no_github_enrichment,
            github_enrich_max_models=args.github_enrich_max_models,
            github_enrich_repos_per_model=args.github_enrich_repos_per_model,
            force_github_enrichment=args.force_github_enrichment,
            refresh_all_github_enrichment=args.refresh_all_github_enrichment,
            qwen_web_enrichment=args.qwen_web_enrichment,
            qwen_web_provider=args.qwen_web_provider,
            qwen_web_model=args.qwen_web_model,
            qwen_web_max_models=args.qwen_web_max_models,
            force_qwen_web_enrichment=args.force_qwen_web_enrichment,
            refresh_all_qwen_web_enrichment=args.refresh_all_qwen_web_enrichment,
        )
        return
    run_pipeline(
        max_results=args.max_results,
        batch_size=args.batch_size,
        year_from=args.year_from,
        year_to=args.year_to,
        provider=args.provider,
        provider_config=Path(args.provider_config),
        planner_agent_dir=Path(args.planner_agent_dir),
        meeting_agent_dir=Path(args.meeting_agent_dir),
        fetch_fulltext=not args.no_fulltext,
        no_planner=args.no_planner,
        reprocess=args.reprocess,
        max_queries=args.max_queries,
        sources=args.sources,
        no_light_filter=args.no_light_filter,
        backsearch_models=not args.no_backsearch_models,
        expand_citations=not args.no_citation_expansion,
        citation_seed_limit=args.citation_seed_limit,
        evidence_compression=not args.no_evidence_compression,
        chunk_target_size=args.chunk_target_size,
        max_chunks=args.max_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
        github_enrichment=not args.no_github_enrichment,
        github_enrich_max_models=args.github_enrich_max_models,
        github_enrich_repos_per_model=args.github_enrich_repos_per_model,
        force_github_enrichment=args.force_github_enrichment,
        refresh_all_github_enrichment=args.refresh_all_github_enrichment,
        qwen_web_enrichment=args.qwen_web_enrichment,
        qwen_web_provider=args.qwen_web_provider,
        qwen_web_model=args.qwen_web_model,
        qwen_web_max_models=args.qwen_web_max_models,
        force_qwen_web_enrichment=args.force_qwen_web_enrichment,
        refresh_all_qwen_web_enrichment=args.refresh_all_qwen_web_enrichment,
    )


if __name__ == '__main__':
    main()
