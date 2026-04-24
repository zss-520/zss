import json
import os
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime
import time
from pathlib import Path
from typing import Any, Dict, List
from serpapi import GoogleSearch
import PyPDF2
from openai import OpenAI

# 假设你的项目中已经有了这些
from agent import Agent
from config import MODEL_NAME
from prompts import BENCHMARK_ARCHITECT_PROMPT

# ==========================================
# 默认兜底策略与 JSON 校验
# ==========================================
DEFAULT_BINARY_AMP_STRATEGY = {
    "task_type": "binary_amp_classification",
    "recommended_datasets": [],
    "label_definition": {
        "positive_rule": "实验支持为 AMP 的肽作为正样本",
        "negative_rule": "优先使用实验支持的无活性肽作为负样本",
        "ambiguous_rule": "灰区样本不参与主测试"
    },
    "deduplication_policy": {
        "exact_dedup": True,
        "near_duplicate_policy": "同一序列必须去重",
        "homology_control_recommended": True
    },
    "export_schema": [
        "id", "sequence", "label", "source_dataset", "evidence_level"
    ],
    "dataset_processing_steps": [
        "download", "normalize_columns", "filter_invalid_sequences",
        "apply_label_rules", "remove_ambiguous", "deduplicate", "export_ground_truth"
    ],
    "metric_weights": {
        "ACC": 0.10, "Recall": 0.15, "MCC": 0.35, "AUROC": 0.10, "AUPRC": 0.30
    },
    "metrics_references": {},
    "reasoning": "默认兜底策略"
}

def safe_float(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

def normalize_weights(metric_weights: Dict[str, Any]) -> Dict[str, float]:
    cleaned = {str(k): safe_float(v, 0.0) for k, v in metric_weights.items()}
    cleaned = {k: v for k, v in cleaned.items() if v > 0}
    total = sum(cleaned.values())
    if total <= 0: return DEFAULT_BINARY_AMP_STRATEGY["metric_weights"].copy()
    return {k: round(v / total, 4) for k, v in cleaned.items()}

def deep_merge(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(default)
    for k, v in custom.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

def validate_strategy(strategy_data: Dict[str, Any]) -> Dict[str, Any]:
    strategy_data = deep_merge(DEFAULT_BINARY_AMP_STRATEGY, strategy_data or {})
    if strategy_data.get("task_type") != "binary_amp_classification":
        strategy_data["task_type"] = "binary_amp_classification"
    strategy_data["metric_weights"] = normalize_weights(strategy_data.get("metric_weights", {}))
    
    if not isinstance(strategy_data.get("recommended_datasets"), list):
        strategy_data["recommended_datasets"] = []
    for ds in strategy_data["recommended_datasets"]:
        if isinstance(ds, dict):
            ds.setdefault("dataset_name", "")
            ds.setdefault("description", "")
            ds.setdefault("source_papers", [])
            ds.setdefault("download_url", "")
            ds.setdefault("role", "auxiliary_source")

    for field in ["label_definition", "sequence_constraints", "deduplication_policy"]:
        if not isinstance(strategy_data.get(field), dict):
            strategy_data[field] = DEFAULT_BINARY_AMP_STRATEGY.get(field, {}).copy()

    if not isinstance(strategy_data.get("export_schema"), list):
        strategy_data["export_schema"] = DEFAULT_BINARY_AMP_STRATEGY["export_schema"].copy()
    if not isinstance(strategy_data.get("dataset_processing_steps"), list):
        strategy_data["dataset_processing_steps"] = DEFAULT_BINARY_AMP_STRATEGY["dataset_processing_steps"].copy()
    if not isinstance(strategy_data.get("metrics_references"), dict):
        strategy_data["metrics_references"] = {}
    if not isinstance(strategy_data.get("reasoning"), str):
        strategy_data["reasoning"] = DEFAULT_BINARY_AMP_STRATEGY["reasoning"]

    return strategy_data


# ==========================================
# 工具函数：通用直链 PDF 下载器与预印本猎手
# ==========================================
def download_direct_pdf(pdf_url: str, output_path: Path) -> bool:
    try:
        req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=30) as response:
            chunk = response.read(2048)
            if b'%PDF' in chunk:
                with open(output_path, 'wb') as f:
                    f.write(chunk)
                    f.write(response.read())
                return True
        return False
    except Exception:
        return False

def hunt_for_preprint_pdf(title: str) -> str:
    """预印本替身猎手：根据标题去免费数据库寻找同名预印本或OA版本的 PDF 直链"""
    print(f"       [替身猎手] 正在为闭源文献寻找免费预印本: {title[:20]}...")
    try:
        safe_title_query = urllib.parse.quote(f'ti:"{title}"')
        arxiv_url = f"http://export.arxiv.org/api/query?search_query={safe_title_query}&max_results=1"
        req = urllib.request.Request(arxiv_url)
        with urllib.request.urlopen(req, timeout=10) as response:
            root = ET.fromstring(response.read())
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            for entry in root.findall('atom:entry', namespace):
                for link in entry.findall('atom:link', namespace):
                    if link.get('title') == 'pdf':
                        pdf_link = link.get('href') + ".pdf"
                        pdf_link = pdf_link.replace("http://", "https://")
                        print(f"       ✅ [替身猎手] 太棒了！在 arXiv 找到了同名预印本！")
                        return pdf_link
    except Exception: pass

    try:
        pmc_title_query = urllib.parse.quote(f'TITLE:"{title}" AND OPEN_ACCESS:y')
        pmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={pmc_title_query}&format=json&resultType=lite"
        req = urllib.request.Request(pmc_url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = data.get("resultList", {}).get("result", [])
            if results:
                pmcid = results[0].get("pmcid")
                if pmcid:
                    print(f"       ✅ [替身猎手] 在 PMC 找到了开源替身 ID: {pmcid}！")
                    return f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML" 
    except Exception: pass

    print("       ⚠️ [替身猎手] 尽力了，各大免费库均未找到同名替代品。")
    return "无"

# 🚨 新增：开源链接硬拦截过滤器
def has_open_source_link(text: str) -> bool:
    if not text: return False
    keywords = ["github.com", "zenodo.org", "gitlab.com", "gitee.com"]
    return any(k in text.lower() for k in keywords)

# 🚨 新增：人机交互文献下载向导
def interactive_paper_download(title: str, url: str, target_path: Path) -> bool:
    """交互式文献下载拦截器：自动下载失败时，引导用户手动下载并重命名放入指定目录"""
    print("\n" + "🚨" * 15)
    print(f"🛑 [需人工介入] 自动下载 PDF 失败 (防盗链/需登录/非直链)！")
    print(f"📄 文献: 【{title}】")
    print(f"🔗 下载链接/参考网页: {url}")
    print(f"📂 请在浏览器中手动下载该 PDF，并【严格重命名保存】到以下确切路径：")
    print(f"   ---> {target_path.absolute()}")
    print("-" * 40)

    while True:
        ans = input(f">>> 📥 PDF 下载并放好了吗？(输入 'y' 继续，输入 'skip' 跳过该 PDF 并仅使用摘要): ").strip().lower()
        if ans == 'y':
            if target_path.exists():
                print("✅ 成功检测到人工存入的 PDF！继续管线...\n")
                return True
            else:
                ans2 = input("⚠️ 目标路径下依然没有检测到该文件！你确定已经放好了吗？要强制继续吗？(y/n): ").strip().lower()
                if ans2 == 'y':
                    print("✅ 强制继续...\n")
                    return True
        elif ans == 'skip':
            print("⏭️ 已跳过 PDF 人工下载，将仅依赖提取到的文献文本信息。\n")
            return False
        else:
            print("❌ 输入无效，请输入 'y' 或 'skip'。")

# ==========================================
# 滚雪球引擎
# ==========================================
def snowball_from_seed_paper(seed_title: str, output_dir: Path, max_results: int = 5):
    """引擎进阶：OpenAlex 向前滚雪球 (寻找引用了某篇神作的最新文章)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [滚雪球引擎] 正在追踪引用了《{seed_title[:20]}...》的最新文献...")
    
    email = "2873691970@qq.com"
    headers = {'User-Agent': f'mailto:{email}'}
    
    safe_query = urllib.parse.quote(seed_title)
    search_url = f"https://api.openalex.org/works?search={safe_query}&mailto={email}"
    
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))
            if not data.get("results"):
                print("    -> [雪球中止] 未能找到种子文献的 OpenAlex ID。")
                return
            seed_id = data["results"][0]["id"].split("/")[-1] 
            print(f"    -> [锁定神作] 成功获取种子 ID: {seed_id}，开始顺藤摸瓜...")
            
        snowball_url = f"https://api.openalex.org/works?filter=cites:{seed_id}&sort=publication_date:desc&per-page={max_results}&mailto={email}"
        req2 = urllib.request.Request(snowball_url, headers=headers)
        with urllib.request.urlopen(req2, timeout=15) as response2:
            snow_data = json.loads(response2.read().decode('utf-8'))
            results = snow_data.get("results", [])
            
            saved_count = 0
            for p in results:
                title = p.get("title") or "Unknown"
                year = p.get("publication_year") or "Unknown"
                pdf_url = p.get("open_access", {}).get("oa_url") or "无"
                
                if pdf_url == "无":
                    pdf_url = hunt_for_preprint_pdf(title)
                
                txt_path = output_dir / f"Snowball_{year}_{re.sub(r'[^a-zA-Z0-9]', '_', title)[:30]}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"【滚雪球来源】: 引用了 {seed_id}\n【标题】: {title}\n【年份】: {year}\n【PDF】: {pdf_url}\n")
                saved_count += 1
                
            print(f"    -> [大丰收] 雪球引擎成功挖出 {saved_count} 篇最新衍生文献！")
    except Exception as e:
        print(f"!!! [滚雪球失败]: {e}")

# ==========================================
# 引擎 A：Semantic Scholar
# ==========================================
def fetch_from_semantic_scholar(topic_name: str, query: str, output_dir: Path, max_results: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [S2 引擎] 正在启动 Semantic Scholar 检索【{topic_name}】...")
    
    encoded_query = urllib.parse.quote(query)
    fields = "title,abstract,year,citationCount,openAccessPdf"
    current_year = datetime.datetime.now().year
    past_year = current_year - 5
    year_filter = f"{past_year}-{current_year}"
    
    fetch_limit = max(50, max_results * 5)
    s2_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={fetch_limit}&fields={fields}&year={year_filter}"
    
    max_retries = 3
    data = None
    
    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) MLOps-Pipeline/1.0'}
            req = urllib.request.Request(s2_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode('utf-8'))
                break 
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = 2 ** attempt
                print(f"    -> [S2 遇阻] 触发限流防御 (429)，正在隐蔽等待 {wait_time} 秒后发起第 {attempt+1} 次突围...")
                time.sleep(wait_time)
            else:
                print(f"!!! [S2 检索失败]: HTTP Error {e.code}")
                return
        except Exception as e:
            print(f"!!! [S2 检索失败]: 发生未知网络异常 -> {e}")
            return
            
    if data is None:
        print("!!! [S2 检索失败]: 多次重试均被拦截，本次暂不使用 S2 引擎。")
        return
        
    papers = data.get("data", [])
    if not papers: return
        
    print(f"    -> [S2] 成功突破！锁定 {len(papers)} 篇初步文献，正在执行开源链接硬拦截...")
    
    saved_count = 0
    for p in papers:
        title = p.get("title") or "Unknown"
        abstract = p.get("abstract") or "No abstract available."
        
        if not has_open_source_link(title + " " + abstract):
            continue

        year = p.get("year") or "Unknown"
        cites = p.get("citationCount") or 0
        
        oa_info = p.get("openAccessPdf")
        pdf_url = oa_info.get("url") if oa_info else "无"
        if pdf_url == "无":
            pdf_url = hunt_for_preprint_pdf(title)
        
        github_links = re.findall(r"https://github\.com/[^\s\)]+", abstract)
        github_str = "\n    - ".join(github_links) if github_links else "无明显的 GitHub 链接"
        
        file_content = f"【文献归属领域】: {topic_name}\n【文献标题】: {title}\n【发表年份】: {year}\n【S2 引用次数】: {cites}\n【PDF 直链】: {pdf_url}\n\n【摘要】: {abstract}\n\n【可能的数据集或开源代码链接提取】:\n    - {github_str}\n"
        
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:30]
        base_filename = f"S2_cites_{cites}_{safe_title}"
        txt_path = output_dir / f"{base_filename}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(file_content)
            
        # 🚨 新增：【拦截机制】S2 替身猎手 XML 截获
        if "fullTextXML" in pdf_url:
            print(f"       [{saved_count+1}/{max_results}] 正在通过替身猎手抓取 PMC 全文...")
            try:
                req_xml = urllib.request.Request(pdf_url)
                with urllib.request.urlopen(req_xml, timeout=20) as resp:
                    root = ET.fromstring(resp.read())
                    body_elem = root.find(".//body")
                    full_text = " ".join("".join(body_elem.itertext()).split()) if body_elem is not None else ""
                    
                    if full_text:
                        with open(txt_path, "a", encoding="utf-8") as f_txt:
                            f_txt.write("\n=========================================\n")
                            f_txt.write("【替身猎手 PMC XML 全文无感提取】:\n")
                            f_txt.write(full_text[:25000])
                            f_txt.write("\n=========================================\n")
                        print("       ✅ 成功免下载提取替身全文！跳过人工介入。")
                        saved_count += 1
                        if saved_count >= max_results: break
                        continue  # 核心：直接跳到下一篇文章
                    else:
                        print("       ⚠️ 替身 XML 无正文(可能仅摘要)。转入后备流程...")
            except Exception as e:
                print(f"       ⚠️ 替身 XML 拉取失败 ({e})。转入后备流程...")

        # 原 PDF 流程
        pdf_path = output_dir / f"{base_filename}.pdf"
        print(f"       [{saved_count+1}/{max_results}] 尝试自动获取 PDF: {title[:15]}... ", end="", flush=True)
        
        if pdf_url != "无" and pdf_url.endswith(".pdf") and download_direct_pdf(pdf_url, pdf_path):
            print("✅ 自动获取成功！存入本地！")
        else:
            print("⚠️ 自动下载失败或非直链！")
            fallback_url = pdf_url if pdf_url != "无" else f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"
            interactive_paper_download(title, fallback_url, pdf_path)
        time.sleep(1.5)

        saved_count += 1
        if saved_count >= max_results: break
        
    print(f"    -> [大丰收] S2 引擎成功处理 {saved_count} 篇【极高含金量】开源文献记录！")

# ==========================================
# 引擎 B：Europe PMC
# ==========================================
def download_pmc_pdf_via_oa(pmcid: str, output_path: Path) -> bool:
    if not pmcid.startswith("PMC"): pmcid = f"PMC{pmcid}"
    oa_api_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    try:
        req = urllib.request.Request(oa_api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            root = ET.fromstring(response.read())
        pdf_link = None
        for link in root.findall(".//link"):
            if link.get("format") == "pdf":
                pdf_link = link.get("href")
                break
        if not pdf_link: return False 
        if pdf_link.startswith("ftp://"): pdf_link = pdf_link.replace("ftp://", "https://")
        
        pdf_req = urllib.request.Request(pdf_link, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(pdf_req, timeout=45) as pdf_response:
            chunk = pdf_response.read(2048)
            if b'%PDF' in chunk:
                with open(output_path, 'wb') as f:
                    f.write(chunk)
                    f.write(pdf_response.read())
                return True
        return False
    except Exception: return False

def fetch_and_save_pmc_papers(topic_name: str, epmc_query: str, output_dir: Path, max_results: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [PMC 引擎] 正在检索并保存【{topic_name}】相关文献至 -> {output_dir}")
    
    fetch_limit = max(50, max_results * 5)
    encoded_query = urllib.parse.quote(epmc_query)
    epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded_query}&format=json&resultType=lite&sort=CITED%20desc&pageSize={fetch_limit}"
    
    try:
        req = urllib.request.Request(epmc_url)
        with urllib.request.urlopen(req, timeout=15) as response:
            search_data = json.loads(response.read().decode('utf-8'))
            
        results = search_data.get("resultList", {}).get("result", [])
        
        pmc_ids = []
        citation_info = {} 
        for r in results:
            pmcid = r.get("pmcid")
            if pmcid:
                clean_id = pmcid.replace("PMC", "")
                pmc_ids.append(clean_id)
                citation_info[clean_id] = r.get("citedByCount", 0)
                
        if not pmc_ids:
            print(f"    -> [警告] 未找到【{topic_name}】的近期高被引文献。")
            return
            
        ids_str = ",".join(pmc_ids)
        print(f"    -> 成功锁定 {len(pmc_ids)} 篇初步文献，正在拉取 XML 并执行开源代码硬拦截...")
        
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={ids_str}&retmode=xml"
        req2 = urllib.request.Request(fetch_url)
        with urllib.request.urlopen(req2, timeout=25) as response2:
            xml_data = response2.read()
            
        root = ET.fromstring(xml_data)
        saved_count = 0
        for article in root.findall(".//article"):
            title_elem = article.find(".//article-title")
            title = "".join(title_elem.itertext()) if title_elem is not None else "Unknown Title"
            abstract_elem = article.find(".//abstract")
            abstract = " ".join(abstract_elem.itertext()) if abstract_elem is not None else "No abstract"
            
            body_elem = article.find(".//body")
            full_text_body = " ".join("".join(body_elem.itertext()).split()) if body_elem is not None else ""

            if not has_open_source_link(title + " " + abstract + " " + full_text_body):
                continue

            year_elem = article.find(".//pub-date/year")
            year = year_elem.text if year_elem is not None else "近期"
            
            cited_count = "未知"
            target_pmcid = ""
            for art_id_node in article.findall(".//article-id"):
                val = (art_id_node.text or "").strip()
                clean_val = val.upper().replace("PMC", "")
                if clean_val and clean_val in citation_info:
                    cited_count = citation_info[clean_val]
                    target_pmcid = f"PMC{clean_val}"
                    break
            
            data_links = []
            for p in article.findall(".//p"):
                p_text = "".join(p.itertext()).strip()
                p_text_lower = p_text.lower()
                if "http" in p_text_lower or "ftp://" in p_text_lower or "data availability" in p_text_lower or "zenodo" in p_text_lower:
                    if 20 < len(p_text) < 1500: data_links.append(p_text)
                        
            data_links = list(set(data_links))[:5]
            data_context = "\n    - ".join(data_links) if data_links else "无明显的外部链接或数据集库。"
            
            file_content = f"【文献归属领域】: {topic_name}\n【文献标题】: {title}\n【发表年份】: {year}\n【引用次数】: {cited_count}\n【PMC ID】: {target_pmcid}\n【摘要】: {abstract}\n\n【可能的数据集或开源代码链接提取】:\n    - {data_context}\n"

            if full_text_body:
                file_content += f"\n【PMC 全文正文提取】:\n{full_text_body}\n" 
            else:
                file_content += f"\n【PMC 全文正文提取】:\n(未获取到 XML 正文，该文献可能为仅摘要版本)\n"

            safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:30]
            base_filename = f"PMC_cites_{cited_count}_{safe_title}"
            txt_path = output_dir / f"{base_filename}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(file_content)
                
            if target_pmcid:
                    pdf_path = output_dir / f"{base_filename}.pdf"
                    print(f"       [{saved_count+1}/{max_results}] 正在窃取开源 PDF: {target_pmcid} ... ", end="", flush=True)
                    if download_pmc_pdf_via_oa(target_pmcid, pdf_path):
                        print("✅ 自动获取成功！存入本地！")
                    else:
                        # 🚨 核心修改：PMC 已经成功拿到 XML 全文，无需强求 PDF，不触发人工阻塞！
                        print("⚠️ 无法获取 PDF，但已提取 XML 全文正文，跳过人工介入。")
                    time.sleep(1)
            saved_count += 1
            if saved_count >= max_results: break
            
        print(f"    -> [大丰收] PMC 引擎成功提取并处理 {saved_count} 篇【极高含金量】文献记录！")
    except Exception as e:
        print(f"!!! [PMC Search] 全文检索发生异常: {e}")

# ==========================================
# 引擎 C：OpenAlex
# ==========================================
def fetch_from_openalex(topic_name: str, query: str, output_dir: Path, max_results: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [OpenAlex 引擎] 正在启动大范围检索【{topic_name}】...")
    
    search_query = urllib.parse.quote(query)
    email = "2873691970@qq.com"
    fetch_limit = max(50, max_results * 5)
    oa_url = f"https://api.openalex.org/works?search={search_query}&per-page={fetch_limit}&mailto={email}"
    headers = {'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 mailto:{email}'}
    
    max_retries = 3
    data = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(oa_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode('utf-8'))
                break 
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = (2 ** attempt) * 5 
                print(f"    -> [OpenAlex 遇阻] 触发限流 (429)，正在隐蔽等待 {wait_time} 秒后发起第 {attempt+1} 次突围...")
                time.sleep(wait_time)
            else:
                print(f"!!! [OpenAlex 检索失败]: HTTP Error {e.code}")
                return
        except Exception as e:
            print(f"!!! [OpenAlex 检索失败]: {e}")
            return
            
    if data is None: return
        
    results = data.get("results", [])
    if not results: return
        
    saved_count = 0
    for p in results:
        title = p.get("title") or "Unknown"
        
        abstract_text = ""
        inv_idx = p.get("abstract_inverted_index")
        if inv_idx:
            words = []
            for word, positions in inv_idx.items():
                for pos in positions: words.append((pos, word))
            words.sort()
            abstract_text = " ".join([w[1] for w in words])

        if not has_open_source_link(title + " " + abstract_text):
            continue

        year = p.get("publication_year") or "Unknown"
        cites = p.get("cited_by_count") or 0
        
        pdf_url = p.get("open_access", {}).get("oa_url") or "无"
        if pdf_url == "无":
            pdf_url = hunt_for_preprint_pdf(title)
        
        file_content = f"【文献归属领域】: {topic_name}\n【文献标题】: {title}\n【发表年份】: {year}\n【引用次数】: {cites}\n【PDF 直链】: {pdf_url}\n\n【摘要】: {abstract_text}\n"
        
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:30]
        txt_path = output_dir / f"OpenAlex_cites_{cites}_{safe_title}.txt"
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(file_content)
            
        # 🚨 新增：【拦截机制】OpenAlex 替身猎手 XML 截获
        if "fullTextXML" in pdf_url:
            print(f"       [{saved_count+1}/{max_results}] 正在通过替身猎手抓取 PMC 全文...")
            try:
                req_xml = urllib.request.Request(pdf_url)
                with urllib.request.urlopen(req_xml, timeout=20) as resp:
                    root = ET.fromstring(resp.read())
                    body_elem = root.find(".//body")
                    full_text = " ".join("".join(body_elem.itertext()).split()) if body_elem is not None else ""
                    
                    if full_text:
                        with open(txt_path, "a", encoding="utf-8") as f_txt:
                            f_txt.write("\n=========================================\n")
                            f_txt.write("【替身猎手 PMC XML 全文无感提取】:\n")
                            f_txt.write(full_text[:25000])
                            f_txt.write("\n=========================================\n")
                        print("       ✅ 成功免下载提取替身全文！跳过人工介入。")
                        saved_count += 1
                        if saved_count >= max_results: break
                        continue  # 核心：直接跳到下一篇文章
                    else:
                        print("       ⚠️ 替身 XML 无正文(可能仅摘要)。转入后备流程...")
            except Exception as e:
                print(f"       ⚠️ 替身 XML 拉取失败 ({e})。转入后备流程...")

        # 原 PDF 流程
        pdf_path = output_dir / f"OpenAlex_cites_{cites}_{safe_title}.pdf"
        print(f"       [{saved_count+1}/{max_results}] 尝试获取 OpenAlex PDF: {safe_title[:15]}... ", end="", flush=True)
        
        download_success = False
        if pdf_url != "无" and pdf_url.endswith(".pdf") and download_direct_pdf(pdf_url, pdf_path):
            print("✅ 自动获取成功！")
            download_success = True
        else:
            print("⚠️ 自动下载失败 (防盗链或非直链)")
            fallback_url = pdf_url if pdf_url != "无" else f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"
            download_success = interactive_paper_download(title, fallback_url, pdf_path)
        
        if download_success and pdf_path.exists():
            print("       正在提取正文...")
            try:
                with open(pdf_path, "rb") as f_pdf:
                    reader = PyPDF2.PdfReader(f_pdf)
                    extracted_text = ""
                    for i, page in enumerate(reader.pages):
                        if i >= 15: break
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
                
                with open(txt_path, "a", encoding="utf-8") as f_txt:
                    f_txt.write("\n=========================================\n")
                    f_txt.write("【OpenAlex PDF 全文/核心正文提取】:\n")
                    clean_text = " ".join(extracted_text.split())
                    f_txt.write(clean_text[:25000]) 
                    f_txt.write("\n=========================================\n")
                    
            except Exception as e:
                print(f"       ⚠️ PDF 文本提取失败 (可能是扫描版或加密): {e}")
        time.sleep(1) 
        
        saved_count += 1
        if saved_count >= max_results: break
        
    print(f"    -> [大丰收] OpenAlex 引擎成功处理 {saved_count} 篇干货文献！")

# ==========================================
# 引擎 D：arXiv
# ==========================================
def fetch_from_arxiv(topic_name: str, query: str, output_dir: Path, max_results: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [arXiv 引擎] 正在扫荡预印本数据库【{topic_name}】...")
    
    fetch_limit = max(50, max_results * 5)
    arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={fetch_limit}&sortBy=submittedDate&sortOrder=descending"
    
    try:
        req = urllib.request.Request(arxiv_url)
        with urllib.request.urlopen(req, timeout=20) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        saved_count = 0
        for entry in root.findall('atom:entry', namespace):
            title = entry.find('atom:title', namespace).text.replace('\n', ' ')
            summary = entry.find('atom:summary', namespace).text.replace('\n', ' ')
            
            if not has_open_source_link(title + " " + summary):
                continue

            published = entry.find('atom:published', namespace).text[:4]
            
            pdf_url = "无"
            for link in entry.findall('atom:link', namespace):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href') + ".pdf"
                    
            github_links = re.findall(r"https://github\.com/[^\s\)]+", summary)
            github_str = "\n    - ".join(github_links) if github_links else "无明显的 GitHub 链接"
            
            file_content = f"【文献归属领域】: {topic_name}\n【文献标题】: {title}\n【发表年份】: {published}\n【PDF 直链】: {pdf_url}\n\n【摘要】: {summary}\n\n【数据集或开源代码链接提取】:\n    - {github_str}\n"
            
            safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:30]
            base_filename = f"arXiv_{published}_{safe_title}"
            txt_path = output_dir / f"{base_filename}.txt"
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            
            if pdf_url != "无" and pdf_url.startswith("http"):
                pdf_url = pdf_url.replace("http://", "https://")
                pdf_path = output_dir / f"{base_filename}.pdf"
                
                print(f"       [{saved_count+1}/{max_results}] 尝试获取 arXiv PDF: {safe_title[:20]}... ", end="", flush=True)
                
                # 🚨 核心修改：先试自动下载，失败后人工介入
                download_success = False
                if download_direct_pdf(pdf_url, pdf_path):
                    print("✅ 自动下载成功！")
                    download_success = True
                else:
                    print("⚠️ 自动下载失败 (网络波动)")
                    download_success = interactive_paper_download(title, pdf_url, pdf_path)

                if download_success and pdf_path.exists():
                    print("       正在提取正文...")
                    try:
                        with open(pdf_path, "rb") as f_pdf:
                            reader = PyPDF2.PdfReader(f_pdf)
                            extracted_text = ""
                            for i, page in enumerate(reader.pages):
                                if i >= 3: break
                                page_text = page.extract_text()
                                if page_text:
                                    extracted_text += page_text + "\n"
                        with open(txt_path, "a", encoding="utf-8") as f_txt:
                            f_txt.write("\n=========================================\n")
                            f_txt.write("【PDF 核心正文提取 (前 3 页)】:\n")
                            f_txt.write(extracted_text[:8000]) 
                            f_txt.write("\n=========================================\n")
                    except Exception as e:
                        print(f"       ⚠️ PDF 文本提取失败 (可能是扫描版或加密): {e}")

            saved_count += 1
            if saved_count >= max_results: break

        print(f"    -> [大丰收] arXiv 引擎成功处理 {saved_count} 篇前沿开源预印本文献！")
    except Exception as e:
        print(f"!!! [arXiv 检索失败]: {e}")

# ==========================================
# 引擎 E：SerpApi
# ==========================================
# ==========================================
# 引擎 E：SerpApi
# ==========================================
def fetch_from_serpapi_scholar(topic_name: str, query: str, output_dir: Path, max_results: int = 5):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [SerpApi 引擎] 正在调用商业接口检索谷歌学术【{topic_name}】...")
    
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        print("!!! [Error] 未找到 SERPAPI_KEY，请在 .env 中配置。")
        return

    current_year = datetime.datetime.now().year
    past_year = current_year - 5
    fetch_num = max(20, max_results * 4) 

    params = {
      "engine": "google_scholar", 
      "q": query,                 
      "api_key": serpapi_key,
      "num": fetch_num,           
      "as_ylo": past_year,        
      "as_yhi": current_year      
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        organic_results = results.get("organic_results", [])
        if not organic_results:
            print("    -> [SerpApi] 未检索到相关文献。")
            return
            
        articles_pool = []
        for pub in organic_results:
            title = pub.get("title", "Unknown")
            pub_info = pub.get("publication_info", {}).get("summary", "Unknown")
            abstract = pub.get("snippet", "No abstract available")
            
            cites = 0
            inline_links = pub.get("inline_links", {})
            cited_by = inline_links.get("cited_by", {})
            if cited_by:
                cites_raw = str(cited_by.get("total", "0"))
                cites_match = re.search(r'\d+', cites_raw)
                if cites_match:
                    cites = int(cites_match.group())
            
            pdf_url = "无"
            resources = pub.get("resources", [])
            for res in resources:
                if res.get("file_format") == "PDF":
                    pdf_url = res.get("link", "无")
                    break
            
            # 【机制 1：替身猎手】
            if pdf_url == "无" or "researchgate" in pdf_url.lower():
                pdf_url = hunt_for_preprint_pdf(title)
                    
            articles_pool.append({
                "title": title, "pub_info": pub_info, "cites": cites,
                "pdf_url": pdf_url, "abstract": abstract, "original_link": pub.get("link", "无")
            })
            
        articles_pool.sort(key=lambda x: x["cites"], reverse=True)
        top_articles = articles_pool[:max_results]
        
        saved_count = 0
        for article in top_articles:
            title = article["title"]
            cites = article["cites"]
            pdf_url = article["pdf_url"]
            original_link = article["original_link"]
            
            file_content = f"【文献归属领域】: {topic_name}\n【文献标题】: {title}\n【出版信息】: {article['pub_info']}\n【引用次数】: {cites}\n【PDF 直链】: {pdf_url}\n\n【摘要】: {article['abstract']}\n"
            
            safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:30]
            txt_path = output_dir / f"Scholar_cites_{cites}_{safe_title}.txt"
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(file_content)
                
            # 🚨 新增：【拦截机制 1.5】如果替身猎手返回的是 PMC XML 链接，直接榨取全文并放行！
            if "fullTextXML" in pdf_url:
                print(f"       [{saved_count+1}/{max_results}] 正在通过替身猎手抓取 PMC 全文...")
                try:
                    req_xml = urllib.request.Request(pdf_url)
                    with urllib.request.urlopen(req_xml, timeout=20) as resp:
                        root = ET.fromstring(resp.read())
                        body_elem = root.find(".//body")
                        full_text = " ".join("".join(body_elem.itertext()).split()) if body_elem is not None else ""
                        
                        if full_text:
                            with open(txt_path, "a", encoding="utf-8") as f_txt:
                                f_txt.write("\n=========================================\n")
                                f_txt.write("【替身猎手 PMC XML 全文无感提取】:\n")
                                f_txt.write(full_text[:25000])
                                f_txt.write("\n=========================================\n")
                            print("       ✅ 成功免下载提取替身全文！跳过人工介入。")
                            saved_count += 1
                            continue  # 核心：直接处理下一篇文章，不往下走 PDF 流程了！
                        else:
                            print("       ⚠️ 替身 XML 无正文(可能仅摘要)。转入后备流程...")
                except Exception as e:
                    print(f"       ⚠️ 替身 XML 拉取失败 ({e})。转入后备流程...")

            # 【机制 2 & 3：常规 PDF 强制下载与人机交互闭环】
            pdf_path = output_dir / f"Scholar_cites_{cites}_{safe_title}.pdf"
            print(f"       [{saved_count+1}/{max_results}] 尝试获取 SerpApi PDF: {safe_title[:15]}... ", end="", flush=True)
            
            download_success = False
            # 如果有真正的 PDF 链接，尝试自动下载
            if pdf_url != "无" and pdf_url.endswith(".pdf") and download_direct_pdf(pdf_url, pdf_path):
                print("✅ 自动获取成功！")
                download_success = True
            else:
                print("⚠️ 无直链或自动下载失败")
                # 就算替身猎手返回无，也要把网页链接或者谷歌学术搜索页发给用户，触发拦截
                fallback_url = pdf_url if pdf_url != "无" else original_link
                if fallback_url == "无": 
                    fallback_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"
                download_success = interactive_paper_download(title, fallback_url, pdf_path)
            
            if download_success and pdf_path.exists():
                print("       正在提取正文...")
                try:
                    with open(pdf_path, "rb") as f_pdf:
                        reader = PyPDF2.PdfReader(f_pdf)
                        extracted_text = ""
                        for i, page in enumerate(reader.pages):
                            if i >= 15: break
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text += page_text + "\n"
                    
                    with open(txt_path, "a", encoding="utf-8") as f_txt:
                        f_txt.write("\n=========================================\n")
                        f_txt.write("【SerpApi PDF 全文/核心正文提取】:\n")
                        clean_text = " ".join(extracted_text.split())
                        f_txt.write(clean_text[:25000]) 
                        f_txt.write("\n=========================================\n")
                except Exception as e:
                    print(f"       ⚠️ PDF 文本提取失败 (可能是扫描版或加密): {e}")
            time.sleep(1) 
            
            saved_count += 1
            
        print(f"    -> [大丰收] SerpApi 引擎成功处理 {saved_count} 篇【近五年最高被引】文献！")
    except Exception as e:
        print(f"!!! [SerpApi 检索失败]: {e}")

# ==========================================
# 本地文献阅读器
# ==========================================
def extract_text_from_papers(directory: Path, max_chars_per_file: int = 8000) -> str:
    if not directory.exists(): return ""
        
    combined_text = []
    valid_extensions = {".txt", ".pdf"}
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    clean_content = " ".join(content.split())[:max_chars_per_file]
                    combined_text.append(f"--- 📄 文献: {file_path.name} ---\n{clean_content}\n")
                    
            elif file_path.suffix.lower() == ".pdf":
                import PyPDF2
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n"
                        if len(content) > max_chars_per_file: break
                    clean_content = " ".join(content.split())[:max_chars_per_file]
                    combined_text.append(f"--- 📄 PDF文献: {file_path.name} ---\n{clean_content}\n")
        except Exception as e:
            pass # 屏蔽损坏的PDF报错
            
    return "\n".join(combined_text)

# ==========================================
# 主力执行函数 
# ==========================================
def generate_benchmark_strategy():
    print("\n========== [Phase 1.5] 首席科学家：融合多源异构数据制定 Benchmark 规范 ==========")
    
    current_year = datetime.datetime.now().year
    past_year = current_year - 5
    
    dir_model_papers = Path("data/papers")
    dir_benchmark_papers = Path("data/benchmark_papers")

    # 1. 基础生信与AI同义词
    amp_synonyms = '("antimicrobial peptide" OR "host defense peptide" OR "antibacterial peptide" OR "AMPs")'
    ai_synonyms = '("deep learning" OR "machine learning" OR "neural network" OR "transformer" OR "graph neural network" OR "language model")'
    # 🚨 新增：强力排他词库（直接在搜索引擎层面拦截跨界文章）
    exclude_terms = '(NOT "anticancer" NOT "ACP" NOT "ACPs" NOT "anti-inflammatory" NOT "AIP" NOT "AIPs" NOT "cell-penetrating" NOT "CPP" NOT "CPPs" NOT "generation" NOT "design")'
    # 2. 任务域拆分
    model_task_synonyms = '("prediction" OR "discovery" OR "identification" OR "classifier" OR "screening")'
    benchmark_task_synonyms = '("benchmark" OR "dataset" OR "gold standard" OR "comprehensive assessment")'

    # 🚨 核心修改：新增【开源特征】强制过滤网！
    # 任何文献只要不包含这些词中的任何一个，搜索引擎连下都不用下载！
    open_source_terms = '("github" OR "repository" OR "source code" OR "web server" OR "zenodo" OR "database")'

    # 🚨 PMC 引擎：模型与数据集都必须加上开源要求
    pmc_model_query = f'(TITLE_ABS:{amp_synonyms} AND {ai_synonyms} AND {model_task_synonyms} AND {open_source_terms} {exclude_terms}) AND OPEN_ACCESS:y AND PUB_YEAR:[{past_year} TO {current_year}]'
    pmc_benchmark_query = f'(TITLE_ABS:{amp_synonyms} AND {ai_synonyms} AND {benchmark_task_synonyms} AND {open_source_terms} {exclude_terms}) AND OPEN_ACCESS:y AND PUB_YEAR:[{past_year} TO {current_year}]'

    # 🚨 Semantic Scholar (S2) 引擎：同样加上开源要求
    s2_model_query = f"{amp_synonyms} AND {ai_synonyms} AND {model_task_synonyms} AND {open_source_terms}"
    s2_benchmark_query = f"{amp_synonyms} AND {ai_synonyms} AND {benchmark_task_synonyms} AND {open_source_terms} {exclude_terms}"
    
    # 🚨 arXiv 和 OpenAlex：不能太宽泛，强行加上开源词汇进行收束
    arxiv_expanded_query = f"%22antimicrobial+peptide%22+AND+%22deep+learning%22+AND+github" 
    openalex_expanded_query = f"{amp_synonyms} AND {ai_synonyms} AND {open_source_terms} {exclude_terms}"
    
    # 🚨 SerpApi 商业引擎：加上 github 和 database 关键词进行精准打击
    serpapi_benchmark_query = f"{amp_synonyms} AND {ai_synonyms} AND (benchmark OR dataset) AND (github OR database OR zenodo)"
    serpapi_model_query = f"{amp_synonyms} AND {ai_synonyms} AND prediction AND github"

    # 执行检索与下载管线
    fetch_and_save_pmc_papers("AMP 预测模型(PMC)", pmc_model_query, dir_model_papers, max_results=20)
    fetch_from_semantic_scholar("AMP 预测模型(S2)", s2_model_query, dir_model_papers, max_results=5)
    fetch_and_save_pmc_papers("AMP 金标准数据集(PMC)", pmc_benchmark_query, dir_benchmark_papers, max_results=5)
    fetch_from_arxiv("AMP 最新预印本模型", arxiv_expanded_query, dir_model_papers, max_results=5)
    fetch_from_openalex("AMP 预测模型(兜底)", openalex_expanded_query, dir_model_papers, max_results=10)
    fetch_from_serpapi_scholar("AMP 预测模型(SerpApi)", serpapi_model_query, dir_model_papers, max_results=5)

    snowball_from_seed_paper("Comprehensive assessment of machine learning-based methods for predicting antimicrobial peptides", dir_model_papers, max_results=5)

    print("\n>>> 正在全局扫描本地文献库...")
    model_context = extract_text_from_papers(dir_model_papers)
    benchmark_context = extract_text_from_papers(dir_benchmark_papers)

    full_context = (
        "📚【领域一：AMP 预测模型核心文献 (用于理解模型参数与开源代码要求)】\n"
        + model_context
        + "\n\n📚【领域二：AMP 基准测试与评测专著 (用于提取金标准链接与打分指标)】\n"
        + benchmark_context
    )
    
    if len(full_context) > 100000:
        print(f"    -> [警告] 知识库过大，正在进行安全压缩...")
        full_context = full_context[:100000]


if __name__ == "__main__":
    generate_benchmark_strategy()