import json
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import datetime
import time
from pathlib import Path
from typing import Any, Dict, List

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
# 工具函数：通用直链 PDF 下载器 (绕过简单防盗链)
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


# ==========================================
# 引擎 A：Semantic Scholar (主攻 AI 模型挖掘)
# ==========================================
def fetch_from_semantic_scholar(topic_name: str, query: str, output_dir: Path, max_results: int = 10):
    """使用 Semantic Scholar API 抓取最新 AI 模型文献与开源代码库（带防限流重试）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [S2 引擎] 正在启动 Semantic Scholar 检索【{topic_name}】...")
    
    encoded_query = urllib.parse.quote(query)
    fields = "title,abstract,year,citationCount,openAccessPdf"
    s2_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={max_results}&fields={fields}"
    
    # 🚨 核心升级：指数退避重试机制
    max_retries = 3
    data = None
    
    for attempt in range(max_retries):
        try:
            # 伪装得更像一点真实的科研脚本
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) MLOps-Pipeline/1.0'}
            req = urllib.request.Request(s2_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.loads(response.read().decode('utf-8'))
                break  # 如果成功拿到数据，立刻打破循环！
                
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = 2 ** attempt  # 1秒, 2秒, 4秒...
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
    if not papers:
        print("    -> [S2] 警告：未检索到相关文献。")
        return
        
    print(f"    -> [S2] 成功突破！锁定 {len(papers)} 篇前沿文献，正在提取数据与 PDF...")
    
    saved_count = 0
    for p in papers:
        title = p.get("title") or "Unknown"
        abstract = p.get("abstract") or "No abstract available."
        year = p.get("year") or "Unknown"
        cites = p.get("citationCount") or 0
        
        oa_info = p.get("openAccessPdf")
        pdf_url = oa_info.get("url") if oa_info else "无"
        
        github_links = re.findall(r"https://github\.com/[^\s\)]+", abstract)
        github_str = "\n    - ".join(github_links) if github_links else "无明显的 GitHub 链接"
        
        file_content = f"【文献归属领域】: {topic_name}\n"
        file_content += f"【文献标题】: {title}\n"
        file_content += f"【发表年份】: {year}\n"
        file_content += f"【S2 引用次数】: {cites}\n"
        file_content += f"【PDF 直链】: {pdf_url}\n\n"
        file_content += f"【摘要】: {abstract}\n\n"
        file_content += f"【可能的数据集或开源代码链接提取】:\n    - {github_str}\n"
        
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:30]
        base_filename = f"S2_cites_{cites}_{safe_title}"
        
        txt_path = output_dir / f"{base_filename}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(file_content)
            
        if pdf_url != "无" and pdf_url.endswith(".pdf"):
            pdf_path = output_dir / f"{base_filename}.pdf"
            print(f"       [{saved_count+1}/{len(papers)}] 正在获取 PDF: {pdf_url.split('/')[-1][:20]}... ", end="", flush=True)
            if download_direct_pdf(pdf_url, pdf_path):
                print("✅ 成功存入本地！")
            else:
                print("⚠️ 下载失败 (可能存在防盗链)")
            time.sleep(1.5) # 给每次下载加上喘息时间

        saved_count += 1
        
    print(f"    -> [大丰收] S2 引擎成功处理 {saved_count} 篇【{topic_name}】文献记录！")


# ==========================================
# 引擎 B：Europe PMC (主攻生物医学金标准与测试集)
# ==========================================
def download_pmc_pdf_via_oa(pmcid: str, output_path: Path) -> bool:
    """调用 NCBI 官方 OA API 获取无防火墙的底层直链"""
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
    
    encoded_query = urllib.parse.quote(epmc_query)
    epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded_query}&format=json&resultType=lite&sort=CITED%20desc&pageSize={max_results}"
    
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
        print(f"    -> 成功锁定 {len(pmc_ids)} 篇高被引文献，正在拉取 XML 并尝试下载 PDF原文...")
        
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
                    if 20 < len(p_text) < 1500:
                        data_links.append(p_text)
                        
            data_links = list(set(data_links))[:5]
            data_context = "\n    - ".join(data_links) if data_links else "无明显的外部链接或数据集库。"
            
            file_content = f"【文献归属领域】: {topic_name}\n"
            file_content += f"【文献标题】: {title}\n"
            file_content += f"【发表年份】: {year}\n"
            file_content += f"【引用次数】: {cited_count}\n"
            file_content += f"【PMC ID】: {target_pmcid}\n"
            file_content += f"【摘要】: {abstract}\n\n"
            file_content += f"【可能的数据集或开源代码链接提取】:\n    - {data_context}\n"

            # 🚨 终极兜底：强行抓取并追加最多 3 万字的正文 Methods 内容
            body_elem = article.find(".//body")
            if body_elem is not None:
                full_text_body = " ".join(body_elem.itertext())
                file_content += f"\n【全文 Methods 段落提取】:\n{full_text_body[:30000]}\n" 
            else:
                file_content += f"\n【全文 Methods 段落提取】:\n(未获取到 XML 正文，可能该文献为仅摘要版本)\n"

            safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)[:30]
            base_filename = f"PMC_cites_{cited_count}_{safe_title}"
            
            txt_path = output_dir / f"{base_filename}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(file_content)
                
            if target_pmcid:
                pdf_path = output_dir / f"{base_filename}.pdf"
                print(f"       [{saved_count+1}/{len(pmc_ids)}] 正在窃取 PDF: {target_pmcid} ... ", end="", flush=True)
                if download_pmc_pdf_via_oa(target_pmcid, pdf_path):
                    print("✅ 成功存入本地！")
                else:
                    print("⚠️ 无法获取开源 PDF (受限或需购买)，但已提取全文 Methods 入 TXT")
                time.sleep(1)

            saved_count += 1
            
        print(f"    -> [大丰收] PMC 引擎成功提取并处理 {saved_count} 篇【{topic_name}】文献记录！")
        
    except Exception as e:
        print(f"!!! [PMC Search] 全文检索发生异常: {e}")


# ==========================================
# 本地文献阅读器 (双重读取)
# ==========================================
def extract_text_from_papers(papers_dir: Path, max_chars_per_paper: int = 15000) -> str:
    valid_extensions = {".pdf", ".txt"}
    if not papers_dir.exists(): return "目录不存在或暂无本地文件。"

    paper_files = [f for f in papers_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    if not paper_files: return "无相关文献。"

    combined_text = ""
    for paper_file in paper_files:
        raw_text = ""
        if paper_file.suffix.lower() == ".pdf":
            try:
                with open(paper_file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        if i > 5: break
                        page_text = page.extract_text()
                        if page_text: raw_text += page_text + "\n"
            except Exception as e:
                print(f"!!! [Local Paper] 读取 PDF 失败: {paper_file.name} -> {e}")
                continue
        else:
            with open(paper_file, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

        clean_text = re.sub(r"\s+", " ", raw_text).strip()
        combined_text += f"\n\n--- 文件源: {paper_file.name} ---\n{clean_text[:max_chars_per_paper]}"
    return combined_text

def build_architect_user_prompt(full_context: str) -> str:
    return f"""请严格审查以下分类后的文献资料，并为我们的 binary AMP classification 评测管线产出最具权威性的 test-only benchmark strategy。

额外要求：
1. 你必须综合【预测模型文献】中的数据需求和【基准评测文献】中的金标准来源，来制定策略。
2. 你必须优先推荐包含真实 URL (如 GitHub) 的可以直接下载的数据集。
3. 必须在 description 中说明数据集的正负样本是如何构造的，以及使用了什么指标(如 MCC, AUPRC)评估。

以下是双线检索融合后的全部可用上下文：

{full_context}
"""

# ==========================================
# 主力执行函数
# ==========================================
def generate_benchmark_strategy():
    print("\n========== [Phase 1.5] 首席科学家：融合多源异构数据制定 Benchmark 规范 ==========")
    
    current_year = datetime.datetime.now().year
    past_year = current_year - 5
    
    dir_model_papers = Path("data/papers")
    dir_benchmark_papers = Path("data/benchmark_papers")
    
    # -------------------------------------------------------------
    # 策略 1：使用 Semantic Scholar 挖掘最新的 AI 预测模型和 GitHub 代码库
    # -------------------------------------------------------------
    s2_query = "antimicrobial peptide deep learning machine learning prediction"
    fetch_from_semantic_scholar("AMP 预测模型", s2_query, dir_model_papers, max_results=10)
    
    # -------------------------------------------------------------
    # 策略 2：使用 Europe PMC 深挖有医学实验背书的金标准基准数据集
    # -------------------------------------------------------------
    benchmark_query = f'((TITLE:"antimicrobial peptide" OR ABSTRACT:"antimicrobial peptide") OR (TITLE:"antimicrobial peptides" OR ABSTRACT:"antimicrobial peptides")) AND (TITLE:"benchmark" OR ABSTRACT:"benchmark" OR TITLE:"dataset" OR ABSTRACT:"dataset" OR TITLE:"gold standard" OR ABSTRACT:"gold standard") AND OPEN_ACCESS:y AND PUB_YEAR:[{past_year} TO {current_year}]'
    fetch_and_save_pmc_papers("AMP 金标准数据集", benchmark_query, dir_benchmark_papers, max_results=10)

    print("\n>>> 正在全局扫描本地文献库并将海量知识喂给 AI 架构师...")
    model_context = extract_text_from_papers(dir_model_papers)
    benchmark_context = extract_text_from_papers(dir_benchmark_papers)

    full_context = (
        "📚【领域一：AMP 预测模型核心文献 (用于理解模型参数与开源代码要求)】\n"
        + model_context
        + "\n\n📚【领域二：AMP 基准测试与评测专著 (用于提取金标准链接与打分指标)】\n"
        + benchmark_context
    )

    architect_agent = Agent(
        title="Benchmark Architect",
        expertise="Computational Biology, Metrics Engineering, Peer Review",
        goal="Design the ultimate test-only evaluation benchmark strictly based on high-impact literature.",
        role="Chief Scientist",
        model=MODEL_NAME,
    )

    client = OpenAI()
    print(">>> AI 正在执行双语料库交叉比对与金标准制定 (由于文献激增，可能需要较长时间)...")

    try:
        response = client.chat.completions.create(
            model=architect_agent.model,
            messages=[
                {"role": "system", "content": BENCHMARK_ARCHITECT_PROMPT},
                {"role": "user", "content": build_architect_user_prompt(full_context)},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content or "{}"
        strategy_data = json.loads(raw_content)
        
        strategy_data = validate_strategy(strategy_data)

        out_path = Path("data/benchmark_strategy.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(strategy_data, f, ensure_ascii=False, indent=4)

        print(f"\n✨ >>> [Success] 测试集策略已生成并保存至: {out_path}")

    except Exception as e:
        print(f"!!! [Error] 生成基准测试策略失败: {e}")

if __name__ == "__main__":
    generate_benchmark_strategy()