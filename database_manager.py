import json
from pathlib import Path
from openai import OpenAI

from agent import Agent
from config import MODEL_NAME
from prompts import PAPER_ANALYST_PROMPT

# 定义记忆库的默认存储路径
DB_PATH = Path("data/model_knowledge_db.json")

def load_db() -> dict:
    """加载本地模型记忆库。如果不存在则初始化空库。"""
    if not DB_PATH.exists():
        return {"papers": [], "models": []}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db_data: dict) -> None:
    """将数据安全地覆写到本地记忆库。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db_data, f, ensure_ascii=False, indent=2)

def extract_model_info_from_text(paper_text: str) -> dict:
    """调用 Paper Analyst 从无结构文本中提取文献与模型元数据"""
    analyst_agent = Agent(
        title="Paper Analyst",
        expertise="Bioinformatics, MLOps, text mining",
        goal="Extract structured model execution metadata from paper text.",
        role="Data Extractor",
        model=MODEL_NAME
    )
    
    client = OpenAI()
    
    try:
        # 使用 response_format 强制 OpenAI API 返回标准 JSON
        response = client.chat.completions.create(
            model=analyst_agent.model,
            messages=[
                {"role": "system", "content": PAPER_ANALYST_PROMPT},
                {"role": "user", "content": f"请解析以下文献或仓库 README 文本，严格输出要求的 JSON 格式：\n\n{paper_text}"}
            ],
            temperature=0.1,  # 保持极低的温度以保证格式稳定性
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content or "{}"
        print(f"    >>> [Debug] AI 原始提取结果:\n{content}")
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        print(f"!!! [Error] 解析失败，Paper Analyst 返回的不是合法的 JSON: {e}")
        return {}
    except Exception as e:
        print(f"!!! [Error] 调用 API 提取文献信息时发生错误: {e}")
        return {}

def ingest_new_paper(paper_text: str, filename: str = "Unknown_File") -> bool: # <--- 增加 filename 参数
    parsed_data = extract_model_info_from_text(paper_text)
    
    if isinstance(parsed_data, list):
        models_list = parsed_data
        # 🚨 修复点 1：如果 AI 返回列表，直接用文件名作为标题
        paper_title = filename 
    elif isinstance(parsed_data, dict) and "models" in parsed_data:
        models_list = parsed_data["models"]
        # 🚨 修复点 2：如果 AI 解析出的标题是 Unknown，则用文件名兜底
        extracted_title = parsed_data.get("paper_title", "Unknown_Paper")
        paper_title = extracted_title if extracted_title != "Unknown_Paper" else filename
    else:
        print("!!! [Error] 解析失败，格式无法识别。")
        return False
        
    db = load_db()
    
    if paper_title not in [p.get("paper_title") for p in db.get("papers", [])]:
        db.setdefault("papers", []).append({"paper_title": paper_title})
        
    existing_models = {m["model_name"] for m in db.get("models", [])}
    new_models_added = 0
    
    # 👇 注意这里改成遍历 models_list
    for model in models_list:
        model["source_paper"] = paper_title
        model["skip_env_setup"] = False
        if not model.get("inference_cmd_template"):
            model["inference_cmd_template"] = "python predict.py --input {fasta_path} --out {output_dir}"
            
        # ==========================================
        # 终极拦截器：无真实代码，坚决不建环境！
        # ==========================================
        repo_url = model.get("repo_url", "")
        is_valid_repo = False
        
        if repo_url:
    # 1. 净化大模型提取出来的链接，并物理切除可能残留的参考文献上标（如果末尾是 62、45 之类的两位数）
            import re
            clean_repo_url = repo_url.strip().replace(" ", "").replace("\n", "")
            zenodo_match = re.match(r"(https?://zenodo\.org/records/)(\d{7,8})", clean_repo_url)
            if zenodo_match:
                clean_repo_url = zenodo_match.group(1) + zenodo_match.group(2) # 强行只保留到 7~8 位正确 ID
    
    # 2. 净化 PDF 原始长文本（消灭所有的空格和换行符带来的排版干扰）
    # 注意：这里的 pdf_text 变量名请替换成你实际的原文变量名（比如 raw_text）
            clean_raw_text = paper_text.replace(" ", "").replace("\n", "")
    
    # 3. 进行无视排版的安全比对！
            if clean_repo_url not in clean_raw_text:
                print(f"    !!! [幻觉拦截] 虚构链接: {repo_url}，原文中并不存在！")
        # 下面保留你原来的 return False 或丢弃模型的逻辑...
            else:
                print(f"    >>> [OK] 链接验证通过: {clean_repo_url}")
                # 【极其关键的修复 1】：把洗干净的完美链接存回字典，覆盖掉带空格的脏链接！
                model["repo_url"] = clean_repo_url 
                
                # 【极其关键的修复 2】：给它发通行证！告诉下面的拦截器它是合法公民！
                is_valid_repo = True
        
        # 如果不是有效的代码链接，直接抛弃这个模型，绝不存入记忆库！
        if not is_valid_repo:
            print(f"    !!! [丢弃] 模型 '{model.get('model_name')}' 缺失真实有效的开源链接，拒绝分配环境并丢弃该模型！")
            continue  # 核心逻辑：直接跳出当前循环，后面的入库代码不执行！
        # ==========================================

        if model["model_name"] not in existing_models:
            db.setdefault("models", []).append(model)
            new_models_added += 1
            print(f"    - [新发现] 模型入库: {model['model_name']} (预期环境: {model.get('env_name')})")
        else:
            print(f"    - [已跳过] 模型已存在于记忆库中: {model['model_name']}")
            
    # 保存数据库。如果所有模型都被丢弃了，db里的 models 不会增加，但我们会把文献标记为“已读”，防止以后每次都重复去读这篇没有代码的废文章。
    save_db(db)
    
    if new_models_added > 0:
        print(f">>> [Knowledge Base] 本次新增 {new_models_added} 个模型的复现记忆。")
    else:
        print(f">>> [Knowledge Base] 该文献未提供有效开源代码，不产生任何复现任务。")
        
    return True

def get_target_models_for_eval(model_names: list[str] = None) -> list[dict]:
    """
    供主工作流调用的接口：获取记忆库中的模型配置。
    如果不传参数，默认返回所有已记忆的模型。
    """
    db = load_db()
    all_models = db.get("models", [])
    
    if not model_names:
        return all_models
        
    # 过滤出指定的模型
    target_models = [m for m in all_models if m["model_name"] in model_names]
    return target_models
def is_paper_processed(filename: str) -> bool:
    """检查该文献文件是否已经被成功解析过"""
    db = load_db()
    return filename in db.get("processed_papers", [])

def mark_paper_processed(filename: str) -> None:
    """将该文献文件标记为已处理"""
    db = load_db()
    if filename not in db.setdefault("processed_papers", []):
        db["processed_papers"].append(filename)
        save_db(db)
# ==========================================
# 独立测试入口
# ==========================================
if __name__ == "__main__":
    # 这是一个用于测试的假想文献片段
    sample_paper_text = """
    We present DeepAMP-Gen, a novel deep learning architecture for antimicrobial peptide prediction. 
    The source code is available at https://github.com/example/DeepAMP-Gen.
    To run the model, you need Python 3.9 and PyTorch 2.0. 
    Users can predict sequences using the following command:
    python run_inference.py --fasta_file input.fa --output_path results/
    """
    
    print("\n========== [测试] 文献提取与入库 ==========")
    ingest_new_paper(sample_paper_text)
    
    print("\n========== [测试] 打印当前记忆库状态 ==========")
    print(json.dumps(load_db(), ensure_ascii=False, indent=2))