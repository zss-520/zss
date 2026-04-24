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

def ingest_new_paper(paper_text: str, filename: str = "Unknown_File") -> bool: 
    parsed_data = extract_model_info_from_text(paper_text)
    
    if isinstance(parsed_data, list):
        models_list = parsed_data
        paper_title = filename 
    elif isinstance(parsed_data, dict) and "models" in parsed_data:
        models_list = parsed_data["models"]
        extracted_title = parsed_data.get("paper_title", "Unknown_Paper")
        paper_title = extracted_title if extracted_title != "Unknown_Paper" else filename
    else:
        print("!!! [Error] 解析失败，格式无法识别。")
        return False
        
    db = load_db()
    
    if paper_title not in [p.get("paper_title") for p in db.get("papers", [])]:
        db.setdefault("papers", []).append({"paper_title": paper_title})
        
    # 🚨 核心修改 1：移除 existing_models 集合，增加 updated_count 计数器
    models_in_db = db.setdefault("models", [])
    new_models_added = 0
    models_updated = 0
    
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
            import re
            clean_repo_url = repo_url.strip().replace(" ", "").replace("\n", "")
            zenodo_match = re.match(r"(https?://zenodo\.org/records/)(\d{7,8})", clean_repo_url)
            if zenodo_match:
                clean_repo_url = zenodo_match.group(1) + zenodo_match.group(2) 
    
            clean_raw_text = paper_text.replace(" ", "").replace("\n", "")
    
            if clean_repo_url not in clean_raw_text:
                print(f"    !!! [幻觉拦截] 虚构链接: {repo_url}，原文中并不存在！")
            else:
                print(f"    >>> [OK] 链接验证通过: {clean_repo_url}")
                model["repo_url"] = clean_repo_url 
                is_valid_repo = True
        
        if not is_valid_repo:
            print(f"    !!! [丢弃] 模型 '{model.get('model_name')}' 缺失真实有效的开源链接，拒绝分配环境并丢弃该模型！")
            continue  
        # ==========================================

        # 🚨 核心修改 2：覆盖更新 (Upsert) 逻辑
        model_name = model.get("model_name")
        # 查找数据库中是否已经存在同名模型
        existing_index = next((i for i, m in enumerate(models_in_db) if m.get("model_name") == model_name), None)
        
        if existing_index is not None:
            # 如果存在，用最新的信息覆盖更新它
            models_in_db[existing_index] = model
            print(f"    - [🔄 已更新] 记忆库中已有该模型，已使用最新配置覆盖: {model_name}")
            models_updated += 1
        else:
            # 如果不存在，直接追加
            models_in_db.append(model)
            print(f"    - [✅ 已新增] 成功将新模型存入记忆库: {model_name} (预期环境: {model.get('env_name', '未知')})")
            new_models_added += 1
            
    # 保存数据库
    save_db(db)
    
    # 🚨 核心修改 3：修复误导人的日志打印
    if new_models_added > 0 or models_updated > 0:
        print(f">>> [Knowledge Base] 数据库同步完成！(新增: {new_models_added} 个, 更新: {models_updated} 个)")
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
def add_models_to_knowledge_db(new_models: list) -> None:
    """将刚发现的新模型（还未复现）加入模型知识库"""
    db = load_db()
    if "models" not in db:
        db["models"] = []
    
    # 获取已存在的模型名称，防止重复录入
    existing_names = {m.get("model_name") for m in db["models"]}
    added_count = 0
    
    for model in new_models:
        name = model.get("model_name")
        if name and name not in existing_names:
            db["models"].append(model)
            added_count += 1
            
    save_db(db)
    if added_count > 0:
        print(f"    -> 🧠 成功将 {added_count} 个新模型情报录入 data/model_knowledge_db.json (待复现队列)。")
    else:
        print("    -> 🧠 提取的模型均已存在于知识库中，无需重复录入。")
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