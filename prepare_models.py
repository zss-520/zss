import json
import os
import re
from pathlib import Path
import PyPDF2

from database_manager import ingest_new_paper, get_target_models_for_eval, is_paper_processed, mark_paper_processed
from vanguard import run_vanguard_exploration

def main():
    print("========== [Ingestion Phase 1] 扫描文献并提取模型元数据 ==========")
    papers_dir = Path("data/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)
    valid_extensions = {".pdf", ".txt"}
    paper_files = [f for f in papers_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    for paper_file in paper_files:
        filename = paper_file.name
        if is_paper_processed(filename):
            print(f">>> [跳过] 文献 '{filename}' 已处理。")
            continue
            
        print(f"\n>>> [新文献] 正在读取: {filename} ...")
        raw_text = ""
        if paper_file.suffix.lower() == ".pdf":
            try:
                with open(paper_file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            raw_text += page_text + "\n"
            except Exception as e:
                print(f"!!! [Error] 读取 PDF 失败: {e}")
                continue
        else:
            with open(paper_file, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
                
        match = re.search(r'\n\s*(references|bibliography|literature cited)\s*\n', raw_text, re.IGNORECASE)
        if match:
            raw_text = raw_text[:match.start()]
            
        if ingest_new_paper(raw_text):
            mark_paper_processed(filename)
            
    models_info = get_target_models_for_eval()

    if not models_info:
        print(">>> 没有提取到需要入库的新模型，流程结束。")
        return

    print(f"\n>>> 准备去超算拉取源码的模型总数: {len(models_info)}")

    print("\n========== [Ingestion Phase 2] 超算物理拉取与勘探 (先遣队) ==========")
    save_directory = Path("data/vlab_discussions")
    save_directory.mkdir(parents=True, exist_ok=True)

    # 找一个本地已有的数据集目录给 vanguard 占位传过去
    base_datasets_dir = Path("data/datasets")
    dataset_dirs = [d for d in base_datasets_dir.iterdir() if d.is_dir()]
    if not dataset_dirs:
        print("!!! [Fatal] 请先在 data/datasets 下准备至少一个测试数据集文件夹！")
        return
    dummy_dataset_dir = str(dataset_dirs[0])
    
    # 执行下载和扫描，返回的 models_info 里将包含极其珍贵的 'repo_structure'！
    models_info = run_vanguard_exploration(
        models_info=models_info,
        sample_dataset_dir=dummy_dataset_dir,
        save_directory=save_directory
    )

    print("\n========== [Ingestion Phase 3] 写入静态注册表 ==========")
    registry_path = "data/local_registry.json"
    
    # 如果已有注册表，则读取并合并，保证之前配好的模型不丢失
    existing_registry = []
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                existing_registry = json.load(f)
        except:
            pass
            
    existing_dict = {m['model_name']: m for m in existing_registry}
    for m in models_info:
        existing_dict[m['model_name']] = m
        
    final_registry = list(existing_dict.values())
    
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(final_registry, f, ensure_ascii=False, indent=4)
        
    print(f"\n✨ >>> [Done] 完美！模型源码已下载，真实物理目录树已挂载！")
    print(f"✨ >>> 静态注册表已更新至: {registry_path}")
    print("✨ >>> 你现在可以去修改 main.py 的 target_model_names 然后跑主程序了！")

if __name__ == "__main__":
    main()