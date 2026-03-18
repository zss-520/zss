import os
import subprocess
import shutil
import glob
import gzip
import sys

# ================= 配置区域 =================
DATA_DIR = "data"
OBSERVATION_LOG = os.path.join(DATA_DIR, "stage1_observation.txt")

# 模型执行配置清单
MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out"',
        "output_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "output_dir": "data/AMP-Scanner-v2_out"
    }
]
# ===========================================

def ensure_data_dir():
    """确保数据目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

def log_message(message, mode='a'):
    """将消息写入观察日志"""
    ensure_data_dir()
    try:
        with open(OBSERVATION_LOG, mode, encoding='utf-8') as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[CRITICAL] Failed to write log: {e}")

def read_file_head(file_path, lines=10):
    """
    读取文件头部内容，支持 .gz 压缩文件
    严格遵守 Stage 1 约束：不使用 pandas，直接读取文本
    """
    content_lines = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_lines.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_lines.append(line.strip())
    except Exception as e:
        return [f"Error reading file: {str(e)}"]
    return content_lines

def execute_model(model_config):
    """
    执行单个模型，包含智能自愈机制 (Self-Healing)
    严格遵守防御性编程规范：绝对隔离，永不连坐
    """
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["output_dir"]
    
    print(f"[INFO] Starting execution of model: {name}")
    log_message(f"--- Model Execution: {name} ---")
    
    try:
        # 第一步：清理 (Clean)
        # 即使目录不存在，ignore_errors=True 也不会报错
        shutil.rmtree(out_dir, ignore_errors=True)
        
        # 第二步：执行前兜底 (Pre-mkdir)
        # 根据规范，执行前必须建好房子
        os.makedirs(out_dir, exist_ok=True)
        
        # 第三步：盲测 (Blind Test)
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 第四步：动态自愈 (Self-Healing)
        if res.returncode != 0:
            # 联合检查 stdout 和 stderr
            combined_output = res.stdout + res.stderr
            print(f"[WARN] {name} initial execution failed. Analyzing error...")
            log_message(f"[ERROR] Initial execution failed for {name}. Returncode: {res.returncode}")
            log_message(f"[DEBUG] Stdout/Stderr snippet: {combined_output[:500]}")
            
            retry_success = False
            
            # 情况 A: 缺少目录 (No such file or directory / Failed to save)
            if "No such file or directory" in combined_output or "Failed to save" in combined_output:
                print(f"[HEALING] {name} detected missing directory error. Attempting to recreate directory...")
                log_message(f"[HEALING] Detected 'No such file' error. Recreating {out_dir}...")
                try:
                    shutil.rmtree(out_dir, ignore_errors=True) # 先清理确保干净
                    os.makedirs(out_dir, exist_ok=True)
                    # 重试
                    res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if res_retry.returncode == 0:
                        retry_success = True
                        print(f"[SUCCESS] {name} healed successfully after mkdir retry.")
                        log_message(f"[SUCCESS] {name} healed successfully after mkdir retry.")
                except Exception as heal_e:
                    log_message(f"[ERROR] Healing process failed for {name}: {str(heal_e)}")
            
            # 情况 B: 目录已存在冲突 (already exists / exists)
            elif "already exists" in combined_output or "exists" in combined_output:
                print(f"[HEALING] {name} detected existing directory conflict. Attempting to remove and retry...")
                log_message(f"[HEALING] Detected 'exists' error. Removing {out_dir} and retrying...")
                try:
                    shutil.rmtree(out_dir, ignore_errors=True)
                    # 注意：这里不再预先 makedirs，让工具自己创建，或者重试前再建
                    # 根据规范：删掉然后再次重试执行 subprocess.run
                    os.makedirs(os.path.dirname(out_dir), exist_ok=True) 
                    
                    res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if res_retry.returncode == 0:
                        retry_success = True
                        print(f"[SUCCESS] {name} healed successfully after rmtree retry.")
                        log_message(f"[SUCCESS] {name} healed successfully after rmtree retry.")
                except Exception as heal_e:
                    log_message(f"[ERROR] Healing process failed for {name}: {str(heal_e)}")
            
            # 如果重试依然失败，记录终极日志
            if not retry_success:
                error_msg = f"[FATAL] Model {name} failed after healing attempts.\nStdout: {res.stdout}\nStderr: {res.stderr}"
                print(f"[FATAL] {name} execution failed permanently.")
                log_message(error_msg)
                # 不 raise，继续执行 (绝对隔离)
                return False
            else:
                return True
        else:
            print(f"[SUCCESS] {name} executed successfully.")
            log_message(f"[SUCCESS] {name} executed successfully.")
            return True
            
    except Exception as e:
        # 绝对隔离，捕获所有未预料的异常
        error_msg = f"[CRITICAL] Unexpected exception during {name} execution: {str(e)}"
        print(error_msg)
        log_message(error_msg)
        return False

def explore_outputs():
    """
    勘探输出文件结构并生成报告
    严格遵守 Stage 1 约束：不许 merge，不许计算指标，只记录文件结构和头部内容
    """
    print("[INFO] Starting output exploration...")
    log_message("\n--- Stage 1 Exploration Report ---\n")
    
    ensure_data_dir()
    
    # 查找所有带有 _out 后缀的目录
    out_dirs = glob.glob(os.path.join(DATA_DIR, "*_out"))
    # 同时也查找 data/ 下可能直接生成的文件
    all_files = glob.glob(os.path.join(DATA_DIR, "*"))
    
    found_files = []
    
    # 收集目录内的文件
    for d in out_dirs:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for file in files:
                    found_files.append(os.path.join(root, file))
    
    # 收集 data 目录下直接的文件 (排除目录本身)
    for f in all_files:
        if os.path.isfile(f) and f not in found_files:
            # 排除日志文件本身以免递归
            if f != OBSERVATION_LOG:
                found_files.append(f)
    
    if not found_files:
        log_message("No output files found in data/ directory.")
        print("[WARN] No output files found.")
        return

    log_message(f"Total files found: {len(found_files)}\n")
    
    for file_path in found_files:
        log_message(f"File Path: {file_path}")
        try:
            head_content = read_file_head(file_path, lines=10)
            log_message("Head Content (first 10 lines):")
            for line in head_content:
                log_message(f"  {line}")
            log_message("-" * 50)
        except Exception as e:
            log_message(f"Error reading content: {str(e)}")
            log_message("-" * 50)

def main():
    """
    主程序入口
    严格遵守代码结构规范：单一脚本，完整 import，def main
    """
    print("=" * 60)
    print("Starting AMP Model Evaluation Stage 1 (Exploration)")
    print("=" * 60)
    
    # 初始化日志
    ensure_data_dir()
    # 初始化日志文件
    with open(OBSERVATION_LOG, 'w', encoding='utf-8') as f:
        f.write("=== Stage 1 Observation Log ===\n")
        try:
            date_res = subprocess.run('date', shell=True, capture_output=True, text=True)
            f.write(f"Start Time: {date_res.stdout.strip()}\n\n")
        except:
            f.write("Start Time: Unknown\n\n")
    
    # 1. 执行模型 (严格隔离)
    for model in MODELS:
        execute_model(model)
        print("-" * 30)
    
    # 2. 勘探输出 (禁止 pandas merge/metrics)
    explore_outputs()
    
    print("=" * 60)
    print("Stage 1 Complete. Check data/stage1_observation.txt for details.")
    print("=" * 60)

if __name__ == '__main__':
    main()
