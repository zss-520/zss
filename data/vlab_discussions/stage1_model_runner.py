import os
import subprocess
import shutil
import glob
import gzip

# ==============================================================================
# 配置区域 (Configuration)
# ==============================================================================

# 确保基础数据目录存在
os.makedirs('data', exist_ok=True)

# 观察报告路径
OBSERVATION_FILE = 'data/stage1_observation.txt'

# 模型执行配置清单
# 注意：命令中的绝对路径严禁修改
MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
        "output_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "output_dir": "data/AMP-Scanner-v2_out"
    }
]

# ==============================================================================
# 工具函数 (Utility Functions)
# ==============================================================================

def log_observation(message):
    """
    将消息追加写入到观察报告文件中。
    """
    try:
        with open(OBSERVATION_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"[CRITICAL] Failed to write to observation log: {e}")

def read_file_header(file_path, lines=10):
    """
    读取文件头部内容，自动处理 .gz 压缩文件。
    """
    content_summary = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_summary.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content_summary.append(line.strip())
    except Exception as e:
        content_summary.append(f"[Error reading file: {str(e)}]")
    
    return content_summary

def execute_model_with_healing(model_config):
    """
    执行单个模型，包含智能自愈机制和异常隔离。
    """
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["output_dir"]
    
    log_observation(f"\n{'='*60}")
    log_observation(f"[START] Model: {name}")
    log_observation(f"[CMD] {cmd}")
    log_observation(f"[OUT] {out_dir}")
    print(f"[INFO] Starting {name}...")

    try:
        # --- 第一步：清理 (Clean) ---
        # 无论之前状态如何，先清理脏数据
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        
        # --- 第二步：执行前兜底 (Pre-flooring) ---
        # 根据防御性编程规范，执行前先建好目录
        os.makedirs(out_dir, exist_ok=True)

        # --- 第三步：盲测 (Blind Test) ---
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # --- 第四步：动态自愈 (Self-Healing) ---
        if res.returncode != 0:
            error_output = res.stdout + res.stderr
            log_observation(f"[WARN] Initial execution failed (Return Code: {res.returncode})")
            log_observation(f"[DEBUG] Error Output Snippet: {error_output[:500]}")
            
            retry_success = False
            
            # 策略 A: 处理 "already exists" 冲突
            # 如果报错包含 exists，说明模型讨厌预存在的目录。我们需要删掉刚才建的空目录，然后重试（不再预建）
            if "already exists" in error_output or "exists" in error_output:
                log_observation("[HEALING] Detected 'exists' error. Removing directory and retrying without pre-mkdir...")
                shutil.rmtree(out_dir, ignore_errors=True)
                # 重试时不再执行 makedirs，让模型自己创建或处理
                res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res_retry.returncode == 0:
                    retry_success = True
                    res = res_retry
            
            # 策略 B: 处理 "No such file" 冲突
            # 如果报错包含 No such file，说明模型需要目录存在但我们可能清理得太干净或路径有误
            elif "No such file or directory" in error_output or "Failed to save" in error_output:
                log_observation("[HEALING] Detected 'No such file' error. Ensuring directory exists and retrying...")
                os.makedirs(out_dir, exist_ok=True)
                res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res_retry.returncode == 0:
                    retry_success = True
                    res = res_retry
            
            # 记录最终状态
            if not retry_success:
                log_observation(f"[FAIL] Model {name} failed after healing attempts.")
                log_observation(f"[STDERR] {res.stderr}")
                log_observation(f"[STDOUT] {res.stdout}")
                print(f"[ERROR] {name} failed. Check observation log.")
            else:
                log_observation(f"[SUCCESS] Model {name} recovered and completed successfully.")
                print(f"[SUCCESS] {name} completed (after healing).")
        else:
            log_observation(f"[SUCCESS] Model {name} completed successfully.")
            print(f"[SUCCESS] {name} completed.")

    except Exception as e:
        # 捕获 Python 层面的异常（如权限错误、命令格式错误等）
        error_msg = f"[CRITICAL EXCEPTION] Model {name} crashed with exception: {str(e)}"
        log_observation(error_msg)
        print(error_msg)

def explore_outputs():
    """
    勘探 data/ 目录下所有 *_out 目录及文件，生成报告。
    """
    log_observation(f"\n{'='*60}")
    log_observation("[STAGE 1 EXPLORATION REPORT]")
    log_observation(f"{'='*60}\n")
    
    # 查找所有包含 _out 的目录
    # 使用 glob 查找 data/ 下的直接子目录，匹配 *_out
    out_dirs = glob.glob('data/*_out')
    
    # 同时也查找 data/ 下可能直接生成的文件 (以防万一)
    # 但主要关注 *_out 目录
    if not out_dirs:
        log_observation("[INFO] No directories matching 'data/*_out' found.")
        # 尝试查找 data 目录下所有文件
        all_files = glob.glob('data/*')
        for f in all_files:
            if os.path.isfile(f):
                log_observation(f"[FILE] {os.path.abspath(f)}")
                headers = read_file_header(f)
                log_observation("[HEADER CONTENT]:")
                for line in headers:
                    log_observation(f"  {line}")
                log_observation("")
        return

    for out_dir in sorted(out_dirs):
        log_observation(f"[DIR] {os.path.abspath(out_dir)}")
        
        # 遍历目录内所有文件
        found_files = False
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                found_files = True
                file_path = os.path.join(root, file)
                log_observation(f"  [FILE] {os.path.abspath(file_path)}")
                
                # 读取头部内容
                headers = read_file_header(file_path)
                log_observation("  [HEADER CONTENT]:")
                for line in headers:
                    # 替换制表符以便日志阅读，限制长度防止日志过大
                    safe_line = line.replace('\t', ' | ').replace('\n', '')
                    if len(safe_line) > 200:
                        safe_line = safe_line[:200] + "..."
                    log_observation(f"    {safe_line}")
                log_observation("")
        
        if not found_files:
            log_observation("  [WARNING] Directory is empty.")
        log_observation("-" * 40 + "\n")

    print("[INFO] Exploration report saved to", os.path.abspath(OBSERVATION_FILE))

# ==============================================================================
# 主程序入口 (Main Entry)
# ==============================================================================

def main():
    print("="*60)
    print("AMP Model Evaluation - Stage 1: Exploration")
    print("="*60)
    
    # 初始化日志文件（清空或创建）
    # 为了保留错误日志，我们使用追加模式，但为了报告清晰，可以在开头加个分隔符
    # 这里为了符合“追加写入错误流”的要求，我们不在这里清空文件，而是直接开始记录
    log_observation(f"\n### NEW SESSION STARTED: {os.path.basename(__file__)} ###\n")

    # 1. 执行模型评估
    for model in MODELS:
        execute_model_with_healing(model)
    
    # 2. 执行输出勘探
    explore_outputs()
    
    print("="*60)
    print("Stage 1 Complete. Check data/stage1_observation.txt")
    print("="*60)

if __name__ == '__main__':
    main()
