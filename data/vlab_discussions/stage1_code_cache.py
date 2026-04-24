import os
import sys
import subprocess
import shutil
import glob
import gzip
import time

MODELS = [
    {
        "name": "Macrel",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives\"",
        "out_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/020419_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv\"",
        "out_dir": "data/AMP-Scanner-v2_out"
    },
    {
        "name": "amPEPpy",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model\"",
        "out_dir": "data/amPEPpy_out"
    },
    {
        "name": "AI4AMP",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv\"",
        "out_dir": "data/AI4AMP_out"
    },
    {
        "name": "AMPlify",
        "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out\"",
        "out_dir": "data/AMPlify_out"
    }
]

MAX_RETRIES = 2

def get_head_content(file_path, max_lines=10):
    lines = []
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.strip())
        return "\n".join(lines)
    except Exception as e:
        return f"[Error reading file: {str(e)}]"

def run_model_with_healing(model_config, task_id):
    name = model_config["name"]
    cmd = model_config["cmd"]
    out_dir = model_config["out_dir"]
    
    print(f"[Task {task_id}] Starting execution for {name}...")
    
    if os.path.exists(out_dir):
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[Task {task_id}] Cleaned existing output directory: {out_dir}")
        except Exception as e:
            print(f"[Task {task_id}] Warning: Failed to clean directory {out_dir}: {e}")

    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            res = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=600
            )
            
            if res.returncode == 0:
                print(f"[Task {task_id}] Success: {name} completed.")
                return True
            
            error_msg = res.stderr + res.stdout
            last_error = error_msg
            
            print(f"[Task {task_id}] Attempt {attempt+1} failed for {name}. Return code: {res.returncode}")
            
            if "already exists" in error_msg.lower() or "exists" in error_msg.lower():
                print(f"[Task {task_id}] Healing: Directory exists conflict detected. Removing and retrying...")
                shutil.rmtree(out_dir, ignore_errors=True)
                continue
                
            elif "No such file" in error_msg or "NotFoundError" in error_msg or "not found" in error_msg:
                print(f"[Task {task_id}] Healing: Missing directory detected. Creating and retrying...")
                os.makedirs(out_dir, exist_ok=True)
                continue
            
            else:
                print(f"[Task {task_id}] Fatal Error for {name}: {error_msg[:200]}...")
                break
                
        except subprocess.TimeoutExpired:
            print(f"[Task {task_id}] Timeout for {name}.")
            last_error = "Timeout"
            break
        except Exception as e:
            print(f"[Task {task_id}] Exception during run: {e}")
            last_error = str(e)
            break

    if last_error:
        err_log_path = f"data/error_task_{task_id}.log"
        os.makedirs("data", exist_ok=True)
        with open(err_log_path, "a") as f:
            f.write(f"--- Task {task_id} ({name}) Failed ---\n")
            f.write(last_error + "\n\n")
        return False
        
    return True

def explore_output(task_id, model_config):
    out_dir = model_config["out_dir"]
    
    os.makedirs("data", exist_ok=True)
    
    # 🚨 ISOLATION LOGIC: Strictly follow the variable assignment requirement
    log_file = f"data/stage1_obs_{task_id}.txt"
    
    report_lines = []
    report_lines.append(f"=== Stage 1 Exploration Report ===")
    report_lines.append(f"Task ID: {task_id}")
    report_lines.append(f"Model: {model_config['name']}")
    report_lines.append(f"Target Dir: {out_dir}")
    report_lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 50 + "\n")
    
    if not os.path.exists(out_dir):
        report_lines.append(f"[WARNING] Output directory does not exist: {out_dir}")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        return

    found_files = []
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            full_path = os.path.join(root, file)
            found_files.append(full_path)
    
    if not found_files:
        report_lines.append("[INFO] No files found in output directory.")
    else:
        report_lines.append(f"[INFO] Found {len(found_files)} files:")
        for f_path in sorted(found_files):
            rel_path = os.path.relpath(f_path, "data")
            size = os.path.getsize(f_path)
            header = get_head_content(f_path)
            
            report_lines.append(f"\nFile: {rel_path}")
            report_lines.append(f"Size: {size} bytes")
            report_lines.append(f"Head Content:\n{header}")
            report_lines.append("-" * 30)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[Task {task_id}] Exploration report saved to: {log_file}")

def main():
    try:
        task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
        task_id = int(task_id_str)
    except ValueError:
        print("Error: SLURM_ARRAY_TASK_ID is not a valid integer. Exiting.")
        sys.exit(1)

    if task_id < 0 or task_id >= len(MODELS):
        print(f"Warning: Task ID {task_id} is out of range [0-{len(MODELS)-1}]. Exiting safely.")
        os.makedirs("data", exist_ok=True)
        log_file = f"data/stage1_obs_{task_id}.txt"
        with open(log_file, "w") as f:
            f.write(f"Task {task_id} skipped: Out of range.\n")
        sys.exit(0)

    current_model = MODELS[task_id]
    print(f"Selected Model for Task {task_id}: {current_model['name']}")

    success = run_model_with_healing(current_model, task_id)

    explore_output(task_id, current_model)

    print(f"Task {task_id} finished.")

if __name__ == '__main__':
    main()