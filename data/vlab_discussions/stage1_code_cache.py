#!/usr/bin/env python3

import os
import subprocess
import shutil
import glob
import gzip

MODELS = [
    {
        "name": "Macrel",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
        "out_dir": "data/Macrel_out"
    },
    {
        "name": "AMP-Scanner-v2",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
        "out_dir": "data/AMP-Scanner-v2_out"
    },
    {
        "name": "amPEPpy",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model"',
        "out_dir": "data/amPEPpy_out"
    },
    {
        "name": "AI4AMP",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv"',
        "out_dir": "data/AI4AMP_out"
    },
    {
        "name": "AMPlify",
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out"',
        "out_dir": "data/AMPlify_out"
    }
]

def run_model(model):
    out_dir = model["out_dir"]
    cmd = model["cmd"]

    # Step 1: Clean up previous outputs
    print(f"[INFO] Removing existing output directory: {out_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)

    # Step 2: First attempt to run the command
    print(f"[INFO] Running model '{model['name']}' with first attempt...")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if res.returncode == 0:
        print("[SUCCESS] Model executed successfully on first try.")
        return True

    stderr_output = res.stderr.lower()
    stdout_output = res.stdout.lower()
    combined_error = stderr_output + stdout_output

    print(f"[ERROR] First execution failed:\nSTDOUT={res.stdout}\nSTDERR={res.stderr}")

    # Step 3: Self-healing logic based on error messages
    if any(keyword in combined_error for keyword in ["no such file", "failed to save", "notfounderror"]):
        print("[HEALING] Detected missing directory issue. Creating output directory...")
        os.makedirs(out_dir, exist_ok=True)
        print("[RETRY] Retrying after creating directory...")
        res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res_retry.returncode == 0:
            print("[SUCCESS] Model executed successfully after retry.")
            return True
        else:
            print(f"[FAIL] Retry also failed:\nSTDOUT={res_retry.stdout}\nSTDERR={res_retry.stderr}")
    
    elif "already exists" in combined_error or "exists" in combined_error:
        print("[HEALING] Detected pre-existing directory conflict. Removing it again...")
        shutil.rmtree(out_dir, ignore_errors=True)
        print("[RETRY] Retrying after removing conflicting directory...")
        res_retry = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res_retry.returncode == 0:
            print("[SUCCESS] Model executed successfully after retry.")
            return True
        else:
            print(f"[FAIL] Retry also failed:\nSTDOUT={res_retry.stdout}\nSTDERR={res_retry.stderr}")

    # Final failure handling
    log_file = f"data/stage1_obs_{int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))}.txt"
    with open(log_file, "w") as f:
        f.write("=== Execution Failed ===\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"Return Code: {res.returncode}\n")
        f.write(f"STDOUT:\n{res.stdout}\n")
        f.write(f"STDERR:\n{res.stderr}\n")
    print(f"[FATAL] All attempts failed. Error details saved to {log_file}")
    return False

def inspect_output(out_dir, log_file):
    print(f"[INSPECT] Inspecting output directory: {out_dir}")
    files = []
    pattern_all = os.path.join(out_dir, "**/*")
    for file_path in glob.glob(pattern_all, recursive=True):
        if os.path.isfile(file_path):
            files.append(file_path)

    with open(log_file, "w") as lf:
        lf.write(f"=== Output Inspection Report ===\n")
        lf.write(f"Output Directory: {out_dir}\n\n")

        for idx, fp in enumerate(files):
            lf.write(f"\n--- File #{idx+1}: {fp} ---\n")
            try:
                if fp.endswith(".gz"):
                    with gzip.open(fp, "rt", encoding="utf-8") as gf:
                        lines = [next(gf) for _ in range(10)]
                        lf.writelines(lines)
                else:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as tf:
                        lines = [next(tf) for _ in range(10)]
                        lf.writelines(lines)
            except Exception as e:
                lf.write(f"[ERROR] Could not read file content: {str(e)}\n")

def main():
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    total_tasks = len(MODELS)

    if task_id >= total_tasks or task_id < 0:
        print(f"[WARNING] Invalid task_id {task_id}, valid range is [0, {total_tasks - 1}]")
        return

    selected_model = MODELS[task_id]
    success = run_model(selected_model)

    if success:
        log_file = f"data/stage1_obs_{task_id}.txt"
        inspect_output(selected_model["out_dir"], log_file)
        print(f"[DONE] Observation report written to {log_file}")

if __name__ == "__main__":
    main()