#!/usr/bin/env python3

import os
import subprocess
import shutil
import glob
import gzip

def main():
    # Step 1: Get task ID from SLURM environment
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    models = [
        {
            "name": "Macrel",
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
            "out_dir": "data/Macrel_out"
        },
        {
            "name": "AMP-Scanner-v2",
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/020419_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"',
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

    if not (0 <= task_id < len(models)):
        print(f"[ERROR] Invalid task_id={task_id}, valid range [0-{len(models)-1}]")
        return

    model_info = models[task_id]
    name = model_info["name"]
    cmd = model_info["cmd"]
    out_dir = model_info["out_dir"]

    log_file = f"data/stage1_obs_{task_id}.txt"

    with open(log_file, 'w') as log:
        def write_log(msg):
            print(msg)
            log.write(msg + "\n")

        write_log(f"[INFO] Running model: {name}")
        write_log(f"[CMD] {cmd}")

        # Step 2: Clean output directory before execution
        write_log("[CLEANUP] Removing previous output directory...")
        shutil.rmtree(out_dir, ignore_errors=True)

        # First attempt
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            write_log("[SUCCESS] Model executed successfully on first try.")
        else:
            stderr_msg = res.stderr.lower()
            stdout_msg = res.stdout.lower()
            combined_msg = stderr_msg + stdout_msg

            write_log("[FIRST FAIL] Command failed:")
            write_log("STDOUT:\n" + res.stdout)
            write_log("STDERR:\n" + res.stderr)

            # Retry logic based on error messages
            retry_needed = False
            if any(keyword in combined_msg for keyword in ['no such file', 'notfounderror', 'failed to save']):
                write_log("[RETRY] Detected missing directory; creating it now...")
                os.makedirs(out_dir, exist_ok=True)
                retry_needed = True
            elif 'already exists' in combined_msg or 'exists' in combined_msg:
                write_log("[RETRY] Detected existing directory conflict; removing it now...")
                shutil.rmtree(out_dir, ignore_errors=True)
                retry_needed = True

            if retry_needed:
                write_log("[RETRYING] Re-executing command after adjustment...")
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res.returncode == 0:
                    write_log("[SUCCESS] Model executed successfully after retry.")
                else:
                    write_log("[FINAL FAIL] Still failing after retry:")
                    write_log("STDOUT:\n" + res.stdout)
                    write_log("STDERR:\n" + res.stderr)
            else:
                write_log("[ABORT] No known fix available for this failure.")

        # Step 3: Explore generated files
        write_log("\n[EXPLORATION] Scanning output directory...")
        all_files = []
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, out_dir)
                all_files.append((full_path, rel_path))

        if not all_files:
            write_log("[WARNING] No output files detected!")
        else:
            for full_path, rel_path in all_files:
                write_log(f"\n[FILE] {rel_path} ({full_path})")
                try:
                    if full_path.endswith('.gz'):
                        with gzip.open(full_path, 'rt', encoding='utf-8') as gzfile:
                            lines = [next(gzfile).strip() for _ in range(5)]
                    else:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as txtfile:
                            lines = [next(txtfile).strip() for _ in range(5)]
                    write_log("[HEAD CONTENT]:\n" + "\n".join(lines))
                except Exception as e:
                    write_log(f"[READ ERROR] Could not read file content: {e}")

if __name__ == "__main__":
    main()