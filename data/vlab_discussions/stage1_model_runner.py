#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Model Execution & Structure Exploration Script
Target Models: Macrel, AMP-Scanner-v2
Constraint: No pandas, No metrics, No plotting. Robust Self-Healing.
Output: data/stage1_observation.txt
"""

import os
import subprocess
import shutil
import glob
import gzip
import sys
import datetime

# ==============================================================================
# Configuration & Constants
# ==============================================================================

DATA_DIR = "data"
OBSERVATION_LOG = os.path.join(DATA_DIR, "stage1_observation.txt")

# Model Definition Matrix
MODELS = [
    {
        "name": "Macrel",
        "output_dir": os.path.join(DATA_DIR, "Macrel_out"),
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out"'
    },
    {
        "name": "AMP-Scanner-v2",
        "output_dir": os.path.join(DATA_DIR, "AMP-Scanner-v2_out"),
        "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"'
    }
]

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_data_dir():
    """Ensure the base data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)

def log_message(message, mode='a'):
    """Append message to the observation log."""
    try:
        # Ensure data dir exists before writing
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OBSERVATION_LOG, mode, encoding='utf-8', errors='ignore') as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[CRITICAL] Failed to write to log: {e}")

def read_file_head(file_path, lines=10):
    """
    Read the first N lines of a file.
    Handles .gz compression automatically.
    Returns a list of strings.
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
        return [f"[Error reading file: {str(e)}]"]
    return content_lines

def run_model_with_healing(model_config):
    """
    Execute a model with strict self-healing logic.
    1. Clean output dir.
    2. Run command.
    3. If fail, check errors and retry based on keywords.
    4. Log final failures.
    """
    name = model_config["name"]
    out_dir = model_config["output_dir"]
    cmd = model_config["cmd"]
    
    print(f"[INFO] Starting execution for model: {name} at {get_timestamp()}")
    log_message(f"\n=== Model Execution: {name} ===\nTime: {get_timestamp()}")
    
    try:
        # Step 1: Clean (Pre-execution hygiene)
        # Even if it doesn't exist, ignore_errors=True prevents crashes
        shutil.rmtree(out_dir, ignore_errors=True)
        
        # Step 2: Blind Test (First Attempt)
        # Ensure parent dir exists just in case
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        
        res = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=3600  # 1 hour timeout safety
        )
        
        # Step 3: Dynamic Self-Healing (If returncode != 0)
        if res.returncode != 0:
            combined_output = res.stdout + res.stderr
            print(f"[WARN] {name} failed initially. Analyzing errors...")
            
            retry_needed = False
            retry_action = ""
            
            # Check for specific error patterns
            if "No such file or directory" in combined_output or "Failed to save" in combined_output:
                # Model complains about missing directory -> Create it
                print(f"[HEALING] {name} needs directory creation. Executing makedirs...")
                os.makedirs(out_dir, exist_ok=True)
                retry_needed = True
                retry_action = "mkdir"
                
            elif "already exists" in combined_output or "exists" in combined_output:
                # Model complains about existing directory -> Remove it
                print(f"[HEALING] {name} hates existing directory. Executing rmtree...")
                shutil.rmtree(out_dir, ignore_errors=True)
                # Re-create empty dir for consistency before retry
                os.makedirs(out_dir, exist_ok=True) 
                retry_needed = True
                retry_action = "rmtree"
            
            if retry_needed:
                # Retry Execution
                print(f"[RETRY] Retrying {name} after {retry_action}...")
                res = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=3600
                )
        
        # Step 4: Ultimate Log (If still failed after retry)
        if res.returncode != 0:
            error_msg = f"[FAILURE] Model {name} failed after healing attempts.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n"
            print(f"[ERROR] {name} execution failed permanently.")
            log_message(error_msg)
        else:
            print(f"[SUCCESS] {name} completed successfully.")
            log_message(f"[SUCCESS] Model {name} completed successfully.\n")

    except Exception as e:
        # Absolute Isolation: Catch ANY exception to prevent blocking main program
        error_msg = f"[CRITICAL EXCEPTION] Model {name} crashed with exception: {str(e)}\n"
        print(f"[CRITICAL] {name} crashed: {e}")
        log_message(error_msg)

def explore_outputs():
    """
    Traverse data/ directory, find output files, read headers, and report.
    Strictly follows Stage 1 constraints: No pandas, no metrics.
    """
    print("[INFO] Starting output structure exploration...")
    log_message("\n=== Stage 1 Exploration Report ===\n")
    log_message(f"Exploration Time: {get_timestamp()}\n")
    
    # Ensure data dir exists before globbing
    if not os.path.exists(DATA_DIR):
        log_message("[ERROR] Data directory does not exist.")
        return

    # Find all directories ending with _out
    out_dirs = glob.glob(os.path.join(DATA_DIR, "*_out"))
    
    found_files = []
    for d in out_dirs:
        if os.path.isdir(d):
            # Walk through the directory
            for root, dirs, files in os.walk(d):
                for file in files:
                    found_files.append(os.path.join(root, file))
    
    # Fallback: Check any file in data/ just in case (excluding log itself)
    if not found_files:
        for f in os.listdir(DATA_DIR):
            full_path = os.path.join(DATA_DIR, f)
            if os.path.isfile(full_path) and not f.endswith('.txt'): 
                found_files.append(full_path)

    if not found_files:
        log_message("[INFO] No output files found in *_out directories or data/ root.")
    else:
        log_message(f"[INFO] Found {len(found_files)} output files.\n")
        for file_path in found_files:
            log_message(f"File Path: {file_path}")
            log_message(f"File Size: {os.path.getsize(file_path)} bytes")
            head_content = read_file_head(file_path, lines=10)
            log_message("Header Content (First 10 lines):")
            for line in head_content:
                log_message(f"  {line}")
            log_message("-" * 50)

    print(f"[INFO] Exploration complete. Report saved to {OBSERVATION_LOG}")

# ==============================================================================
# Main Execution Flow
# ==============================================================================

def main():
    print("=" * 60)
    print("Stage 1: Model Execution & Exploration Pipeline")
    print("=" * 60)
    
    # 0. Initialization
    ensure_data_dir()
    
    # Clear previous observation log for fresh start
    if os.path.exists(OBSERVATION_LOG):
        os.remove(OBSERVATION_LOG)
        
    log_message(f"Pipeline Start Time: {get_timestamp()}")
    
    # 1. Execute Models (Isolated & Self-Healing)
    for model in MODELS:
        run_model_with_healing(model)
        # Small delay to ensure file system sync
        subprocess.run("sleep 1", shell=True)
        
    # 2. Explore Outputs (Reconnaissance)
    explore_outputs()
    
    log_message(f"Pipeline End Time: {get_timestamp()}")
    print("=" * 60)
    print("Pipeline Finished. Check data/stage1_observation.txt for details.")
    print("=" * 60)

if __name__ == '__main__':
    main()
