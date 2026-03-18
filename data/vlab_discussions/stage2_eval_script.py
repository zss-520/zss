#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMP Model Evaluation Pipeline (Stage 2)
Based on Stage 1 Exploration Report.
Strictly follows PI constraints regarding data parsing, merging, and scoring.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, matthews_corrcoef, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ==========================================
# Configuration & Constants
# ==========================================
DATA_DIR = "data"
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
OUTPUT_JSON = "eval_result.json"
OUTPUT_CSV = "final_results_with_predictions.csv"
OUTPUT_PNG = "evaluation_curves.png"

# Model Output Directories (Identified from Stage 1 Report)
MODEL_DIRS = {
    "Macrel": os.path.join(DATA_DIR, "Macrel_out"),
    "AMP-Scanner-v2": os.path.join(DATA_DIR, "AMP-Scanner-v2_out")
}

# ==========================================
# Helper Functions
# ==========================================

def clean_id_series(series):
    """
    Strong Clean ID: Convert to string, remove '>', split by space, strip.
    Compliant with PI Requirement 3.
    """
    return series.apply(lambda x: str(x).replace('>', '').split()[0].strip())

def find_model_output_file(directory, extensions):
    """
    Dynamic Data Extraction: Find the most likely output file using glob.
    Compliant with PI Requirement "Dynamic Data Extraction".
    """
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        files = glob.glob(pattern)
        # Filter out logs/readme if possible by size or name, but here we look for data files
        # Prioritize files that are not .md or .log
        valid_files = [f for f in files if not f.endswith('.md') and not f.endswith('.log')]
        if valid_files:
            # Return the largest file usually containing data
            return max(valid_files, key=os.path.getsize)
    return None

def load_ground_truth():
    """
    Load Ground Truth strictly from CSV.
    """
    if not os.path.exists(GROUND_TRUTH_PATH):
        raise FileNotFoundError(f"Ground Truth not found at {GROUND_TRUTH_PATH}")
    
    # Read without low_memory=False engine conflicts
    df = pd.read_csv(GROUND_TRUTH_PATH, sep=',')
    
    # Validate columns based on Report: Sequence, label
    if 'Sequence' not in df.columns or 'label' not in df.columns:
        # Fuzzy match if exact names differ slightly
        seq_col = [c for c in df.columns if 'seq' in c.lower()]
        lab_col = [c for c in df.columns if 'label' in c.lower() or 'truth' in c.lower()]
        if seq_col and lab_col:
            df = df.rename(columns={seq_col[0]: 'Sequence', lab_col[0]: 'label'})
        else:
            raise ValueError("Cannot identify Sequence or label columns in ground_truth.csv")
            
    df['ID'] = clean_id_series(df['Sequence'])
    return df[['ID', 'label']].set_index('ID')

def load_macrel_output(directory):
    """
    Load Macrel output based on Report: .gz, tab-separated, comment='#'.
    """
    filepath = find_model_output_file(directory, ['.gz', '.csv', '.tsv', '.out'])
    if not filepath:
        return None
    
    # Report indicates: macrel.out.prediction.gz
    # Strict reading protocol for .gz
    try:
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, sep='\t', comment='#', compression='gzip')
        else:
            df = pd.read_csv(filepath, sep='\t', comment='#')
            
        # Report Columns: Access, Sequence, AMP_family, AMP_probability...
        # Identify Probability Column
        prob_cols = [c for c in df.columns if 'prob' in c.lower() and 'amp' in c.lower()]
        if not prob_cols:
            prob_cols = [c for c in df.columns if 'prob' in c.lower()]
        
        if not prob_cols:
            return None
            
        prob_col = prob_cols[0]
        
        # Identify ID Column (Sequence)
        id_cols = [c for c in df.columns if 'sequence' in c.lower()]
        if not id_cols:
            return None
        id_col = id_cols[0]
        
        df['ID'] = clean_id_series(df[id_col])
        df['Pred_Prob'] = pd.to_numeric(df[prob_col], errors='coerce')
        return df[['ID', 'Pred_Prob']]
        
    except Exception as e:
        print(f"Error loading Macrel: {e}", file=sys.stderr)
        return None

def load_amp_scanner_output(directory):
    """
    Load AMP-Scanner-v2 output based on Report: .csv, comma-separated.
    """
    filepath = find_model_output_file(directory, ['.csv', '.tsv', '.txt', '.out'])
    if not filepath:
        return None
        
    # Report indicates: ampscanner_out.csv
    # Strict reading protocol for .csv (Default C engine, NO low_memory=False)
    try:
        df = pd.read_csv(filepath, sep=',')
        
        # Report Columns: SeqID, Prediction_Class, Prediction_Probability, Sequence
        # Identify Probability Column
        prob_cols = [c for c in df.columns if 'prob' in c.lower()]
        if not prob_cols:
            return None
        prob_col = prob_cols[0]
        
        # Identify ID Column (Sequence)
        id_cols = [c for c in df.columns if 'sequence' in c.lower()]
        if not id_cols:
            return None
        id_col = id_cols[0]
        
        df['ID'] = clean_id_series(df[id_col])
        df['Pred_Prob'] = pd.to_numeric(df[prob_col], errors='coerce')
        return df[['ID', 'Pred_Prob']]
        
    except Exception as e:
        print(f"Error loading AMP-Scanner: {e}", file=sys.stderr)
        return None

def calculate_metrics(y_true, y_prob):
    """
    Calculate all required metrics using sklearn.
    """
    # Binary prediction based on threshold 0.5
    y_pred = (y_prob > 0.5).astype(int)
    
    # Handle case where all probabilities are NaN or 0 leading to constant input
    try:
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        acc, recall, mcc = 0.0, 0.0, 0.0
        
    try:
        auroc = auc(*roc_curve(y_true, y_prob)[:2])
    except Exception:
        auroc = 0.5
        
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = 0.0
        
    return {
        "ACC": float(acc),
        "Recall": float(recall),
        "MCC": float(mcc),
        "AUROC": float(auroc),
        "AUPRC": float(auprc)
    }

def plot_curves(model_results, output_path):
    """
    Plot ROC and PR curves with publication quality.
    Uses numpy.interp instead of scipy.interp.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Color map for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))
    
    # --- ROC Curve ---
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    
    # --- PR Curve ---
    ax_pr = axes[1]
    
    for i, (model_name, data) in enumerate(model_results.items()):
        y_true = data['y_true']
        y_prob = data['y_prob']
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[i], lw=2, label=f'{model_name} (AUC={roc_auc:.2f})')
        
        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax_pr.plot(recall, precision, color=colors[i], lw=2, label=f'{model_name} (AP={pr_auc:.2f})')
        
    # Styling ROC
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.grid(True, linestyle='--', alpha=0.7)
    
    # Styling PR
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Curve (PR)', fontsize=14, fontweight='bold')
    ax_pr.legend(loc="lower left", fontsize=10)
    ax_pr.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Main Execution Logic
# ==========================================

def main():
    print("=== Starting AMP Evaluation Pipeline ===")
    
    # 1. Load Ground Truth
    print("Loading Ground Truth...")
    gt_df = load_ground_truth()
    if gt_df.empty:
        raise ValueError("Ground Truth is empty after loading.")
    
    # Initialize Final Results DataFrame with Ground Truth
    # We start with ID and Label
    final_results = gt_df.reset_index() # ID is index, reset to column
    
    # Store results for JSON and Plotting
    eval_results = {}
    plot_data = {}
    
    # 2. Define Model Loaders
    models_config = [
        {"name": "Macrel", "loader": load_macrel_output, "dir": MODEL_DIRS["Macrel"]},
        {"name": "AMP-Scanner-v2", "loader": load_amp_scanner_output, "dir": MODEL_DIRS["AMP-Scanner-v2"]}
    ]
    
    # 3. Process Each Model
    for model_cfg in models_config:
        model_name = model_cfg["name"]
        print(f"Processing {model_name}...")
        
        # Load Model Predictions
        model_df = model_cfg["loader"](model_cfg["dir"])
        
        if model_df is None or model_df.empty:
            print(f"Warning: No valid data found for {model_name}. Skipping.")
            eval_results[model_name] = {"Error": "No data parsed"}
            continue
        
        # --- SAFETY MERGE & SCORING DISCIPLINE ---
        # 3.1 Merge with Ground Truth (Left Join on ID)
        # Ensure ID column is consistent
        merged = final_results[['ID', 'label']].merge(model_df, on='ID', how='left')
        
        # 3.2 Fill NaN Probabilities with 0.0
        merged['Pred_Prob'] = pd.to_numeric(merged['Pred_Prob'], errors='coerce').fillna(0.0)
        
        # 3.3 Calculate Metrics IMMEDIATELY (Before Renaming)
        # Extract numpy arrays for sklearn
        y_true = merged['label'].values
        y_prob = merged['Pred_Prob'].values
        
        metrics = calculate_metrics(y_true, y_prob)
        eval_results[model_name] = metrics
        plot_data[model_name] = {'y_true': y_true, 'y_prob': y_prob}
        
        print(f"  Metrics for {model_name}: {metrics}")
        
        # 3.4 Rename Columns for Final CSV Accumulation
        # Now safe to rename for the final big table
        prob_col_name = f"{model_name}_Prob"
        pred_col_name = f"{model_name}_Pred"
        
        merged[prob_col_name] = merged['Pred_Prob']
        merged[pred_col_name] = (merged['Pred_Prob'] > 0.5).astype(int)
        
        # 3.5 Merge into Final Results Table
        # We only need to add the new columns to the main final_results dataframe
        # Since 'ID' is unique in final_results, we can join directly
        final_results = final_results.merge(
            merged[['ID', prob_col_name, pred_col_name]], 
            on='ID', 
            how='left'
        )
        
    # 4. Save Outputs
    print("Saving Outputs...")
    
    # 4.1 Save JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"  Saved {OUTPUT_JSON}")
    
    # 4.2 Save CSV
    final_results.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {OUTPUT_CSV}")
    
    # 4.3 Save Plots
    if plot_data:
        plot_curves(plot_data, OUTPUT_PNG)
        print(f"  Saved {OUTPUT_PNG}")
    else:
        print("  Skipping plots due to lack of data.")
        
    print("=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
