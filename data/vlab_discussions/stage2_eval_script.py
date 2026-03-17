#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMP Model Evaluation Pipeline (Stage 2)
Based on Stage 1 Exploration Report.
Strictly follows PI constraints regarding data cleaning, merging, and visualization.
"""

import os
import sys
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Must declare before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, matthews_corrcoef,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# Suppress specific warnings to keep output clean, but not errors
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# 1. Configuration & Paths
# ==============================================================================

DATA_DIR = "data"
GT_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
OUTPUT_DIR = "."  # Save results in current directory

# Expected patterns based on Exploration Report
MACREL_PATTERN = os.path.join(DATA_DIR, "Macrel_out", "*.gz")
AMPSCANNER_PATTERN = os.path.join(DATA_DIR, "AMP-Scanner-v2_out", "*.csv")

# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def clean_id(x):
    """
    Strict ID cleaning as per PI requirements:
    - Convert to string
    - Remove '>'
    - Split by space and take first part
    - Strip whitespace
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.startswith('>'):
        s = s[1:]
    return s.split()[0].strip()

def find_probability_column(df):
    """
    Fuzzy lock probability column based on discipline.
    """
    candidates = [col for col in df.columns if 'prob' in col.lower() or 'score' in col.lower()]
    # Prioritize columns explicitly named with probability
    for col in candidates:
        if 'prob' in col.lower():
            return col
    if candidates:
        return candidates[0]
    return None

def find_label_column(df):
    """
    Identify ground truth label column.
    """
    # Look for common names
    possible_names = ['label', 'true_label', 'class', 'target', 'y']
    for col in df.columns:
        if any(name in col.lower() for name in possible_names):
            return col
    # Fallback: find binary column (0/1) that isn't ID
    for col in df.columns:
        if df[col].dtype in [np.int64, np.float64, int, float]:
            unique_vals = df[col].unique()
            if len(unique_vals) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in unique_vals):
                return col
    # Last resort: assume second column
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]

def find_id_column(df, model_name=None):
    """
    Identify ID column based on model specifics or general heuristics.
    """
    if model_name == 'Macrel':
        if 'Access' in df.columns:
            return 'Access'
    if model_name == 'AMP-Scanner':
        if 'SeqID' in df.columns:
            return 'SeqID'
    
    # General heuristics
    possible_names = ['id', 'seqid', 'access', 'name', 'sequence_id']
    for col in df.columns:
        if any(name in col.lower() for name in possible_names):
            return col
    # Fallback: first column
    return df.columns[0]

# ==============================================================================
# 3. Data Loading (Strict Discipline)
# ==============================================================================

def load_ground_truth():
    """
    Load ground truth. Must be the source of truth for labels.
    """
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Ground Truth file not found at {GT_PATH}")
    
    # Default CSV reading, no low_memory=False
    df = pd.read_csv(GT_PATH)
    
    id_col = find_id_column(df)
    label_col = find_label_column(df)
    
    df = df.rename(columns={id_col: 'ID', label_col: 'True_Label'})
    df['ID'] = df['ID'].apply(clean_id)
    df['True_Label'] = pd.to_numeric(df['True_Label'], errors='coerce').fillna(0).astype(int)
    
    return df[['ID', 'True_Label']].drop_duplicates(subset=['ID'])

def load_macrel():
    """
    Load Macrel output based on Exploration Report (.gz, tab-separated, comments).
    """
    files = glob.glob(MACREL_PATTERN)
    if not files:
        # Fallback to exact path from report if glob fails
        exact_path = os.path.join(DATA_DIR, "Macrel_out", "macrel.out.prediction.gz")
        if os.path.exists(exact_path):
            files = [exact_path]
        else:
            return None
            
    filepath = files[0] # Take the first match
    
    # STRICT READ COMMAND FOR .gz
    df = pd.read_csv(filepath, sep='\t', comment='#', compression='gzip')
    
    id_col = find_id_column(df, model_name='Macrel')
    prob_col = find_probability_column(df)
    
    if not prob_col:
        return None
        
    df = df.rename(columns={id_col: 'ID', prob_col: 'Macrel_Prob'})
    df['ID'] = df['ID'].apply(clean_id)
    df['Macrel_Prob'] = pd.to_numeric(df['Macrel_Prob'], errors='coerce')
    
    return df[['ID', 'Macrel_Prob']]

def load_ampscanner():
    """
    Load AMP-Scanner-v2 output based on Exploration Report (.csv, comma-separated).
    """
    files = glob.glob(AMPSCANNER_PATTERN)
    if not files:
        # Fallback to exact path from report if glob fails
        exact_path = os.path.join(DATA_DIR, "AMP-Scanner-v2_out", "ampscanner_out.csv")
        if os.path.exists(exact_path):
            files = [exact_path]
        else:
            return None
            
    filepath = files[0]
    
    # STRICT READ COMMAND FOR .csv
    df = pd.read_csv(filepath, sep=',')
    
    id_col = find_id_column(df, model_name='AMP-Scanner')
    prob_col = find_probability_column(df)
    
    if not prob_col:
        return None
        
    df = df.rename(columns={id_col: 'ID', prob_col: 'AMPScanner_Prob'})
    df['ID'] = df['ID'].apply(clean_id)
    df['AMPScanner_Prob'] = pd.to_numeric(df['AMPScanner_Prob'], errors='coerce')
    
    return df[['ID', 'AMPScanner_Prob']]

# ==============================================================================
# 4. Merging & Cleaning
# ==============================================================================

def merge_data(gt_df, macrel_df, amp_df):
    """
    Merge all dataframes based on Ground Truth (Left Join).
    Fill missing probabilities with 0.0.
    """
    # Start with Ground Truth
    merged = gt_df.copy()
    
    # Merge Macrel
    if macrel_df is not None:
        merged = merged.merge(macrel_df, on='ID', how='left')
    else:
        merged['Macrel_Prob'] = np.nan
        
    # Merge AMP-Scanner
    if amp_df is not None:
        merged = merged.merge(amp_df, on='ID', how='left')
    else:
        merged['AMPScanner_Prob'] = np.nan
        
    # Fill Missing Probs with 0.0 (PI Requirement)
    prob_cols = ['Macrel_Prob', 'AMPScanner_Prob']
    for col in prob_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)
            
    # Generate Binary Predictions (Threshold 0.5)
    for col in prob_cols:
        if col in merged.columns:
            pred_col = col.replace('_Prob', '_Pred')
            merged[pred_col] = (merged[col] > 0.5).astype(int)
            
    return merged

# ==============================================================================
# 5. Evaluation Metrics
# ==============================================================================

def calculate_metrics(y_true, y_prob, y_pred):
    """
    Calculate ACC, Recall, MCC, AUROC, AUPRC.
    """
    metrics = {}
    
    # ACC
    metrics['ACC'] = float(accuracy_score(y_true, y_pred))
    
    # Recall
    metrics['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    
    # MCC
    try:
        metrics['MCC'] = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        metrics['MCC'] = 0.0
        
    # AUROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics['AUROC'] = float(auc(fpr, tpr))
    
    # AUPRC
    metrics['AUPRC'] = float(average_precision_score(y_true, y_prob))
    
    return metrics, fpr, tpr

def evaluate_models(merged_df):
    """
    Evaluate all models and return results + curve data.
    """
    results = {}
    curve_data = {}
    
    y_true = merged_df['True_Label'].values
    
    models = {
        'Macrel': 'Macrel_Prob',
        'AMP-Scanner-v2': 'AMPScanner_Prob'
    }
    
    for model_name, prob_col in models.items():
        if prob_col not in merged_df.columns:
            continue
            
        y_prob = merged_df[prob_col].values
        pred_col = prob_col.replace('_Prob', '_Pred')
        y_pred = merged_df[pred_col].values
        
        metrics, fpr, tpr = calculate_metrics(y_true, y_prob, y_pred)
        results[model_name] = metrics
        
        # Store curve data for plotting
        curve_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'prob': y_prob
        }
        
    return results, curve_data

# ==============================================================================
# 6. Visualization (Publication Quality)
# ==============================================================================

def plot_curves(merged_df, curve_data, output_path):
    """
    Plot ROC and PR curves using numpy.interp for smoothing/alignment.
    High quality settings for publication.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # Professional style
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color map for models
    colors = {'Macrel': '#E63946', 'AMP-Scanner-v2': '#457B9D'}
    linestyles = {'Macrel': '-', 'AMP-Scanner-v2': '--'}
    
    y_true = merged_df['True_Label'].values
    
    # --- ROC Curve ---
    ax = axes[0]
    # Standard grid for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    
    for model_name, data in curve_data.items():
        fpr = data['fpr']
        tpr = data['tpr']
        auc_val = auc(fpr, tpr)
        
        # Interpolate TPR to mean FPR using numpy.interp (PI Requirement)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        
        ax.plot(mean_fpr, tpr_interp, color=colors.get(model_name, 'gray'), 
                linestyle=linestyles.get(model_name, '-'), linewidth=2.5,
                label=f'{model_name} (AUC = {auc_val:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # --- PR Curve ---
    ax = axes[1]
    # For PR curve, we interpolate Precision over Recall
    mean_recall = np.linspace(0, 1, 100)
    
    for model_name, data in curve_data.items():
        y_prob = data['prob']
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        # Interpolate Precision to mean Recall using numpy.interp (PI Requirement)
        # Note: precision_recall_curve returns decreasing recall, so we sort for interp
        # But np.interp expects increasing x. 
        # Standard practice: interpolate precision at specific recall points
        # To avoid complexity, we plot raw steps but ensure numpy.interp is used for smoothing if needed.
        # Here we use np.interp to align recall points for smooth visualization.
        precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
        
        ax.plot(mean_recall, precision_interp, color=colors.get(model_name, 'gray'), 
                linestyle=linestyles.get(model_name, '-'), linewidth=2.5,
                label=f'{model_name} (AP = {ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 7. Main Execution
# ==============================================================================

def main():
    print("[INFO] Starting Evaluation Pipeline...")
    
    # 1. Load Data
    print("[INFO] Loading Ground Truth...")
    gt_df = load_ground_truth()
    print(f"[INFO] Ground Truth loaded: {len(gt_df)} samples.")
    
    print("[INFO] Loading Macrel...")
    macrel_df = load_macrel()
    if macrel_df is not None:
        print(f"[INFO] Macrel loaded: {len(macrel_df)} samples.")
    else:
        print("[WARNING] Macrel output not found.")
        
    print("[INFO] Loading AMP-Scanner-v2...")
    amp_df = load_ampscanner()
    if amp_df is not None:
        print(f"[INFO] AMP-Scanner-v2 loaded: {len(amp_df)} samples.")
    else:
        print("[WARNING] AMP-Scanner-v2 output not found.")
    
    # 2. Merge Data
    print("[INFO] Merging datasets (Left Join on Ground Truth)...")
    merged_df = merge_data(gt_df, macrel_df, amp_df)
    print(f"[INFO] Merged dataset size: {len(merged_df)} samples.")
    
    # 3. Evaluate
    print("[INFO] Calculating metrics...")
    eval_results, curve_data = evaluate_models(merged_df)
    
    # 4. Save Results (JSON)
    json_path = os.path.join(OUTPUT_DIR, "eval_result.json")
    with open(json_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    print(f"[SUCCESS] Metrics saved to {json_path}")
    
    # 5. Save Predictions (CSV)
    csv_path = os.path.join(OUTPUT_DIR, "final_results_with_predictions.csv")
    merged_df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Predictions saved to {csv_path}")
    
    # 6. Plot Curves
    png_path = os.path.join(OUTPUT_DIR, "evaluation_curves.png")
    plot_curves(merged_df, curve_data, png_path)
    print(f"[SUCCESS] Curves saved to {png_path}")
    
    # 7. Print Summary
    print("\n=== Evaluation Summary ===")
    for model, metrics in eval_results.items():
        print(f"\n{model}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    print("\n[INFO] Pipeline completed successfully.")

if __name__ == "__main__":
    main()
