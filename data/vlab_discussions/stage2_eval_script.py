import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# ============================================================
# Configuration & Paths
# ============================================================
BASE_DIR = "data"
PATH_GROUND_TRUTH = os.path.join(BASE_DIR, "ground_truth.csv")
PATH_FASTA = os.path.join(BASE_DIR, "combined_test.fasta")
PATH_OUTPUT_JSON = "eval_result.json"
PATH_OUTPUT_CSV = "final_results_with_predictions.csv"
PATH_OUTPUT_PNG = "evaluation_curves.png"

# Model Output Paths based on Session 1 Exploration Report
MODEL_CONFIGS = {
    "Macrel": {
        "path": os.path.join(BASE_DIR, "Macrel_out", "macrel.out.prediction.gz"),
        "type": "gz",
        "sep": "\t",
        "comment": "#"
    },
    "AMP-Scanner-v2": {
        "path": os.path.join(BASE_DIR, "AMP-Scanner-v2_out", "ampscanner_out.csv"),
        "type": "csv",
        "sep": ",",
        "comment": None
    }
}

# ============================================================
# Helper Functions
# ============================================================

def load_fasta_map(fasta_path):
    """
    🚨 FASTA Mapping 必杀技：
    读取 FASTA 文件，构建 {Header_ID: Sequence} 字典。
    用于将模型输出的 ID 映射为真实氨基酸序列。
    """
    fasta_map = {}
    if not os.path.exists(fasta_path):
        print(f"[WARN] FASTA file not found: {fasta_path}")
        return fasta_map
    
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    fasta_map[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            fasta_map[current_id] = "".join(current_seq)
            
    return fasta_map

def sniff_column(columns, keywords):
    """
    🚨 动态嗅探真实列名：
    寻找包含 keywords 中任意一个词的列名。
    """
    for col in columns:
        col_lower = str(col).lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None

def clean_id_value(val):
    """
    🚨 ID 清洗纪律：
    必须进行 split()[0].strip() 强清洗。
    """
    if pd.isna(val):
        return ""
    return str(val).split()[0].strip()

def load_ground_truth(gt_path):
    """
    读取 Ground Truth，动态 sniff 列名，统一重命名为 'ID', 'True_Label'。
    严禁解析 FASTA 获取标签。
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground Truth not found: {gt_path}")
    
    df = pd.read_csv(gt_path)
    
    # Sniff ID Column (id, name, seq, sequence)
    id_col = sniff_column(df.columns, ['id', 'name', 'seq', 'sequence'])
    # Sniff Label Column (label, class, target)
    label_col = sniff_column(df.columns, ['label', 'class', 'target'])
    
    if id_col is None or label_col is None:
        raise ValueError(f"Could not sniff columns in Ground Truth. Found: {df.columns}")
    
    df = df.rename(columns={id_col: 'ID', label_col: 'True_Label'})
    
    # 🚨 ID 清洗纪律
    df['ID'] = df['ID'].apply(clean_id_value)
    
    # Ensure Label is binary (0/1)
    if df['True_Label'].dtype == object:
        df['True_Label'] = df['True_Label'].apply(lambda x: 1 if str(x).lower() in ['amp', '1', 'true', 'positive'] else 0)
    
    df['True_Label'] = df['True_Label'].astype(int)
    
    return df[['ID', 'True_Label']].drop_duplicates(subset=['ID'])

def load_model_output(model_name, config, fasta_map):
    """
    读取模型输出，动态 sniff 概率列，并将 ID 映射为序列。
    """
    path = config["path"]
    f_type = config["type"]
    sep = config["sep"]
    comment = config["comment"]
    
    if not os.path.exists(path):
        print(f"[WARN] Model output not found: {path}")
        return None
    
    try:
        if f_type == "gz":
            # 🚨 防崩溃死命令：特定 gzip 读取方式
            df = pd.read_csv(path, sep=sep, comment=comment, compression='gzip')
        else:
            # 🚨 防崩溃死命令：特定 csv 读取方式
            df = pd.read_csv(path, sep=sep)
    except Exception as e:
        print(f"[ERROR] Failed to read {model_name}: {e}")
        return None
    
    if df.empty:
        return None

    # 1. Sniff Probability Column ('prob', 'score')
    prob_col = sniff_column(df.columns, ['prob', 'score'])
    if prob_col is None:
        print(f"[WARN] No probability column found for {model_name}. Columns: {df.columns}")
        return None
    
    # 2. 🚨 致命的 ID 与序列错位陷阱处理
    # 优先寻找 'Sequence' 列，如果没有，则寻找 ID 列并通过 fasta_map 映射
    seq_col_candidate = sniff_column(df.columns, ['sequence'])
    final_id_col = None
    
    if seq_col_candidate:
        # Use existing sequence column as ID
        final_id_col = seq_col_candidate
        print(f"[INFO] Using column '{seq_col_candidate}' as ID for {model_name}")
    else:
        # Fallback: Find ID column and map via FASTA
        id_col_candidate = sniff_column(df.columns, ['id', 'access', 'seqid'])
        if id_col_candidate and fasta_map:
            print(f"[INFO] Mapping {model_name} IDs via FASTA map...")
            df['ID'] = df[id_col_candidate].astype(str).map(fasta_map)
            final_id_col = 'ID'
        else:
            print(f"[ERROR] Cannot resolve ID to Sequence for {model_name}")
            return None

    # Rename the identified ID column to 'ID' for merging
    if final_id_col != 'ID':
        df = df.rename(columns={final_id_col: 'ID'})
    
    # 🚨 ID 清洗纪律
    df['ID'] = df['ID'].apply(clean_id_value)
    
    # 3. 🚨 概率列校验与清洗
    # 必须校验 dtype == float，避开读取英文字符串标签导致的 DataFrame 灾难
    df['Pred_Prob'] = pd.to_numeric(df[prob_col], errors='coerce')
    # 检查是否全部为 NaN
    if df['Pred_Prob'].isna().all():
        print(f"[WARN] Probability column for {model_name} contains only NaNs after conversion.")
        return None
        
    # 🚨 合并后必须用 fillna(0.0) 兜底 (Applied later during merge, but good to ensure here too)
    df['Pred_Prob'] = df['Pred_Prob'].fillna(0.0)
    
    return df[['ID', 'Pred_Prob']].drop_duplicates(subset=['ID'])

def calculate_metrics(y_true, y_prob):
    """
    计算 ACC, Recall, MCC, AUROC, AUPRC。
    🚨 保存 JSON 前，一定要记得将所有指标通过 float(val) 转为 Python 原生类型！
    """
    # Threshold 0.5 for classification metrics
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = {}
    
    try:
        metrics["ACC"] = float(accuracy_score(y_true, y_pred))
    except:
        metrics["ACC"] = 0.0
        
    try:
        metrics["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except:
        metrics["Recall"] = 0.0
        
    try:
        metrics["MCC"] = float(matthews_corrcoef(y_true, y_pred))
    except:
        metrics["MCC"] = 0.0
        
    try:
        metrics["AUROC"] = float(roc_auc_score(y_true, y_prob))
    except:
        metrics["AUROC"] = 0.0
        
    try:
        metrics["AUPRC"] = float(average_precision_score(y_true, y_prob))
    except:
        metrics["AUPRC"] = 0.0
        
    return metrics

def plot_curves(results_dict, output_path):
    """
    绘制专业的 ROC 和 PR 曲线。
    🚫 禁止使用 scipy.interp，必须使用 numpy.interp
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # ROC Curve
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Chance', linewidth=2)
    
    # PR Curve
    ax_pr = axes[1]
    
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, (model_name, data) in enumerate(results_dict.items()):
        y_true = np.array(data['True_Label'])
        y_prob = np.array(data['Pred_Prob'])
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        # Interpolate to mean_fpr using numpy.interp
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        ax_roc.plot(mean_fpr, tpr_interp, color=colors[i], label=f"{model_name} (AUROC={data['Metrics']['AUROC']:.3f})", linewidth=2)
        
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(rec, prec, color=colors[i], label=f"{model_name} (AUPRC={data['Metrics']['AUPRC']:.3f})", linewidth=2)

    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax_pr.legend(loc="lower left", fontsize=10)
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved curves to {output_path}")

# ============================================================
# Main Execution
# ============================================================

def main():
    print("============================================================")
    print("[START] AMP Model Evaluation Pipeline")
    print("============================================================")
    
    # 1. Load FASTA Map (🚨 Mandatory Step for Safety)
    print("[INFO] Loading FASTA map for ID unification...")
    fasta_map = load_fasta_map(PATH_FASTA)
    
    # 2. Load Ground Truth
    print("[INFO] Loading Ground Truth...")
    try:
        gt_df = load_ground_truth(PATH_GROUND_TRUTH)
        print(f"[INFO] Ground Truth loaded: {len(gt_df)} samples. Columns: {gt_df.columns.tolist()}")
    except Exception as e:
        print(f"[ERROR] Failed to load Ground Truth: {e}")
        sys.exit(1)
    
    # 3. Process Models
    final_results = gt_df.copy() # Start with GT to ensure all IDs are present
    eval_results = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n[START] Processing Model: {model_name}")
        
        model_df = load_model_output(model_name, config, fasta_map)
        
        if model_df is None:
            print(f"[WARN] Skipping {model_name} due to loading errors.")
            eval_results[model_name] = {"Error": "Failed to load/output"}
            continue
        
        print(f"[INFO] Model output loaded: {len(model_df)} samples.")
        
        # 🚨 安全合并与算分纪律
        # 1. Merge (how='left' to keep GT structure)
        merged = pd.merge(gt_df, model_df, on='ID', how='left')
        
        # 2. Clean Prob (fillna 0.0)
        merged['Pred_Prob'] = pd.to_numeric(merged['Pred_Prob'], errors='coerce').fillna(0.0)
        
        # 3. Calculate Metrics (🚨 Must calculate BEFORE renaming)
        metrics = calculate_metrics(merged['True_Label'].values, merged['Pred_Prob'].values)
        print(f"[METRICS] {model_name}: {metrics}")
        
        # 4. Rename columns for final CSV
        prob_col_name = f"{model_name}_Prob"
        pred_col_name = f"{model_name}_Pred"
        
        merged[prob_col_name] = merged['Pred_Prob']
        merged[pred_col_name] = (merged['Pred_Prob'] >= 0.5).astype(int)
        
        # 5. Merge into final_results
        # Only add the new columns to the main table
        final_results = pd.merge(final_results, merged[['ID', prob_col_name, pred_col_name]], on='ID', how='left')
        
        # Store results for JSON and Plotting
        eval_results[model_name] = {
            "Metrics": metrics,
            "True_Label": merged['True_Label'].tolist(),
            "Pred_Prob": merged['Pred_Prob'].tolist()
        }
        
    # 4. Save Outputs
    print("\n[SAVE] Saving Results...")
    
    # Save JSON (Nested Dict Structure)
    # Convert to clean metrics dict for JSON
    json_save_dict = {}
    for m_name, data in eval_results.items():
        if "Metrics" in data:
            json_save_dict[m_name] = data["Metrics"]
        else:
            json_save_dict[m_name] = data
            
    with open(PATH_OUTPUT_JSON, 'w') as f:
        json.dump(json_save_dict, f, indent=4)
    print(f"[SUCCESS] Saved {PATH_OUTPUT_JSON}")
    
    # Save CSV
    final_results.to_csv(PATH_OUTPUT_CSV, index=False)
    print(f"[SUCCESS] Saved {PATH_OUTPUT_CSV}")
    
    # Save Plots
    # Filter only models with valid data for plotting
    plot_data = {k: v for k, v in eval_results.items() if "Metrics" in v}
    if plot_data:
        plot_curves(plot_data, PATH_OUTPUT_PNG)
    else:
        print("[WARN] No valid data to plot curves.")
    
    print("\n============================================================")
    print("[END] Evaluation Pipeline Completed")
    print("============================================================")

if __name__ == '__main__':
    main()
