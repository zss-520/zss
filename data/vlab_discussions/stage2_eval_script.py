import os
import sys
import shutil
import subprocess
import json
import gzip
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, recall_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

# ==============================================================================
# 配置与路径常量
# ==============================================================================
DATA_DIR = 'data'
GT_PATH = os.path.join(DATA_DIR, 'ground_truth.csv')
MACREL_PATH = os.path.join(DATA_DIR, 'macrel_out', 'macrel.out.prediction.gz')
AMPSCANNER_PATH = os.path.join(DATA_DIR, 'ampscanner_out.csv')
AMPNET_OUT_DIR = os.path.join(DATA_DIR, 'AmpNet_out')
AMPNET_CMD = 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_ampnet && python inference.py -i data/combined_test.fasta -o data/AmpNet_out"'

OUTPUT_JSON = 'eval_result.json'
OUTPUT_PNG = 'evaluation_curves.png'
OUTPUT_CSV = 'final_results_with_predictions.csv'

# ==============================================================================
# 辅助函数
# ==============================================================================

def clean_id_series(series):
    """
    强制清洗 ID 列：转字符串，去 '>'，取空格前第一部分，去首尾空白。
    """
    return series.astype(str).apply(lambda x: x.replace('>', '').split()[0].strip())

def load_ground_truth():
    """加载 ground truth，仅保留 id 和 label"""
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Ground Truth not found at {GT_PATH}")
    df = pd.read_csv(GT_PATH)
    # 确保列名小写标准化
    df.columns = [c.strip().lower() for c in df.columns]
    df['id'] = clean_id_series(df['id'])
    return df[['id', 'label']]

def load_macrel():
    """加载 Macrel 结果 (gzip)"""
    if not os.path.exists(MACREL_PATH):
        raise FileNotFoundError(f"Macrel output not found at {MACREL_PATH}")
    # 根据勘探报告，Macrel 输出为 gzipped csv
    df = pd.read_csv(MACREL_PATH, compression='gzip', sep='\t')
    # 标准化列名
    df.columns = [c.strip() for c in df.columns]
    # 映射到标准列名
    # 报告头：Access	Sequence	AMP_family	AMP_probability ...
    if 'Access' not in df.columns or 'AMP_probability' not in df.columns:
        raise ValueError("Macrel output columns mismatch")
    
    res = pd.DataFrame()
    res['id'] = clean_id_series(df['Access'])
    res['prob'] = pd.to_numeric(df['AMP_probability'], errors='coerce')
    return res

def load_ampscanner():
    """加载 AmpScanner 结果"""
    if not os.path.exists(AMPSCANNER_PATH):
        raise FileNotFoundError(f"AmpScanner output not found at {AMPSCANNER_PATH}")
    df = pd.read_csv(AMPSCANNER_PATH)
    df.columns = [c.strip() for c in df.columns]
    # 报告头：SeqID,Prediction_Class,Prediction_Probability,Sequence
    if 'SeqID' not in df.columns or 'Prediction_Probability' not in df.columns:
        raise ValueError("AmpScanner output columns mismatch")
    
    res = pd.DataFrame()
    res['id'] = clean_id_series(df['SeqID'])
    res['prob'] = pd.to_numeric(df['Prediction_Probability'], errors='coerce')
    return res

def load_ampnet():
    """加载 AmpNet 结果 (本次运行生成)"""
    if not os.path.exists(AMPNET_OUT_DIR):
        raise FileNotFoundError(f"AmpNet output directory not found at {AMPNET_OUT_DIR}")
    
    # 查找目录下的 CSV 文件
    csv_files = glob.glob(os.path.join(AMPNET_OUT_DIR, '*.csv'))
    if not csv_files:
        # 尝试查找 txt 或其他常见格式，但根据常规推断为 csv
        raise FileNotFoundError(f"No CSV files found in {AMPNET_OUT_DIR}")
    
    # 假设取第一个找到的 CSV，通常 inference 输出单一结果文件
    file_path = csv_files[0]
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    
    # 假设列名为 ID, Probability 或 seq_id, prob 等，需做模糊匹配或标准化
    # 这里为了代码健壮性，尝试匹配常见列名
    id_col = None
    prob_col = None
    
    for c in df.columns:
        cl = c.lower()
        if 'id' in cl or 'seq' in cl or 'access' in cl:
            id_col = c
        if 'prob' in cl or 'score' in cl:
            prob_col = c
            
    if not id_col or not prob_col:
        #  fallback 假设前两列
        id_col = df.columns[0]
        prob_col = df.columns[1]
        
    res = pd.DataFrame()
    res['id'] = clean_id_series(df[id_col])
    res['prob'] = pd.to_numeric(df[prob_col], errors='coerce')
    return res

def merge_data(gt_df, model_df, model_name):
    """
    以 GT 为左表合并，填充缺失概率为 0.0
    """
    merged = gt_df.merge(model_df, on='id', how='left')
    # 重命名概率列以便区分
    merged = merged.rename(columns={'prob': f'{model_name}_prob'})
    # 强制数值化并填充 0.0
    merged[f'{model_name}_prob'] = pd.to_numeric(merged[f'{model_name}_prob'], errors='coerce').fillna(0.0)
    return merged

def calculate_metrics(y_true, y_prob):
    """计算 sklearn 指标"""
    y_pred = (y_prob > 0.5).astype(int)
    
    # 处理边缘情况：如果只有一类，某些指标可能报错，但 AMP 评估通常正负样本都有
    try:
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError as e:
        # 如果样本单一导致无法计算
        acc = 0.0
        recall = 0.0
        mcc = 0.0
        auroc = 0.0
        auprc = 0.0
        print(f"Warning: Metric calculation issue - {e}")
        
    return {
        'ACC': float(acc),
        'Recall': float(recall),
        'MCC': float(mcc),
        'AUROC': float(auroc),
        'AUPRC': float(auprc)
    }

def plot_curves(results_dict, gt_labels):
    """
    绘制 ROC 和 PR 曲线，使用 numpy.interp
    """
    models = list(results_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 统一 FPR 网格用于 ROC 插值
    fpr_grid = np.linspace(0, 1, 100)
    
    # --- ROC Curve ---
    ax = axes[0]
    for i, model in enumerate(models):
        y_prob = results_dict[model]['prob']
        y_true = gt_labels
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # 使用 numpy.interp 进行插值 (严禁 scipy.interp)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        
        ax.plot(fpr_grid, tpr_interp, color=colors[i], label=f"{model} (AUROC={results_dict[model]['metrics']['AUROC']:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- PR Curve ---
    ax = axes[1]
    for i, model in enumerate(models):
        y_prob = results_dict[model]['prob']
        y_true = gt_labels
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        ax.plot(recall, precision, color=colors[i], label=f"{model} (AUPRC={results_dict[model]['metrics']['AUPRC']:.3f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve Comparison')
    ax.legend(loc="lower left")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

def main():
    print("=" * 50)
    print("AMP Evaluation Pipeline Started")
    print("=" * 50)
    
    # 1. 历史数据清理机制 (强制要求)
    print(f"Cleaning existing AmpNet output at {AMPNET_OUT_DIR}...")
    shutil.rmtree(AMPNET_OUT_DIR, ignore_errors=True)
    
    # 2. 强校验命令执行 (强制要求，逐字复制)
    print("开始运行 AmpNet...")
    # 确保目录存在以便后续写入，虽然 subprocess 可能会创建，但以防万一
    os.makedirs(DATA_DIR, exist_ok=True)
    
    res_AmpNet = subprocess.run(AMPNET_CMD, shell=True, capture_output=True, text=True)
    if res_AmpNet.returncode != 0:
        print(f"!!! AmpNet 真实报错日志:\n{res_AmpNet.stderr}")
        raise RuntimeError("AmpNet 预测执行失败，已阻断程序！")
    print("AmpNet 执行成功。")
    
    # 3. 数据加载与清洗
    print("Loading Ground Truth...")
    gt_df = load_ground_truth()
    
    print("Loading Macrel...")
    macrel_df = load_macrel()
    
    print("Loading AmpScanner...")
    ampscanner_df = load_ampscanner()
    
    print("Loading AmpNet...")
    ampnet_df = load_ampnet()
    
    # 4. 数据合并 (左连接兜底)
    print("Merging data...")
    # 以 GT 为基准
    merged_df = gt_df.copy()
    merged_df = merge_data(merged_df, macrel_df, 'Macrel')
    merged_df = merge_data(merged_df, ampscanner_df, 'AmpScanner')
    merged_df = merge_data(merged_df, ampnet_df, 'AmpNet')
    
    # 5. 评估计算
    print("Calculating metrics...")
    y_true = merged_df['label'].values
    models_data = {}
    eval_result = {}
    
    for model in ['Macrel', 'AmpScanner', 'AmpNet']:
        prob_col = f'{model}_prob'
        probs = merged_df[prob_col].values
        metrics = calculate_metrics(y_true, probs)
        
        models_data[model] = {
            'prob': probs,
            'metrics': metrics
        }
        eval_result[model] = metrics
        print(f"{model}: ACC={metrics['ACC']:.4f}, AUROC={metrics['AUROC']:.4f}")
    
    # 6. 保存结果
    # 6.1 JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(eval_result, f, indent=4)
    print(f"Saved metrics to {OUTPUT_JSON}")
    
    # 6.2 CSV (包含预测概率)
    # 生成二值预测列
    for model in ['Macrel', 'AmpScanner', 'AmpNet']:
        merged_df[f'{model}_pred'] = (merged_df[f'{model}_prob'] > 0.5).astype(int)
    
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to {OUTPUT_CSV}")
    
    # 6.3 绘图
    print("Plotting curves...")
    plot_curves(models_data, y_true)
    print(f"Saved curves to {OUTPUT_PNG}")
    
    print("=" * 50)
    print("Pipeline Completed Successfully")
    print("=" * 50)

if __name__ == '__main__':
    main()
