## User

现在开始会议。请先由 PI 审阅任务并提出明确要求，然后由代码工程师给出最终完整代码。

会议任务如下：
当前需要评测的计算生物学模型清单：Macrel, AMP-Scanner-v2。


【这次是第二次会议（利用评测阶段），请严格执行以下要求】：
1. PI 请先阅读下方【第一阶段超算传回的勘探报告】，明确第一次会议实际生成的文件名和数据结构。
2. 工程师必须根据勘探报告揭示的“真实文件路径和列名”编写 pandas 解析代码。绝对禁止重新运行任何模型！
3. 【数据清洗与合并（必须严格遵守）】：
   - 严禁解析 FASTA 获取标签：必须且只能读取 `data/ground_truth.csv` 获取真实标签。
   - 强清洗 ID：在 merge 之前，必须对所有表的 ID 列转换为字符串，剔除 '>' 符号，并使用 `apply(lambda x: str(x).split()[0].strip())` 只保留干净 ID。
   - 🚨【安全合并与算分纪律 (防 Key 报错)】：
     1. 遍历每个模型时，只提取该模型的 ['ID', 'Pred_Prob'] 两列与 `ground_truth` 进行 `how='left'` 的 merge。
     2. merge 后，立即使用 `pd.to_numeric(..., errors='coerce').fillna(0.0)` 将 `Pred_Prob` 里的空值填为 0.0。
     3. 接着，基于 `Pred_Prob` 动态寻找最优阈值生成预测标签，并计算出所有 metrics 字典！**必须先算分！**
     4. 算完分之后，你**再**把当前模型的 `Pred_Prob` 和标签列重命名为带有模型前缀的名字（如 `Macrel_Prob`, `Macrel_Pred`），然后合并到那个最终的用于保存 CSV 的 `final_results` 大表里！绝对不允许先重命名再去算分，会导致 Key 找不到！
   - 统一使用浮点数概率值（`prob > 0.5`）生成最终的 0 和 1 预测标签。
4. 必须使用 sklearn 计算 ACC, Recall, MCC, AUROC, AUPRC，并将结果保存为嵌套字典结构的 `eval_result.json`。
5. 必须保存 `evaluation_curves.png`（需声明 `import matplotlib; matplotlib.use('Agg')`）和 `final_results_with_predictions.csv`。
6. 🚫【代码兼容性极速警告】：执行环境是 Python 3.11+。绝对禁止使用 `from scipy import interp`！绘制 ROC/PR 曲线插值必须使用 `numpy.interp`！
7. 要求做的图表要极具专业性并且全面，满足能在顶刊顶会发表的要求。全面对比出每个模型在每个指标上的表现差异。

【🔴 极度重要的数据嗅探与 Pandas 提取纪律 (Dynamic Data Extraction)】
当你编写读取模型输出的 Python 脚本时，必须具备“极客级”的解析能力：
1. **多格式兼容**：使用 `glob.glob` 寻找输出目录下最大的 `.csv`, `.tsv`, `.txt`, `.out`, 甚至 `.gz` 文件！
2. **解压与注释跳过（终极死纪律）**：
   - 🚨 **防崩溃死命令**：绝对禁止在 `pd.read_csv` 中使用 `low_memory=False` 或 `engine='python'`！这会引发致命的引擎冲突！默认使用 C 引擎即可！
   - 如果文件是 `.gz` 结尾（如 Macrel），你必须严格使用这套代码去读，千万不要乱改：`pd.read_csv(filepath, sep='\t', comment='#', compression='gzip')`
   - 如果文件是 `.csv` 结尾（如 AMP-Scanner），你必须严格使用这套代码去读：`pd.read_csv(filepath, sep=',')`
3. **模糊列名锁定**：读取文件后，使用类似 `[col for col in df.columns if 'prob' in col.lower() or 'score' in col.lower()]` 的逻辑锁定概率列。如果找到了正确的概率列，必须用 `pd.to_numeric(..., errors='coerce')` 将其强转为浮点数。
4. **优雅降级**：只有当所有读取方式都失败时，才赋值为 `None`。

以下是第一阶段传回的勘探报告：
--------------------------------
=== Stage 1 Observation Log ===
Start Time: Wed Mar 18 13:38:22 CST 2026

--- Model Execution: Macrel ---
[ERROR] Initial execution failed for Macrel. Returncode: 1
[DEBUG] Stdout/Stderr snippet: Output folder [data/Macrel_out] already exists

[HEALING] Detected 'exists' error. Removing data/Macrel_out and retrying...
[SUCCESS] Macrel healed successfully after rmtree retry.
--- Model Execution: AMP-Scanner-v2 ---
[SUCCESS] AMP-Scanner-v2 executed successfully.

--- Stage 1 Exploration Report ---

Total files found: 5

File Path: data/Macrel_out/README.md
Head Content (first 10 lines):
  If you find Macrel useful, please cite:
  
  > Santos-Junior, C.D. et al. Macrel: antimicrobial peptide screening in
  > genomes and metagenomes. The PeerJ 8:e10555
  > https://doi.org/10.7717/peerj.10555
  
  For more information, please read [the macrel
  documentation](https://macrel.readthedocs.io) and use the [AMPSphere mailing
  list](https://groups.google.com/g/ampsphere-users) for questions.
  
--------------------------------------------------
File Path: data/Macrel_out/macrel.out.prediction.gz
Head Content (first 10 lines):
  # Prediction from macrel v1.6.0
  Access	Sequence	AMP_family	AMP_probability	Hemolytic	Hemolytic_probability
  93	IIIQYEGHKH	CLP	0.812	NonHemo	0.119
  94	AAGMGFFGAR	CLP	0.693	Hemo	0.515
  97	GILDAITGLL	ALP	0.941	NonHemo	0.436
  98	LRPAILVRIK	CLP	0.693	Hemo	0.772
  99	GLLKRIKTLL	CLP	0.941	Hemo	0.970
  101	SRSELIVHQR	CLP	0.743	NonHemo	0.020
  102	LLGGLLQSLL	CLP	0.802	NonHemo	0.396
  103	CTHPLDCSN	CDP	0.644	NonHemo	0.000
--------------------------------------------------
File Path: data/AMP-Scanner-v2_out/ampscanner_out.csv
Head Content (first 10 lines):
  SeqID,Prediction_Class,Prediction_Probability,Sequence
  93,Non-AMP,0.0389,IIIQYEGHKH
  94,AMP,0.5786,AAGMGFFGAR
  95,Non-AMP,0.0126,KQTENLADTY
  96,AMP,0.9189,FIMDLLGKIF
  97,AMP,0.9534,GILDAITGLL
  98,AMP,0.7697,LRPAILVRIK
  99,AMP,0.9972,GLLKRIKTLL
  100,AMP,0.8792,NAGSLLSGWG
  101,Non-AMP,0.0092,SRSELIVHQR
--------------------------------------------------
File Path: data/ground_truth.csv
Head Content (first 10 lines):
  Sequence,label
  IIIQYEGHKH,1
  AAGMGFFGAR,1
  KQTENLADTY,1
  FIMDLLGKIF,1
  GILDAITGLL,1
  LRPAILVRIK,1
  GLLKRIKTLL,1
  NAGSLLSGWG,1
  SRSELIVHQR,1
--------------------------------------------------
File Path: data/combined_test.fasta
Head Content (first 10 lines):
  >93
  IIIQYEGHKH
  >94
  AAGMGFFGAR
  >95
  KQTENLADTY
  >96
  FIMDLLGKIF
  >97
  GILDAITGLL
--------------------------------------------------
--------------------------------

## PI

# PI 审阅与任务指令 (PI Review & Instructions)

各位工程师，注意。根据第一阶段超算传回的【勘探报告】，我们已经掌握了模型输出的真实文件结构。本次任务的核心是**基于现有文件进行严格评测**，严禁重新运行模型。

请仔细阅读以下基于勘探报告得出的**硬性执行要求**，代码必须完全符合以下规范：

### 1. 文件路径与格式确认 (基于勘探报告)
*   **Ground Truth**: `data/ground_truth.csv`。
    *   关键列：`Sequence` (作为 ID), `label` (真实标签)。
*   **Macrel**: `data/Macrel_out/macrel.out.prediction.gz`。
    *   格式：Gzip 压缩，Tab 分隔 (`sep='\t'`)，含注释行 (`comment='#'`)。
    *   关键列：`Sequence` (作为 ID), `AMP_probability` (概率)。
    *   **注意**：必须使用 `compression='gzip'` 读取，禁止解压到临时文件。
*   **AMP-Scanner-v2**: `data/AMP-Scanner-v2_out/ampscanner_out.csv`。
    *   格式：CSV (`sep=','`)。
    *   关键列：`Sequence` (作为 ID), `Prediction_Probability` (概率)。

### 2. 数据清洗与合并纪律 (Data Cleaning & Merge Discipline)
*   **ID 统一**：所有表格必须使用 `Sequence` 列作为 merge 的 Key。严禁使用 `Access` 或 `SeqID` 数字索引，因为 Ground Truth 仅包含序列字符串。
*   **强清洗 (Strong Clean)**：在 merge 之前，必须对所有 ID 列执行：`str(x).split()[0].strip()` 并剔除 `>` 符号。即使当前数据看似干净，也必须执行此步骤以防万一。
*   **安全合并流程 (防 Key 报错)**：
    1.  加载 Ground Truth 作为主表。
    2.  遍历模型，提取 `[ID, Prob]`。
    3.  `left` merge 到主表。
    4.  `Pred_Prob` 空值填 0.0。
    5.  **立即算分**：基于当前的 `Pred_Prob` 和 `label` 计算 ACC, Recall, MCC, AUROC, AUPRC。
    6.  **算分后重命名**：将 `Pred_Prob` 重命名为 `Model_Prob`，生成 `Model_Pred` (阈值 0.5)，再合并到最终 CSV 表。
    7.  **禁止**先重命名再算分，这会导致后续模型 merge 时找不到统一的 `Pred_Prob` 列。

### 3. 技术指标与兼容性 (Metrics & Compatibility)
*   **Python 3.11+**：禁止使用已废弃的 `scipy.interp`。ROC/PR 曲线插值必须使用 `numpy.interp`。
*   **Pandas 引擎**：禁止使用 `low_memory=False` 或 `engine='python'`。
*   **指标**：必须包含 MCC (`matthews_corrcoef`)。
*   **图表**：`evaluation_curves.png` 必须达到顶刊发表标准（高分辨率，清晰图例，网格线）。

### 4. 输出交付物
*   `eval_result.json`: 嵌套字典，包含每个模型的所有指标。
*   `final_results_with_predictions.csv`: 包含所有模型的预测概率和二元标签。
*   `evaluation_curves.png`: ROC 和 PR 曲线对比图。

---

# 工程师代码实现 (Engineer Code Implementation)

```python
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
```

## User

下面请严格根据 PI 的要求输出最终结果。

【这是第二次会议，必须严格基于第一次会议的标准化输出进行评测】
1. 优先复用第一次会议已经生成的标准化输出文件。
2. 如果第一次会议产物中已包含模型预测结果，则禁止重新运行模型命令。
3. 本次必须产出：eval_result.json、evaluation_curves.png、final_results_with_predictions.csv。
4. 本次要计算最终评测指标，并输出严格格式结果。

【PI_PROMPT 原文】
你是一位顶级的计算生物学 PI。当前的评测任务是：【{task_desc}】。
请引导 MLOps 工程师写出评测的 Python 脚本和 Slurm 提交脚本。特别提醒工程师：
1. 严禁编写从 FASTA 提取标签的代码，必须直接读取 ground_truth.csv！
2. 合并前 ID 必须进行 `split()[0].strip()` 强清洗，合并后必须用 `fillna(0.0)` 兜底。
3. 寻找概率列时必须校验 `dtype == float`，避开读取英文字符串标签导致的 DataFrame 灾难！
4. 保存 JSON 前，一定要记得将所有指标通过 `float(val)` 转为 Python 原生类型，防止 Numpy 序列化报错！同时规定 JSON 的双层字典嵌套结构。
5. 评测脚本必须是一个完整可运行的 Python 脚本，而不能只有片段。

【CODER_PROMPT 原文】
你是一位精通超算的 MLOps 工程师。根据 PI 的要求编写评测代码。

【代码输出的格式底线（极其重要）】：
1. 你提供的 Python 代码必须是一个**单一的、完全完整的脚本**，全部包含在一个 ` ```python ... ``` ` 代码块中！绝对不允许把代码切分成多个小块。
2. 脚本必须包含所有的 `import`，必须有明确定义的 `def main():` 函数，并以 `if __name__ == '__main__':` 作为程序入口去调用 `main()`。
3. 🚨 **基础语法死纪律**：**绝对不允许在定义变量名时包含空格！** 例如 `PATH Ground Truth = ...` 是极其低级的致命语法错误！所有的变量名必须使用下划线连接（如 `PATH_GROUND_TRUTH`）。

【评测功能要求】：
1. 必须保存严格嵌套结构的 `eval_result.json`，以及 `evaluation_curves.png` 和 `final_results_with_predictions.csv`。
2. 必须提供 `run_eval.sh`，Bash 脚本必须写在单独的 ` ```bash ... ``` ` 块中，并参考以下模板：
   ```bash
   #!/bin/bash
   #SBATCH -J amp_eval
   #SBATCH -N 1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=4
   #SBATCH --gres=gpu:1
   #SBATCH -p gpu
   #SBATCH -o amp_eval.%j.out
   #SBATCH -e amp_eval.%j.err

   cd /share/home/zhangss/vlab_workspace
   source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
   conda activate eval_env
   python eval_script.py
   echo "finish"
   ```

## Code Engineer

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMP Model Evaluation Pipeline (Stage 2)
Strictly follows PI requirements based on Stage 1 Exploration Report.
No model re-execution. Uses existing output files.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, matthews_corrcoef,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ==========================================
# Configuration Constants
# ==========================================
DATA_DIR = "data"
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
OUTPUT_JSON = "eval_result.json"
OUTPUT_CSV = "final_results_with_predictions.csv"
OUTPUT_PNG = "evaluation_curves.png"

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
    Compliant with PI Requirement.
    """
    return series.apply(lambda x: str(x).replace('>', '').split()[0].strip())

def find_model_output_file(directory, extensions):
    """
    Dynamic Data Extraction: Find the most likely output file using glob.
    """
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        files = glob.glob(pattern)
        valid_files = [f for f in files if not f.endswith('.md') and not f.endswith('.log')]
        if valid_files:
            return max(valid_files, key=os.path.getsize)
    return None

def load_ground_truth():
    """
    Load Ground Truth strictly from CSV.
    """
    if not os.path.exists(GROUND_TRUTH_PATH):
        raise FileNotFoundError(f"Ground Truth not found at {GROUND_TRUTH_PATH}")
    
    df = pd.read_csv(GROUND_TRUTH_PATH, sep=',')
    
    # Identify columns based on Report: Sequence, label
    seq_col = [c for c in df.columns if 'sequence' in c.lower()]
    lab_col = [c for c in df.columns if 'label' in c.lower() or 'truth' in c.lower()]
    
    if not seq_col or not lab_col:
        raise ValueError("Cannot identify Sequence or label columns in ground_truth.csv")
        
    df = df.rename(columns={seq_col[0]: 'Sequence', lab_col[0]: 'label'})
    df['ID'] = clean_id_series(df['Sequence'])
    
    # Ensure label is numeric
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    return df[['ID', 'label']].set_index('ID')

def load_macrel_output(directory):
    """
    Load Macrel output based on Report: .gz, tab-separated, comment='#'.
    """
    filepath = find_model_output_file(directory, ['.gz', '.csv', '.tsv', '.out'])
    if not filepath:
        return None
    
    try:
        if filepath.endswith('.gz'):
            df = pd.read_csv(filepath, sep='\t', comment='#', compression='gzip')
        else:
            df = pd.read_csv(filepath, sep='\t', comment='#')
            
        # Identify Probability Column (AMP_probability)
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
        
        # Force numeric probability
        df['Pred_Prob'] = pd.to_numeric(df[prob_col], errors='coerce')
        
        # Check dtype as per PI requirement
        if not np.issubdtype(df['Pred_Prob'].dtype, np.number):
            return None
            
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
        
    try:
        df = pd.read_csv(filepath, sep=',')
        
        # Identify Probability Column (Prediction_Probability)
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
        
        # Force numeric probability
        df['Pred_Prob'] = pd.to_numeric(df[prob_col], errors='coerce')
        
        # Check dtype as per PI requirement
        if not np.issubdtype(df['Pred_Prob'].dtype, np.number):
            return None
            
        return df[['ID', 'Pred_Prob']]
        
    except Exception as e:
        print(f"Error loading AMP-Scanner: {e}", file=sys.stderr)
        return None

def calculate_metrics(y_true, y_prob):
    """
    Calculate all required metrics using sklearn.
    Returns native Python floats for JSON serialization.
    """
    y_pred = (y_prob > 0.5).astype(int)
    
    try:
        acc = float(accuracy_score(y_true, y_pred))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        acc, recall, mcc = 0.0, 0.0, 0.0
        
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = float(auc(fpr, tpr))
    except Exception:
        auroc = 0.5
        
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = 0.0
        
    return {
        "ACC": acc,
        "Recall": recall,
        "MCC": mcc,
        "AUROC": auroc,
        "AUPRC": auprc
    }

def plot_curves(model_results, output_path):
    """
    Plot ROC and PR curves with publication quality.
    Uses numpy.interp if interpolation is needed (though sklearn auc handles raw).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))
    
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    
    ax_pr = axes[1]
    
    for i, (model_name, data) in enumerate(model_results.items()):
        y_true = data['y_true']
        y_prob = data['y_prob']
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[i], lw=2, label=f'{model_name} (AUC={roc_auc:.3f})')
        
        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax_pr.plot(recall, precision, color=colors[i], lw=2, label=f'{model_name} (AP={pr_auc:.3f})')
        
    # Styling ROC
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    ax_roc.legend(loc="lower right", fontsize=12)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    ax_roc.tick_params(axis='both', which='major', labelsize=12)
    
    # Styling PR
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax_pr.set_title('Precision-Recall Curve (PR)', fontsize=16, fontweight='bold')
    ax_pr.legend(loc="lower left", fontsize=12)
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    ax_pr.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# Main Execution Logic
# ==========================================

def main():
    print("=== Starting AMP Evaluation Pipeline (Stage 2) ===")
    
    # 1. Load Ground Truth
    print("Loading Ground Truth...")
    try:
        gt_df = load_ground_truth()
    except Exception as e:
        print(f"Critical Error loading Ground Truth: {e}", file=sys.stderr)
        sys.exit(1)
        
    if gt_df.empty:
        raise ValueError("Ground Truth is empty after loading.")
    
    # Initialize Final Results DataFrame
    final_results = gt_df.reset_index()
    
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
        
        model_df = model_cfg["loader"](model_cfg["dir"])
        
        if model_df is None or model_df.empty:
            print(f"Warning: No valid data found for {model_name}. Skipping.")
            eval_results[model_name] = {"Error": "No data parsed"}
            continue
        
        # --- SAFETY MERGE & SCORING DISCIPLINE ---
        # 3.1 Merge with Ground Truth (Left Join on ID)
        merged = final_results[['ID', 'label']].merge(model_df, on='ID', how='left')
        
        # 3.2 Fill NaN Probabilities with 0.0
        merged['Pred_Prob'] = pd.to_numeric(merged['Pred_Prob'], errors='coerce').fillna(0.0)
        
        # 3.3 Calculate Metrics IMMEDIATELY (Before Renaming)
        y_true = merged['label'].values
        y_prob = merged['Pred_Prob'].values
        
        metrics = calculate_metrics(y_true, y_prob)
        eval_results[model_name] = metrics
        plot_data[model_name] = {'y_true': y_true, 'y_prob': y_prob}
        
        print(f"  Metrics for {model_name}: {metrics}")
        
        # 3.4 Rename Columns for Final CSV Accumulation
        prob_col_name = f"{model_name}_Prob"
        pred_col_name = f"{model_name}_Pred"
        
        merged[prob_col_name] = merged['Pred_Prob']
        merged[pred_col_name] = (merged['Pred_Prob'] > 0.5).astype(int)
        
        # 3.5 Merge into Final Results Table
        final_results = final_results.merge(
            merged[['ID', prob_col_name, pred_col_name]], 
            on='ID', 
            how='left'
        )
        
    # 4. Save Outputs
    print("Saving Outputs...")
    
    # 4.1 Save JSON (Ensure native types)
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

if __name__ == '__main__':
    main()
```

```bash
#!/bin/bash
#SBATCH -J amp_eval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o amp_eval.%j.out
#SBATCH -e amp_eval.%j.err

cd /share/home/zhangss/vlab_workspace
source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
conda activate eval_env
python eval_script.py
echo "finish"
```

