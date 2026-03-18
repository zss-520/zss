## User

现在开始会议。请先由 PI 审阅任务并提出明确要求，然后由代码工程师给出最终完整代码。

会议任务如下：
当前需要评测的计算生物学模型清单：Macrel, AMP-Scanner-v2。


【这次是第二次会议（利用评测阶段），请严格执行以下要求】：
1. PI 请先阅读下方【第一阶段超算传回的勘探报告】，明确第一次会议实际生成的文件名和数据结构。
2. 工程师必须根据勘探报告揭示的“真实文件路径和列名”编写 pandas 解析代码。绝对禁止重新运行任何模型！
3. 【数据清洗与合并（必须严格遵守）】：
   - 严禁解析 FASTA 获取标签：必须且只能读取 `data/ground_truth.csv` 获取真实标签。
   - 🚨【动态嗅探真实列名（防崩溃纪律）】：绝对禁止在代码里写死 ground truth 的列名！你必须写一段模糊匹配代码，动态寻找 ID 列（列名包含 'id', 'name', 'seq', 'sequence'）和真实标签列（列名包含 'label', 'class', 'target'），找到后统一重命名为 'ID' 和 'True_Label'。
   - 🚨【致命的 ID 与序列错位陷阱 (FASTA Mapping 必杀技)】：
     注意！`ground_truth.csv` 里的 ID 列很可能是**真实的氨基酸序列**（如 `IIIQYEGHKH`），而模型输出的标识符是 FASTA header（如 `seq_1`）。如果直接 merge 绝对全部失败！
     你【必须】写一个单独的函数读取 `data/combined_test.fasta`，构建一个字典 `fasta_id_to_seq`（键是清洗掉 `>` 符号的 FASTA ID，值是下一行的真实氨基酸序列）。
     在读取每个模型的输出 df 后，你【必须】将 df 里的 FASTA ID 映射替换为真实的序列字符串！只有把两边的键都统一成【氨基酸序列文本】后，才能和 `ground_truth.csv` 进行 merge！
   - 🚨【安全合并与算分纪律 (防 Key 报错)】：
     1. 遍历每个模型时，只提取该模型的 ['ID', 'Pred_Prob'] 两列与 `ground_truth` 进行 `how='left'` 的 merge。
     2. merge 后，立即使用 `pd.to_numeric(..., errors='coerce').fillna(0.0)` 将 `Pred_Prob` 里的空值填为 0.0。
     3. 接着，基于 `Pred_Prob` 生成标签，并计算出所有 metrics 字典！**必须先算分！**
     4. 算完分之后，你**再**把当前模型的 `Pred_Prob` 和标签列重命名为带有模型前缀的名字（如 `Macrel_Prob`, `Macrel_Pred`），然后合并到最终保存 CSV 的 `final_results` 大表里。
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
### NEW SESSION STARTED: eval_script.py ###


============================================================
[START] Model: Macrel
[CMD] bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"
[OUT] data/Macrel_out
[WARN] Initial execution failed (Return Code: 1)
[DEBUG] Error Output Snippet: Output folder [data/Macrel_out] already exists

[HEALING] Detected 'exists' error. Removing directory and retrying without pre-mkdir...
[SUCCESS] Model Macrel recovered and completed successfully.

============================================================
[START] Model: AMP-Scanner-v2
[CMD] bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv"
[OUT] data/AMP-Scanner-v2_out
[SUCCESS] Model AMP-Scanner-v2 completed successfully.

============================================================
[STAGE 1 EXPLORATION REPORT]
============================================================

[DIR] /share/home/zhangss/vlab_workspace/data/AMP-Scanner-v2_out
  [FILE] /share/home/zhangss/vlab_workspace/data/AMP-Scanner-v2_out/ampscanner_out.csv
  [HEADER CONTENT]:
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

----------------------------------------

[DIR] /share/home/zhangss/vlab_workspace/data/Macrel_out
  [FILE] /share/home/zhangss/vlab_workspace/data/Macrel_out/README.md
  [HEADER CONTENT]:
    If you find Macrel useful, please cite:
    
    > Santos-Junior, C.D. et al. Macrel: antimicrobial peptide screening in
    > genomes and metagenomes. The PeerJ 8:e10555
    > https://doi.org/10.7717/peerj.10555
    
    For more information, please read [the macrel
    documentation](https://macrel.readthedocs.io) and use the [AMPSphere mailing
    list](https://groups.google.com/g/ampsphere-users) for questions.
    

  [FILE] /share/home/zhangss/vlab_workspace/data/Macrel_out/macrel.out.prediction.gz
  [HEADER CONTENT]:
    # Prediction from macrel v1.6.0
    Access | Sequence | AMP_family | is_AMP | AMP_probability | Hemolytic | Hemolytic_probability
    93 | IIIQYEGHKH | CLP | True | 0.812 | NonHemo | 0.119
    94 | AAGMGFFGAR | CLP | True | 0.693 | Hemo | 0.515
    95 | KQTENLADTY | ALP | False | 0.178 | NonHemo | 0.000
    96 | FIMDLLGKIF | CLP | False | 0.337 | NonHemo | 0.396
    97 | GILDAITGLL | ALP | True | 0.941 | NonHemo | 0.436
    98 | LRPAILVRIK | CLP | True | 0.693 | Hemo | 0.772
    99 | GLLKRIKTLL | CLP | True | 0.941 | Hemo | 0.970
    100 | NAGSLLSGWG | CLP | False | 0.228 | NonHemo | 0.208

----------------------------------------
--------------------------------

## PI

# PI 审阅与任务重申 (PI Review & Task Restatement)

各位工程师，注意。这是第二次会议，我们处于**评测阶段**。根据第一阶段超算传回的勘探报告，模型已经运行完毕，我们现在的任务是**数据清洗、合并与评测**。

我仔细审阅了勘探报告，发现几个关键的数据结构特征，必须在代码中严格执行，否则评测结果将全部作废。请仔细阅读以下**硬性要求**：

1.  **禁止重新运行模型**：
    *   代码中绝对不允许出现调用 `macrel` 或 `amp_scanner_v2_predict_tf1.py` 的命令。
    *   直接读取勘探报告确认的输出文件：
        *   Macrel: `data/Macrel_out/macrel.out.prediction.gz`
        *   AMP-Scanner-v2: `data/AMP-Scanner-v2_out/ampscanner_out.csv`

2.  **🚨 核心陷阱：ID 与序列的对齐 (ID-Sequence Alignment)**：
    *   勘探报告显示，模型输出中包含 `SeqID`/`Access` (数字，如 93, 94) 和 `Sequence` (氨基酸序列)。
    *   `ground_truth.csv` 的 ID 列极大概率是**氨基酸序列**。
    *   **强制要求**：你必须编写 `load_fasta_map` 函数读取 `data/combined_test.fasta`。在处理模型 DataFrame 时，**必须**利用该映射表，将模型输出中的 ID 列（如 `SeqID` 或 `Access`）转换为真实的氨基酸序列，并将该列统一重命名为 `ID`，以此作为与 Ground Truth 合并的唯一键 (Key)。**严禁直接使用数字 ID 进行 merge！**

3.  **🚨 动态列名嗅探 (Dynamic Column Sniffing)**：
    *   **Ground Truth**: 禁止写死列名。必须遍历列名，寻找包含 `['id', 'name', 'seq', 'sequence']` 的列作为 ID，寻找包含 `['label', 'class', 'target']` 的列作为真实标签。统一重命名为 `ID` 和 `True_Label`。
    *   **模型输出**: 必须寻找包含 `['prob', 'score']` 的列作为预测概率。
    *   **文件读取**:
        *   Macrel (`.gz`): 必须使用 `compression='gzip'`, `sep='\t'`, `comment='#'`。
        *   AMP-Scanner (`.csv`): 必须使用 `sep=','`。
        *   **禁止**使用 `low_memory=False` 或 `engine='python'` 导致引擎冲突。

4.  **🚨 计算与保存纪律 (Calculation & Saving Discipline)**：
    *   **顺序**：先 Merge -> 清洗概率 (fillna 0.0) -> **计算 Metrics** -> 重命名列 -> 合并到 `final_results`。
    *   **指标**: ACC, Recall, MCC, AUROC, AUPRC (sklearn)。
    *   **绘图**: `evaluation_curves.png` 必须专业 (dpi=300)，使用 `numpy.interp` (禁止 `scipy.interp`)。
    *   **输出**: `eval_result.json` (嵌套字典), `final_results_with_predictions.csv`。

5.  **环境兼容性**:
    *   Python 3.11+。
    *   设置 `matplotlib.use('Agg')`。

工程师，请根据以上要求，编写最终的 `eval_pipeline.py` 代码。确保代码健壮，能够处理列名微调的情况，并严格遵循数据清洗流程。

---

# 工程师最终代码 (Engineer Final Code)

```python
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
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "ground_truth.csv")
FASTA_PATH = os.path.join(BASE_DIR, "combined_test.fasta")
OUTPUT_JSON = "eval_result.json"
OUTPUT_CSV = "final_results_with_predictions.csv"
OUTPUT_PNG = "evaluation_curves.png"

# Model Output Paths based on Exploration Report
MODEL_CONFIGS = {
    "Macrel": {
        "path": os.path.join(BASE_DIR, "Macrel_out", "macrel.out.prediction.gz"),
        "type": "gz"
    },
    "AMP-Scanner-v2": {
        "path": "data/AMP-Scanner-v2_out/ampscanner_out.csv", # Using relative path consistent with BASE_DIR logic if needed, but report says absolute. Adjusting to match script context.
        "type": "csv"
    }
}
# Correcting paths to be relative to script execution assuming script runs in workspace root
MODEL_CONFIGS["AMP-Scanner-v2"]["path"] = os.path.join(BASE_DIR, "AMP-Scanner-v2_out", "ampscanner_out.csv")

# ============================================================
# Helper Functions
# ============================================================

def load_fasta_map(fasta_file):
    """
    🚨 FASTA Mapping 必杀技：
    读取 FASTA 文件，构建 {Header_ID: Sequence} 字典。
    用于将模型输出的 ID 映射为真实氨基酸序列。
    """
    fasta_map = {}
    if not os.path.exists(fasta_file):
        print(f"[WARN] FASTA file not found: {fasta_file}")
        return fasta_map
    
    current_id = None
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # Clean header: remove '>' and split by space to get ID
                current_id = line[1:].split()[0]
            else:
                if current_id is not None:
                    # In case sequence is multi-line, append (though usually single line in test sets)
                    if current_id in fasta_map:
                        fasta_map[current_id] += line
                    else:
                        fasta_map[current_id] = line
    return fasta_map

def sniff_column(columns, keywords):
    """
    🚨 动态嗅探真实列名：
    寻找包含 keywords 中任意一个词的列名。
    """
    for col in columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None

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
    
    # Ensure ID is string (Sequence)
    df['ID'] = df['ID'].astype(str).str.strip()
    # Ensure Label is binary (0/1)
    # Assuming labels might be 'AMP'/'Non-AMP' or 1/0. Convert to 0/1.
    if df['True_Label'].dtype == object:
        # Map common strings to 1 (AMP) and 0 (Non-AMP)
        df['True_Label'] = df['True_Label'].apply(lambda x: 1 if str(x).lower() in ['amp', '1', 'true', 'positive'] else 0)
    
    return df[['ID', 'True_Label']]

def load_model_output(model_name, config, fasta_map):
    """
    读取模型输出，动态 sniff 概率列，并将 ID 映射为序列。
    """
    path = config["path"]
    f_type = config["type"]
    
    if not os.path.exists(path):
        print(f"[WARN] Model output not found: {path}")
        return None
    
    try:
        if f_type == "gz":
            # 🚨 防崩溃死命令：特定 gzip 读取方式
            df = pd.read_csv(path, sep='\t', comment='#', compression='gzip')
        else:
            # 🚨 防崩溃死命令：特定 csv 读取方式
            df = pd.read_csv(path, sep=',')
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
    # 策略：优先寻找 'Sequence' 列，如果没有，则寻找 ID 列并通过 fasta_map 映射
    # 勘探报告显示两者都有 'Sequence' 列，但为了严格遵守 PI 指令 "将 df 里的 FASTA ID 映射替换为真实的序列字符串"
    # 我们检查是否有类似 'SeqID', 'Access', 'ID' 的列，并尝试映射。
    # 如果已有 'Sequence' 列，直接用它作为 ID 最安全，但为了体现 mapping 逻辑：
    
    final_id_col = None
    
    # Check if direct Sequence column exists
    seq_col_candidate = sniff_column(df.columns, ['sequence'])
    
    if seq_col_candidate:
        # Use existing sequence column as ID
        final_id_col = seq_col_candidate
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
    
    # Ensure ID is string
    df['ID'] = df['ID'].astype(str).str.strip()
    
    # Extract Prob
    df['Pred_Prob'] = pd.to_numeric(df[prob_col], errors='coerce').fillna(0.0)
    
    return df[['ID', 'Pred_Prob']]

def calculate_metrics(y_true, y_prob):
    """
    计算 ACC, Recall, MCC, AUROC, AUPRC。
    """
    # Threshold 0.5 for classification metrics
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Handle edge cases for MCC/Recall
    try:
        acc = accuracy_score(y_true, y_pred)
    except:
        acc = 0.0
        
    try:
        recall = recall_score(y_true, y_pred, zero_division=0)
    except:
        recall = 0.0
        
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0
        
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.0
        
    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0
        
    return {
        "ACC": float(acc),
        "Recall": float(recall),
        "MCC": float(mcc),
        "AUROC": float(auroc),
        "AUPRC": float(auprc)
    }

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
    
    # We need to store curves for interpolation to align points if needed, 
    # but for publication quality, plotting raw curves is often better if sampled densely.
    # However, to compare at specific FPR points, interpolation is used.
    # PI Requirement: Use numpy.interp.
    
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, (model_name, data) in enumerate(results_dict.items()):
        y_true = data['True_Label']
        y_prob = data['Pred_Prob']
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        # Interpolate to mean_fpr
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        ax_roc.plot(mean_fpr, tpr_interp, color=colors[i], label=f"{model_name} (AUROC={data['Metrics']['AUROC']:.3f})", linewidth=2)
        
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        # PR curves usually plotted as rec vs prec
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
    
    # 1. Load FASTA Map (🚨 Mandatory Step)
    print("[INFO] Loading FASTA map for ID unification...")
    fasta_map = load_fasta_map(FASTA_PATH)
    if not fasta_map:
        print("[WARN] FASTA map is empty. Relying on direct Sequence columns if available.")
    
    # 2. Load Ground Truth
    print("[INFO] Loading Ground Truth...")
    try:
        gt_df = load_ground_truth(GROUND_TRUTH_PATH)
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
        
        # Clean up temp columns for next iteration
        # (Not strictly needed as we merge back to final_results, but good practice)
        
    # 4. Save Outputs
    print("\n[SAVE] Saving Results...")
    
    # Save JSON (Remove large lists for clean JSON if needed, but PI asked for nested dict structure)
    # To keep JSON readable, we might store only metrics in a summary, but keeping data for reproducibility.
    # Let's create a summary JSON for metrics and a full one for data if needed. 
    # PI Requirement: "nested dictionary structure eval_result.json".
    # We will save metrics primarily to keep it clean, or include data if not too large.
    # Given "evaluation" context, metrics are key. I will save metrics + sample count.
    
    json_save_dict = {}
    for m_name, data in eval_results.items():
        if "Metrics" in data:
            json_save_dict[m_name] = data["Metrics"]
        else:
            json_save_dict[m_name] = data
            
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_save_dict, f, indent=4)
    print(f"[SUCCESS] Saved {OUTPUT_JSON}")
    
    # Save CSV
    final_results.to_csv(OUTPUT_CSV, index=False)
    print(f"[SUCCESS] Saved {OUTPUT_CSV}")
    
    # Save Plots
    # Filter only models with valid data for plotting
    plot_data = {k: v for k, v in eval_results.items() if "Metrics" in v}
    if plot_data:
        plot_curves(plot_data, OUTPUT_PNG)
    else:
        print("[WARN] No valid data to plot curves.")
    
    print("\n============================================================")
    print("[END] Evaluation Pipeline Completed")
    print("============================================================")

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

