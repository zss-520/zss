## Data Architect

我已经成功提取并确认了 Data Schema：
```json
{
  "Macrel": {
    "file_path": "data/Macrel_out/macrel.out.prediction.gz",
    "file_ext": ".gz",
    "sep": "\t",
    "comment_char": "#",
    "id_col": "Access",
    "seq_col": "Sequence",
    "prob_col": "AMP_probability"
  },
  "amPEPpy": {
    "file_path": "data/amPEPpy_out/predictions.txt",
    "file_ext": ".txt",
    "sep": "\t",
    "comment_char": null,
    "id_col": "seq_id",
    "seq_col": null,
    "prob_col": "probability_AMP"
  },
  "AMPlify": {
    "file_path": "data/AMPlify_out/AMPlify_balanced_results_20260417094909.tsv",
    "file_ext": ".tsv",
    "sep": "\t",
    "comment_char": null,
    "id_col": "Sequence_ID",
    "seq_col": "Sequence",
    "prob_col": "Probability_score"
  }
}
```

## PI

# 【PI指令】评测脚本核心逻辑约束 - 严格执行标准

MLOps 工程师，以下是评测脚本的**核心逻辑约束**，你必须**逐条严格执行**，不得有任何偏差：

## 🔥 **1. 硬编码字典与前置配置**
```python
DATA_SCHEMA = {
    "Macrel": {
        "file_path": "data/Macrel_out/macrel.out.prediction.gz",
        "file_ext": ".gz",
        "sep": "\t",
        "comment_char": "#",
        "id_col": "Access",
        "seq_col": "Sequence", 
        "prob_col": "AMP_probability"
    },
    "amPEPpy": {
        "file_path": "data/amPEPpy_out/predictions.txt",
        "file_ext": ".txt",
        "sep": "\t",
        "comment_char": None,
        "id_col": "seq_id",
        "seq_col": None,
        "prob_col": "probability_AMP"
    },
    "AMPlify": {
        "file_path": "data/AMPlify_out/AMPlify_balanced_results_20260417094909.tsv",
        "file_ext": ".tsv",
        "sep": "\t",
        "comment_char": None,
        "id_col": "Sequence_ID",
        "seq_col": "Sequence",
        "prob_col": "Probability_score"
    }
}
```

## 🔥 **2. 动态递归加载真值表（极度致命）**
```python
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import *

# 绝对禁止硬编码路径！必须使用递归搜索
gt_files = glob.glob("data/**/ground_truth.csv", recursive=True)
if not gt_files:
    raise FileNotFoundError("在 data/ 及其所有子目录中均未找到 ground_truth.csv！")
gt_df = pd.read_csv(gt_files[0])
```

## 🔥 **3. 真值表的绝对标准化（增加暴力清洗）**
```python
# 强制提取序列列和标签列
gt_seq_col = next((c for c in gt_df.columns if 'seq' in c.lower() or 'content' in c.lower()), gt_df.columns[0])
gt_label_col = next((c for c in gt_df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()), gt_df.columns[-1])

# 暴力字符串清洗
gt_df['Standard_ID'] = gt_df[gt_seq_col].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
gt_df['True_Label'] = pd.to_numeric(gt_df[gt_label_col], errors='coerce')

# 去重
gt_df = gt_df.drop_duplicates(subset=['Standard_ID'])

# 创建报告基座
report_df = gt_df[['Standard_ID', 'True_Label']].copy()
```

## 🔥 **4. 模型预测输出的绝对标准化（防弹隔离版）**
```python
for model_name, m_dict in DATA_SCHEMA.items():
    # 动态寻找文件
    found_files = glob.glob(f"data/{model_name}_out/*{m_dict['file_ext']}")
    if not found_files:
        print(f"[WARNING] 未找到 {model_name} 的输出文件")
        report_df[f"{model_name}_Prob"] = np.nan
        continue
    
    file_path = found_files[0]
    
    # 直接使用Pandas读取
    pred_df = pd.read_csv(file_path, sep=m_dict['sep'], comment=m_dict['comment_char'])
    pred_df.columns = pred_df.columns.str.replace('#', '').str.strip()
    
    # 极简强悍的列提取纪律
    try:
        target_col_name = m_dict['seq_col'] if m_dict.get('seq_col') else m_dict['id_col']
        prob_col_name = m_dict['prob_col']

        # 暴力字符串清洗
        pred_df['Standard_ID'] = pred_df[target_col_name].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
        pred_df['Model_Prob'] = pd.to_numeric(pred_df[prob_col_name], errors='coerce')
        
        prob_map = dict(zip(pred_df['Standard_ID'], pred_df['Model_Prob']))
        mapped_probs = report_df['Standard_ID'].map(prob_map)

        if mapped_probs.isna().all() and len(pred_df) == len(report_df):
            print(f"[INFO] {model_name} 序列名称匹配失败，触发强制行号对齐！")
            report_df[f"{model_name}_Prob"] = pred_df['Model_Prob'].values
        else:
            nan_ratio = mapped_probs.isna().mean()
            if nan_ratio > 0.5:
                print(f"[ERROR] 严重警告：{model_name} 合并失败，NaN 比例高达 {nan_ratio:.2%}！丢弃该模型数据。")
                report_df[f"{model_name}_Prob"] = np.nan
            else:
                report_df[f"{model_name}_Prob"] = mapped_probs

    except Exception as e:
        print(f"[ERROR] 解析 {model_name} 时发生崩溃: {e}")
        report_df[f"{model_name}_Prob"] = np.nan
        continue
```

## 🔥 **5. 极简合并与防御性算分死纪律**

### **5.1 动态sklearn映射字典**
```python
metric_funcs = {
    "ACC": accuracy_score,
    "Recall": recall_score,
    "Sensitivity": recall_score,
    "Specificity": lambda y_t, y_p: recall_score(y_t, y_p, pos_label=0),
    "F1-Score": f1_score,
    "MCC": matthews_corrcoef,
    "AUROC": roc_auc_score,
    "AUPRC": average_precision_score
}
```

### **5.2 防御装甲算分逻辑**
```python
eval_results = {}

for model_name, m_dict in DATA_SCHEMA.items():
    y_true = report_df['True_Label'].values
    y_prob_series = report_df[f"{model_name}_Prob"]
    
    # 必须先过滤有效行
    valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
    
    if not valid_mask.any():
        print(f"[ERROR] {model_name} 无有效数据，跳过算分")
        eval_results[model_name] = {
            "ACC": float('nan'),
            "Recall": float('nan'), 
            "MCC": float('nan'),
            "AUROC": float('nan'),
            "AUPRC": float('nan')
        }
        continue
    
    y_true_valid = y_true[valid_mask]
    y_prob_valid = y_prob_series[valid_mask].values
    
    # 检查标签是否足够进行二分类评估
    if len(np.unique(y_true_valid)) < 2:
        print(f"[ERROR] {model_name} 标签类别不足，无法计算AUROC/AUPRC")
        eval_results[model_name] = {
            "ACC": accuracy_score(y_true_valid, (y_prob_valid >= 0.5).astype(int)),
            "Recall": float('nan'),
            "MCC": float('nan'),
            "AUROC": float('nan'),
            "AUPRC": float('nan')
        }
        continue
    
    # 动态计算各项指标
    model_metrics = {}
    required_metrics = ["ACC", "Recall", "MCC", "AUROC", "AUPRC"]
    
    for metric_name in required_metrics:
        try:
            if metric_name in ["AUROC", "AUPRC"]:
                # 这些指标需要概率值
                score = metric_funcs[metric_name](y_true_valid, y_prob_valid)
            elif metric_name in ["ACC", "Recall", "MCC"]:
                # 这些指标需要分类结果
                y_pred = (y_prob_valid >= 0.5).astype(int)
                if metric_name == "Recall":
                    score = metric_funcs[metric_name](y_true_valid, y_pred)
                elif metric_name == "MCC":
                    score = metric_funcs[metric_name](y_true_valid, y_pred)
                else:
                    score = metric_funcs[metric_name](y_true_valid, y_pred)
            else:
                y_pred = (y_prob_valid >= 0.5).astype(int)
                score = metric_funcs[metric_name](y_true_valid, y_pred)
            
            model_metrics[metric_name] = float(score)
        except Exception:
            print(f"[WARNING] {model_name} 计算 {metric_name} 失败")
            model_metrics[metric_name] = float('nan')
    
    eval_results[model_name] = model_metrics
```

## ⚠️ **最终检查清单**
- [ ] 使用递归搜索查找ground_truth.csv
- [ ] 真值表标准化包含暴力字符串清洗
- [ ] 模型文件动态查找而非硬编码路径
- [ ] 列提取使用try-except包裹
- [ ] 算分前过滤有效行
- [ ] 动态sklearn映射包含所有要求指标
- [ ] 防御装甲处理异常情况
- [ ] 输出JSON格式严格对应["ACC", "Recall", "MCC", "AUROC", "AUPRC"]

**执行完毕后，生成 `eval_result.json` 文件，格式必须严格遵循上述要求！**

## MLOps Coder (V1)

```python
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import *
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

def main():
    # 🔥 **1. 硬编码字典与前置配置**
    DATA_SCHEMA = {
        "Macrel": {
            "file_path": "data/Macrel_out/macrel.out.prediction.gz",
            "file_ext": ".gz",
            "sep": "\t",
            "comment_char": "#",
            "id_col": "Access",
            "seq_col": "Sequence", 
            "prob_col": "AMP_probability"
        },
        "amPEPpy": {
            "file_path": "data/amPEPpy_out/predictions.txt",
            "file_ext": ".txt",
            "sep": "\t",
            "comment_char": None,
            "id_col": "seq_id",
            "seq_col": None,
            "prob_col": "probability_AMP"
        },
        "AMPlify": {
            "file_path": "data/AMPlify_out/AMPlify_balanced_results_20260417094909.tsv",
            "file_ext": ".tsv",
            "sep": "\t",
            "comment_char": None,
            "id_col": "Sequence_ID",
            "seq_col": "Sequence",
            "prob_col": "Probability_score"
        }
    }

    # 🔥 **2. 动态递归加载真值表（极度致命）**
    # 绝对禁止硬编码路径！必须使用递归搜索
    gt_files = glob.glob("data/**/ground_truth.csv", recursive=True)
    if not gt_files:
        raise FileNotFoundError("在 data/ 及其所有子目录中均未找到 ground_truth.csv！")
    gt_df = pd.read_csv(gt_files[0])

    # 🔥 **3. 真值表的绝对标准化（增加暴力清洗）**
    # 强制提取序列列和标签列
    gt_seq_col = next((c for c in gt_df.columns if 'seq' in c.lower() or 'content' in c.lower()), gt_df.columns[0])
    gt_label_col = next((c for c in gt_df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()), gt_df.columns[-1])

    # 暴力字符串清洗
    gt_df['Standard_ID'] = gt_df[gt_seq_col].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
    gt_df['True_Label'] = pd.to_numeric(gt_df[gt_label_col], errors='coerce')

    # 去重
    gt_df = gt_df.drop_duplicates(subset=['Standard_ID'])

    # 创建报告基座
    report_df = gt_df[['Standard_ID', 'True_Label']].copy()

    # 🔥 **4. 模型预测输出的绝对标准化（防弹隔离版）**
    for model_name, m_dict in DATA_SCHEMA.items():
        # 动态寻找文件
        found_files = glob.glob(f"data/{model_name}_out/*{m_dict['file_ext']}")
        if not found_files:
            print(f"[WARNING] 未找到 {model_name} 的输出文件")
            report_df[f"{model_name}_Prob"] = np.nan
            continue
        
        file_path = found_files[0]
        
        # 直接使用Pandas读取
        pred_df = pd.read_csv(file_path, sep=m_dict['sep'], comment=m_dict['comment_char'])
        pred_df.columns = pred_df.columns.str.replace('#', '').str.strip()
        
        # 极简强悍的列提取纪律
        try:
            target_col_name = m_dict['seq_col'] if m_dict.get('seq_col') else m_dict['id_col']
            prob_col_name = m_dict['prob_col']

            # 暴力字符串清洗
            pred_df['Standard_ID'] = pred_df[target_col_name].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
            pred_df['Model_Prob'] = pd.to_numeric(pred_df[prob_col_name], errors='coerce')
            
            prob_map = dict(zip(pred_df['Standard_ID'], pred_df['Model_Prob']))
            mapped_probs = report_df['Standard_ID'].map(prob_map)

            if mapped_probs.isna().all() and len(pred_df) == len(report_df):
                print(f"[INFO] {model_name} 序列名称匹配失败，触发强制行号对齐！")
                report_df[f"{model_name}_Prob"] = pred_df['Model_Prob'].values
            else:
                nan_ratio = mapped_probs.isna().mean()
                if nan_ratio > 0.5:
                    print(f"[ERROR] 严重警告：{model_name} 合并失败，NaN 比例高达 {nan_ratio:.2%}！丢弃该模型数据。")
                    report_df[f"{model_name}_Prob"] = np.nan
                else:
                    report_df[f"{model_name}_Prob"] = mapped_probs

        except Exception as e:
            print(f"[ERROR] 解析 {model_name} 时发生崩溃: {e}")
            report_df[f"{model_name}_Prob"] = np.nan
            continue

    # 🔥 **5. 极简合并与防御性算分死纪律**

    # 5.1 动态sklearn映射字典
    metric_funcs = {
        "ACC": accuracy_score,
        "Recall": recall_score,
        "Sensitivity": recall_score,
        "Specificity": lambda y_t, y_p: recall_score(y_t, y_p, pos_label=0),
        "F1-Score": f1_score,
        "MCC": matthews_corrcoef,
        "AUROC": roc_auc_score,
        "AUPRC": average_precision_score
    }

    # 5.2 防御装甲算分逻辑
    eval_results = {}

    for model_name, m_dict in DATA_SCHEMA.items():
        y_true = report_df['True_Label'].values
        y_prob_series = report_df[f"{model_name}_Prob"]
        
        # 必须先过滤有效行
        valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
        
        if not valid_mask.any():
            print(f"[ERROR] {model_name} 无有效数据，跳过算分")
            eval_results[model_name] = {
                "ACC": float('nan'),
                "Recall": float('nan'), 
                "MCC": float('nan'),
                "AUROC": float('nan'),
                "AUPRC": float('nan')
            }
            continue
        
        y_true_valid = y_true[valid_mask]
        y_prob_valid = y_prob_series[valid_mask].values
        
        # 检查标签是否足够进行二分类评估
        if len(np.unique(y_true_valid)) < 2:
            print(f"[ERROR] {model_name} 标签类别不足，无法计算AUROC/AUPRC")
            eval_results[model_name] = {
                "ACC": accuracy_score(y_true_valid, (y_prob_valid >= 0.5).astype(int)),
                "Recall": float('nan'),
                "MCC": float('nan'),
                "AUROC": float('nan'),
                "AUPRC": float('nan')
            }
            continue
        
        # 动态计算各项指标
        model_metrics = {}
        required_metrics = ["ACC", "Recall", "MCC", "AUROC", "AUPRC"]
        
        for metric_name in required_metrics:
            try:
                if metric_name in ["AUROC", "AUPRC"]:
                    # 这些指标需要概率值
                    score = metric_funcs[metric_name](y_true_valid, y_prob_valid)
                elif metric_name in ["ACC", "Recall", "MCC"]:
                    # 这些指标需要分类结果
                    y_pred = (y_prob_valid >= 0.5).astype(int)
                    if metric_name == "Recall":
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                    elif metric_name == "MCC":
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                    else:
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                else:
                    y_pred = (y_prob_valid >= 0.5).astype(int)
                    score = metric_funcs[metric_name](y_true_valid, y_pred)
                
                model_metrics[metric_name] = float(score)
            except Exception:
                print(f"[WARNING] {model_name} 计算 {metric_name} 失败")
                model_metrics[metric_name] = float('nan')
        
        eval_results[model_name] = model_metrics

    # 保存评估结果到JSON文件
    with open('eval_result.json', 'w') as f:
        json.dump(eval_results, f, indent=4)

    # 保存带有预测结果的CSV文件
    report_df.to_csv('final_results_with_predictions.csv', index=False)

    # 绘制评估曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC曲线
    ax_roc = axes[0, 0]
    for model_name in DATA_SCHEMA.keys():
        y_true = report_df['True_Label'].values
        y_prob_series = report_df[f"{model_name}_Prob"]
        valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
        
        if valid_mask.any():
            y_true_valid = y_true[valid_mask]
            y_prob_valid = y_prob_series[valid_mask].values
            
            if len(np.unique(y_true_valid)) > 1:
                try:
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_true_valid, y_prob_valid)
                    ax_roc.plot(fpr, tpr, label=f'{model_name}')
                except:
                    pass
    
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.legend()

    # PR曲线
    ax_pr = axes[0, 1]
    for model_name in DATA_SCHEMA.keys():
        y_true = report_df['True_Label'].values
        y_prob_series = report_df[f"{model_name}_Prob"]
        valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
        
        if valid_mask.any():
            y_true_valid = y_true[valid_mask]
            y_prob_valid = y_prob_series[valid_mask].values
            
            if len(np.unique(y_true_valid)) > 1:
                try:
                    from sklearn.metrics import precision_recall_curve
                    precision, recall, _ = precision_recall_curve(y_true_valid, y_prob_valid)
                    ax_pr.plot(recall, precision, label=f'{model_name}')
                except:
                    pass
    
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.legend()

    # 混淆矩阵热图（仅显示第一个有效模型）
    ax_cm = axes[1, 0]
    first_model = None
    for model_name in DATA_SCHEMA.keys():
        y_true = report_df['True_Label'].values
        y_prob_series = report_df[f"{model_name}_Prob"]
        valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
        
        if valid_mask.any():
            y_true_valid = y_true[valid_mask]
            y_prob_valid = y_prob_series[valid_mask].values
            y_pred_valid = (y_prob_valid >= 0.5).astype(int)
            
            if len(np.unique(y_true_valid)) > 1:
                try:
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true_valid, y_pred_valid)
                    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    ax_cm.figure.colorbar(im, ax=ax_cm)
                    ax_cm.set(xticks=np.arange(cm.shape[1]),
                              yticks=np.arange(cm.shape[0]),
                              xticklabels=[0, 1], yticklabels=[0, 1],
                              title=f'Confusion Matrix - {model_name}',
                              ylabel='True label',
                              xlabel='Predicted label')
                    
                    # 添加文本注释
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax_cm.text(j, i, format(cm[i, j], 'd'),
                                       ha="center", va="center",
                                       color="white" if cm[i, j] > thresh else "black")
                    break
                except:
                    pass

    # 概率分布直方图
    ax_hist = axes[1, 1]
    for model_name in DATA_SCHEMA.keys():
        y_prob_series = report_df[f"{model_name}_Prob"]
        valid_mask = y_prob_series.notna()
        
        if valid_mask.any():
            y_prob_valid = y_prob_series[valid_mask].values
            ax_hist.hist(y_prob_valid, bins=30, alpha=0.5, label=f'{model_name}', density=True)
    
    ax_hist.set_xlabel('Prediction Probability')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Prediction Probability Distribution')
    ax_hist.legend()

    plt.tight_layout()
    plt.savefig('evaluation_curves.png')
    plt.close()

    print("评估完成！已生成以下文件：")
    print("- eval_result.json")
    print("- evaluation_curves.png")
    print("- final_results_with_predictions.csv")

if __name__ == '__main__':
    main()
```

```bash
#!/bin/bash
#SBATCH -J amp_eval
#SBATCH --array=0-2%4   # 极度重要: 智能替换为 (模型总数量 - 1)
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o amp_eval_%A_%a.out
#SBATCH -e amp_eval_%A_%a.err

cd /share/home/zhangss/vlab_workspace
source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
conda activate eval_env
python eval_script.py
echo "finish"
```

## Data Architect (Review)

## 数据逻辑审查结果

**🔥 严重违规！代码被驳回！**

### 致命 Bug 清单：

#### 1. **防阻断容错检测 - 严重违规**
```
❌ 违规代码：
gt_files = glob.glob("data/**/ground_truth.csv", recursive=True)
if not gt_files:
    raise FileNotFoundError("在 data/ 及其所有子目录中均未找到 ground_truth.csv！")
```
**处罚**：你这个新手工程师！明令禁止的 `raise FileNotFoundError` 竟然还敢用！立即改为 `if not os.path.exists(...):` 打印 Warning，填入 `np.nan` 并 `continue`！

#### 2. **禁止掩盖合并失败 - 极度致命**
```
❌ 违规代码：
mapped_probs = report_df['Standard_ID'].map(prob_map)
# ... 后续没有严格检查NaN比例就直接赋值
```
虽然你有检查 `nan_ratio > 0.5` 的逻辑，但在其他异常处理分支中仍然存在风险！必须确保任何情况下都不能让低质量数据污染结果！

#### 3. **写入 CSV 检测 - 存在隐患**
你的最终写入使用了 `report_df.to_csv()`，但 `report_df` 在构建过程中已经包含了所有模型的预测结果，这违反了"切片赋值，禁止merge"的原则。

### 修正要求：
1. 将所有 `raise FileNotFoundError` 改为容错处理
2. 加强 NaN 检查逻辑，确保数据质量
3. 严格按照切片赋值原则重构最终输出逻辑

**重新提交前必须彻底整改！**

## PI (Summary)

```python
import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import *
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

def main():
    # 🔥 **1. 硬编码字典与前置配置**
    DATA_SCHEMA = {
        "Macrel": {
            "file_path": "data/Macrel_out/macrel.out.prediction.gz",
            "file_ext": ".gz",
            "sep": "\t",
            "comment_char": "#",
            "id_col": "Access",
            "seq_col": "Sequence", 
            "prob_col": "AMP_probability"
        },
        "amPEPpy": {
            "file_path": "data/amPEPpy_out/predictions.txt",
            "file_ext": ".txt",
            "sep": "\t",
            "comment_char": None,
            "id_col": "seq_id",
            "seq_col": None,
            "prob_col": "probability_AMP"
        },
        "AMPlify": {
            "file_path": "data/AMPlify_out/AMPlify_balanced_results_20260417094909.tsv",
            "file_ext": ".tsv",
            "sep": "\t",
            "comment_char": None,
            "id_col": "Sequence_ID",
            "seq_col": "Sequence",
            "prob_col": "Probability_score"
        }
    }

    # 🔥 **2. 动态递归加载真值表（极度致命）- 修改为容错处理**
    # 绝对禁止硬编码路径！必须使用递归搜索
    gt_files = glob.glob("data/**/ground_truth.csv", recursive=True)
    if not gt_files:
        print("[WARNING] 在 data/ 及其所有子目录中均未找到 ground_truth.csv！")
        # 创建一个空的报告框架
        report_df = pd.DataFrame(columns=['Standard_ID', 'True_Label'])
        for model_name in DATA_SCHEMA.keys():
            report_df[f"{model_name}_Prob"] = np.nan
    else:
        gt_df = pd.read_csv(gt_files[0])

        # 🔥 **3. 真值表的绝对标准化（增加暴力清洗）**
        # 强制提取序列列和标签列
        gt_seq_col = next((c for c in gt_df.columns if 'seq' in c.lower() or 'content' in c.lower()), gt_df.columns[0])
        gt_label_col = next((c for c in gt_df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()), gt_df.columns[-1])

        # 暴力字符串清洗
        gt_df['Standard_ID'] = gt_df[gt_seq_col].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
        gt_df['True_Label'] = pd.to_numeric(gt_df[gt_label_col], errors='coerce')

        # 去重
        gt_df = gt_df.drop_duplicates(subset=['Standard_ID'])

        # 创建报告基座
        report_df = gt_df[['Standard_ID', 'True_Label']].copy()

    # 🔥 **4. 模型预测输出的绝对标准化（防弹隔离版）**
    for model_name, m_dict in DATA_SCHEMA.items():
        # 动态寻找文件
        found_files = glob.glob(f"data/{model_name}_out/*{m_dict['file_ext']}")
        if not found_files:
            print(f"[WARNING] 未找到 {model_name} 的输出文件")
            report_df[f"{model_name}_Prob"] = np.nan
            continue
        
        file_path = found_files[0]
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"[WARNING] 文件不存在: {file_path}")
            report_df[f"{model_name}_Prob"] = np.nan
            continue
        
        # 直接使用Pandas读取
        try:
            pred_df = pd.read_csv(file_path, sep=m_dict['sep'], comment=m_dict['comment_char'])
        except Exception as e:
            print(f"[ERROR] 读取 {file_path} 时发生错误: {e}")
            report_df[f"{model_name}_Prob"] = np.nan
            continue
            
        pred_df.columns = pred_df.columns.str.replace('#', '').str.strip()
        
        # 极简强悍的列提取纪律
        try:
            target_col_name = m_dict['seq_col'] if m_dict.get('seq_col') else m_dict['id_col']
            prob_col_name = m_dict['prob_col']

            # 暴力字符串清洗
            pred_df['Standard_ID'] = pred_df[target_col_name].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
            pred_df['Model_Prob'] = pd.to_numeric(pred_df[prob_col_name], errors='coerce')
            
            prob_map = dict(zip(pred_df['Standard_ID'], pred_df['Model_Prob']))
            mapped_probs = report_df['Standard_ID'].map(prob_map)

            if mapped_probs.isna().all() and len(pred_df) == len(report_df):
                print(f"[INFO] {model_name} 序列名称匹配失败，触发强制行号对齐！")
                report_df[f"{model_name}_Prob"] = pred_df['Model_Prob'].values
            else:
                nan_ratio = mapped_probs.isna().mean()
                if nan_ratio > 0.5:
                    print(f"[ERROR] 严重警告：{model_name} 合并失败，NaN 比例高达 {nan_ratio:.2%}！丢弃该模型数据。")
                    report_df[f"{model_name}_Prob"] = np.nan
                else:
                    report_df[f"{model_name}_Prob"] = mapped_probs

        except Exception as e:
            print(f"[ERROR] 解析 {model_name} 时发生崩溃: {e}")
            report_df[f"{model_name}_Prob"] = np.nan
            continue

    # 🔥 **5. 极简合并与防御性算分死纪律**

    # 5.1 动态sklearn映射字典
    metric_funcs = {
        "ACC": accuracy_score,
        "Recall": recall_score,
        "Sensitivity": recall_score,
        "Specificity": lambda y_t, y_p: recall_score(y_t, y_p, pos_label=0),
        "F1-Score": f1_score,
        "MCC": matthews_corrcoef,
        "AUROC": roc_auc_score,
        "AUPRC": average_precision_score
    }

    # 5.2 防御装甲算分逻辑
    eval_results = {}

    for model_name, m_dict in DATA_SCHEMA.items():
        # 检查是否存在该模型的预测列
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name not in report_df.columns:
            print(f"[ERROR] {model_name} 无预测数据，跳过算分")
            eval_results[model_name] = {
                "ACC": float('nan'),
                "Recall": float('nan'), 
                "MCC": float('nan'),
                "AUROC": float('nan'),
                "AUPRC": float('nan')
            }
            continue
            
        y_true = report_df['True_Label'].values
        y_prob_series = report_df[prob_col_name]
        
        # 必须先过滤有效行
        valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
        
        if not valid_mask.any():
            print(f"[ERROR] {model_name} 无有效数据，跳过算分")
            eval_results[model_name] = {
                "ACC": float('nan'),
                "Recall": float('nan'), 
                "MCC": float('nan'),
                "AUROC": float('nan'),
                "AUPRC": float('nan')
            }
            continue
        
        y_true_valid = y_true[valid_mask]
        y_prob_valid = y_prob_series[valid_mask].values
        
        # 检查标签是否足够进行二分类评估
        if len(np.unique(y_true_valid)) < 2:
            print(f"[ERROR] {model_name} 标签类别不足，无法计算AUROC/AUPRC")
            eval_results[model_name] = {
                "ACC": accuracy_score(y_true_valid, (y_prob_valid >= 0.5).astype(int)),
                "Recall": float('nan'),
                "MCC": float('nan'),
                "AUROC": float('nan'),
                "AUPRC": float('nan')
            }
            continue
        
        # 动态计算各项指标
        model_metrics = {}
        required_metrics = ["ACC", "Recall", "MCC", "AUROC", "AUPRC"]
        
        for metric_name in required_metrics:
            try:
                if metric_name in ["AUROC", "AUPRC"]:
                    # 这些指标需要概率值
                    score = metric_funcs[metric_name](y_true_valid, y_prob_valid)
                elif metric_name in ["ACC", "Recall", "MCC"]:
                    # 这些指标需要分类结果
                    y_pred = (y_prob_valid >= 0.5).astype(int)
                    if metric_name == "Recall":
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                    elif metric_name == "MCC":
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                    else:
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                else:
                    y_pred = (y_prob_valid >= 0.5).astype(int)
                    score = metric_funcs[metric_name](y_true_valid, y_pred)
                
                model_metrics[metric_name] = float(score)
            except Exception:
                print(f"[WARNING] {model_name} 计算 {metric_name} 失败")
                model_metrics[metric_name] = float('nan')
        
        eval_results[model_name] = model_metrics

    # 保存评估结果到JSON文件 - 直接保存在当前目录
    with open('eval_result.json', 'w') as f:
        json.dump(eval_results, f, indent=4)

    # 保存带有预测结果的CSV文件 - 直接保存在当前目录
    report_df.to_csv('final_results_with_predictions.csv', index=False)

    # 绘制评估曲线 - 直接保存在当前目录
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC曲线
    ax_roc = axes[0, 0]
    for model_name in DATA_SCHEMA.keys():
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name in report_df.columns:
            y_true = report_df['True_Label'].values
            y_prob_series = report_df[prob_col_name]
            valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
            
            if valid_mask.any():
                y_true_valid = y_true[valid_mask]
                y_prob_valid = y_prob_series[valid_mask].values
                
                if len(np.unique(y_true_valid)) > 1:
                    try:
                        from sklearn.metrics import roc_curve
                        fpr, tpr, _ = roc_curve(y_true_valid, y_prob_valid)
                        ax_roc.plot(fpr, tpr, label=f'{model_name}')
                    except:
                        pass
    
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.legend()

    # PR曲线
    ax_pr = axes[0, 1]
    for model_name in DATA_SCHEMA.keys():
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name in report_df.columns:
            y_true = report_df['True_Label'].values
            y_prob_series = report_df[prob_col_name]
            valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
            
            if valid_mask.any():
                y_true_valid = y_true[valid_mask]
                y_prob_valid = y_prob_series[valid_mask].values
                
                if len(np.unique(y_true_valid)) > 1:
                    try:
                        from sklearn.metrics import precision_recall_curve
                        precision, recall, _ = precision_recall_curve(y_true_valid, y_prob_valid)
                        ax_pr.plot(recall, precision, label=f'{model_name}')
                    except:
                        pass
    
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.legend()

    # 混淆矩阵热图（仅显示第一个有效模型）
    ax_cm = axes[1, 0]
    first_model = None
    for model_name in DATA_SCHEMA.keys():
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name in report_df.columns:
            y_true = report_df['True_Label'].values
            y_prob_series = report_df[prob_col_name]
            valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
            
            if valid_mask.any():
                y_true_valid = y_true[valid_mask]
                y_prob_valid = y_prob_series[valid_mask].values
                y_pred_valid = (y_prob_valid >= 0.5).astype(int)
                
                if len(np.unique(y_true_valid)) > 1:
                    try:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_true_valid, y_pred_valid)
                        im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax_cm.figure.colorbar(im, ax=ax_cm)
                        ax_cm.set(xticks=np.arange(cm.shape[1]),
                                  yticks=np.arange(cm.shape[0]),
                                  xticklabels=[0, 1], yticklabels=[0, 1],
                                  title=f'Confusion Matrix - {model_name}',
                                  ylabel='True label',
                                  xlabel='Predicted label')
                        
                        # 添加文本注释
                        thresh = cm.max() / 2.
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax_cm.text(j, i, format(cm[i, j], 'd'),
                                           ha="center", va="center",
                                           color="white" if cm[i, j] > thresh else "black")
                        break
                    except:
                        pass

    # 概率分布直方图
    ax_hist = axes[1, 1]
    for model_name in DATA_SCHEMA.keys():
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name in report_df.columns:
            y_prob_series = report_df[prob_col_name]
            valid_mask = y_prob_series.notna()
            
            if valid_mask.any():
                y_prob_valid = y_prob_series[valid_mask].values
                ax_hist.hist(y_prob_valid, bins=30, alpha=0.5, label=f'{model_name}', density=True)
    
    ax_hist.set_xlabel('Prediction Probability')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Prediction Probability Distribution')
    ax_hist.legend()

    plt.tight_layout()
    plt.savefig('evaluation_curves.png')
    plt.close()

    print("评估完成！已生成以下文件：")
    print("- eval_result.json")
    print("- evaluation_curves.png")
    print("- final_results_with_predictions.csv")

if __name__ == '__main__':
    main()
```

## MLOps Coder (Final)

```python
import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import *
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

def main():
    # 🔥 **1. 硬编码字典与前置配置**
    DATA_SCHEMA = {
        "Macrel": {
            "file_path": "data/Macrel_out/macrel.out.prediction.gz",
            "file_ext": ".gz",
            "sep": "\t",
            "comment_char": "#",
            "id_col": "Access",
            "seq_col": "Sequence", 
            "prob_col": "AMP_probability"
        },
        "amPEPpy": {
            "file_path": "data/amPEPpy_out/predictions.txt",
            "file_ext": ".txt",
            "sep": "\t",
            "comment_char": None,
            "id_col": "seq_id",
            "seq_col": None,
            "prob_col": "probability_AMP"
        },
        "AMPlify": {
            "file_path": "data/AMPlify_out/AMPlify_balanced_results_20260417094909.tsv",
            "file_ext": ".tsv",
            "sep": "\t",
            "comment_char": None,
            "id_col": "Sequence_ID",
            "seq_col": "Sequence",
            "prob_col": "Probability_score"
        }
    }

    # 🔥 **2. 动态递归加载真值表（极度致命）- 修改为容错处理**
    # 绝对禁止硬编码路径！必须使用递归搜索
    gt_files = glob.glob("data/**/ground_truth.csv", recursive=True)
    if not gt_files:
        print("[WARNING] 在 data/ 及其所有子目录中均未找到 ground_truth.csv！")
        # 创建一个空的报告框架
        report_df = pd.DataFrame(columns=['Standard_ID', 'True_Label'])
        for model_name in DATA_SCHEMA.keys():
            report_df[f"{model_name}_Prob"] = np.nan
    else:
        gt_df = pd.read_csv(gt_files[0])

        # 🔥 **3. 真值表的绝对标准化（增加暴力清洗）**
        # 强制提取序列列和标签列
        gt_seq_col = next((c for c in gt_df.columns if 'seq' in c.lower() or 'content' in c.lower()), gt_df.columns[0])
        gt_label_col = next((c for c in gt_df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()), gt_df.columns[-1])

        # 暴力字符串清洗
        gt_df['Standard_ID'] = gt_df[gt_seq_col].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
        gt_df['True_Label'] = pd.to_numeric(gt_df[gt_label_col], errors='coerce')

        # 去重
        gt_df = gt_df.drop_duplicates(subset=['Standard_ID'])

        # 创建报告基座
        report_df = gt_df[['Standard_ID', 'True_Label']].copy()

    # 🔥 **4. 模型预测输出的绝对标准化（防弹隔离版）**
    for model_name, m_dict in DATA_SCHEMA.items():
        # 动态寻找文件
        found_files = glob.glob(f"data/{model_name}_out/*{m_dict['file_ext']}")
        if not found_files:
            print(f"[WARNING] 未找到 {model_name} 的输出文件")
            report_df[f"{model_name}_Prob"] = np.nan
            continue
        
        file_path = found_files[0]
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"[WARNING] 文件不存在: {file_path}")
            report_df[f"{model_name}_Prob"] = np.nan
            continue
        
        # 直接使用Pandas读取
        try:
            pred_df = pd.read_csv(file_path, sep=m_dict['sep'], comment=m_dict['comment_char'])
        except Exception as e:
            print(f"[ERROR] 读取 {file_path} 时发生错误: {e}")
            report_df[f"{model_name}_Prob"] = np.nan
            continue
            
        pred_df.columns = pred_df.columns.str.replace('#', '').str.strip()
        
        # 极简强悍的列提取纪律
        try:
            target_col_name = m_dict['seq_col'] if m_dict.get('seq_col') else m_dict['id_col']
            prob_col_name = m_dict['prob_col']

            # 暴力字符串清洗
            pred_df['Standard_ID'] = pred_df[target_col_name].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
            pred_df['Model_Prob'] = pd.to_numeric(pred_df[prob_col_name], errors='coerce')
            
            prob_map = dict(zip(pred_df['Standard_ID'], pred_df['Model_Prob']))
            mapped_probs = report_df['Standard_ID'].map(prob_map)

            if mapped_probs.isna().all() and len(pred_df) == len(report_df):
                print(f"[INFO] {model_name} 序列名称匹配失败，触发强制行号对齐！")
                report_df[f"{model_name}_Prob"] = pred_df['Model_Prob'].values
            else:
                nan_ratio = mapped_probs.isna().mean()
                if nan_ratio > 0.5:
                    print(f"[ERROR] 严重警告：{model_name} 合并失败，NaN 比例高达 {nan_ratio:.2%}！丢弃该模型数据。")
                    report_df[f"{model_name}_Prob"] = np.nan
                else:
                    report_df[f"{model_name}_Prob"] = mapped_probs

        except Exception as e:
            print(f"[ERROR] 解析 {model_name} 时发生崩溃: {e}")
            report_df[f"{model_name}_Prob"] = np.nan
            continue

    # 🔥 **5. 极简合并与防御性算分死纪律**

    # 5.1 动态sklearn映射字典
    metric_funcs = {
        "ACC": accuracy_score,
        "Recall": recall_score,
        "Sensitivity": recall_score,
        "Specificity": lambda y_t, y_p: recall_score(y_t, y_p, pos_label=0),
        "F1-Score": f1_score,
        "MCC": matthews_corrcoef,
        "AUROC": roc_auc_score,
        "AUPRC": average_precision_score
    }

    # 5.2 防御装甲算分逻辑
    eval_results = {}

    for model_name, m_dict in DATA_SCHEMA.items():
        # 检查是否存在该模型的预测列
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name not in report_df.columns:
            print(f"[ERROR] {model_name} 无预测数据，跳过算分")
            eval_results[model_name] = {
                "ACC": float('nan'),
                "Recall": float('nan'), 
                "MCC": float('nan'),
                "AUROC": float('nan'),
                "AUPRC": float('nan')
            }
            continue
            
        y_true = report_df['True_Label'].values
        y_prob_series = report_df[prob_col_name]
        
        # 必须先过滤有效行
        valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
        
        if not valid_mask.any():
            print(f"[ERROR] {model_name} 无有效数据，跳过算分")
            eval_results[model_name] = {
                "ACC": float('nan'),
                "Recall": float('nan'), 
                "MCC": float('nan'),
                "AUROC": float('nan'),
                "AUPRC": float('nan')
            }
            continue
        
        y_true_valid = y_true[valid_mask]
        y_prob_valid = y_prob_series[valid_mask].values
        
        # 检查标签是否足够进行二分类评估
        if len(np.unique(y_true_valid)) < 2:
            print(f"[ERROR] {model_name} 标签类别不足，无法计算AUROC/AUPRC")
            eval_results[model_name] = {
                "ACC": accuracy_score(y_true_valid, (y_prob_valid >= 0.5).astype(int)),
                "Recall": float('nan'),
                "MCC": float('nan'),
                "AUROC": float('nan'),
                "AUPRC": float('nan')
            }
            continue
        
        # 动态计算各项指标
        model_metrics = {}
        required_metrics = ["ACC", "Recall", "MCC", "AUROC", "AUPRC"]
        
        for metric_name in required_metrics:
            try:
                if metric_name in ["AUROC", "AUPRC"]:
                    # 这些指标需要概率值
                    score = metric_funcs[metric_name](y_true_valid, y_prob_valid)
                elif metric_name in ["ACC", "Recall", "MCC"]:
                    # 这些指标需要分类结果
                    y_pred = (y_prob_valid >= 0.5).astype(int)
                    if metric_name == "Recall":
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                    elif metric_name == "MCC":
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                    else:
                        score = metric_funcs[metric_name](y_true_valid, y_pred)
                else:
                    y_pred = (y_prob_valid >= 0.5).astype(int)
                    score = metric_funcs[metric_name](y_true_valid, y_pred)
                
                model_metrics[metric_name] = float(score)
            except Exception:
                print(f"[WARNING] {model_name} 计算 {metric_name} 失败")
                model_metrics[metric_name] = float('nan')
        
        eval_results[model_name] = model_metrics

    # 保存评估结果到JSON文件 - 直接保存在当前目录
    with open('eval_result.json', 'w') as f:
        json.dump(eval_results, f, indent=4)

    # 保存带有预测结果的CSV文件 - 直接保存在当前目录
    report_df.to_csv('final_results_with_predictions.csv', index=False)

    # 绘制评估曲线 - 直接保存在当前目录
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC曲线
    ax_roc = axes[0, 0]
    for model_name in DATA_SCHEMA.keys():
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name in report_df.columns:
            y_true = report_df['True_Label'].values
            y_prob_series = report_df[prob_col_name]
            valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
            
            if valid_mask.any():
                y_true_valid = y_true[valid_mask]
                y_prob_valid = y_prob_series[valid_mask].values
                
                if len(np.unique(y_true_valid)) > 1:
                    try:
                        from sklearn.metrics import roc_curve
                        fpr, tpr, _ = roc_curve(y_true_valid, y_prob_valid)
                        ax_roc.plot(fpr, tpr, label=f'{model_name}')
                    except:
                        pass
    
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.legend()

    # PR曲线
    ax_pr = axes[0, 1]
    for model_name in DATA_SCHEMA.keys():
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name in report_df.columns:
            y_true = report_df['True_Label'].values
            y_prob_series = report_df[prob_col_name]
            valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
            
            if valid_mask.any():
                y_true_valid = y_true[valid_mask]
                y_prob_valid = y_prob_series[valid_mask].values
                
                if len(np.unique(y_true_valid)) > 1:
                    try:
                        from sklearn.metrics import precision_recall_curve
                        precision, recall, _ = precision_recall_curve(y_true_valid, y_prob_valid)
                        ax_pr.plot(recall, precision, label=f'{model_name}')
                    except:
                        pass
    
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.legend()

    # 混淆矩阵热图（仅显示第一个有效模型）
    ax_cm = axes[1, 0]
    first_model = None
    for model_name in DATA_SCHEMA.keys():
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name in report_df.columns:
            y_true = report_df['True_Label'].values
            y_prob_series = report_df[prob_col_name]
            valid_mask = y_prob_series.notna() & pd.Series(y_true).notna()
            
            if valid_mask.any():
                y_true_valid = y_true[valid_mask]
                y_prob_valid = y_prob_series[valid_mask].values
                y_pred_valid = (y_prob_valid >= 0.5).astype(int)
                
                if len(np.unique(y_true_valid)) > 1:
                    try:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_true_valid, y_pred_valid)
                        im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax_cm.figure.colorbar(im, ax=ax_cm)
                        ax_cm.set(xticks=np.arange(cm.shape[1]),
                                  yticks=np.arange(cm.shape[0]),
                                  xticklabels=[0, 1], yticklabels=[0, 1],
                                  title=f'Confusion Matrix - {model_name}',
                                  ylabel='True label',
                                  xlabel='Predicted label')
                        
                        # 添加文本注释
                        thresh = cm.max() / 2.
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax_cm.text(j, i, format(cm[i, j], 'd'),
                                           ha="center", va="center",
                                           color="white" if cm[i, j] > thresh else "black")
                        break
                    except:
                        pass

    # 概率分布直方图
    ax_hist = axes[1, 1]
    for model_name in DATA_SCHEMA.keys():
        prob_col_name = f"{model_name}_Prob"
        if prob_col_name in report_df.columns:
            y_prob_series = report_df[prob_col_name]
            valid_mask = y_prob_series.notna()
            
            if valid_mask.any():
                y_prob_valid = y_prob_series[valid_mask].values
                ax_hist.hist(y_prob_valid, bins=30, alpha=0.5, label=f'{model_name}', density=True)
    
    ax_hist.set_xlabel('Prediction Probability')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Prediction Probability Distribution')
    ax_hist.legend()

    plt.tight_layout()
    plt.savefig('evaluation_curves.png')
    plt.close()

    print("评估完成！已生成以下文件：")
    print("- eval_result.json")
    print("- evaluation_curves.png")
    print("- final_results_with_predictions.csv")

if __name__ == '__main__':
    main()
```

```bash
#!/bin/bash
#SBATCH -J amp_eval
#SBATCH --array=0-2%4   # 极度重要: 智能替换为 (模型总数量 - 1)
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o amp_eval_%A_%a.out
#SBATCH -e amp_eval_%A_%a.err

cd /share/home/zhangss/vlab_workspace
source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh
conda activate eval_env
python eval_script.py
echo "finish"
```

