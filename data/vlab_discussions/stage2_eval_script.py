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

            if mapped_probs.isna().all():
                print(f"[ERROR] {model_name} 的预测 ID 与真值完全不匹配；禁止无证据的行号对齐，丢弃该模型数据。")
                report_df[f"{model_name}_Prob"] = np.nan
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
    ax_roc.set_title('A. ROC Curves')
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
    ax_pr.set_title('B. Precision-Recall Curves')
    ax_pr.legend()

    # Precision bar chart at the same 0.5 threshold used by metrics.
    precision_models = []
    precision_values = []
    recall_models = []
    recall_values = []
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
                    precision_models.append(model_name)
                    precision_values.append(float(precision_score(y_true_valid, y_pred_valid, zero_division=0)))
                    recall_models.append(model_name)
                    recall_values.append(float(recall_score(y_true_valid, y_pred_valid, zero_division=0)))
                except Exception:
                    pass

    ax_precision = axes[1, 0]
    if precision_models:
        bars = ax_precision.bar(precision_models, precision_values)
        ax_precision.bar_label(bars, fmt='%.3f', padding=3)
    ax_precision.set_ylim(0, 1.05)
    ax_precision.set_ylabel('Precision')
    ax_precision.set_title('C. Precision at 0.5 Threshold')
    ax_precision.tick_params(axis='x', rotation=20)

    ax_recall = axes[1, 1]
    if recall_models:
        bars = ax_recall.bar(recall_models, recall_values)
        ax_recall.bar_label(bars, fmt='%.3f', padding=3)
    ax_recall.set_ylim(0, 1.05)
    ax_recall.set_ylabel('Recall')
    ax_recall.set_title('D. Recall at 0.5 Threshold')
    ax_recall.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig('evaluation_curves.png')
    plt.close()

    print("评估完成！已生成以下文件：")
    print("- eval_result.json")
    print("- evaluation_curves.png")
    print("- final_results_with_predictions.csv")

if __name__ == '__main__':
    main()
