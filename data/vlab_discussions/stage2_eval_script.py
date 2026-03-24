import pandas as pd
import glob
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, roc_auc_score, average_precision_score
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


# 【硬编码】数据 Schema
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
    "AMP-Scanner-v2": {
        "file_path": "data/AMP-Scanner-v2_out/ampscanner_out.csv",
        "file_ext": ".csv",
        "sep": ",",
        "comment_char": None,
        "id_col": "SeqID",
        "seq_col": "Sequence",
        "prob_col": "Prediction_Probability"
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
    "AI4AMP": {
        "file_path": "data/AI4AMP_out/predictions.csv",
        "file_ext": ".csv",
        "sep": ",",
        "comment_char": None,
        "id_col": "Peptide",
        "seq_col": None,
        "prob_col": "Score"
    },
    "AMPlify": {
        "file_path": "data/AMPlify_out/AMPlify_balanced_results_20260324100314.tsv",
        "file_ext": ".tsv",
        "sep": "\t",
        "comment_char": None,
        "id_col": "Sequence_ID",
        "seq_col": "Sequence",
        "prob_col": "Probability_score"
    }
}

GROUND_TRUTH_PATH = "data/ground_truth.csv"


def main():
    # 加载真值表
    gt_df = pd.read_csv(GROUND_TRUTH_PATH)

    # 自动识别序列列 & 标签列
    gt_seq_col = next((c for c in gt_df.columns if 'seq' in c.lower() or 'content' in c.lower()), gt_df.columns[0])
    gt_label_col = next((c for c in gt_df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()), gt_df.columns[-1])

    # 强制标准化 ID 和 Label 列
    gt_df['Standard_ID'] = gt_df[gt_seq_col].astype(str).str.strip().str.upper()
    gt_df['True_Label'] = pd.to_numeric(gt_df[gt_label_col], errors='coerce')

    # 去重保留唯一序列
    gt_df = gt_df.drop_duplicates(subset=['Standard_ID'])

    # 创建初始报告基座 DataFrame
    report_df = gt_df[['Standard_ID', 'True_Label']].copy()

    # 模型预测输出的绝对标准化（防弹隔离）
    for model_name, m_dict in DATA_SCHEMA.items():
        try:
            # 动态查找文件
            found_files = glob.glob(f"data/{model_name}_out/*{m_dict['file_ext']}")
            if not found_files:
                print(f"[WARNING] 未找到 {model_name} 的输出文件")
                report_df[f"{model_name}_Prob"] = 0.0
                continue
            file_path = found_files[0]

            # 使用 Pandas 原生读取器（自动处理压缩和注释）
            pred_df = pd.read_csv(file_path, sep=m_dict['sep'], comment=m_dict['comment_char'])

            # 清洗列名
            pred_df.columns = pred_df.columns.str.replace('#', '').str.strip()

            # 提取主键列（优先使用 seq_col，否则 fallback 到 id_col）
            target_col_name = m_dict['seq_col'] if m_dict.get('seq_col') else m_dict['id_col']
            prob_col_name = m_dict['prob_col']

            # 构造 Standard_ID 并提取概率
            pred_df['Standard_ID'] = pred_df[target_col_name].astype(str).str.strip().str.upper()
            pred_df['Model_Prob'] = pd.to_numeric(pred_df[prob_col_name], errors='coerce')

            # 映射到 report_df 上
            prob_map = dict(zip(pred_df['Standard_ID'], pred_df['Model_Prob']))
            mapped_probs = report_df['Standard_ID'].map(prob_map)

            # 如果映射失败但长度一致，则启用强制行号对齐模式
            if mapped_probs.isna().all() and len(pred_df) == len(report_df):
                print(f"[INFO] {model_name} 序列名称匹配失败，触发强制行号对齐！")
                report_df[f"{model_name}_Prob"] = pred_df['Model_Prob'].values
            else:
                report_df[f"{model_name}_Prob"] = mapped_probs.fillna(0.0)

        except KeyError as e:
            print(f"[ERROR] {model_name} 找不到指定的列名: {e}")
            report_df[f"{model_name}_Prob"] = 0.0
            continue
        except Exception as e:
            print(f"[ERROR] 解析 {model_name} 时发生未知崩溃: {e}")
            report_df[f"{model_name}_Prob"] = 0.0
            continue

    # 极简合并与算分死纪律
    results = {}
    y_true = report_df['True_Label']

    # 初始化绘图对象
    fig, axes = plt.subplots(2, len(DATA_SCHEMA), figsize=(5*len(DATA_SCHEMA), 10))
    if len(DATA_SCHEMA) == 1:
        axes = axes.reshape(-1, 1)

    for idx, model_name in enumerate(DATA_SCHEMA.keys()):
        try:
            y_prob = report_df[f"{model_name}_Prob"]
            y_pred = (y_prob >= 0.5).astype(int)

            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc_roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
            auc_pr = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0

            results[model_name] = {
                "ACC": round(acc, 4),
                "Recall": round(rec, 4),
                "MCC": round(mcc, 4),
                "AUROC": round(auc_roc, 4),
                "AUPRC": round(auc_pr, 4)
            }

            # 绘制 ROC 曲线
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            axes[0, idx].plot(fpr, tpr, label=f'{model_name} (AUC={auc_roc:.2f})')
            axes[0, idx].set_title(f'{model_name} ROC Curve')
            axes[0, idx].set_xlabel('False Positive Rate')
            axes[0, idx].set_ylabel('True Positive Rate')
            axes[0, idx].legend(loc='lower right')

            # 绘制 PR 曲线
            prec, recall, _ = precision_recall_curve(y_true, y_prob)
            axes[1, idx].plot(recall, prec, label=f'{model_name} (AP={auc_pr:.2f})')
            axes[1, idx].set_title(f'{model_name} Precision-Recall Curve')
            axes[1, idx].set_xlabel('Recall')
            axes[1, idx].set_ylabel('Precision')
            axes[1, idx].legend(loc='upper right')

        except Exception as e:
            print(f"[ERROR] 计算 {model_name} 分数时出错: {e}")
            results[model_name] = {
                "ACC": 0.0,
                "Recall": 0.0,
                "MCC": 0.0,
                "AUROC": 0.0,
                "AUPRC": 0.0
            }

    plt.tight_layout()
    plt.savefig("evaluation_curves.png")
    plt.close()

    # 输出结果至 JSON 文件
    with open("eval_result.json", "w") as f:
        json.dump(results, f, indent=4)

    # 保存带预测结果的 CSV 文件
    report_df.to_csv("final_results_with_predictions.csv", index=False)


if __name__ == '__main__':
    main()
