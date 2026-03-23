import os
import json
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# === 配置区 ===
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
        "file_path": "data/AMPlify_out/AMPlify_balanced_results_20260323140222.tsv",
        "file_ext": ".tsv",
        "sep": "\t",
        "comment_char": None,
        "id_col": "Sequence_ID",
        "seq_col": "Sequence",
        "prob_col": "Probability_score"
    }
}

GROUND_TRUTH_PATH = "data/ground_truth.csv"

def load_ground_truth(path):
    gt_df = pd.read_csv(path)
    gt_seq_col = next((c for c in gt_df.columns if 'seq' in c.lower() or 'content' in c.lower()), gt_df.columns[0])
    gt_label_col = next((c for c in gt_df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()), gt_df.columns[-1])
    gt_df['Standard_ID'] = gt_df[gt_seq_col].astype(str).str.strip().str.upper()
    gt_df['True_Label'] = pd.to_numeric(gt_df[gt_label_col], errors='coerce')
    gt_df = gt_df.drop_duplicates(subset=['Standard_ID'])
    return gt_df[['Standard_ID', 'True_Label']].copy()

def standardize_model_prediction(model_name, m_dict, report_df):
    found_files = glob.glob(f"data/{model_name}_out/*{m_dict['file_ext']}")
    if not found_files:
        print(f"[WARNING] 未找到 {model_name} 的输出文件")
        report_df[f"{model_name}_Prob"] = 0.0
        return report_df
    file_path = found_files[0]

    pred_df = pd.read_csv(file_path, sep=m_dict['sep'], comment=m_dict['comment_char'])
    pred_df.columns = pred_df.columns.str.replace('#', '').str.strip()

    try:
        target_col_name = m_dict['seq_col'] if m_dict.get('seq_col') else m_dict['id_col']
        prob_col_name = m_dict['prob_col']

        pred_df['Standard_ID'] = pred_df[target_col_name].astype(str).str.strip().str.upper()
        pred_df['Model_Prob'] = pd.to_numeric(pred_df[prob_col_name], errors='coerce')

        prob_map = dict(zip(pred_df['Standard_ID'], pred_df['Model_Prob']))
        mapped_probs = report_df['Standard_ID'].map(prob_map)

        if mapped_probs.isna().all() and len(pred_df) == len(report_df):
            print(f"[INFO] {model_name} 序列名称匹配失败，触发强制行号对齐！")
            report_df[f"{model_name}_Prob"] = pred_df['Model_Prob'].values
        else:
            report_df[f"{model_name}_Prob"] = mapped_probs.fillna(0.0)

    except KeyError as e:
        print(f"[ERROR] {model_name} 找不到指定的列名: {e}")
        report_df[f"{model_name}_Prob"] = 0.0
    except Exception as e:
        print(f"[ERROR] 解析 {model_name} 时发生未知崩溃: {e}")
        report_df[f"{model_name}_Prob"] = 0.0

    return report_df

def evaluate_and_plot(report_df):
    results = {}
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name in DATA_SCHEMA.keys():
        try:
            y_true = report_df['True_Label'].values
            y_pred_prob = report_df[f"{model_name}_Prob"].values
            y_pred = (y_pred_prob >= 0.5).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.0

            results[model_name] = {
                "ACC": round(acc, 4),
                "PREC": round(prec, 4),
                "REC": round(rec, 4),
                "F1": round(f1, 4),
                "AUC": round(auc, 4)
            }

            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')

        except Exception as e:
            print(f"[ERROR] 计算 {model_name} 分数时出错: {e}")
            results[model_name] = {"ACC": 0.0, "PREC": 0.0, "REC": 0.0, "F1": 0.0, "AUC": 0.0}

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves of Models')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("evaluation_curves.png")
    plt.close()

    with open("eval_result.json", "w") as f:
        json.dump(results, f, indent=4)

    report_df.to_csv("final_results_with_predictions.csv", index=False)
    print("[SUCCESS] 评测完成，报告已生成")

def main():
    report_df = load_ground_truth(GROUND_TRUTH_PATH)
    for model_name, m_dict in DATA_SCHEMA.items():
        report_df = standardize_model_prediction(model_name, m_dict, report_df)
    evaluate_and_plot(report_df)

if __name__ == '__main__':
    main()
