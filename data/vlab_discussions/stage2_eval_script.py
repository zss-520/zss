import pandas as pd
import numpy as np
import glob
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

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

def main():
    # 约束2：动态递归加载真值表
    gt_files = glob.glob("data/**/ground_truth.csv", recursive=True)
    if not gt_files:
        raise FileNotFoundError("在 data/ 及其所有子目录中均未找到 ground_truth.csv！")
    gt_df = pd.read_csv(gt_files[0])

    gt_seq_col = next((c for c in gt_df.columns if 'seq' in c.lower() or 'content' in c.lower()), gt_df.columns[0])
    gt_label_col = next((c for c in gt_df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()), gt_df.columns[-1])

    gt_df['Standard_ID'] = gt_df[gt_seq_col].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
    gt_df['True_Label'] = pd.to_numeric(gt_df[gt_label_col], errors='coerce')
    gt_df = gt_df.drop_duplicates(subset=['Standard_ID'])

    report_df = gt_df[['Standard_ID', 'True_Label']].copy()

    # 约束3：模型预测输出的绝对标准化
    for model_name, m_dict in DATA_SCHEMA.items():
        found_files = glob.glob(f"data/{model_name}_out/*{m_dict['file_ext']}")
        if not found_files:
            print(f"[WARNING] 未找到 {model_name} 的输出文件")
            report_df[f"{model_name}_Prob"] = np.nan
            continue
        file_path = found_files[0]

        pred_df = pd.read_csv(file_path, sep=m_dict['sep'], comment=m_dict['comment_char'])
        pred_df.columns = pred_df.columns.str.replace('#', '').str.strip()

        try:
            target_col_name = m_dict['seq_col'] if m_dict.get('seq_col') else m_dict['id_col']
            prob_col_name = m_dict['prob_col']

            pred_df['Standard_ID'] = pred_df[target_col_name].astype(str).str.strip().str.upper().str.replace('>', '', regex=False)
            pred_df['Model_Prob'] = pd.to_numeric(pred_df[prob_col_name], errors='coerce')

            prob_map = dict(zip(pred_df['Standard_ID'], pred_df['Model_Prob']))
            mapped_probs = report_df['Standard_ID'].map(prob_map)

            # 匹配失败降级策略：强制行号对齐
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

    # 约束4：极简合并与防御性算分死纪律
    metric_funcs = {
        "ACC": accuracy_score,
        "Recall": recall_score,
        "MCC": matthews_corrcoef,
        "AUROC": roc_auc_score,
        "AUPRC": average_precision_score
    }

    eval_result = {}

    for model_name in DATA_SCHEMA.keys():
        y_true = report_df['True_Label'].values
        y_prob = report_df[f"{model_name}_Prob"].values

        valid_mask = pd.notna(y_true) & pd.notna(y_prob)
        y_t_valid = y_true[valid_mask]
        y_p_valid = y_prob[valid_mask]

        model_metrics = {}
        for metric_name, func in metric_funcs.items():
            try:
                if len(np.unique(y_t_valid)) < 2:
                    raise ValueError("真实标签类别数不足2，无法计算区分度指标")
                score = func(y_t_valid, y_p_valid)
                model_metrics[metric_name] = float(score)
            except Exception:
                model_metrics[metric_name] = float('nan')

        eval_result[model_name] = model_metrics

    # 持久化输出
    with open("eval_result.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=4, ensure_ascii=False)
    print("[SUCCESS] 评测完成，结果已保存至 eval_result.json")

    report_df.to_csv("final_results_with_predictions.csv", index=False)
    print("[SUCCESS] 预测结果与真值已保存至 final_results_with_predictions.csv")

    # 生成评估曲线
    plt.figure(figsize=(12, 5))
    for i, model_name in enumerate(DATA_SCHEMA.keys()):
        y_true = report_df['True_Label'].values
        y_prob = report_df[f"{model_name}_Prob"].values
        valid_mask = pd.notna(y_true) & pd.notna(y_prob)
        y_t_v = y_true[valid_mask]
        y_p_v = y_prob[valid_mask]

        if len(np.unique(y_t_v)) < 2 or len(y_t_v) == 0:
            continue

        ax1 = plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y_t_v, y_p_v)
        auc_val = roc_auc_score(y_t_v, y_p_v)
        ax1.plot(fpr, tpr, label=f'{model_name} (AUC={auc_val:.2f})')
        ax1.set_title('ROC Curve')
        ax1.legend()

        ax2 = plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y_t_v, y_p_v)
        ap_val = average_precision_score(y_t_v, y_p_v)
        ax2.plot(recall, precision, label=f'{model_name} (AP={ap_val:.2f})')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()

    plt.tight_layout()
    plt.savefig("evaluation_curves.png", dpi=150)
    plt.close()
    print("[SUCCESS] 评估曲线已保存至 evaluation_curves.png")

if __name__ == '__main__':
    main()
