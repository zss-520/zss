# MLOps 工程师，以下是你的核心逻辑约束，请严格遵守！

import pandas as pd
import glob
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, roc_auc_score, average_precision_score
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

# 🚨 硬编码字典与前置配置
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
        "file_path": "data/AMPlify_out/AMPlify_balanced_results_20260325092223.tsv",
        "file_ext": ".tsv",
        "sep": "\t",
        "comment_char": None,
        "id_col": "Sequence_ID",
        "seq_col": "Sequence",
        "prob_col": "Probability_score"
    }
}

def main():
    # 🚨 真值表的绝对标准化（以序列为王）
    gt_df = pd.read_csv("data/ground_truth.csv")
    
    gt_seq_col = next((c for c in gt_df.columns if 'seq' in c.lower() or 'content' in c.lower()), gt_df.columns[0])
    gt_label_col = next((c for c in gt_df.columns if 'label' in c.lower() or 'target' in c.lower() or 'class' in c.lower()), gt_df.columns[-1])
    
    gt_df['Standard_ID'] = gt_df[gt_seq_col].astype(str).str.strip().str.upper()
    gt_df['True_Label'] = pd.to_numeric(gt_df[gt_label_col], errors='coerce')
    gt_df = gt_df.drop_duplicates(subset=['Standard_ID'])
    
    # 创建报告基座
    report_df = gt_df[['Standard_ID', 'True_Label']].copy()
    
    # 🚨 模型预测输出的绝对标准化
    for model_name, m_dict in DATA_SCHEMA.items():
        found_files = glob.glob(f"data/{model_name}_out/*{m_dict['file_ext']}")
        if not found_files:
            print(f"[WARNING] 未找到 {model_name} 的输出文件")
            report_df[f"{model_name}_Prob"] = 0.0
            continue
        
        file_path = found_files[0]
        
        # 🚨 强制使用 Pandas 原生读取
        pred_df = pd.read_csv(file_path, sep=m_dict['sep'], comment=m_dict['comment_char'])
        
        # 暴力清洗表头
        pred_df.columns = pred_df.columns.str.replace('#', '').str.strip()
        
        # 🚨 极简强悍的列提取纪律
        try:
            target_col_name = m_dict['seq_col'] if m_dict.get('seq_col') else m_dict['id_col']
            prob_col_name = m_dict['prob_col']

            pred_df['Standard_ID'] = pred_df[target_col_name].astype(str).str.strip().str.upper()
            pred_df['Model_Prob'] = pd.to_numeric(pred_df[prob_col_name], errors='coerce')
            
            # 尝试通过"序列/ID"进行精准映射
            prob_map = dict(zip(pred_df['Standard_ID'], pred_df['Model_Prob']))
            mapped_probs = report_df['Standard_ID'].map(prob_map)

            # 🚨 终极绝招：如果匹配全部失败（全是 NaN），且输出行数与真值表完全一致，触发强制行号对齐！
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
    
    # 🚨 极简合并与算分死纪律
    results = {}
    
    # 用于绘制曲线的数据
    curve_data = {}
    
    for model_name in DATA_SCHEMA.keys():
        y_true = report_df['True_Label']
        y_prob = report_df[f"{model_name}_Prob"]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # 过滤掉 NaN 值
        mask = ~y_prob.isna()
        y_true_filtered = y_true[mask]
        y_prob_filtered = y_prob[mask]
        y_pred_filtered = y_pred[mask]
        
        try:
            ACC = accuracy_score(y_true_filtered, y_pred_filtered)
            Recall = recall_score(y_true_filtered, y_pred_filtered)
            MCC = matthews_corrcoef(y_true_filtered, y_pred_filtered)
            AUROC = roc_auc_score(y_true_filtered, y_prob_filtered)
            AUPRC = average_precision_score(y_true_filtered, y_prob_filtered)
            
            results[model_name] = {
                "ACC": float(ACC),
                "Recall": float(Recall),
                "MCC": float(MCC),
                "AUROC": float(AUROC),
                "AUPRC": float(AUPRC)
            }
            
            # 保存用于绘制曲线的数据
            curve_data[model_name] = {
                'y_true': y_true_filtered,
                'y_prob': y_prob_filtered
            }
        except Exception:
            # 如果算分时抛出异常，必须在 except 块中将这几个指标全部赋值 0.0
            results[model_name] = {
                "ACC": 0.0,
                "Recall": 0.0,
                "MCC": 0.0,
                "AUROC": 0.0,
                "AUPRC": 0.0
            }
            
            # 即使计算失败也尝试保存曲线数据
            curve_data[model_name] = {
                'y_true': y_true_filtered,
                'y_prob': y_prob_filtered
            }
    
    # 🚨 绘制评估曲线
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制 ROC 曲线
    for model_name in DATA_SCHEMA.keys():
        if model_name in curve_data:
            data = curve_data[model_name]
            y_true = data['y_true']
            y_prob = data['y_prob']
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = roc_auc_score(y_true, y_prob)
                axes[0].plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
            except Exception:
                pass  # 如果无法计算 ROC 曲线，则跳过
    
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制 PR 曲线
    for model_name in DATA_SCHEMA.keys():
        if model_name in curve_data:
            data = curve_data[model_name]
            y_true = data['y_true']
            y_prob = data['y_prob']
            
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
                axes[1].plot(recall, precision, label=f'{model_name} (AUPRC = {auprc:.3f})')
            except Exception:
                pass  # 如果无法计算 PR 曲线，则跳过
    
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🚨 保存最终结果 CSV
    report_df.to_csv('final_results_with_predictions.csv', index=False)
    
    # 🚨 强制输出文件名纪律：必须严格命名为 eval_result.json
    with open('eval_result.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
