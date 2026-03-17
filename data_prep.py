import os
import re
import pandas as pd
from pathlib import Path

# =====================================================================
# 通用数据清洗与列名嗅探工具 (保留了你原本优秀的逻辑，完全不动)
# =====================================================================

def clean_sequence(seq: str) -> str:
    return str(seq).strip().upper().replace(" ", "")

def is_valid_peptide_sequence(seq: str) -> bool:
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq))

def parse_label_value(x):
    label_str = str(x).strip().lower()
    pos_set = {"1", "1.0", "amp", "amps", "positive", "true", "pos", "yes", "active", "toxin"}
    neg_set = {"0", "0.0", "non-amp", "non-amps", "negative", "false", "neg", "no", "decoy", "non", "inactive", "non-toxin"}

    if label_str in pos_set: return 1
    if label_str in neg_set: return 0
    try:
        return 1 if float(label_str) > 0 else 0
    except Exception:
        return 0

def infer_label_from_filename(filename: str):
    lower_name = filename.lower()
    if "amp" in lower_name and "non" not in lower_name and "decoy" not in lower_name: return 1
    if "pos" in lower_name: return 1
    if "neg" in lower_name or "decoy" in lower_name or "non" in lower_name: return 0
    return None

def find_sequence_column(df: pd.DataFrame):
    candidates = []
    for c in df.columns:
        c_low = str(c).lower()
        if any(k in c_low for k in ["sequence", "seq", "peptide", "aa_seq", "protein"]):
            candidates.append(c)
    return candidates[0] if candidates else None

def find_id_column(df: pd.DataFrame):
    candidates = []
    for c in df.columns:
        c_low = str(c).lower().strip()
        if (c_low == "id" or c_low.endswith("_id") or c_low.endswith(" id") or 
            "identifier" in c_low or "name" in c_low or "entry" in c_low or "accession" in c_low):
            candidates.append(c)
    return candidates[0] if candidates else None

def normalize_colname(name: str) -> str:
    name = str(name)
    name = name.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    name = name.replace("\u00A0", " ")  
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name

def find_true_label_column(df: pd.DataFrame):
    cols = list(df.columns)
    bad_keywords = ["pred", "predict", "prediction", "prob", "score", "logit", "output", "model", "classifier", "confidence"]

    def is_bad(col_name: str) -> bool:
        col_low = normalize_colname(col_name)
        return any(k in col_low for k in bad_keywords)

    strong_candidates = ["true labels", "true label", "ground truth", "ground_truth", "real label", "gold label", "labels", "label", "target", "class", "y_true", "activity", "type", "category", "is_amp"]

    for candidate in strong_candidates:
        for c in cols:
            c_low = normalize_colname(c)
            if is_bad(c_low): continue
            if c_low == candidate or candidate in c_low: return c

    for c in cols:
        c_low = normalize_colname(c)
        if is_bad(c_low): continue
        if "true" in c_low and "label" in c_low: return c
        if "ground" in c_low and "truth" in c_low: return c

    candidate_cols = []
    for c in cols:
        c_low = normalize_colname(c)
        if is_bad(c_low): continue
        series = df[c].dropna()
        if len(series) == 0: continue
        unique_vals = set(str(v).strip().lower() for v in series.unique())
        allowed = {"0", "1", "0.0", "1.0", "amp", "non-amp", "amps", "non-amps", "positive", "negative", "true", "false", "pos", "neg", "active", "inactive", "toxin", "non-toxin"}
        overlap_ratio = len(unique_vals & allowed) / max(len(unique_vals), 1)
        if overlap_ratio >= 0.5: candidate_cols.append(c)

    if candidate_cols:
        scored = []
        for c in candidate_cols:
            c_low = normalize_colname(c)
            score = 0
            if "true" in c_low: score += 4
            if "label" in c_low: score += 4
            if "target" in c_low: score += 3
            if "class" in c_low or "type" in c_low: score += 2
            scored.append((score, c))
        scored.sort(reverse=True)
        return scored[0][1]
    return None

def read_mixed_table(file_path: str):
    lower = file_path.lower()
    if lower.endswith(".csv"):
        return [("CSV", pd.read_csv(file_path))]
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        xls = pd.read_excel(file_path, sheet_name=None)
        return list(xls.items())
    raise ValueError(f"不支持的表格格式: {file_path}")

# =====================================================================
# 核心业务逻辑：独立处理单个文件夹 (不合并外部数据，不去重，不聚类)
# =====================================================================

def process_single_folder(input_dir: str, dataset_name: str, min_length=10):
    print(f"\n========== 开始处理数据集: [{dataset_name}] ==========")
    if not os.path.exists(input_dir):
        print(f"!!! [错误] 找不到输入目录: {input_dir}")
        return False

    all_records = []
    seq_counter = 1

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            continue

        label = infer_label_from_filename(file)
        print(f"[*] 解析文件: {file} ", end="")

        # 1. 解析 FASTA / TXT
        if file.endswith((".fa", ".fasta", ".txt")):
            if label is None:
                print("-> [跳过] 无法通过文件名判断正负样本。")
                continue
            print(f"-> 判定为: {'🟢 正样本' if label == 1 else '🔴 负样本'}")

            with open(file_path, "r", encoding="utf-8") as f:
                current_id, current_seq = "", ""
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.startswith(">"):
                        if current_seq and len(current_seq) >= min_length:
                            current_seq = clean_sequence(current_seq)
                            if is_valid_peptide_sequence(current_seq):
                                all_records.append({"id": current_id, "sequence": current_seq, "label": label})
                        raw_id = line[1:].split()[0] if line[1:].strip() else f"{'pos' if label == 1 else 'neg'}_seq_{seq_counter}"
                        current_id = raw_id.replace(" ", "_")
                        current_seq = ""
                    else:
                        if not current_id:
                            current_id = f"{'pos' if label == 1 else 'neg'}_seq_{seq_counter}"
                            seq_counter += 1
                        current_seq += line

                if current_seq and len(current_seq) >= min_length:
                    current_seq = clean_sequence(current_seq)
                    if is_valid_peptide_sequence(current_seq):
                        all_records.append({"id": current_id, "sequence": current_seq, "label": label})

        # 2. 解析 CSV / Excel (多 Sheet 支持)
        elif file.endswith((".csv", ".xlsx", ".xls")):
            print("-> 📊 混合表格文件")
            try:
                sheets_data = read_mixed_table(file_path)
                for sheet_name, df in sheets_data:
                    df.columns = [str(c).strip() for c in df.columns]
                    seq_col = find_sequence_column(df)
                    label_col = find_true_label_column(df)
                    id_col = find_id_column(df)

                    if seq_col and label_col:
                        added_count = 0
                        for _, row in df.iterrows():
                            seq = clean_sequence(row[seq_col])
                            if not seq or seq.lower() == "nan": continue
                            if len(seq) >= min_length and is_valid_peptide_sequence(seq):
                                raw_id = str(row[id_col]).strip() if id_col and pd.notna(row[id_col]) else f"table_{sheet_name}_seq_{seq_counter}"
                                seq_id = raw_id.replace(" ", "_")
                                lab = parse_label_value(row[label_col])
                                all_records.append({"id": seq_id, "sequence": seq, "label": lab})
                                seq_counter += 1
                                added_count += 1
                        print(f"    -> [成功] Sheet '{sheet_name}' 提取了 {added_count} 条序列。")
                    else:
                        print(f"    -> [跳过] Sheet '{sheet_name}' 缺少 sequence 或 label 列。")
            except Exception as e:
                print(f"  -> [错误] 读取表格失败: {e}")
        else:
            print("-> [跳过] 格式不支持")

    if not all_records:
        print(f"!!! [{dataset_name}] 未提取到任何有效序列，跳过生成。")
        return False

    df_records = pd.DataFrame(all_records)
    
    # 【改动核心】：完全移除了基于 sequence 的去重，也移除了 CD-HIT。
    # 仅作唯一一步妥协：处理 ID 重名。因为如果不对相同 ID 加后缀，后续写入 FASTA 会导致很多模型报错！
    dup_count = df_records["id"].duplicated().sum()
    if dup_count > 0:
        counts = {}
        new_ids = []
        for seq_id in df_records["id"]:
            if seq_id not in counts:
                counts[seq_id] = 0
                new_ids.append(seq_id)
            else:
                counts[seq_id] += 1
                new_ids.append(f"{seq_id}__dup{counts[seq_id]}")
        df_records["id"] = new_ids

    # 建立输出目录，把结果严格限定在 data/datasets/{dataset_name}/
    out_dir = Path(f"data/datasets/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fasta_path = out_dir / "combined_test.fasta"
    csv_path = out_dir / "ground_truth.csv"

    # 写入最终文件
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df_records.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")

    df_records[["id", "sequence", "label"]].to_csv(csv_path, index=False)

    print(f"\n[+] [{dataset_name}] 处理完成！保留原始条数: {len(df_records)} 条。")
    print(f"    🟢 正样本: {sum(df_records['label'] == 1)} 条")
    print(f"    🔴 负样本: {sum(df_records['label'] == 0)} 条")
    print(f"    📁 输出目录: {out_dir}")
    return True

def main():
    print(">>> 启动独立数据预处理流水线 (无去重、无聚类模式)...")
    
    # 定义你要处理的源文件夹和对应的输出数据集名称
    # 按照你说的目录：part2/data/test1 到 test3
    datasets_to_process = {
        "data/test1": "Dataset_1_test1",
        "data/test2": "Dataset_2_test2",
        "data/test3": "Dataset_3_test3",
    }
    
    for input_folder, dataset_name in datasets_to_process.items():
        process_single_folder(input_folder, dataset_name)
        
    print("\n🎉 所有数据集预处理完毕！你可以去 data/datasets/ 目录下查看生成的文件。")

if __name__ == "__main__":
    main()