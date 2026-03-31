import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =====================================================================
# 通用工具
# =====================================================================

def clean_sequence(seq: str) -> str:
    return str(seq).strip().upper().replace(" ", "")


def is_valid_peptide_sequence(seq: str) -> bool:
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq))


def parse_label_value(x):
    label_str = str(x).strip().lower()
    pos_set = {"1", "1.0", "amp", "amps", "positive", "true", "pos", "yes", "active", "toxin"}
    neg_set = {"0", "0.0", "non-amp", "non-amps", "negative", "false", "neg", "no", "decoy", "non", "inactive", "non-toxin"}

    if label_str in pos_set:
        return 1
    if label_str in neg_set:
        return 0
    try:
        return 1 if float(label_str) > 0 else 0
    except Exception:
        return None


def infer_label_from_filename(filename: str):
    lower_name = filename.lower()
    if "amp" in lower_name and "non" not in lower_name and "decoy" not in lower_name:
        return 1
    if "pos" in lower_name:
        return 1
    if "neg" in lower_name or "decoy" in lower_name or "non" in lower_name:
        return 0
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
        if (
            c_low == "id"
            or c_low.endswith("_id")
            or c_low.endswith(" id")
            or "identifier" in c_low
            or "name" in c_low
            or "entry" in c_low
            or "accession" in c_low
        ):
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

    strong_candidates = [
        "true labels", "true label", "ground truth", "ground_truth", "real label",
        "gold label", "labels", "label", "target", "class", "y_true",
        "activity", "type", "category", "is_amp"
    ]

    for candidate in strong_candidates:
        for c in cols:
            c_low = normalize_colname(c)
            if is_bad(c_low):
                continue
            if c_low == candidate or candidate in c_low:
                return c

    for c in cols:
        c_low = normalize_colname(c)
        if is_bad(c_low):
            continue
        if "true" in c_low and "label" in c_low:
            return c
        if "ground" in c_low and "truth" in c_low:
            return c

    candidate_cols = []
    for c in cols:
        c_low = normalize_colname(c)
        if is_bad(c_low):
            continue
        series = df[c].dropna()
        if len(series) == 0:
            continue
        unique_vals = set(str(v).strip().lower() for v in series.unique())
        allowed = {
            "0", "1", "0.0", "1.0", "amp", "non-amp", "amps", "non-amps",
            "positive", "negative", "true", "false", "pos", "neg", "active", "inactive",
            "toxin", "non-toxin"
        }
        overlap_ratio = len(unique_vals & allowed) / max(len(unique_vals), 1)
        if overlap_ratio >= 0.5:
            candidate_cols.append(c)

    if candidate_cols:
        scored = []
        for c in candidate_cols:
            c_low = normalize_colname(c)
            score = 0
            if "true" in c_low:
                score += 4
            if "label" in c_low:
                score += 4
            if "target" in c_low:
                score += 3
            if "class" in c_low or "type" in c_low:
                score += 2
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
# Strategy 读取
# =====================================================================

DEFAULT_STRATEGY = {
    "task_type": "binary_amp_classification",
    "label_definition": {
        "positive_rule": "实验支持为 AMP 的肽作为正样本",
        "negative_rule": "优先实验低活性/无活性肽作为负样本；不把未注释序列当负样本",
        "ambiguous_rule": "灰区样本单独保存，不参与主测试"
    },
    "sequence_constraints": {
        "allowed_alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "min_len": 5,
        "max_len": 50,
        "allow_modification": False,
        "allow_noncanonical": False
    },
    "deduplication_policy": {
        "exact_dedup": True,
        "near_duplicate_policy": "same_sequence_only",
        "homology_control_recommended": True
    },
    "export_schema": [
        "id", "sequence", "label", "source_dataset", "evidence_level"
    ]
}


def load_benchmark_strategy(path: str = "data/benchmark_strategy.json") -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                merged = DEFAULT_STRATEGY.copy()
                for k, v in data.items():
                    if isinstance(v, dict) and isinstance(merged.get(k), dict):
                        merged[k] = {**merged[k], **v}
                    else:
                        merged[k] = v
                return merged
        except Exception as e:
            print(f"!!! [Warning] 读取 benchmark strategy 失败，使用默认策略: {e}")
    return DEFAULT_STRATEGY.copy()


# =====================================================================
# 读取原始文件
# =====================================================================

def parse_fasta_file(file_path: str, fallback_label: Optional[int], min_length: int) -> List[Dict]:
    records = []
    seq_counter = 1

    if fallback_label is None:
        print(f"    -> [跳过] FASTA/TXT 文件无法通过文件名判断正负样本: {os.path.basename(file_path)}")
        return records

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        current_id, current_seq = "", ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    current_seq = clean_sequence(current_seq)
                    if len(current_seq) >= min_length and is_valid_peptide_sequence(current_seq):
                        records.append({
                            "id": current_id,
                            "sequence": current_seq,
                            "label": fallback_label,
                            "source_dataset": os.path.basename(file_path),
                            "evidence_level": "filename_inferred",
                        })
                raw_id = line[1:].split()[0] if line[1:].strip() else f"seq_{seq_counter}"
                current_id = raw_id.replace(" ", "_")
                current_seq = ""
                seq_counter += 1
            else:
                if not current_id:
                    current_id = f"seq_{seq_counter}"
                    seq_counter += 1
                current_seq += line

        if current_seq:
            current_seq = clean_sequence(current_seq)
            if len(current_seq) >= min_length and is_valid_peptide_sequence(current_seq):
                records.append({
                    "id": current_id,
                    "sequence": current_seq,
                    "label": fallback_label,
                    "source_dataset": os.path.basename(file_path),
                    "evidence_level": "filename_inferred",
                })
    return records


def standardize_table_records(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    seq_col = find_sequence_column(df)
    if not seq_col:
        return pd.DataFrame()

    id_col = find_id_column(df)
    label_col = find_true_label_column(df)

    out = pd.DataFrame()
    out["sequence"] = df[seq_col].astype(str).map(clean_sequence)

    if id_col:
        out["id"] = df[id_col].astype(str).str.split().str[0].str.strip()
    else:
        out["id"] = [f"{Path(file_name).stem}_seq_{i+1}" for i in range(len(df))]

    if label_col:
        out["label"] = df[label_col].map(parse_label_value)
        out["evidence_level"] = "table_label"
    else:
        fallback_label = infer_label_from_filename(file_name)
        out["label"] = fallback_label
        out["evidence_level"] = "filename_inferred" if fallback_label is not None else "unknown"

    out["source_dataset"] = Path(file_name).stem
    return out


def ingest_single_folder(input_dir: str, dataset_name: str, min_length: int) -> pd.DataFrame:
    print(f"\n========== 开始导入数据集: [{dataset_name}] ==========")

    if not os.path.exists(input_dir):
        print(f"!!! [错误] 找不到输入目录: {input_dir}")
        return pd.DataFrame()

    all_records: List[Dict] = []

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            continue

        print(f"[*] 导入文件: {file}")

        if file.endswith((".fa", ".fasta", ".txt")):
            fallback_label = infer_label_from_filename(file)
            records = parse_fasta_file(file_path, fallback_label, min_length)
            all_records.extend(records)
            print(f"    -> FASTA/TXT 导入 {len(records)} 条")
            continue

        if file.endswith((".csv", ".xlsx", ".xls")):
            try:
                sheets_data = read_mixed_table(file_path)
                total = 0
                for sheet_name, df in sheets_data:
                    std_df = standardize_table_records(df, f"{file}__{sheet_name}")
                    if not std_df.empty:
                        total += len(std_df)
                        all_records.extend(std_df.to_dict(orient="records"))
                print(f"    -> 表格导入 {total} 条")
            except Exception as e:
                print(f"    -> [跳过] 表格读取失败: {e}")
            continue

        print("    -> [跳过] 不支持的文件类型")

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)


# =====================================================================
# 测试集处理逻辑
# =====================================================================

def filter_sequences(df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
    seq_cfg = strategy["sequence_constraints"]
    min_len = int(seq_cfg.get("min_len", 5))
    max_len = int(seq_cfg.get("max_len", 60))

    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).map(clean_sequence)
    df = df[df["sequence"].str.len().between(min_len, max_len, inclusive="both")]
    df = df[df["sequence"].map(is_valid_peptide_sequence)]

    if "id" not in df.columns:
        df["id"] = [f"seq_{i+1}" for i in range(len(df))]
    df["id"] = df["id"].astype(str).str.split().str[0].str.strip()

    return df.reset_index(drop=True)


def apply_gold_label_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    def normalize_binary_label(x):
        if x in [0, 1]:
            return x
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return parse_label_value(x)

    df["label"] = df["label"].map(normalize_binary_label)

    ambiguous_df = df[df["label"].isna()].copy()
    gold_df = df[df["label"].isin([0, 1])].copy()

    return gold_df.reset_index(drop=True), ambiguous_df.reset_index(drop=True)


def deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    priority = {"table_label": 2, "filename_inferred": 1, "unknown": 0}
    df["__prio"] = df["evidence_level"].map(lambda x: priority.get(str(x), 0))

    df = df.sort_values(by=["sequence", "__prio"], ascending=[True, False])
    df = df.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
    df = df.drop(columns=["__prio"])

    return df


def assign_unique_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    counts = {}
    new_ids = []
    for seq_id in df["id"].astype(str):
        base = seq_id if seq_id else "seq"
        if base not in counts:
            counts[base] = 0
            new_ids.append(base)
        else:
            counts[base] += 1
            new_ids.append(f"{base}__dup{counts[base]}")
    df["id"] = new_ids
    return df


# =====================================================================
# 导出
# =====================================================================

def write_fasta(df: pd.DataFrame, fasta_path: Path):
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")


def export_testset(df: pd.DataFrame, ambiguous_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    write_fasta(df, out_dir / "combined_test.fasta")
    df[["id", "sequence", "label"]].to_csv(out_dir / "ground_truth.csv", index=False)

    if not ambiguous_df.empty:
        ambiguous_df.to_csv(out_dir / "ambiguous.csv", index=False)


# =====================================================================
# 主流程
# =====================================================================

def process_single_folder(input_dir: str, dataset_name: str):
    strategy = load_benchmark_strategy()
    min_len = int(strategy["sequence_constraints"].get("min_len", 5))

    raw_df = ingest_single_folder(input_dir, dataset_name, min_length=min_len)
    if raw_df.empty:
        print(f"!!! [{dataset_name}] 未读取到有效记录")
        return False

    filtered_df = filter_sequences(raw_df, strategy)
    gold_df, ambiguous_df = apply_gold_label_rules(filtered_df)
    gold_df = deduplicate_records(gold_df)
    gold_df = assign_unique_ids(gold_df)

    out_dir = Path(f"data/datasets/{dataset_name}")
    export_testset(gold_df, ambiguous_df, out_dir)

    print(f"\n[+] [{dataset_name}] 金标准测试集构建完成")
    print(f"    总记录数: {len(filtered_df)}")
    print(f"    保留 gold: {len(gold_df)}")
    print(f"    ambiguous: {len(ambiguous_df)}")
    print(f"    正样本: {sum(gold_df['label'] == 1)}")
    print(f"    负样本: {sum(gold_df['label'] == 0)}")
    print(f"    输出目录: {out_dir}")
    return True


def main():
    print(">>> 启动二分类 AMP 金标准测试集预处理流水线（全自动对接模式）。")
    
    # 1. 动态读取 AI 首席科学家生成的策略文件
    strategy = load_benchmark_strategy()
    datasets = strategy.get("recommended_datasets", [])

    if not datasets:
        print("!!! [Error] benchmark_strategy.json 中没有 recommended_datasets，请先执行 prepare_models.py")
        return

    # 2. 遍历策略中的每一个顶刊数据集
    for ds in datasets:
        # 名字必须和 dataset_fetcher.py 保持一致（替换空格为下划线）
        ds_name = ds.get("dataset_name", "Unknown_Dataset").replace(" ", "_")
        
        # 指向刚刚从 GitHub 下载提取的原始数据文件夹
        input_folder = f"data/datasets/{ds_name}"

        if os.path.exists(input_folder) and os.path.isdir(input_folder):
            print(f"\n[🚀] 发现目标数据集目录: {input_folder}，准备启动数据炼金炉...")
            # 3. 原地执行清洗，洗好的 combined_test.fasta 也会存在这个文件夹里
            process_single_folder(input_folder, ds_name)
        else:
            print(f"\n!!! [Skip] 找不到原始数据文件夹: {input_folder} (可能是 Web 数据库，需手动放入数据)")

    print("\n🎉 所有测试集物理清洗、长度裁切与标签对齐完毕！")
    print("👉 最终的金标准评估文件 (combined_test.fasta) 已生成，随时可以发往超算进行打分！")

if __name__ == "__main__":
    main()