import os
import re
import pandas as pd
import subprocess
import tempfile
import shutil
import paramiko

# 引入项目配置中的超算环境变量
from config import (
    HPC_HOST,
    HPC_PORT,
    HPC_USER,
    HPC_PASS,
    HPC_TARGET_DIR,
    CONDA_SH_PATH,
    VLAB_ENV
)

def clean_sequence(seq: str) -> str:
    return str(seq).strip().upper().replace(" ", "")

def is_valid_peptide_sequence(seq: str) -> bool:
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq))

def parse_label_value(x):
    label_str = str(x).strip().lower()

    # 加入了常见的复数 (amps)、活性描述 (active) 和毒性描述 (toxin)
    pos_set = {"1", "1.0", "amp", "amps", "positive", "true", "pos", "yes", "active", "toxin"}
    neg_set = {"0", "0.0", "non-amp", "non-amps", "negative", "false", "neg", "no", "decoy", "non", "inactive", "non-toxin"}

    if label_str in pos_set:
        return 1
    if label_str in neg_set:
        return 0

    try:
        return 1 if float(label_str) > 0 else 0
    except Exception:
        return 0

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
        # 加入 entry, accession，并兼容 "sequence id" 这种带空格的命名
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
    name = name.replace("\u00A0", " ")  # 不可见空格
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name

def find_true_label_column(df: pd.DataFrame):
    cols = list(df.columns)

    bad_keywords = [
        "pred", "predict", "prediction", "prob", "score", "logit",
        "output", "model", "classifier", "confidence"
    ]

    def is_bad(col_name: str) -> bool:
        col_low = normalize_colname(col_name)
        return any(k in col_low for k in bad_keywords)

    strong_candidates = [
        "true labels", "true label", "ground truth", "ground_truth",
        "real label", "gold label", "labels", "label", "target", "class", "y_true",
        "activity", "type", "category", "is_amp"
    ]

    for candidate in strong_candidates:
        for c in cols:
            c_low = normalize_colname(c)
            if is_bad(c_low): continue
            if c_low == candidate or candidate in c_low:
                return c

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
        allowed = {
            "0", "1", "0.0", "1.0",
            "amp", "non-amp", "amps", "non-amps",
            "positive", "negative",
            "true", "false",
            "pos", "neg",
            "active", "inactive", "toxin", "non-toxin"
        }
        overlap_ratio = len(unique_vals & allowed) / max(len(unique_vals), 1)

        if overlap_ratio >= 0.5:
            candidate_cols.append(c)

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


def run_cdhit_clustering_on_hpc(df: pd.DataFrame, similarity_threshold=0.9, word_length=5) -> pd.DataFrame:
    """
    通过 SSH 将数据推送到远端超算，调用超算上的 CD-HIT 进行同源性聚类，并拉回结果。
    """
    print(f"\n>>> [远端协同] 开始通过超算进行 CD-HIT 同源性聚类 (相似度阈值 {similarity_threshold})...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None

    try:
        # 连接超算
        ssh.connect(HPC_HOST, HPC_PORT, HPC_USER, HPC_PASS)
        sftp = ssh.open_sftp()
        
        # 在本地创建临时目录处理数据
        with tempfile.TemporaryDirectory() as tmpdir:
            local_in = os.path.join(tmpdir, "local_input.fasta")
            local_out = os.path.join(tmpdir, "local_output.fasta")

            # 写入临时 FASTA 文件
            with open(local_in, "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    f.write(f">{row['id']}\n{row['sequence']}\n")

            remote_in = f"{HPC_TARGET_DIR}/tmp_cdhit_in.fasta"
            remote_out = f"{HPC_TARGET_DIR}/tmp_cdhit_out.fasta"

            print("   -> [SSH] 正在推送临时序列到超算节点...")
            ssh.exec_command(f"mkdir -p {HPC_TARGET_DIR}")
            sftp.put(local_in, remote_in)

            print("   -> [SSH] 正在超算执行 CD-HIT 聚类计算 (这可能需要几分钟)...")
            # 构建并在超算执行 cd-hit 命令
            cmd = f"source {CONDA_SH_PATH} && conda activate {VLAB_ENV} && cd-hit -i {remote_in} -o {remote_out} -c {similarity_threshold} -n {word_length} -d 0 -M 0"
            stdin, stdout, stderr = ssh.exec_command(cmd)
            
            # 等待远端执行完成并获取状态码
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status != 0:
                err_msg = stderr.read().decode("utf-8")
                print(f"   [错误] 超算 CD-HIT 执行失败，将保留原始数据。错误信息:\n{err_msg}")
                return df

            print("   -> [SSH] 聚类完成，正在拉取幸存代表序列...")
            sftp.get(remote_out, local_out)

            # 清理超算远端的临时文件，保持整洁
            ssh.exec_command(f"rm -f {remote_in} {remote_out} {remote_out}.clstr")

            # 从拉回的本地聚类结果中，提取幸存的代表性序列 ID
            surviving_ids = set()
            with open(local_out, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(">"):
                        surviving_ids.add(line.strip()[1:])
            
            # 使用幸存的 ID 过滤原始 DataFrame
            clustered_df = df[df["id"].isin(surviving_ids)].copy()
            print(f"   [成功] 远端 CD-HIT 协同聚类完美收官！序列总数由 {len(df)} 骤减至 {len(clustered_df)} 条精英序列。")
            return clustered_df

    except Exception as e:
        print(f"   [异常] 远端超算协同交互时发生意外: {e}")
        print("   -> 已安全回退：跳过同源聚类，目前仅保留了 100% 序列完全一致的去重结果。")
        return df
    finally:
        if sftp:
            sftp.close()
        ssh.close()


def auto_prepare_local_data(data_dir="data", min_length=10):
    print("\n========== [Phase 0] 本地数据自动嗅探与预处理 ==========")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"!!! 未找到 {data_dir} 目录，已自动创建。请将序列文件放入后重试。")
        return False

    all_records = []
    seq_counter = 1

    print(">>> 开始扫描数据目录...")
    for file in os.listdir(data_dir):
        if file in [
            "combined_test.fasta",
            "ground_truth.csv",
            "final_results_with_predictions.csv",
            "evaluation_curves.png",
            "eval_result.json",
        ]:
            continue

        file_path = os.path.join(data_dir, file)

        if os.path.isdir(file_path):
            print(f"[*] 发现文件: {file} -> [跳过] 目录")
            continue

        label = infer_label_from_filename(file)
        print(f"[*] 发现文件: {file} ", end="")

        if file.endswith((".fa", ".fasta", ".txt")):
            if label is None:
                print("-> [警告] 无法通过文件名判断正负样本，已跳过。建议文件名中包含 pos/amp/neg/decoy/non")
                continue

            print(f"-> 判定为: {'🟢 正样本(Label=1)' if label == 1 else '🔴 负样本(Label=0)'}")

            with open(file_path, "r", encoding="utf-8") as f:
                current_id = ""
                current_seq = ""

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith(">"):
                        if current_seq and len(current_seq) >= min_length:
                            current_seq = clean_sequence(current_seq)
                            if is_valid_peptide_sequence(current_seq):
                                all_records.append({
                                    "id": current_id,
                                    "sequence": current_seq,
                                    "label": label
                                })

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
                        all_records.append({
                            "id": current_id,
                            "sequence": current_seq,
                            "label": label
                        })

        elif file.endswith((".csv", ".xlsx", ".xls")):
            print("-> 判定为: 📊 混合表格文件")
            try:
                sheets_data = read_mixed_table(file_path)
                
                for sheet_name, df in sheets_data:
                    df.columns = [str(c).strip() for c in df.columns]

                    seq_col = find_sequence_column(df)
                    label_col = find_true_label_column(df)
                    id_col = find_id_column(df)

                    if len(sheets_data) > 1:
                        print(f"   [Sheet: {sheet_name}] sequence列: {seq_col} | 真实标签列: {label_col} | ID列: {id_col}")
                    else:
                        print(f"   [识别] sequence列: {seq_col} | 真实标签列: {label_col} | ID列: {id_col}")

                    if seq_col and label_col:
                        added_count = 0
                        for _, row in df.iterrows():
                            seq = clean_sequence(row[seq_col])
                            if not seq or seq.lower() == "nan":
                                continue
                            if len(seq) >= min_length and is_valid_peptide_sequence(seq):
                                raw_id = str(row[id_col]).strip() if id_col and pd.notna(row[id_col]) else f"table_{sheet_name}_seq_{seq_counter}"
                                seq_id = raw_id.replace(" ", "_")
                                lab = parse_label_value(row[label_col])

                                all_records.append({
                                    "id": seq_id,
                                    "sequence": seq,
                                    "label": lab
                                })
                                seq_counter += 1
                                added_count += 1
                        print(f"     -> [成功] 从该表提取了 {added_count} 条有效序列数据。")
                    else:
                        print("     -> [跳过] 该表缺少 sequence 或 真实 label 列，无法提取。")

            except Exception as e:
                print(f"  -> [错误] 读取表格失败: {e}")
        else:
            print("-> [跳过] 非支持文件类型")

    if not all_records:
        print("!!! 未提取到任何序列数据，流程终止。")
        return False

    df_records = pd.DataFrame(all_records)
    df_records["sequence"] = df_records["sequence"].astype(str).str.strip()
    df_records = df_records[df_records["sequence"] != ""].copy()

    # ==========================================
    # 1. 100% 序列完全一致去重 (Exact Deduplication)
    # ==========================================
    initial_count = len(df_records)
    df_records = df_records.drop_duplicates(subset=["sequence"], keep="first").copy()
    exact_dedup_count = initial_count - len(df_records)
    if exact_dedup_count > 0:
        print(f"\n>>> [清理] 100% 序列精确去重：剔除了 {exact_dedup_count} 条完全重复的序列。")

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

    # ==========================================
    # 2. 调用远端超算协同 CD-HIT 聚类去重
    # ==========================================
    # 默认 0.9 (90% 相似度), word_length=5
    df_records = run_cdhit_clustering_on_hpc(df_records, similarity_threshold=0.9, word_length=5)

    # ==========================================
    # 输出最终清洗完毕的数据
    # ==========================================
    fasta_path = os.path.join(data_dir, "combined_test.fasta")
    csv_path = os.path.join(data_dir, "ground_truth.csv")

    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df_records.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")

    df_records[["id", "sequence", "label"]].to_csv(csv_path, index=False)

    print(f"\n[+] 数据预处理、清洗与聚类完成！最终保留 {len(df_records)} 条高质量测试序列。")
    print(f"    🟢 正样本: {sum(df_records['label'] == 1)} 条")
    print(f"    🔴 负样本: {sum(df_records['label'] == 0)} 条")
    print(f"    📄 已生成模型测试库: {fasta_path}")
    print(f"    📊 已生成真值与合并表格: {csv_path}")
    return True


if __name__ == "__main__":
    auto_prepare_local_data(data_dir="data")