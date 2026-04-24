import os
import re
import zipfile
import tarfile
import importlib.util
from openai import OpenAI
from config import MODEL_NAME
from prompts import DATASET_ETL_AGENT_PROMPT
import shutil
import json
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
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
    # 🚨 补丁：增加 'acp'（抗癌肽）作为正样本关键词
    if any(k in lower_name for k in ["amp", "pos", "acp", "active"]) and \
       not any(k in lower_name for k in ["non", "decoy", "neg"]):
        return 1
    if any(k in lower_name for k in ["neg", "decoy", "non", "inactive"]):
        return 0
    return None


def find_sequence_column(df: pd.DataFrame):
    """表头模糊匹配 + 序列特征自动识别"""
    candidates = []
    for c in df.columns:
        c_low = str(c).lower()
        if any(k in c_low for k in ["sequence", "seq", "peptide", "aa_seq"]):
            candidates.append(c)
    
    if candidates: return candidates[0]
    
    # 🚨 核心逻辑：如果没有匹配到任何列名，但第一列的内容全都是合法的氨基酸序列
    if not df.empty:
        # 检查前 10 行（或全部）是否为合法序列
        test_rows = df.iloc[:, 0].astype(str).map(clean_sequence).head(10)
        if all(is_valid_peptide_sequence(s) for s in test_rows if s):
            return df.columns[0]
            
    return None


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


def sniff_file_format(file_path: str) -> str:
    """智能文件格式嗅探器：解决后缀名欺骗问题"""
    lower_name = file_path.lower()
    if lower_name.endswith((".fa", ".fasta")): return "fasta"
    if lower_name.endswith(".json"): return "json"
    if lower_name.endswith((".csv", ".xlsx", ".xls", ".tsv")): return "table"
    
    # 🚨 核心修复：把生信老古董后缀 .data 和 .dat 也加入拆箱嗅探范围
    if lower_name.endswith((".txt", ".data", ".dat")):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in range(5):
                    line = f.readline().strip()
                    if not line: continue
                    if line.startswith(">"): return "fasta"
                    if "\t" in line or "," in line: return "table"
        except: pass
        return "table" # 如果看不出，丢给表格解析器去碰运气
        
    return "unknown"
def read_mixed_table(file_path: str):
    """增强版表格读取器：支持无表头纯序列文件"""
    lower = file_path.lower()
    if lower.endswith((".csv", ".tsv", ".txt", ".data", ".dat")):
        for sep in [',', '\t', ';', ' ']: # 增加空格分隔符支持
            for enc in ['utf-8', 'gbk', 'latin1']:
                try:
                    # 🚨 核心逻辑：先不带表头尝试读取前 5 行
                    df_sample = pd.read_csv(file_path, encoding=enc, sep=sep, header=None, nrows=5)
                    first_val = str(df_sample.iloc[0, 0]).strip().upper()
                    
                    # 如果第一行第一列就符合氨基酸特征，说明是无表头纯数据
                    if is_valid_peptide_sequence(first_val):
                        df = pd.read_csv(file_path, encoding=enc, sep=sep, header=None, on_bad_lines='skip')
                        return [("RawTable", df)]
                    else:
                        # 否则按照有表头正常读取
                        df = pd.read_csv(file_path, encoding=enc, sep=sep, on_bad_lines='skip')
                        return [("StandardTable", df)]
                except:
                    continue
        
    if lower.endswith((".xlsx", ".xls")):
        try:
            xls = pd.read_excel(file_path, sheet_name=None)
            return list(xls.items())
        except Exception as e:
            print(f"    ⚠️ Excel 读取异常: {e}")
    return []


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
    # 🚨 补丁：强制转换所有表头为字符串，并填充空值，防止类型崩溃
    df.columns = [str(c).strip() for c in df.columns]
    
    seq_col = find_sequence_column(df)
    if not seq_col: return pd.DataFrame()

    id_col = find_id_column(df)
    label_col = find_true_label_column(df)

    out = pd.DataFrame()
    # 🚨 补丁：在调用 .str 之前先用 .astype(str) 确保万无一失
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
def decompress_archives(target_dir: str):
    """递归解压目录下所有的 zip, tar.gz, tar 文件 (带防假死修复)"""
    path = Path(target_dir)
    found_new = True
    while found_new:
        found_new = False
        for ext in ["*.zip", "*.tar.gz", "*.tgz", "*.tar"]:
            for archive in path.rglob(ext):
                # 智能推断解压目标文件夹名称
                if archive.name.endswith(".tar.gz"): folder_name = archive.name[:-7]
                elif archive.name.endswith(".tgz"): folder_name = archive.name[:-4]
                else: folder_name = archive.stem
                
                extract_to = archive.parent / folder_name
                
                # 🚨 核心修复：如果文件夹不存在，或者虽然存在但里面是空的，就强行解压！
                if not extract_to.exists() or not any(extract_to.iterdir()):
                    print(f"    -> 📦 自动解压: {archive.name} ...")
                    try:
                        extract_to.mkdir(parents=True, exist_ok=True)
                        shutil.unpack_archive(str(archive), str(extract_to))
                        found_new = True
                    except Exception as e:
                        print(f"    ⚠️ 解压失败: {e}")

# =====================================================================
# 🧬 JSON 智能转换引擎 (修复版：加入递归吸尘器)
# =====================================================================

def _recursive_find_sequences(data, results, current_id=""):
    """递归爬虫：穿透任何深度的嵌套 JSON，吸取多肽序列"""
    if isinstance(data, dict):
        # 尝试提取 MIBiG 特有的 accession ID 作为标识
        next_id = data.get("mibig_accession", current_id)
        
        for k, v in data.items():
            k_lower = str(k).lower()
            # 命中核心关键词：只要键名带有这些字眼，就把值当作序列吸出
            if isinstance(v, str) and any(kw in k_lower for kw in ['translation', 'sequence', 'peptide']):
                # 先不管合不合法，统一吸出来，后续清洗引擎会过滤脏数据
                results.append({"id": next_id, "sequence": v})
            elif isinstance(v, (dict, list)):
                _recursive_find_sequences(v, results, next_id)
                
    elif isinstance(data, list):
        for i, item in enumerate(data):
            # 对于列表里的项，加上索引防止 ID 冲突
            _recursive_find_sequences(item, results, f"{current_id}_{i}")


def parse_json_to_df(file_path: str) -> pd.DataFrame:
    """尝试将各种格式的 JSON 转换为 DataFrame"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        
        # 🚨 1. 启动“递归吸尘器” (专治 MIBiG 等深层基因组 JSON)
        results = []
        _recursive_find_sequences(data, results, Path(file_path).stem)
        
        # 如果吸到了哪怕一条序列，直接构造成 DataFrame 返回！
        if results:
            df = pd.DataFrame(results)
            # 为了骗过后续的标准清洗引擎，人为补充一个 label 列（设为 None 等待二次推断）
            df['label'] = None 
            return df

        # 2. 如果没吸到，回退到普通的列表/扁平化解析 (针对 GRAMPA 等常规 JSON)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list) and len(v) > 0:
                    return pd.DataFrame(v)
            return pd.json_normalize(data)
            
    except Exception as e:
        print(f"    ⚠️ JSON 转换失败: {e}")
    return pd.DataFrame()

def ingest_single_folder(input_dir: str, dataset_name: str, min_length: int) -> pd.DataFrame:
    print(f"\n========== 开始导入数据集: [{dataset_name}] ==========")
    if not os.path.exists(input_dir): return pd.DataFrame()

    decompress_archives(input_dir)
    all_records = []

    for root, _, files in os.walk(input_dir):
        # 🚨 核心修复：屏蔽 Mac 电脑压缩包带出来的垃圾文件
        if "__MACOSX" in root: continue
        
        for file in files:
            if file.startswith("."): continue # 屏蔽隐藏文件
            
            file_path = os.path.join(root, file)
            fmt = sniff_file_format(file_path)
            
            if fmt == "unknown": continue
            print(f"[*] 处理文件: {file} [{fmt}]")

            if fmt == "fasta":
                fallback_label = infer_label_from_filename(file)
                records = parse_fasta_file(file_path, fallback_label, min_length)
                all_records.extend(records)
                print(f"    -> FASTA 导入 {len(records)} 条")
                
            elif fmt == "table":
                try:
                    for sheet_name, df in read_mixed_table(file_path):
                        std_df = standardize_table_records(df, f"{file}__{sheet_name}")
                        if not std_df.empty:
                            all_records.extend(std_df.to_dict(orient="records"))
                            print(f"    -> 表格导入 {len(std_df)} 条")
                except Exception as e:
                    print(f"    -> [跳过] 表格解析报错: {e}")

            elif fmt == "json":
                try:
                    df_json = parse_json_to_df(file_path)
                    if not df_json.empty:
                        std_df = standardize_table_records(df_json, file)
                        if not std_df.empty:
                            all_records.extend(std_df.to_dict(orient="records"))
                            print(f"    -> JSON 导入 {len(std_df)} 条")
                except Exception as e:
                    print(f"    -> [跳过] JSON 报错: {e}")

    return pd.DataFrame(all_records)

# =====================================================================
# 🧠 Agentic ETL: 动态嗅探与代码生成引擎
# =====================================================================

def sniff_dataset_directory(dataset_dir: str) -> str:
    """自动探查目录，生成数据样本报告供 Agent 阅读"""
    target_path = Path(dataset_dir)
    if not target_path.exists(): return ""

    # 强制解压所有深层压缩包，防止有遗漏
    decompress_archives(dataset_dir)

    report = f"📂 目标数据集目录: {dataset_dir}\n"
    report += "-" * 40 + "\n"

    valid_exts = {".csv", ".tsv", ".txt", ".xlsx", ".json", ".fasta", ".fa", ".dat", ".data"}
    for root, _, files in os.walk(dataset_dir):
        if "__MACOSX" in root: continue
        for file in files:
            if file.startswith("."): continue
            ext = Path(file).suffix.lower()
            if ext not in valid_exts: continue

            file_path = Path(root) / file
            rel_path = file_path.relative_to(target_path)
            report += f"📄 文件: {rel_path}\n"

            # 读取前 5 行纯文本特征
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    head_lines = [next(f) for _ in range(5)]
                report += "【前5行数据内容】:\n"
                report += "".join(head_lines) + "\n"
            except StopIteration:
                report += "【空文件】\n"
            except Exception as e:
                report += f"【读取失败】: {e}\n"
            report += "-" * 40 + "\n"

    return report

def agentic_process_dataset(dataset_name: str, dataset_dir: str) -> pd.DataFrame:
    """Agent 动态生成 ETL 代码，提取并返回 raw_df"""
    print(f"\n🤖 召唤 Data Architect Agent 针对 [{dataset_name}] 编写动态解析代码...")

    sniff_report = sniff_dataset_directory(dataset_dir)
    if not sniff_report:
        print("    ⚠️ 嗅探报告为空，无法启动 Agent。")
        return pd.DataFrame()

    client = OpenAI()
    prompt = DATASET_ETL_AGENT_PROMPT.format(sniff_report=sniff_report)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1
        )
        code_text = response.choices[0].message.content

        # 提取 Python 代码块
        match = re.search(r'```python\n(.*?)\n```', code_text, re.DOTALL)
        if not match:
            print("    ❌ Agent 未返回合法的 Python 代码块。")
            return pd.DataFrame()

        generated_code = match.group(1)
        
        # 固化代码为本地独立脚本，方便留痕和排错
        script_path = Path(dataset_dir) / f"{dataset_name}_etl.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(generated_code)
        print(f"    ✅ Agent 代码已生成并存入: {script_path.name}，正在尝试执行...")

        # 动态沙盒加载模块并执行
        spec = importlib.util.spec_from_file_location("custom_etl", script_path)
        custom_etl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_etl)

        raw_df = custom_etl.custom_extract_data(dataset_dir)
        
        if raw_df is not None and not raw_df.empty:
            print(f"    🌟 Agent 大显神威！成功提取出 {len(raw_df)} 条潜在记录。")
            return raw_df
            
    except Exception as e:
        print(f"    ❌ Agent ETL 管线执行崩溃: {e}")

    return pd.DataFrame()
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

def process_single_folder(input_dir: str, dataset_name: str) -> Optional[pd.DataFrame]:
    """
    处理单一数据集文件夹：
    1. 优先使用启发式引擎提取。
    2. 如果失败或数据过少，触发大模型编写 ETL 代码强制提取。
    3. 最后统一经过物理底线防御系统（质控、过滤、去重）。
    """
    strategy = load_benchmark_strategy()
    min_len = int(strategy["sequence_constraints"].get("min_len", 5))

    # 第一重防线：尝试常规的启发式读取
    raw_df = ingest_single_folder(input_dir, dataset_name, min_length=min_len)

    # 第二重防线：如果常规方法拿不到数据（或者数据过少判定为漏读），立刻触发 Agent ETL 抢救
    if raw_df.empty or len(raw_df) < 5:
        print(f"⚠️ [{dataset_name}] 启发式引擎未能提取有效数据，启动 Agentic ETL 抢救机制...")
        raw_df = agentic_process_dataset(dataset_name, input_dir)

    if raw_df.empty:
        print(f"!!! [{dataset_name}] 彻底宣告失败：启发式与 Agent 均未能提取出有效数据。")
        return None

    # 第三重（最终）防线：不管上面是谁拿到的数据，必须强制过一遍清洗和质控系统！
    filtered_df = filter_sequences(raw_df, strategy)
    gold_df, ambiguous_df = apply_gold_label_rules(filtered_df)
    gold_df = deduplicate_records(gold_df)
    gold_df = assign_unique_ids(gold_df)
    
    if gold_df.empty:
        print(f"!!! [{dataset_name}] 提取的数据未能通过生信质控规则（可能格式无效或全部被过滤）。")
        return None

    out_dir = Path(f"data/datasets/{dataset_name}")
    export_testset(gold_df, ambiguous_df, out_dir)

    print(f"\n[+] [{dataset_name}] 单体金标准测试集构建完成")
    print(f"    总原始/提取记录数: {len(raw_df)}")
    print(f"    质控后保留 gold: {len(gold_df)}")
    print(f"    正样本: {sum(gold_df['label'] == 1)}")
    print(f"    负样本: {sum(gold_df['label'] == 0)}")
    print(f"    输出目录: {out_dir}")
    
    return gold_df


def main():
    print(">>> 启动二分类 AMP 金标准测试集预处理流水线（全自动对接模式）。")
    
    # 1. 动态读取 AI 首席科学家生成的策略文件
    strategy = load_benchmark_strategy()
    datasets = strategy.get("recommended_datasets", [])

    if not datasets:
        print("!!! [Error] benchmark_strategy.json 中没有 recommended_datasets，请先执行 prepare_models.py")
        return

    all_gold_dfs = []

    # 2. 遍历策略中的每一个顶刊数据集，进行单源清洗
    for ds in datasets:
        # 名字必须和 dataset_fetcher.py 保持一致（替换空格为下划线）
        ds_name = ds.get("dataset_name", "Unknown_Dataset").replace(" ", "_")
        
        # 指向刚刚从 GitHub 下载提取的原始数据文件夹
        input_folder = f"data/datasets/{ds_name}"

        if os.path.exists(input_folder) and os.path.isdir(input_folder):
            print(f"\n[🚀] 发现目标数据集目录: {input_folder}，准备启动数据炼金炉...")
            # 3. 原地执行清洗，同时收集返回的纯净 DataFrame
            gold_df = process_single_folder(input_folder, ds_name)
            if gold_df is not None and not gold_df.empty:
                all_gold_dfs.append(gold_df)
        else:
            print(f"\n!!! [Skip] 找不到原始数据文件夹: {input_folder} (可能是 Web 数据库，需手动放入数据)")

    print("\n" + "="*60)
    print("🎉 所有独立测试集物理清洗、长度裁切与标签对齐完毕！")

    # 4. 🚨 核心新功能：执行全局跨数据集大一统合并 (Master Benchmark)
    if all_gold_dfs:
        print("\n========== [Phase 7] 数据大一统：构建终极 Master Benchmark ==========")
        
        # 将所有清洗好的数据集合并成一张大表
        master_df = pd.concat(all_gold_dfs, ignore_index=True)
        initial_master_len = len(master_df)
        
        # 全局去重法则：优先保留 evidence_level 更可靠的记录
        priority = {"table_label": 2, "filename_inferred": 1, "unknown": 0}
        master_df["__prio"] = master_df["evidence_level"].map(lambda x: priority.get(str(x), 0))
        
        # 排序：序列相同的，置信度高的排前面
        master_df = master_df.sort_values(by=["sequence", "__prio"], ascending=[True, False])
        
        # 执行全局暴力去重 (只要序列一模一样，跨库重复也只保留一条！)
        master_df = master_df.drop_duplicates(subset=["sequence"], keep="first").reset_index(drop=True)
        master_df = master_df.drop(columns=["__prio"])
        
        # 为防止合并后 ID 冲突，重新赋予绝对唯一的全局 ID
        master_df["id"] = [f"Master_Seq_{i+1:05d}_{row['source_dataset'][:5]}" for i, row in master_df.iterrows()]
        
        dropped_count = initial_master_len - len(master_df)
        
        # 统一存入最高级别的独立标准库中
        master_dir = Path("data/standardized_datasets")
        master_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出供模型预测的综合 FASTA 和供评估脚本核对的最终 CSV
        master_csv_path = master_dir / "MASTER_BENCHMARK.csv"
        master_fasta_path = master_dir / "MASTER_BENCHMARK.fasta"
        
        master_df[["id", "sequence", "label", "source_dataset"]].to_csv(master_csv_path, index=False)
        write_fasta(master_df, master_fasta_path)
        
        print(f"    -> 🌍 [跨库融合] 成功汇聚 {len(all_gold_dfs)} 个独立基准集的优质数据。")
        print(f"    -> ✂️ [全局去重] 清洗了 {dropped_count} 条跨库重复/冗余数据。")
        print(f"    -> 🏆 [终极产出] 诞生 Master Benchmark！共含 {len(master_df)} 条绝对纯净序列！")
        print(f"       -> 正样本: {sum(master_df['label'] == 1)} | 负样本: {sum(master_df['label'] == 0)}")
        print(f"    -> 📁 {master_fasta_path}")
        print("👉 整个 MLOps 评估管线的数据准备工作现已达到完美闭环！可以提交超算打分了！")
    else:
        print("⚠️ 全局合并失败：没有采集到任何有效的数据集 DataFrame。")

if __name__ == "__main__":
    main()