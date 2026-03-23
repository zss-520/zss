import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-coder-plus")

HPC_HOST = os.getenv("HPC_HOST")
HPC_PORT = int(os.getenv("HPC_PORT", "22"))
HPC_USER = os.getenv("HPC_USER")
HPC_PASS = os.getenv("HPC_PASS")
HPC_TARGET_DIR = os.getenv("HPC_TARGET_DIR")

CONDA_SH_PATH = os.getenv("CONDA_SH_PATH")
MACREL_ENV = os.getenv("MACREL_ENV")
AMPSCANNER_ENV = os.getenv("AMPSCANNER_ENV")
VLAB_ENV = os.getenv("VLAB_ENV")

MACREL_CMD = os.getenv("MACREL_CMD", "macrel")
AMPSCANNER_CMD = os.getenv(
    "AMPSCANNER_CMD",
    "python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py",
)

SLURM_PARTITION = os.getenv("SLURM_PARTITION", "gpu")
SLURM_JOB_NAME = os.getenv("SLURM_JOB_NAME", "amp_eval")
SLURM_GPUS = os.getenv("SLURM_GPUS", "4")
SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK", "16")

PIP_INDEX_URL = os.getenv(
    "PIP_INDEX_URL",
    "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
)
PIP_EXTRA_INDEX_URL = os.getenv(
    "PIP_EXTRA_INDEX_URL",
    "https://mirrors.aliyun.com/pypi/simple/",
)

MEETING_OUTPUT_DIR = os.getenv("MEETING_OUTPUT_DIR", "meeting_outputs")
FIRST_MEETING_NAME = os.getenv("FIRST_MEETING_NAME", "meeting_stage1_model_run")
SECOND_MEETING_NAME = os.getenv("SECOND_MEETING_NAME", "meeting_stage2_eval_codegen")
FIRST_STAGE_PY_NAME = os.getenv("FIRST_STAGE_PY_NAME", "stage1_model_runner.py")
FIRST_STAGE_SH_NAME = os.getenv("FIRST_STAGE_SH_NAME", "stage1_run_model.sh")
SECOND_STAGE_PY_NAME = os.getenv("SECOND_STAGE_PY_NAME", "stage2_eval_script.py")
SECOND_STAGE_SH_NAME = os.getenv("SECOND_STAGE_SH_NAME", "stage2_run_eval.sh")

# 新增：第一阶段只需要生成一份勘探报告
FIRST_STAGE_OBSERVATION_TXT = os.getenv("FIRST_STAGE_OBSERVATION_TXT", "stage1_observation.txt")
# =========================
# 评测量化打分权重配置 (供 Critic 和计算使用)
# =========================
# 在 AMP 预测等类别极度不平衡的任务中，AUPRC 和 MCC 通常比 ACC 更重要。
# 你可以根据未来不同领域任务的需求，自由调整这里的指标和权重。
# 注意：所有加权指标的和最好为 1.0 (或 100%)
METRIC_WEIGHTS = {
    "ACC": 0.10,      # 准确率权重 
    "Recall": 0.15,   # 召回率权重 
    "MCC": 0.35,      # 马修斯相关系数权重 
    "AUROC": 0.10,    # ROC曲线下侧面积权重 
    "AUPRC": 0.30,    # PR曲线下侧面积权重 
}

def validate_runtime_config(require_hpc: bool = False) -> None:
    required = {
        "OPENAI_API_KEY / DASHSCOPE_API_KEY": DASHSCOPE_API_KEY,
    }

    if require_hpc:
        required.update(
            {
                "HPC_HOST": HPC_HOST,
                "HPC_USER": HPC_USER,
                "HPC_PASS": HPC_PASS,
                "HPC_TARGET_DIR": HPC_TARGET_DIR,
                "CONDA_SH_PATH": CONDA_SH_PATH,
                "MACREL_ENV": MACREL_ENV,
                "AMPSCANNER_ENV": AMPSCANNER_ENV,
                "VLAB_ENV": VLAB_ENV,
                "AMPSCANNER_CMD": AMPSCANNER_CMD,
            }
        )

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"缺少必要配置: {', '.join(missing)}")

    print(">>> [Config] 配置检查通过")
    print(f"    MODEL_NAME                 = {MODEL_NAME}")
    print(f"    MEETING_OUTPUT_DIR         = {MEETING_OUTPUT_DIR}")
    print(f"    FIRST_STAGE_OBSERVATION_TXT= {FIRST_STAGE_OBSERVATION_TXT}")
    if require_hpc:
        print(f"    HPC_HOST                   = {HPC_HOST}")
        print(f"    HPC_TARGET_DIR             = {HPC_TARGET_DIR}")