import os
import subprocess
import shutil
import glob
import gzip

def main():
    # 获取当前任务 ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    # 定义模型配置列表
    models = [
        {
            "name": "Macrel",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives\"",
            "output_dir": "data/Macrel_out"
        },
        {
            "name": "AMP-Scanner-v2",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/AMP-Scanner-v2_out/ampscanner_out.csv\"",
            "output_dir": "data/AMP-Scanner-v2_out"
        },
        {
            "name": "amPEPpy",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model\"",
            "output_dir": "data/amPEPpy_out"
        },
        {
            "name": "AI4AMP",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ai4amp && python /share/home/zhangss/AI4AMP_predictor/PC6/PC6_predictor.py -f data/combined_test.fasta -o data/AI4AMP_out/predictions.csv\"",
            "output_dir": "data/AI4AMP_out"
        },
        {
            "name": "AMPlify",
            "cmd": "bash -c \"source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out\"",
            "output_dir": "data/AMPlify_out"
        }
    ]

    # 检查任务 ID 是否合法
    if task_id < 0 or task_id >= len(models):
        print(f"[ERROR] Invalid task_id {task_id}. Must be between 0 and {len(models)-1}.")
        return

    model = models[task_id]
    name = model["name"]
    cmd = model["cmd"]
    output_dir = model["output_dir"]

    # 设置日志文件路径（每个任务独立）
    log_file = f"data/stage1_obs_{task_id}.txt"

    with open(log_file, "w") as log:
        log.write(f"[INFO] Running model: {name}\n")
        log.write(f"[CMD] {cmd}\n\n")

        # 第一步：清理旧目录
        log.write("[CLEANUP] Removing previous output directory...\n")
        shutil.rmtree(output_dir, ignore_errors=True)

        # 第二步：尝试首次执行
        log.write("[EXECUTE] First attempt to run command...\n")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # 第三步：判断是否需要自愈重试
        retry_needed = False
        if res.returncode != 0:
            stderr_content = res.stderr.lower()
            stdout_content = res.stdout.lower()
            combined_error = stderr_content + stdout_content

            if "no such file" in combined_error or "not found" in combined_error or "failed to save" in combined_error:
                log.write("[RETRY] Detected missing directory error. Creating directory and retrying...\n")
                os.makedirs(output_dir, exist_ok=True)
                retry_needed = True
            elif "already exists" in combined_error or "exists" in combined_error:
                log.write("[RETRY] Detected existing directory conflict. Removing and retrying...\n")
                shutil.rmtree(output_dir, ignore_errors=True)
                retry_needed = True

        # 如果需要重试则再执行一次
        if retry_needed:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # 记录执行结果
        log.write(f"[RESULT] Return code: {res.returncode}\n")
        if res.returncode != 0:
            log.write("[STDERR]\n")
            log.write(res.stderr)
            log.write("\n[STDOUT]\n")
            log.write(res.stdout)
        else:
            log.write("[SUCCESS] Command executed successfully.\n")

        # 第四步：勘探输出目录结构
        log.write("\n[EXPLORATION] Scanning output directory...\n")
        if not os.path.exists(output_dir):
            log.write(f"[WARNING] Output directory does not exist: {output_dir}\n")
        else:
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                log.write(f"{indent}{os.path.basename(root)}/\n")
                subindent = ' ' * 2 * (level + 1)
                for f in files:
                    file_path = os.path.join(root, f)
                    log.write(f"{subindent}{f}\n")

                    # 尝试读取文件头部内容
                    try:
                        if f.endswith(".gz"):
                            with gzip.open(file_path, 'rt', encoding='utf-8') as gf:
                                lines = [next(gf) for _ in range(5)]
                                log.write("".join(lines[:5]))
                        else:
                            with open(file_path, 'r', encoding='utf-8') as rf:
                                lines = [next(rf) for _ in range(5)]
                                log.write("".join(lines[:5]))
                    except Exception as e:
                        log.write(f"[READ ERROR] Could not read file: {str(e)}\n")
                log.write("\n")


if __name__ == "__main__":
    main()
