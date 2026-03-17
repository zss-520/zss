import os
import shutil
import subprocess
import glob
import gzip

def explore_output_directory(base_dir='data'):
    """
    勘探 data 目录下所有 *_out 相关的文件，读取头部内容。
    """
    observation_log = []
    observation_log.append("=" * 50)
    observation_log.append("Stage 1: Model Output Observation Report")
    observation_log.append("=" * 50)
    observation_log.append("")

    # 查找所有 *_out 结尾的目录或文件
    # 使用 glob 查找 data/ 下直接子目录或文件匹配 *_out
    pattern = os.path.join(base_dir, '*_out')
    found_items = glob.glob(pattern)
    
    # 同时也查找 data/ 下可能存在的子目录中的 *_out (递归搜索以防万一)
    # 但根据任务描述，主要是模型直接生成的 output 目录
    # 这里我们主要关注找到的 items 是目录还是文件
    
    if not found_items:
        observation_log.append("警告：未在 data/ 目录下找到任何 *_out 后缀的目录或文件。")
    else:
        for item_path in found_items:
            observation_log.append(f"勘探对象：{item_path}")
            observation_log.append("-" * 30)
            
            files_to_read = []
            if os.path.isdir(item_path):
                # 如果是目录，遍历内部文件
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        files_to_read.append(os.path.join(root, file))
            else:
                # 如果是文件
                files_to_read.append(item_path)
            
            for file_path in files_to_read:
                observation_log.append(f"  文件路径：{file_path}")
                try:
                    lines = []
                    # 处理 .gz 文件
                    if file_path.endswith('.gz'):
                        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f):
                                if i >= 10: break
                                lines.append(line.strip())
                    else:
                        # 处理普通文本文件
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f):
                                if i >= 10: break
                                lines.append(line.strip())
                    
                    if lines:
                        observation_log.append("  文件头内容摘要 (前 10 行):")
                        for line in lines:
                            observation_log.append(f"    | {line}")
                    else:
                        observation_log.append("  文件头内容摘要：[文件为空]")
                except Exception as e:
                    observation_log.append(f"  读取失败：{str(e)}")
            
            observation_log.append("")

    observation_log.append("=" * 50)
    observation_log.append("End of Observation")
    observation_log.append("=" * 50)
    
    return "\n".join(observation_log)

def main():
    # 0. 确保 data 目录存在，以便后续保存报告
    os.makedirs('data', exist_ok=True)

    # 1. 【历史数据清理机制】
    # 在执行模型预测前，必须执行清理逻辑
    print("正在执行历史数据清理...")
    shutil.rmtree('data/PepNet_out', ignore_errors=True)
    # 注：任务要求中列出了两次 PepNet 清理，逻辑上清理一次即可确保目录干净
    print("清理完成。")

    # 2. 【强校验命令执行】
    # 必须严格照抄以下代码块来执行预测（切勿自行添加 try-except 吞咽错误）
    # 注：虽然任务列表提到两次 PepNet，但因输出路径相同且伴随清理，重复执行会导致数据丢失。
    # 此处严格执行一次预测以保留结果供勘探。
    print("开始运行 PepNet...")
    PepNet_cmd = 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate pepnet_env && python predict.py --input data/combined_test.fasta --output data/PepNet_out"'
    res_PepNet = subprocess.run(PepNet_cmd, shell=True, capture_output=True, text=True)
    if res_PepNet.returncode != 0:
        print(f"!!! PepNet 真实报错日志:\n{res_PepNet.stderr}")
        raise RuntimeError("PepNet 预测执行失败，已阻断程序！")
    
    print("PepNet 运行成功。")

    # 3. 【勘探逻辑】
    # 运行完模型后，遍历 data/ 目录下所有带有 _out 后缀的目录或新生成的文件
    print("开始勘探模型输出结构...")
    report_content = explore_output_directory('data')
    
    # 4. 【保存勘探报告】
    # 勘探报告必须保存为：data/stage1_observation.txt
    report_path = 'data/stage1_observation.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"勘探报告已保存至：{report_path}")
    print("Stage 1 任务完成。")

if __name__ == '__main__':
    main()
