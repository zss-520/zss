import os
import subprocess
import shutil
import glob
import gzip

def read_file_head(filepath, lines=10):
    """读取文件头部内容，支持gzip压缩文件"""
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt') as f:
                content = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content.append(line.rstrip())
                return '\n'.join(content)
        else:
            with open(filepath, 'r') as f:
                content = []
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    content.append(line.rstrip())
                return '\n'.join(content)
    except Exception as e:
        return f"Error reading file: {str(e)}"

def explore_output_directory(output_dir, model_name):
    """勘探输出目录中的所有文件并生成报告"""
    report_lines = [f"=== Model: {model_name} Output Exploration Report ==="]
    
    if not os.path.exists(output_dir):
        report_lines.append(f"Output directory {output_dir} does not exist!")
        return '\n'.join(report_lines)
    
    # 查找所有文件（包括子目录）
    all_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    if not all_files:
        report_lines.append(f"No files found in {output_dir}")
        return '\n'.join(report_lines)
    
    report_lines.append(f"Found {len(all_files)} files in {output_dir}:")
    
    for filepath in all_files:
        rel_path = os.path.relpath(filepath, output_dir)
        abs_path = os.path.abspath(filepath)
        file_size = os.path.getsize(filepath)
        
        report_lines.append(f"\n--- File: {rel_path} ---")
        report_lines.append(f"Absolute path: {abs_path}")
        report_lines.append(f"Size: {file_size} bytes")
        
        # 读取文件头部内容
        head_content = read_file_head(filepath, 10)
        report_lines.append("Head content:")
        report_lines.append(head_content)
        report_lines.append("-" * 50)
    
    return '\n'.join(report_lines)

def main():
    # 获取任务ID
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    # 定义模型配置
    models = [
        {
            "name": "Macrel",
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/Macrel_out --keep-negatives"',
            "output_dir": "data/Macrel_out"
        },
        {
            "name": "amPEPpy", 
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amPEP && ampep predict -i data/combined_test.fasta -o data/amPEPpy_out/predictions.txt -m /share/home/zhangss/amPEPpy/pretrained_models/amPEP.model"',
            "output_dir": "data/amPEPpy_out"
        },
        {
            "name": "AMPlify",
            "cmd": 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate amplify && AMPlify -s data/combined_test.fasta -m balanced -of tsv -od data/AMPlify_out"',
            "output_dir": "data/AMPlify_out"
        }
    ]
    
    # 检查task_id是否有效
    if task_id >= len(models):
        print(f"Task ID {task_id} exceeds available models (0-{len(models)-1}). Exiting.")
        return
    
    model = models[task_id]
    model_name = model["name"]
    cmd = model["cmd"]
    output_dir = model["output_dir"]
    
    print(f"Starting task {task_id}: Running {model_name}")
    
    # 执行模型并实现自愈机制
    try:
        # 第一步：清理历史数据（防止冲突）
        print(f"Cleaning old output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
        
        # 执行模型
        print(f"Executing {model_name}...")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 检查是否执行失败
        if res.returncode != 0:
            print(f"{model_name} execution failed. Return code: {res.returncode}")
            print(f"Stderr: {res.stderr}")
            
            # 检查错误类型并尝试自愈
            error_msg = res.stderr + res.stdout
            if "No such file" in error_msg or "NotFoundError" in error_msg or "not found" in error_msg:
                print("Detected missing directory error. Creating output directory and retrying...")
                os.makedirs(output_dir, exist_ok=True)
                
                # 重新执行
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if res.returncode != 0:
                    print(f"Retry failed for {model_name}. Final stderr: {res.stderr}")
                    
                    # 再次检查错误类型
                    final_error_msg = res.stderr + res.stdout
                    if "already exists" in final_error_msg or "exists" in final_error_msg:
                        print("Detected directory exists conflict. Removing and retrying...")
                        shutil.rmtree(output_dir, ignore_errors=True)
                        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 记录执行结果
        log_file = f"data/stage1_obs_{task_id}.txt"
        os.makedirs("data", exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"Model Execution Report for Task {task_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Return Code: {res.returncode}\n")
            f.write(f"Stdout:\n{res.stdout}\n")
            f.write(f"Stderr:\n{res.stderr}\n")
            f.write("="*60 + "\n\n")
            
            # 勘探输出目录
            exploration_report = explore_output_directory(output_dir, model_name)
            f.write(exploration_report)
        
        print(f"Exploration report saved to {log_file}")
        
    except Exception as e:
        print(f"Exception occurred during execution of {model_name}: {str(e)}")
        log_file = f"data/stage1_obs_{task_id}.txt"
        os.makedirs("data", exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write(f"Model Execution Error Report for Task {task_id}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Exception: {str(e)}\n")
            f.write("="*60 + "\n\n")
            
            # 即使出错也要尝试勘探目录
            exploration_report = explore_output_directory(output_dir, model_name)
            f.write(exploration_report)

if __name__ == '__main__':
    main()
