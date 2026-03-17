import os
import shutil
import subprocess

def main():
    # 历史清理
    print("正在清理历史输出文件...")
    shutil.rmtree('data/macrel_out', ignore_errors=True)
    if os.path.exists('data/ampscanner_out.csv'):
        os.remove('data/ampscanner_out.csv')
    
    # Macrel 调用代码模板
    print("开始运行Macrel...")
    macrel_cmd = 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate env_macrel && macrel peptides --fasta data/combined_test.fasta --output data/macrel_out"'
    res_macrel = subprocess.run(macrel_cmd, shell=True, capture_output=True, text=True)
    if res_macrel.returncode != 0:
        print(f"\n=== Macrel 报错详情 ===\n{res_macrel.stderr}")
        raise RuntimeError("Macrel预测执行失败，已阻断程序！")

    # AMP-Scanner-v2 调用代码模板
    print("开始运行AMP-Scanner-v2...")
    ascan_cmd = 'bash -c "source /share/home/zhangss/miniconda3/etc/profile.d/conda.sh && conda activate ascan2_tf1 && python /share/home/zhangss/amp-scanner-v2/amp_scanner_v2_predict_tf1.py -f data/combined_test.fasta -m /share/home/zhangss/amp-scanner-v2/trained-models/021820_FULL_MODEL.h5 -p data/ampscanner_out.csv"'
    res_ascan = subprocess.run(ascan_cmd, shell=True, capture_output=True, text=True)
    if res_ascan.returncode != 0:
        print(f"\n=== AMP-Scanner 报错详情 ===\n{res_ascan.stderr}")
        raise RuntimeError("AMP-Scanner预测执行失败，已阻断程序！")
    
    print("所有预测任务完成！")

if __name__ == '__main__':
    main()