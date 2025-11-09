import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from utils import plot_curves
except ImportError:
    print("Error: 无法从 'src' 文件夹导入 'utils'。")
    print("请确保此脚本位于您的项目根目录 (与 'src' 文件夹同级)。")
    sys.exit(1)

EXPERIMENTS = {
    'Baseline (PE Enabled)': 'results/baseline_losses.json',
    'Ablation (No PE)': 'results/no_pe_losses.json'
}


def main():
    loss_data_for_plot = {}

    for name, path in EXPERIMENTS.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                loss_data_for_plot[name] = data
        else:
            print(f"Error: 找不到损失文件 {path}")
            print("请确保您已成功运行 'baseline' 和 'no_pe' 实验。")
            return

    plot_curves(
        loss_data_for_plot,
        filename='results/comparison_PE_ablation.png',
        title='Loss Comparison (Seq2Seq): Baseline Transformer vs. No Positional Encoding'
    )
    print("\nSeq2Seq (IWSLT2017) 对比图表已生成。")


if __name__ == '__main__':
    main()