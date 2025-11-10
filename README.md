# LLM-Assignment
Homework

大型模型基础与应用 - 期中作业

本项目根据课程要求，从零开始手工搭建了 Transformer 模型，并完成了两个核心实验：

Encoder-Only (类 GPT) 语言模型：在 Tiny Shakespeare 数据集上进行字符级语言建模。

Encoder-Decoder (完整 Transformer)：在 IWSLT2017 (英-德) 数据集子集上进行机器翻译。

项目完整实现了多头注意力、位置前馈网络、位置编码、残差连接、层归一化以及掩码机制，并包括了对位置编码的消融实验。

仓库结构

readme.md
		requirements.txt
		
		yuyan/
		
		└── pythonProject1
		
			├── models
			
			├── data
			
			├── results
			
			├── plot_comparison.py
			
			├── run.sh
			
			├── train_encoder_only.py
			
			├── train.py
			
			└──src/
			
				├── transformer_blocks.py
				
				├── models.py
				
				├── data_loaders.py
				
				└── utils.py
				
		xulie/
		
		└── pythonProject1
		
			 └──src/
			 
			 	├── models
				
			 	├── data
				
			 	├── results
				
			 	└──src/
				
			 		├── run.sh
					
			 		├── plot_comparison.py
					
			 		├── seq2seq_dataset.py
					
			 		├── seq2seq_model.py
					
			 		├── train.py
					
			 		├── train_seq2seq.py
					
			 		├── transformer_blocks.py
					
			 		└── utils.py


复现实验

1. 环境设置 (使用 Conda)

请确保您已安装 Conda。

（1）创建环境 (我们调试时用的 Python 3.8)
conda create -n transformer_env python=3.8

（2）激活环境
conda activate transformer_env

（3）安装核心依赖
 (注意：根据您的服务器网络情况，您可能需要配置代理)
pip install -r requirements.txt

（4）下载 Spacy 语言模型 (用于 Seq2Seq 分词)
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm


2. 硬件要求

Encoder-Only (Tiny Shakespeare): 可以在 CPU 上运行，但在 GPU (如 T4, RTX 3060) 上会快得多。

Encoder-Decoder (IWSLT2017): 强烈推荐使用 GPU (CUDA)。使用 CPU 训练一个 Epoch 需要数小时，而使用 GPU (T4) 仅需约 1-2 分钟。

3. 运行实验

本项目包含两个实验，每个实验都需要运行两次（一次基线，一次消融）。

注意：如果您的服务器网络受限，请在安装前设置 HTTP\_PROXY 和 HTTPS\_PROXY 环境变量

# 实验一：Encoder-Only (Tiny Shakespeare)

我们提供了 run.sh 脚本来自动运行基线和消融实验。

增加执行权限
通过cd命令进入相应文件夹
$chmod +x run.sh

运行此脚本将执行两次训练，并生成对比图表
bash run.sh


手动复现（作业要求的 exact 命令行）：

1. 运行基线模型 (带 PE)
(随机种子 42, 结果保存为 baseline_enc_losses.json)
python train.py --experiment_name "baseline_enc" --seed 42

2. 运行消融实验 (不带 PE)
(随机种子 42, 结果保存为 no_pe_enc_losses.json)
python train_encoder_only.py --experiment_name "no_pe_enc" --seed 42 --disable_pe

3. 生成对比图表
python plot_comparison.py



# 实验二：Encoder-Decoder (IWSLT2017)

我们提供了 run.sh 脚本来自动运行基线和消融实验。

运行此脚本将执行两次训练，并生成对比图表
通过cd命令进入相应文件夹
bash run.sh


手动复现（作业要求的 exact 命令行）：

1. 运行基线模型 (带 PE)
(随机种子 42, 结果保存为 baseline_seq2seq_losses.json)
python train_seq2seq.py --experiment_name "baseline_seq2seq" --seed 42

2. 运行消融实验 (不带 PE)
(随机种子 42, 结果保存为 no_pe_seq2seq_losses.json)
python train.py --experiment_name "no_pe_seq2seq" --seed 42 --disable_pe

3. 生成对比图表
python plot_comparison.py
