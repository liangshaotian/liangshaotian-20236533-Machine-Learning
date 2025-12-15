📝 README.md
# 跨域自适应U-Net：多模态心脏分割

基于U-Net架构的多模态医学图像心脏分割项目，通过在MMWHS数据集上联合训练CT和MRI两种成像模态，实现单一模型对不同模态图像的统一分割。

## MMWHS数据集链接:https://mega.nz/folder/UNMF2YYI#1cqJVzo4p_wESv9P_pc8uA

## 📊 主要结果

在MMWHS验证集上的性能：
- Dice系数: 0.9210
- IoU: 0.8553
- Precision: 0.9761
- Recall: 0.9134

## 🛠️ 环境要求

### 硬件
- GPU: NVIDIA RTX 3070 Ti (8GB) 或同等配置
- 内存: 至少32GB RAM

### 软件依赖


pip install -r requirements.txt
主要依赖：

Python 3.12.3
PyTorch 2.6.0
CUDA 12.1
NumPy 1.26.4
scikit-image 0.24.0
📂 项目结构
.
├── train_cross_domain.py        # 跨域训练脚本
├── mmwhs_dataset.py             # MMWHS数据集加载
├── MMWHS_process.py             # 数据预处理
├── requirements.txt             # 依赖列表
├── data/                        # 数据目录
└── checkpoints_mmwhs_optimized/ # 训练权重

🚀 使用方法
1. 数据准备
从MMWHS Challenge下载数据集，然后运行预处理：

python MMWHS_process.py
2. 训练模型
python train_cross_domain.py
训练配置：

Batch size: 8
学习率: 7e-4
优化器: AdamW
损失权重: 0.55(分割) + 0.25(先验) + 0.20(域对齐)

📈 核心功能
多模态联合训练: 单一模型处理CT和MRI两种模态
跨域自适应: 通过对抗训练实现模态特征对齐
先验引导: 多尺度先验掩码生成与融合
标签平滑: 防止过拟合（ε=0.05）
边界损失: 提升分割边界质量