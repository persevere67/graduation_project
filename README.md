# 基于联邦学习的隐私保护新闻推荐算法研究

本项目为毕业设计实验代码仓库。

## 1. 项目简介
使用 **MIND** 数据集，基于 **Flower** 联邦学习框架，实现了结合 **SBERT** 新闻编码与 **Multi-Head Attention** 用户编码的双塔推荐模型。

## 2. 实验进展
- [x] 数据预处理与特征提取 (SBERT)
- [x] 中心化 Baseline 训练 (Loss: 1.31)
- [x] 联邦学习环境搭建 (Flower)
- [x] 联邦学习模型收敛实验 (50 Clients, 10 Rounds, Loss: 1.515)

## 3. 运行环境
- Python 3.11+
- PyTorch
- flwr (Flower)
- sentence-transformers