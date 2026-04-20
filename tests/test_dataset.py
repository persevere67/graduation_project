import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import random # 别忘了加上这个，dataset.py 里用到了

# 导入你刚刚写的类
from dataset import MINDDataset

def test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(current_dir, 'processed')

    # 1. 加载必要的字典和数据
    print("正在加载数据...")
    with open(os.path.join(processed_path, 'news_id_dict.pkl'), 'rb') as f:
        news_id_dict = pickle.load(f)
    
    with open(os.path.join(processed_path, 'federated_data.pkl'), 'rb') as f:
        federated_data = pickle.load(f)

    # 取出一个客户端的数据进行测试
    client_0_df = federated_data['client_0']
    
    # 2. 实例化 Dataset
    print("正在初始化 Dataset...")
    dataset = MINDDataset(client_0_df, news_id_dict, max_hist_len=50)
    
    # 3. 测试单条数据
    print(f"Dataset 长度: {len(dataset)}")
    history, candidates, label = dataset[0]
    
    print("\n--- 单条数据维度测试 ---")
    print(f"History 形状: {history.shape} (预期: torch.Size([50]))")
    print(f"Candidates 形状: {candidates.shape} (预期: torch.Size([5]))")
    print(f"Label: {label} (预期: 0)")
    
    # 4. 测试 DataLoader (模拟 batch 训练)
    print("\n正在测试 DataLoader...")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 取一个 batch
    batch_hist, batch_cand, batch_label = next(iter(loader))
    
    print("--- Batch 数据维度测试 ---")
    print(f"Batch History 形状: {batch_hist.shape} (预期: [32, 50])")
    print(f"Batch Candidates 形状: {batch_cand.shape} (预期: [32, 5])")
    print(f"Batch Labels 形状: {batch_label.shape} (预期: [32])")
    
    print("\n✅ 数据管道测试通过！")

if __name__ == "__main__":
    test()