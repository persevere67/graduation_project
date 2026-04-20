import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle

# 导入你之前的类
from models import NewsRecommender
from dataset import MINDDataset

def train():
    # 1. 配置路径与设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

    # 2. 加载 Embedding 矩阵 (核心！)
    print("正在加载全量 Embedding 矩阵...")
    news_embeddings = np.load(os.path.join(PROCESSED_DIR, 'news_embeddings.npy'))
    # 转为 Tensor 并转到 GPU，之后通过索引直接取值
    news_embeddings = torch.FloatTensor(news_embeddings).to(device)

    # 3. 整合所有客户端数据用于中心化训练
    print("正在整合训练数据...")
    with open(os.path.join(PROCESSED_DIR, 'federated_data.pkl'), 'rb') as f:
        federated_data = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, 'news_id_dict.pkl'), 'rb') as f:
        news_id_dict = pickle.load(f)

    # 将 50 个客户端的数据合并成一个大的 DataFrame
    all_train_df = pd.concat(federated_data.values())
    train_dataset = MINDDataset(all_train_df, news_id_dict)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 4. 初始化模型
    model = NewsRecommender(embedding_dim=384).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # 因为我们的 label 永远是 0 (正样本在第一位)

    # 5. 训练循环
    model.train()
    print("开始训练...")
    for epoch in range(5): # 先跑 5 个 Epoch 看看
        total_loss = 0
        for batch_idx, (hist_idx, cand_idx, label) in enumerate(train_loader):
            hist_idx, cand_idx, label = hist_idx.to(device), cand_idx.to(device), label.to(device)
            
            # --- 关键步：根据索引查 Embedding ---
            # news_embeddings 是 [65238, 384]
            # hist_idx 是 [64, 50] -> 取完后变 [64, 50, 384]
            hist_vecs = news_embeddings[hist_idx]
            cand_vecs = news_embeddings[cand_idx]
            
            # 前向传播
            optimizer.zero_grad()
            scores = model(hist_vecs, cand_vecs) # [64, 5]
            
            loss = criterion(scores, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} 完成，平均 Loss: {total_loss/len(train_loader):.4f}")

    # 6. 保存模型
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'checkpoints', 'centralized_model.pth'))
    print("✅ 模型已保存至 checkpoints 文件夹")

if __name__ == "__main__":
    import pandas as pd # 临时加一下，确保合并数据成功
    train()