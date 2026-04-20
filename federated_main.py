import flwr as fl
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
import os
os.environ["RAY_CHOSEN_GPU"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 强制 Ray 在 Windows 下不使用多进程启动，而是顺序执行
os.environ["RAY_worker_register_timeout_seconds"] = "600"
# 导入你之前的核心组件
from models import NewsRecommender
from dataset import MINDDataset

# --- 1. 环境与资源配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

# 加载全局 Embedding 矩阵（只读，不参与梯度更新）
print("正在载入全局数据...")
news_embeddings = torch.FloatTensor(np.load(os.path.join(PROCESSED_DIR, 'news_embeddings.npy'))).to(device)
with open(os.path.join(PROCESSED_DIR, 'news_id_dict.pkl'), 'rb') as f:
    news_id_dict = pickle.load(f)
with open(os.path.join(PROCESSED_DIR, 'federated_data.pkl'), 'rb') as f:
    federated_data = pickle.load(f)
with open(os.path.join(PROCESSED_DIR, 'dev_data_all.pkl'), 'rb') as f:
    dev_data = pickle.load(f)

# --- 2. 定义 Flower 客户端逻辑 ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for hist_idx, cand_idx, label in self.train_loader:
            hist_idx, cand_idx, label = hist_idx.to(device), cand_idx.to(device), label.to(device)
            # 从全局矩阵查表
            hist_vecs, cand_vecs = news_embeddings[hist_idx], news_embeddings[cand_idx]
            
            optimizer.zero_grad()
            output = self.model(hist_vecs, cand_vecs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

# --- 3. 定义服务器端评估逻辑 ---
def get_evaluate_fn():
    """该函数会被服务器在每轮聚合结束后调用"""
    # 准备验证集加载器
    val_dataset = MINDDataset(dev_data, news_id_dict)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    def evaluate(server_round, parameters, config):
        model = NewsRecommender(embedding_dim=384).to(device)
        # 加载最新聚合的模型参数
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        with torch.no_grad():
            for hist_idx, cand_idx, label in val_loader:
                hist_idx, cand_idx, label = hist_idx.to(device), cand_idx.to(device), label.to(device)
                hist_vecs, cand_vecs = news_embeddings[hist_idx], news_embeddings[cand_idx]
                output = model(hist_vecs, cand_vecs)
                total_loss += criterion(output, label).item()
        
        # 返回 loss 给服务器记录
        return total_loss / len(val_loader), {}
    
    return evaluate

# --- 4. 模拟器启动配置 ---
def client_fn(cid: str) -> FlowerClient:
    model = NewsRecommender(embedding_dim=384).to(device)
    # 确保 cid 对应的 key 在 federated_data 中存在
    client_df = federated_data[f"client_{cid}"]
    dataset = MINDDataset(client_df, news_id_dict)
    # 本地 Batch 不要设太大，防止单客户端训练时间过长
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return FlowerClient(cid, model, train_loader)

# 设置聚合策略
strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_evaluate_fn(), # 开启服务器端评估
    fraction_fit=0.2,             # 每轮随机选 20% 的客户端（50*0.2=10个）参与训练
    min_available_clients=10,      # 至少有 10 个客户端在线才开始
)

if __name__ == "__main__":
    # 将启动代码放进这个判断里
    print("\n🚀 联邦学习模拟启动...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=50,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={"num_gpus": 0.1, "num_cpus": 2},
    )