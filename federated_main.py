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

SIGMA = 0.01 # 设置为 0.001 或 0.01 来启用差分隐私增强模式，设为 0 则为原始联邦学习模式

# --- 1. 环境与资源配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
# 【关键补丁】加上这一行！
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

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
        
        # 1. 执行本地训练
        for hist_idx, cand_idx, label in self.train_loader:
            hist_idx, cand_idx, label = hist_idx.to(device), cand_idx.to(device), label.to(device)
            hist_vecs, cand_vecs = news_embeddings[hist_idx], news_embeddings[cand_idx]
            
            optimizer.zero_grad()
            output = self.model(hist_vecs, cand_vecs)
            loss = criterion(output, label)
            loss.backward()
            
            # --- 差分隐私关键步骤 A：梯度裁剪 (Clipping) ---
            # 限制梯度模长，防止单个样本对模型产生过大影响（隐私保护的核心）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
        # 2. 获取训练后的参数
        new_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        # --- 差分隐私关键步骤 B：加注噪声 (Adding Noise) ---
        # SIGMA 为 0 时即为原始联邦学习
        # SIGMA 为 0.001 或 0.01 时即为隐私增强模式
        if SIGMA > 0:
            dp_params = []
            for p in new_params:
                # 生成与参数形状一致的高斯噪声
                noise = np.random.normal(0, SIGMA, p.shape)
                dp_params.append(p + noise)
            return dp_params, len(self.train_loader.dataset), {}
            
        return new_params, len(self.train_loader.dataset), {}

# --- 3. 定义服务器端评估逻辑 ---
# --- 3. 定义服务器端评估逻辑 ---
from sklearn.metrics import roc_auc_score

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[order]
    ranks = np.where(y_true_sorted == 1)[0]
    return 1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0

def get_evaluate_fn():
    # 在这里初始化，确保 evaluate 函数一定能找到它
    print("正在初始化验证集...")
    local_val_dataset = MINDDataset(dev_data, news_id_dict)
    
    def evaluate(server_round, parameters, config):
        # 设定关键评估节点：0轮(初始), 1, 5, 10, 15, 20轮
        eval_rounds = [0, 1, 5, 10, 15, 20]
        if server_round not in eval_rounds:
            return 0.0, {} 

        model = NewsRecommender(embedding_dim=384).to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        all_auc, all_mrr = [], []
        # 使用更大的 Batch 提速验证，这里使用 local_val_dataset
        test_loader = DataLoader(local_val_dataset, batch_size=256, shuffle=False)
        
        print(f"正在进行第 {server_round} 轮全量评估...")
        with torch.no_grad():
            for hist_idx, cand_idx, label in test_loader:
                hist_idx, cand_idx, label = hist_idx.to(device), cand_idx.to(device), label.to(device)
                hist_vecs, cand_vecs = news_embeddings[hist_idx], news_embeddings[cand_idx]
                output = model(hist_vecs, cand_vecs)
                
                scores_batch = output.cpu().numpy()
                labels_batch = label.cpu().numpy()
                for i in range(len(labels_batch)):
                    try:
                        y_true = np.zeros(scores_batch[i].shape)
                        y_true[labels_batch[i]] = 1
                        all_auc.append(roc_auc_score(y_true, scores_batch[i]))
                        all_mrr.append(mrr_score(y_true, scores_batch[i]))
                    except:
                        continue

        mean_auc, mean_mrr = np.mean(all_auc), np.mean(all_mrr)
        print(f"\n📈 [Round {server_round}] 关键节点评估: AUC: {mean_auc:.4f} | MRR: {mean_mrr:.4f}")
        
        # 保存模型逻辑
        if server_round % 10 == 0 and server_round > 0:
            if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
            save_path = os.path.join(CHECKPOINT_DIR, f'fed_model_dp_0.01.pth')
            torch.save(model.state_dict(), save_path)
            print(f"💾 模型已存档至: {save_path}")

        return 0.0, {"AUC": mean_auc, "MRR": mean_mrr}
    
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
        client_resources={"num_gpus": 0.15, "num_cpus": 3},
    )