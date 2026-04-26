import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from models import NewsRecommender

# --- 1. 基础配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

def mrr_score(y_true, y_score):
    """计算单条 impression 的 MRR"""
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[order]
    ranks = np.where(y_true_sorted == 1)[0]
    if len(ranks) > 0:
        return 1.0 / (ranks[0] + 1)
    return 0.0

def evaluate_model(model_path, model_name="Model"):
    print(f"\n🚀 正在评估模型: {model_name}")
    print(f"加载权重: {model_path}")
    
    # --- 2. 加载数据 ---
    news_embeddings = torch.FloatTensor(np.load(os.path.join(PROCESSED_DIR, 'news_embeddings.npy'))).to(device)
    with open(os.path.join(PROCESSED_DIR, 'dev_data_all.pkl'), 'rb') as f:
        dev_data = pickle.load(f)
    # 【新增】加载字典用于 History 翻译
    with open(os.path.join(PROCESSED_DIR, 'news_id_dict.pkl'), 'rb') as f:
        news_id_dict = pickle.load(f)
        
    # --- 3. 初始化并加载模型 ---
    model = NewsRecommender(embedding_dim=384).to(device)
    # 加入 weights_only=True 消除安全警告
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
    model.eval()

    all_auc = []
    all_mrr = []
    
    print(f"开始遍历验证集，共 {len(dev_data)} 条测试数据...")
    
    with torch.no_grad():
        for idx, row in dev_data.iterrows():
            # 1. 处理历史记录
            raw_hist = row['History'].split() if isinstance(row['History'], str) else row['History']
            # 【关键修改】利用字典将字符串转为索引，过滤掉不认识的ID
            hist_ids = [news_id_dict[nid] for nid in raw_hist if nid in news_id_dict]
            
            if len(hist_ids) > 50:
                hist_ids = hist_ids[-50:]
            else:
                hist_ids = [0] * (50 - len(hist_ids)) + hist_ids
                
            hist_tensor = torch.tensor([hist_ids], dtype=torch.long).to(device)
            hist_vecs = news_embeddings[hist_tensor] # [1, 50, 384]

            # 2. 处理候选新闻
            impressions = row['parsed_impressions']
            cand_ids = [imp[0] for imp in impressions]
            labels = [imp[1] for imp in impressions]
            
            if sum(labels) == 0 or sum(labels) == len(labels):
                continue
                
            cand_tensor = torch.tensor([cand_ids], dtype=torch.long).to(device)
            cand_vecs = news_embeddings[cand_tensor] # [1, num_cand, 384]
            
            # 3. 模型打分
            scores = model(hist_vecs, cand_vecs).squeeze(0).cpu().numpy() # [num_cand]
            
            # 4. 计算指标
            auc = roc_auc_score(labels, scores)
            mrr = mrr_score(labels, scores)
            
            all_auc.append(auc)
            all_mrr.append(mrr)
            
            if (idx + 1) % 5000 == 0:
                print(f"已评估 {idx + 1} 条，当前平均 AUC: {np.mean(all_auc):.4f} | MRR: {np.mean(all_mrr):.4f}")

    final_auc = np.mean(all_auc)
    final_mrr = np.mean(all_mrr)
    print(f"\n✅ {model_name} 评估完成！")
    print(f"🏆 最终 AUC: {final_auc:.4f}")
    print(f"🏆 最终 MRR: {final_mrr:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    centralized_model_path = os.path.join(CHECKPOINT_DIR, 'centralized_model.pth')
    if os.path.exists(centralized_model_path):
        evaluate_model(centralized_model_path, model_name="中心化模型 (Centralized Baseline)")
    else:
        print(f"未找到模型文件：{centralized_model_path}，请确认路径！")