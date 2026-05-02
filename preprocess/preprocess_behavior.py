import os
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

behavior_path = os.path.join(DATASET_DIR, "MINDsmall_train", "behaviors.tsv")

# 1. 加载之前存好的 news_id_dict
with open(os.path.join(PROCESSED_DIR, "news_id_dict.pkl"), "rb") as f:
    news_id_dict = pickle.load(f)

# 2. 读取行为数据
behavior_columns = ["ImpressionID", "UserID", "Time", "History", "Impressions"]
df = pd.read_csv(behavior_path, sep="\t", names=behavior_columns)

# 3. 数据清洗
# 去掉历史记录为空的用户，并限制每个用户的历史记录长度（建议最近 50 条），防止爆显存
df = df.dropna(subset=["History"])

def process_impressions(imp_str):
    """将 'N1-1 N2-0' 转换成 list: [([N1_idx], 1), ([N2_idx], 0)]"""
    pairs = [pair.split("-") for pair in imp_str.split()]
    return [(news_id_dict[p[0]], int(p[1])) for p in pairs if p[0] in news_id_dict]

print("正在解析用户行为...")
df["parsed_impressions"] = df["Impressions"].apply(process_impressions)
# 过滤掉解析后为空的行为（可能因为新闻ID没在训练集中出现）
df = df[df["parsed_impressions"].map(len) > 0]

# 4. 模拟联邦学习：分发给 50 个客户端
num_clients = 50
unique_users = df["UserID"].unique()
np.random.shuffle(unique_users)
user_groups = np.array_split(unique_users, num_clients)

client_datasets = {}
for i in range(num_clients):
    client_users = set(user_groups[i])
    client_df = df[df["UserID"].isin(client_users)]
    # 只保留训练需要的核心列：UserID, History, parsed_impressions
    client_datasets[f"client_{i}"] = client_df[["UserID", "History", "parsed_impressions"]]

# 5. 保存
with open(os.path.join(PROCESSED_DIR, "federated_data.pkl"), "wb") as f:
    pickle.dump(client_datasets, f)

print(f"✅ 成功！已生成 {num_clients} 个客户端的数据。")
