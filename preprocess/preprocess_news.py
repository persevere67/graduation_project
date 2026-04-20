import pandas as pd
import numpy as np
import os
import torch
import pickle
from sentence_transformers import SentenceTransformer
# 第一层获取 preprocess/ 文件夹路径
# 第二层获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_path = os.path.join(BASE_DIR, 'processed')
# 设置镜像环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
train_news_path = os.path.join(current_dir, 'dataset', 'MINDsmall_train', 'news.tsv')
dev_news_path = os.path.join(current_dir, 'dataset', 'MINDsmall_dev', 'news.tsv')
save_path = os.path.join(current_dir, 'processed')
if not os.path.exists(save_path): os.makedirs(save_path)

# 1. 加载数据并合并
news_columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
print("正在读取并合并训练集与验证集新闻...")

train_news = pd.read_csv(train_news_path, sep='\t', names=news_columns, encoding='utf-8', keep_default_na=False)
dev_news = pd.read_csv(dev_news_path, sep='\t', names=news_columns, encoding='utf-8', keep_default_na=False)

# 合并后按 NewsID 去重
full_news_df = pd.concat([train_news, dev_news]).drop_duplicates(subset=['NewsID'])
print(f"去重后总新闻数: {len(full_news_df)}")

# 2. 准备文本
full_texts = (full_news_df['Title'] + " " + full_news_df['Abstract']).tolist()

# 3. 加载 SBERT 并推理
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name).to(device)

print("开始全量提取特征向量（3060 加速中）...")
with torch.no_grad():
    # 6GB 显存，batch_size=128 比较稳妥
    embeddings = model.encode(full_texts, batch_size=128, show_progress_bar=True, device=device)

# 4. 保存核心文件
# 保存全量 Embedding 矩阵
np.save(os.path.join(save_path, 'news_embeddings.npy'), embeddings)

# 保存全量 ID 到索引的映射字典
news_id_dict = {nid: i for i, nid in enumerate(full_news_df['NewsID'])}
with open(os.path.join(save_path, 'news_id_dict.pkl'), 'wb') as f:
    pickle.dump(news_id_dict, f)

print(f"✅ 做法A完成！")
print(f"已保存 {len(embeddings)} 个新闻向量，映射字典已同步更新。")