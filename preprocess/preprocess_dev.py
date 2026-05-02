import os
import pickle

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

dev_behavior_path = os.path.join(DATASET_DIR, "MINDsmall_dev", "behaviors.tsv")

# 2. 加载“做法A”生成的全量新闻映射表
# 现在的 news_id_dict 已经是全量的了，包含了 train 和 dev
with open(os.path.join(PROCESSED_DIR, "news_id_dict.pkl"), "rb") as f:
    full_news_id_dict = pickle.load(f)

# 3. 读取行为数据
behavior_columns = ["ImpressionID", "UserID", "Time", "History", "Impressions"]
dev_df = pd.read_csv(dev_behavior_path, sep="\t", names=behavior_columns)

# 4. 数据清洗
# 必须有历史记录才能进行用户画像建模
dev_df = dev_df.dropna(subset=["History"])

# 5. 解析 Impressions（评估时需要看到所有的点击和未点击情况）
def process_dev_impressions(imp_str):
    if pd.isna(imp_str):
        return []
    pairs = [pair.split("-") for pair in imp_str.split()]
    # 只要新闻 ID 在我们的大字典里，就记录下来
    return [(full_news_id_dict[p[0]], int(p[1])) for p in pairs if p[0] in full_news_id_dict]

print("正在解析验证集行为数据...")
dev_df["parsed_impressions"] = dev_df["Impressions"].apply(process_dev_impressions)
# 过滤掉解析后为空的行
dev_df = dev_df[dev_df["parsed_impressions"].map(len) > 0]

# 6. 保存为验证集专用文件
# 评估时不需要分客户端，直接存为一个整体
dev_data_final = dev_df[["UserID", "History", "parsed_impressions"]]

with open(os.path.join(PROCESSED_DIR, "dev_data_all.pkl"), "wb") as f:
    pickle.dump(dev_data_final, f)

print(f"✅ 验证集处理完成！")
print(f"当前全量字典大小: {len(full_news_id_dict)}")
print(f"生成的验证集行为记录数: {len(dev_data_final)}")
