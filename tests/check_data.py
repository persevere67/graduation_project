import pickle
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkl_path = os.path.join(base_dir, "processed", "federated_data.pkl")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 取出第一个客户端的数据看一眼
first_client = list(data.keys())[0]
df = data[first_client]

print(f"--- 客户端: {first_client} 的样例数据 ---")
print(df.head(1))

# 重点检查 History 和 parsed_impressions
sample_history = df.iloc[0]['History']
sample_imp = df.iloc[0]['parsed_impressions']

print("\n--- 详细字段检查 ---")
print(f"History 类型: {type(sample_history)}")
print(f"History 内容样例: {sample_history[:5]}...") # 只看前5个
print(f"Parsed Impressions 内容: {sample_imp[:2]}...")
