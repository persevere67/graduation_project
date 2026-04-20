import pandas as pd
import os
# 第一层获取 preprocess/ 文件夹路径
# 第二层获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_path = os.path.join(BASE_DIR, 'processed')
# 1. 获取当前脚本 main.py 所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接完整的 news.tsv 路径
# 确保文件夹名字 MINDsmall_train 与你截图中的大小写完全一致
news_path = os.path.join(current_dir, 'dataset', 'MINDsmall_train', 'news.tsv')

print(f"正在尝试读取文件: {news_path}")

news_columns = [
    'NewsID', 'Category', 'SubCategory', 'Title', 
    'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'
]

if os.path.exists(news_path):
    try:
        # 读取文件
        news_df = pd.read_csv(news_path, sep='\t', names=news_columns, encoding='utf-8', keep_default_na=False)
        print("✅ 读取成功！数据样例如下：")
        print(news_df[['NewsID', 'Category', 'Title']].head())
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
else:
    print("❌ 错误：文件不存在！请检查：")
    print(f"1. 路径是否正确: {news_path}")
    print(f"2. dataset 文件夹里是否有 MINDsmall_train 文件夹")