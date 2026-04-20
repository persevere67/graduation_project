import os
# 获取 main.py 所在的文件夹，即项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
processed_path = os.path.join(BASE_DIR, 'processed')
dataset_path = os.path.join(BASE_DIR, 'dataset')