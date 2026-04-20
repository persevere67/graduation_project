import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import random
class MINDDataset(Dataset):
    def __init__(self, dataframe, news_id_dict, max_hist_len=50):
        """
        dataframe: 包含 UserID, History, parsed_impressions 的数据框
        news_id_dict: 全量新闻 ID 到索引的映射字典
        max_hist_len: 用户历史记录的最大长度（多删少补）
        """
        self.data = dataframe
        self.news_id_dict = news_id_dict
        self.max_hist_len = max_hist_len
        
        # 预处理：将 History 字符串转为索引列表，方便快速读取
        self.processed_histories = []
        for hist_str in self.data['History']:
            # 拆分字符串并查表
            hist_ids = [self.news_id_dict[nid] for nid in hist_str.split() if nid in self.news_id_dict]
            # Padding 逻辑：取最近的 max_hist_len 个，不足则补 0
            if len(hist_ids) > self.max_hist_len:
                hist_ids = hist_ids[-self.max_hist_len:]
            else:
                hist_ids = [0] * (self.max_hist_len - len(hist_ids)) + hist_ids
            self.processed_histories.append(hist_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 获取历史记录索引
        history = torch.tensor(self.processed_histories[idx], dtype=torch.long)
        
        # 2. 获取本次点击情况 (impressions)
        # 注意：训练时我们通常进行负采样，这里先取出一个正样本和几个负样本
        impressions = self.data.iloc[idx]['parsed_impressions']
        
        # 为了简化实验，我们假设每次只训练一个正样本和一个负样本的对比
        pos_samples = [i[0] for i in impressions if i[1] == 1]
        neg_samples = [i[0] for i in impressions if i[1] == 0]
        
        # 如果没有正样本，随机塞一个（鲁棒性处理）
        target_pos = pos_samples[0] if len(pos_samples) > 0 else 0
        # 负采样：随机选 4 个负样本
        if len(neg_samples) >= 4:
            target_negs = random.sample(neg_samples, 4)
        else:
            target_negs = neg_samples + [0] * (4 - len(neg_samples))
            
        candidates = torch.tensor([target_pos] + target_negs, dtype=torch.long)
        # 标签：正样本在第 0 位
        labels = torch.tensor(0, dtype=torch.long) 
        
        return history, candidates, labels