import torch
import torch.nn as nn
import torch.nn.functional as F

class UserEncoder(nn.Module):
    def __init__(self, embedding_dim=384, num_heads=8):
        super(UserEncoder, self).__init__()
        # 1. 多头自注意力：学习历史记录之间的关联
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        # 2. Add & Norm 层
        self.layer_norm = nn.LayerNorm(embedding_dim)
        # 3. 最终投影，得到用户兴趣表示
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, history_embeddings):
        """
        history_embeddings shape: [batch_size, history_len, embedding_dim]
        """
        # 注意力计算：让历史记录互相“交谈”
        # 比如看了“梅西”会加强对“足球”记录的关注
        attn_output, _ = self.multihead_attn(
            history_embeddings, history_embeddings, history_embeddings
        )
        
        # 残差连接与归一化
        out = self.layer_norm(attn_output + history_embeddings)
        
        # 聚合历史信息：这里采用最简单的平均聚合（Pooling）
        # 也可以改用更复杂的 Attention Pooling
        user_vector = torch.mean(out, dim=1)
        
        return self.fc(user_vector)

class NewsRecommender(nn.Module):
    def __init__(self, embedding_dim=384):
        super(NewsRecommender, self).__init__()
        # 新闻编码：因为 SBERT 向量已经是预训练好的，
        # 我们这里加一个线性层做微调（Feature Adaptation）
        self.news_proj = nn.Linear(embedding_dim, embedding_dim)
        self.user_encoder = UserEncoder(embedding_dim)

    def forward(self, history_vecs, candidate_vecs):
        """
        history_vecs: 用户历史新闻向量 [batch_size, hist_len, 384]
        candidate_vecs: 候选新闻向量 [batch_size, num_candidates, 384]
        """
        # 1. 获取用户向量
        user_vector = self.user_encoder(history_vecs) # [batch_size, 384]
        
        # 2. 对候选新闻进行投影
        cand_proj = self.news_proj(candidate_vecs) # [batch_size, num_cand, 384]
        
        # 3. 计算匹配得分 (点积)
        # 维度变换以便进行批量矩阵乘法
        user_vector = user_vector.unsqueeze(2) # [batch_size, 384, 1]
        scores = torch.bmm(cand_proj, user_vector) # [batch_size, num_cand, 1]
        
        return scores.squeeze(2) # [batch_size, num_cand]