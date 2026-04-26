import torch
import torch.nn as nn

class UserEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, use_attention=True):
        super(UserEncoder, self).__init__()
        self.use_attention = use_attention
        
        if self.use_attention:
            # 原有的多头注意力机制
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim, 
                num_heads=num_heads, 
                batch_first=True
            )
        
    def forward(self, history_vecs):
        """
        history_vecs: [batch_size, hist_len, embedding_dim]
        """
        if self.use_attention:
            # 完整模型：使用注意力机制聚合兴趣
            attn_output, _ = self.multihead_attention(
                history_vecs, history_vecs, history_vecs
            )
            # 取序列平均作为用户表示
            user_vector = torch.mean(attn_output, dim=1)
        else:
            # 消融实验：直接取平均，不进行注意力加权
            # 这种方式假设所有历史新闻对用户兴趣的贡献是一样的
            user_vector = torch.mean(history_vecs, dim=1)
            
        return user_vector

class NewsRecommender(nn.Module):
    def __init__(self, embedding_dim=384, use_attention=True):
        super(NewsRecommender, self).__init__()
        # 传递开关给 UserEncoder
        self.user_encoder = UserEncoder(embedding_dim, use_attention=use_attention)
        
    def forward(self, history_vecs, candidate_vecs):
        """
        history_vecs: [batch_size, hist_len, embedding_dim]
        candidate_vecs: [batch_size, num_candidates, embedding_dim]
        """
        # 1. 得到用户兴趣向量
        user_vector = self.user_encoder(history_vecs) # [batch_size, embedding_dim]
        
        # 2. 得到候选新闻向量 (这里简化处理，直接使用)
        # user_vector: [batch_size, 1, dim]
        user_vector = user_vector.unsqueeze(1)
        
        # 3. 计算点击概率（点积排序）
        # candidate_vecs: [batch_size, num_candidates, dim]
        scores = torch.bmm(candidate_vecs, user_vector.transpose(1, 2)).squeeze(2)
        
        return scores