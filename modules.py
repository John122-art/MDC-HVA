import math
import torch
import torch.nn as nn
import torch.nn.functional as F




@torch.no_grad()
def find_knn_neighbors_batched(batches,actorB, agentn, rpes_mask, K =15, distance_threshold=2.6):



    device = batches.device
   # batch_size = len(batches)
   # M = batches.shape[0]  # 每个批次的代理数量
   # result, rpen, result_rpe_mask =[],[],[]
    # 创建结果张量
    result = torch.zeros((agentn, K, 128), device=device)

    rpen = torch.zeros((agentn, K, 5), device=device)
    rpen_mask_part = torch.zeros((agentn, K), device=device)
   # padding = torch.zeros((K, 128), device=device)

    embeddings =actorB
       # positions = batches[b]

        # 计算距离
        #distances = torch.cdist(embeddings, embeddings)  # (M, M)
    batches_new = batches.clone()
    distances = batches_new[...,-1]

    M = distances.shape[0]
        # 设置对角线的相对位置为 inf，以排除自身
    distances[torch.arange(M), torch.arange(M)] = float('inf')

        # 应用距离阈值
    valid_mask = distances <= distance_threshold
    distances = torch.where(valid_mask, distances, float('inf'))#(191,191)
    batches_new[..., -1] = distances
    #a= distances[40,:]

        # 寻找 K 个最近邻
    for i in range(agentn):

        # i_tensor = torch.tensor([i], device=device)
        # neighbor_indices = torch.argsort(batches_new[i,:,-1])[:K]
        # neighbor_indices_with_i = torch.cat((i_tensor, neighbor_indices))
        # rpen.append(batches[neighbor_indices_with_i[:,None],neighbor_indices_with_i,:])
        # result.append(embeddings[neighbor_indices_with_i])

        # result_rpe_mask.append(rpes_mask[neighbor_indices_with_i[:,None],neighbor_indices_with_i])

        neighbor_indices = torch.argsort(distances[i])[:K]


        result[i, :len(neighbor_indices), :] = embeddings[neighbor_indices]
        #rpen[i, :len(neighbor_indices), :] = batches[neighbor_indices]
        rpen_mask_part[i, :len(neighbor_indices)] = rpes_mask[i,neighbor_indices]
        rpen_mask_part = rpen_mask_part.to(torch.bool)

    return result, rpen_mask_part





class CrossAttention(nn.Module):
    def __init__(self,device):
        super(CrossAttention, self).__init__()
        heads, dim, dropout = 8, 128, 0.1
        self.device=device
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, device=self.device)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim),
                                 nn.Dropout(dropout))

    def forward(self, query, val, mask):

        # query = B[:tok,:]
        N, D = query.shape
        query = query.view(N,1,D)
        # query = query.permute(1,0,2)
        # val = val.permute(1,0,2)

        attn_output, attn_weights = self.multihead_attn(query, val, val, key_padding_mask=mask)
        attn_output = attn_output.squeeze(1)  # [9, 128])
        attention_output = self.norm_1(attn_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output