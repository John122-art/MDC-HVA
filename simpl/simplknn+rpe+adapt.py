from typing import Any, Dict, List, Tuple, Union, Optional
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from fractions import gcd
#
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from .transformer import TransformerBlock
from utils.utils import gpu, init_weights

@torch.no_grad()
def get_tgt_knn_idx(
      rel_pose: Optional[Tensor], rel_dist: Tensor, n_tgt_knn: int,dist_limit: Union[float, Tensor]
) -> Tuple[Optional[Tensor], Tensor, Optional[Tensor]]:
    """
    Args:
        tgt_invalid: [n_scene, n_tgt](3,1024)表示目标元素的有效性
        rel_pose: [n_scene, n_src, n_tgt, 3](3,1024,1024,3)
        rel_dist: [n_scene, n_src, n_tgt]##表示源元素与目标元素之间的相对距离。
        knn: int, set to <=0 to skip knn, i.e. n_tgt_knn=n_tgt
        dist_limit: float, or Tensor [n_scene, n_tgt, 1] 1500

    Returns:
        idx_tgt: [n_scene, n_src, n_tgt_knn], or None  #表示每个源元素对应的最近目标元素的索引
        tgt_invalid_knn: [n_scene, n_src, n_tgt_knn] 布尔张量，表示每个源元素对应的目标元素的无效性。
        rpe: [n_scene, n_src, n_tgt_knn, 3]
    """
    n_src, n_tag, _ = rel_dist.shape#(23,162,1)
    #idx_scene = torch.arange(n_scene)[:, None, None]  # [n_scene, 1, 1]就是0,1,2
    idx_src = torch.arange(n_src)[:,None] # [1, n_src, 1]就是0,1,2,...1024

    if 0 < n_tgt_knn < rel_pose.shape[1]:
        # [n_scene, n_src, n_tgt_knn]
        n_tgt_knn = min(n_tgt_knn, rel_dist.size(-2))
        rel_dist = rel_dist.squeeze(-1)
        dist_knn, idx_tgt = torch.topk(rel_dist, n_tgt_knn, dim=-1, largest=False, sorted=False)#
        #idx_tgt = idx_tgt.squeeze(-1)
        # [n_scene, n_src, n_tgt_knn]
        # tgt_invalid_knn = tgt_invalid[idx_src, idx_tgt]
        # [n_batch, n_src, n_tgt_knn, 3]
        if rel_pose is None:
            rpe = None
        else:
            rpe = rel_pose[idx_src, idx_tgt]##高级索引就是很神奇，idx_tgt在rel_Pose的第三维度，就用idx_tgt的第三个维度去索引  （3,1024,36,3）
    else:
        dist_knn = rel_dist
       # tgt_invalid_knn = tgt_invalid # [n_scene, n_src, n_tgt](3,1024,36)
        rpe = rel_pose
        idx_tgt = None

    #tgt_invalid_knn = tgt_invalid_knn | (dist_knn > dist_limit)
    # if rpe is not None:
    #     rpe = rpe.masked_fill(tgt_invalid_knn.unsqueeze(-1), 0)

    return idx_tgt,  rpe

class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(
            int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, n_in=3, hidden_size=128, n_fpn_scale=4):
        super(ActorNet, self).__init__()
        norm = "GN"
        ng = 1

        n_out = [2**(5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128]
        blocks = [Res1d] * n_fpn_scale
        num_blocks = [2] * n_fpn_scale

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out


class PointAggregateBlock(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointAggregateBlock, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _global_maxpool_aggre(self, feat):
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x_inp):
        x = self.fc1(x_inp)  # [N_{lane}, 10, hidden_size]
        x_aggre = self._global_maxpool_aggre(x)
        x_aggre = torch.cat([x, x_aggre.repeat([1, x.shape[1], 1])], dim=-1)

        out = self.norm(x_inp + self.fc2(x_aggre))
        if self.aggre_out:
            return self._global_maxpool_aggre(out).squeeze()
        else:
            return out


class LaneNet(nn.Module):
    def __init__(self, device, in_size=10, hidden_size=128, dropout=0.1):
        super(LaneNet, self).__init__()
        self.device = device

        self.proj = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.aggre1 = PointAggregateBlock(hidden_size=hidden_size, aggre_out=False, dropout=dropout)
        self.aggre2 = PointAggregateBlock(hidden_size=hidden_size, aggre_out=True, dropout=dropout)

    def forward(self, feats):
        x = self.proj(feats)  # [N_{lane}, 10, hidden_size]
        x = self.aggre1(x)
        x = self.aggre2(x)  # [N_{lane}, hidden_size]
        return x


class SftLayer(nn.Module):
    def __init__(self,
                 device,
                 d_edge: int = 128,
                 d_model: int = 128,
                 d_ffn: int = 2048,
                 n_head: int = 8,
                 dropout: float = 0.1,
                 update_edge: bool = True) -> None:
        super(SftLayer, self).__init__()
        self.device = device
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                node: Tensor,
                edge: Tensor,
                edge_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                node:       (N, d_model)
                edge:       (N, N, d_model)
                edge_mask:  (N, N)
        '''
        # update node
        x, edge, memory = self._build_memory(node, edge)
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask)
        x = self.norm2(x + x_prime).squeeze()
        x = self.norm3(x + self._ff_block(x))
        return x, edge, None

    def _build_memory(self,
                      node: Tensor,
                      edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            input:
                node:   (N, d_model)
                edge:   (N, N, d_edge)
            output:
                :param  (1, N, d_model)
                :param  (N, N, d_edge)
                :param  (N, N, d_model)
        '''
        n_token = node.shape[0]

        # 1. build memory
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(dim=0), edge, memory

    # multihead attention block
    def _mha_block(self,
                   x: Tensor,
                   mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                x:                  [1, N, d_model]
                mem:                [N, N, d_model]
                attn_mask:          [N, N]
                key_padding_mask:   [N, N]
            output:
                :param      [1, N, d_model]
                :param      [N, N]
        '''
        x, _ = self.multihead_attn(x, mem, mem,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SymmetricFusionTransformer(nn.Module):
    def __init__(self,
                 device,
                 d_model: int = 128,
                 d_edge: int = 128,
                 n_head: int = 8,
                 n_layer: int = 6,
                 dropout: float = 0.1,
                 update_edge: bool = True):
        super(SymmetricFusionTransformer, self).__init__()
        self.device = device

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(SftLayer(device=device,
                                   d_edge=d_edge,
                                   d_model=d_model,
                                   d_ffn=d_model*2,
                                   n_head=n_head,
                                   dropout=dropout,
                                   update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
        '''
            x: (N, d_model)
            edge: (d_model, N, N)
            edge_mask: (N, N)
        '''
        # attn_multilayer = []
        for mod in self.fusion:
            x, edge, _ = mod(x, edge, edge_mask)
            # attn_multilayer.append(attn)
        return x, None


class FusionNet(nn.Module):
    def __init__(self, device, config):
        super(FusionNet, self).__init__()
        self.device = device

        d_embed = config['d_embed']
        dropout = config['dropout']
        update_edge = config['update_edge']

        self.proj_actor = nn.Sequential(
            nn.Linear(config['d_actor'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_lane = nn.Sequential(
            nn.Linear(config['d_lane'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(config['d_rpe_in'], config['d_rpe']),
            nn.LayerNorm(config['d_rpe']),
            nn.ReLU(inplace=True)
        )
        self.proj_memory = nn.Sequential(
            nn.Linear(d_embed + d_embed + d_embed, d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.fuse_scene = SymmetricFusionTransformer(self.device,
                                                     d_model=d_embed,
                                                     d_edge=config['d_rpe'],
                                                     n_head=config['n_scene_head'],
                                                     n_layer=config['n_scene_layer'],
                                                     dropout=dropout,
                                                     update_edge=update_edge)

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Dict[str, Tensor]):
        # print('actors: ', actors.shape)
        # print('actor_idcs: ', [x.shape for x in actor_idcs])
        # print('lanes: ', lanes.shape)
        # print('lane_idcs: ', [x.shape for x in lane_idcs])

        # projection
        actors = self.proj_actor(actors)
        lanes = self.proj_lane(lanes)

        actors_new, lanes_new = list(), list()
        for a_idcs, l_idcs, rpes in zip(actor_idcs, lane_idcs, rpe_prep):
            # * fusion - scene
            _actors = actors[a_idcs]
            _lanes = lanes[l_idcs]
            tokens = torch.cat([_actors, _lanes], dim=0)  # (N, d_model)
            rpe = self.proj_rpe_scene(rpes['scene'].permute(1, 2, 0))  # (N, N, d_rpe)
            RPE_maxpooled = F.max_pool1d(rpe.permute(0, 2, 1), kernel_size=rpe.shape[1]).squeeze(-1)
            RPE_avgpooled = F.avg_pool1d(rpe.permute(0, 2, 1), kernel_size=rpe.shape[1]).squeeze(-1)
            tokens_new = torch.cat([tokens, RPE_maxpooled, RPE_avgpooled], dim=1)
            tokens = self.proj_memory(tokens_new)
            out, _ = self.fuse_scene(tokens, rpe, rpes['scene_mask'])

            actors_new.append(out[:len(a_idcs)])
            lanes_new.append(out[len(a_idcs):])
        # print('actors: ', [x.shape for x in actors_new])
        # print('lanes: ', [x.shape for x in lanes_new])
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        # print('actors: ', actors.shape)
        # print('lanes: ', lanes.shape)
        return actors, lanes, None


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        # [dim]
        freqs = freqs.repeat_interleave(2, 0)
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor):
        """
        Args:
            x: [...]
        Returns:
            pos_enc: [..., dim]
        """
        # [..., dim]
        pos_enc = x.unsqueeze(-1) * self.freqs.view([1] * x.dim() + [-1])
        pos_enc = torch.cat([torch.cos(pos_enc[..., ::2]), torch.sin(pos_enc[..., 1::2])], dim=-1)
        return pos_enc

class PosePE(nn.Module):
    def __init__(self, pe_dim: int = 128, theta_xy: float = 1e3, theta_cs: float = 1e1):
        super().__init__()


        self.out_dim = pe_dim
        self.pe_xy = PositionalEmbedding(dim=pe_dim // 4, theta=theta_xy)
        self.pe_dir = PositionalEmbedding(dim=pe_dim // 4, theta=theta_cs)
        # elif self.mode == "pe_xy_yaw":
        #     self.out_dim = pe_dim
        #     self.pe_xy = PositionalEmbedding(dim=pe_dim // 4, theta=theta_xy)
        #     self.pe_yaw = PositionalEmbeddingRad(dim=pe_dim // 2)


    def forward(self, xy: Tensor, dir: Tensor):
        """
        Args: input either dir or yaw.
            xy: [..., 2]
            dir: cos/sin [..., 2] or yaw [..., 1]

        Returns:
            pos_out: [..., self.out_dim]
        """

        if dir.shape[-1] == 1:
            dir = torch.cat([dir.cos(), dir.sin()], dim=-1)
        pos_out = torch.cat(
            [self.pe_xy(xy[..., 0]), self.pe_xy(xy[..., 1]), self.pe_dir(dir[..., 0]), self.pe_dir(dir[..., 1])],
            dim=-1,
        )

        return pos_out

class Trajectory_Decoder(nn.Module):
    def __init__(self, device, config):
        super(Trajectory_Decoder, self).__init__()
        self.hidden_size = config['d_embed']
        self.device =device
        self.num_modes = config['g_num_modes']
        self.endpoint_predictor = Dynamic_Trajectory_Decoder(self.device,self.hidden_size, 6*2)
        #self.covert = MLPEND(self.device,self.hidden_size*6, self.hidden_size, residual=True)

        self.get_trajectory = MLPEND( self.device,self.hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLPEND(self.device,self.hidden_size +2, 2, residual=True)
        self.get_prob = MLPEND(self.device,self.hidden_size + 2, 1, residual=True)
        # dim_mm = self.hidden_size * self.num_modes
        # dim_inter = dim_mm // 2
        # self.aban_k = nn.Sequential(
        #     nn.Linear(dim_mm, dim_inter),
        #     nn.LayerNorm(dim_inter),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim_inter, self.hidden_size),
        #     nn.LayerNorm(self.hidden_size),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.multihead_proj = nn.Sequential(
        #     nn.Linear(self.hidden_size, dim_inter),
        #     nn.LayerNorm(dim_inter),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim_inter, dim_mm),
        #     nn.LayerNorm(dim_mm),
        #     nn.ReLU(inplace=True)
        # )
        self.proj_actor = nn.Sequential(
            nn.Linear(config['d_actor'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True)
        )
    def forward(self, embed,actor_idcs):
        # agent_features.shape = (N, k, 128)
        #res_cls2, res_reg2, res_aux2 = [], [], []
        agent_features = self.proj_actor(embed)
       # agent_features = torch.cat(agent_features,dim=0)
        N = agent_features.shape[0]
        D = agent_features.shape[1]
       # D = agent_features.shape[2]
        #agent_features.view(N,-1)
        #agent_features = self.aban_k(agent_features.view(N,-1))

        endpoints = self.endpoint_predictor(agent_features).view(N, 6, 2)
        agent_features = agent_features.unsqueeze(dim=1).expand(N, 6, D)
        offsets = self.endpoint_refiner(torch.cat([agent_features, endpoints.detach()], dim=-1))
        endpoints += offsets
        agent_features = torch.cat([agent_features, endpoints.detach()], dim=-1)

        predictions = self.get_trajectory(agent_features).view(N, 6, 29, 2)
        logits = self.get_prob(agent_features).view(N,  6)

        logits = F.softmax(logits * 1, dim=1)

        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        assert predictions.shape == (N, 6, 30, 2)
        res_cls2, res_reg2 = [], []
        for i in range(len(actor_idcs)):


           idcs = actor_idcs[i]
           res_cls2.append(logits[idcs])
           res_reg2.append(predictions[idcs])

        return res_cls2, res_reg2
class MLPDecoder(nn.Module):
    def __init__(self,
                 device,
                 config) -> None:
        super(MLPDecoder, self).__init__()
        self.device = device
        self.config = config
        self.hidden_size = config['d_embed']
        self.future_steps = config['g_pred_len']
        self.num_modes = config['g_num_modes']
        self.param_out = config['param_out']  # parametric output: bezier/monomial/none
        self.N_ORDER = config['param_order']

        dim_mm = self.hidden_size * self.num_modes
        dim_inter = dim_mm // 2
        self.multihead_proj = nn.Sequential(
            nn.Linear(self.hidden_size, dim_inter),
            nn.LayerNorm(dim_inter),
            nn.ReLU(inplace=True),
            nn.Linear(dim_inter, dim_mm),
            nn.LayerNorm(dim_mm),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)
        )

        if self.param_out == 'bezier':
            self.mat_T = self._get_T_matrix_bezier(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)
            self.mat_Tp = self._get_Tp_matrix_bezier(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2)
            )
        elif self.param_out == 'monomial':
            self.mat_T = self._get_T_matrix_monomial(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)
            self.mat_Tp = self._get_Tp_matrix_monomial(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2)
            )
        elif self.param_out == 'none':
            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2)
            )
        else:
            raise NotImplementedError

    def _get_T_matrix_bezier(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = math.comb(n_order, i) * (1.0 - ts)**(n_order - i) * ts**i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_bezier(self, n_order, n_step):
        # ~ 1st derivatives
        # ! NOTICE: we multiply n_order inside of the Tp matrix
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = n_order * math.comb(n_order - 1, i) * (1.0 - ts)**(n_order - 1 - i) * ts**i
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def _get_T_matrix_monomial(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = ts ** i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_monomial(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = (i + 1) * (ts ** i)
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def forward(self,
                embed: torch.Tensor,
                actor_idcs: List[Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # input embed: [159, 128]
        embed = self.multihead_proj(embed).view(-1, self.num_modes, self.hidden_size).permute(1, 0, 2)
        # print('embed: ', embed.shape)  # e.g., [6, 159, 128]

        cls = self.cls(embed).view(self.num_modes, -1).permute(1, 0)  # e.g., [159, 6]
        cls = F.softmax(cls * 1.0, dim=1)  # e.g., [159, 6]

        if self.param_out == 'bezier':
            param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 2)  # e.g., [6, 159, N_ORDER + 1, 2]
            param = param.permute(1, 0, 2, 3)  # e.g., [159, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [159, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, torch.diff(param, dim=2)) / (self.future_steps * 0.1)
        elif self.param_out == 'monomial':
            param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 2)  # e.g., [6, 159, N_ORDER + 1, 2]
            param = param.permute(1, 0, 2, 3)  # e.g., [159, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [159, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, param[:, :, 1:, :]) / (self.future_steps * 0.1)
        elif self.param_out == 'none':
            reg = self.reg(embed).view(self.num_modes, -1, self.future_steps, 2)  # e.g., [6, 159, 30, 2]
            reg = reg.permute(1, 0, 2, 3)  # e.g., [159, 6, 30, 2]
            vel = torch.gradient(reg, dim=-2)[0] / 0.1  # vel is calculated from pos

        # print('reg: ', reg.shape, 'cls: ', cls.shape)
        # de-batchify
        res_cls, res_reg, res_aux = [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            res_cls.append(cls[idcs])
            res_reg.append(reg[idcs])

            if self.param_out == 'none':
                res_aux.append((vel[idcs], None))  # ! None is a placeholder
            else:
                res_aux.append((vel[idcs], param[idcs]))  # List[Tuple[Tensor,...]]

        return res_cls, res_reg, res_aux
class MLPEND(nn.Module):
    def __init__(self,device, input_dim, output_dim, p_drop=0.0, hidden_dim=None, residual=False):
        super(MLPEND, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(layer2_dim, output_dim)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)
        self.device = device
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(torch.cat([out, x], dim=-1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out

class Dynamic_Trajectory_Decoder(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super(Dynamic_Trajectory_Decoder, self).__init__()
        self.device = device
        self.D_in = input_dim #128
        self.D_out = output_dim #12

        self.mlp = MLPEND( self.device,self.D_in, self.D_in, residual=True)

        self.weight_layer1 = nn.Linear(self.D_in, self.D_in*self.D_out)
        self.bias1 = nn.Linear(self.D_in, self.D_out)

        self.weight_layer2 = nn.Linear(self.D_in, self.D_out*self.D_out)
        self.bias2 = nn.Linear(self.D_in, self.D_out)

        self.norm1 = nn.LayerNorm(self.D_out)

    def forward(self, agent_features):
        # agent_features.shape = (N, M, D)(batch,agent,d)
        N = agent_features.shape[0]
        D = agent_features.shape[1]
       # D = agent_features.shape[2]

        assert D == self.D_in

        D_in = self.D_in
        D_out = self.D_out

        # agent_features_weights.shape = (N*M, D_in)
        w_source = self.mlp(agent_features)#eg(145,128)
        # agent_features.shape = (N*M, D_in, adapt+mtr最后结果)
        agent_features = agent_features.view(N, D_in, 1)#(145,128,1)

        # === Weight Calculation ===
        # W_1.shape = (N*M, D_out, D_in)
        W_1 = self.weight_layer1(w_source).view(-1, D_out, D_in)#(145,12,128)
        # b_1.shape = (N, M, D_out)
        b_1 = self.bias1(w_source).view(N, D_out)#(145,12)

        # W_2.shape = (N*M, D_out, D_out)
        W_2 = self.weight_layer2(w_source).view(-1, D_out, D_out)#(145,12,12)
        # b_2.shape = (N, M, D_out)
        b_2 = self.bias2(w_source).view(N,  D_out)#(145,12)
        # === === ===

        # agent_features.shape = (N, M, D_out)
        out = torch.bmm(W_1, agent_features).view(N,  D_out)#(145,12)
        out += b_1
        out = self.norm1(out)
        out = F.relu(out)

        # out.shape = (N*M, D_out, adapt+mtr最后结果)
        out = out.view(N, D_out, 1)#(145,12,1)

        # agent_features.shape = (N, M, D_out)
        out = torch.bmm(W_2, out).view(N,  D_out)
        out += b_2

        return out






class SceneRealitivePose(nn.Module):
    def __init__(self, device,
                 config, n_layer_map: int,n_layer_agent: int):
        super(SceneRealitivePose, self).__init__()
        self.device = device
        self.config = config
        self.hidden_size = config['d_embed']
        self.pose = PosePE(pe_dim=self.hidden_size)
        self.intra_class_encoder = IntraClassEncoder(device=self.device,config=self.config,n_layer_map=1,n_layer_agent=1,pose_rpe= self.pose, n_tgt_knn=20)

    def forward(self,actors: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Dict[str, Tensor],
                rel_pose:List[Tensor],
                inference_repeat_n: int = 1):
        actors_new, lanes_new = list(), list()
        for a_idcs, l_idcs, rpes ,rel_po in zip(actor_idcs, lane_idcs, rpe_prep,rel_pose):
            # * fusion - scene
            _actors = actors[a_idcs]
            #n_agent = _actors.shape[0]
            _lanes = lanes[l_idcs]
            tokens_all = torch.cat([_actors, _lanes], dim=0)  # (N, d_model)
            rel_dist = rpes['scene'].permute(1, 2, 0)[...,-1]
            rel_dist.fill_diagonal_(30)
            rel_dist = rel_dist.unsqueeze(-1)

            for _ in range(inference_repeat_n):
                 actor= self.intra_class_encoder(


                    agent_attr=_actors,

                    map_attr=_lanes,

                    rel_pose=rel_po,
                    rel_dist=rel_dist,
                    dist_limit_map=8,
                    tokens_all=tokens_all
                )

            actors_new.append(actor.squeeze(0))
            lanes_new.append(_lanes)
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        return actors,lanes



            # rpe = self.proj_rpe_scene(rpes['scene'].permute(1, 2, 0))  # (N, N, d_rpe)
            # out, _ = self.fuse_scene(tokens, rpe, rpes['scene_mask'])
class IntraClassEncoder(nn.Module):
    def __init__(self,device,
                 config,n_layer_map: int,n_layer_agent:int, n_tgt_knn: int, pose_rpe: nn.Module) -> None:
        super(IntraClassEncoder, self).__init__()
        self.device = device
        self.config = config
        self.pose_rpe = pose_rpe
        self.hidden_size = config['d_embed']
        self.n_tgt_knn = 20
        self.n_agent_knn = 10
        if n_layer_map > 0:
            self.agent_map = TransformerBlock(
                        d_model= self.hidden_size, d_feedforward=2048, d_rpe=self.pose_rpe.out_dim
                    )
        # if n_layer_agent > 0:
        #     self.agent_agent =TransformerBlock(d_model=   self.hidden_size, d_feedforward=2048, d_rpe=self.pose_rpe.out_dim)



    def forward(self,map_attr,agent_attr,rel_pose,rel_dist,dist_limit_map,tokens_all):


        n_map = map_attr.shape[0]
        n_agent = agent_attr.shape[0]
        n_all = tokens_all.shape[0]

        _map_idx_knn, _map_rpe_knn = get_tgt_knn_idx( rel_pose[:n_agent, n_agent:, :], rel_dist[:n_agent, n_agent:, :],self.n_tgt_knn, dist_limit=dist_limit_map)
        _rpe = self.pose_rpe(xy=_map_rpe_knn[..., :2], dir=_map_rpe_knn[..., [2]])  #_rpe（23，20，128）
        _idx_agent = torch.arange(n_agent)[:, None].expand(-1, self.n_tgt_knn)  #(23,20)
        _map_idx_knn_adjusted = _map_idx_knn+n_agent#(23,20)
       # _idx_map=torch.arange(_map_idx_knn.shape[1])[None,:,None]

        _tgt = tokens_all.unsqueeze(1).expand(-1, n_all, -1)
        _tgt = _tgt[_idx_agent, _map_idx_knn_adjusted]
        agentone, _ = self.agent_map(
            src=agent_attr.unsqueeze(0),  # [n_scene, n_map, hidden_dim]
            src_padding_mask=None,  # [n_scene, n_map]
            tgt=_tgt.unsqueeze(0),
            tgt_padding_mask=None,  # [n_scene, n_map, n_tgt_knn]
            rpe=_rpe,  # [n_scene, n_map, n_tgt_knn, d_rpe]
                            )
        return agentone



        # if n_agent > self.n_agent_knn+1:
        #
        #     _agent_idx_knn, _agent_rpe_knn = get_tgt_knn_idx(
        #
        #       rel_pose[:n_agent, :n_agent, :],  # rel_pose[n_scene, n_map+n_tl+n_agent, n_map+n_tl+n_agent, 3],故只取
        #       rel_dist[:n_agent, :n_agent, :],
        #       self.n_agent_knn ,
        #       dist_limit=dist_limit_map,
        #      )
        #
        #     _rpeap = self.pose_rpe(xy=_agent_rpe_knn[..., :2], dir=_agent_rpe_knn[..., [2]])  # _rpe（23，20，128）
        #     _idx_agento = torch.arange(n_agent)[:, None].expand(-1,  self.n_agent_knn )  # [n_map, 1,1]
        # #_map_idx_knn_adjusted = _map_idx_knn + n_agent  # (23,20)
        # # _idx_map=torch.arange(_map_idx_knn.shape[1])[None,:,None]
        #
        #     _tgtap = tokens_all.unsqueeze(1).expand(-1, n_all, -1)
        #     _tgtap = _tgtap[_idx_agento, _agent_idx_knn]
        #     agentwo, _ = self.agent_agent(
        #      src=agentone,  # [n_scene, n_map, hidden_dim]
        #      src_padding_mask=None,  # [n_scene, n_map]
        #      tgt=_tgtap.unsqueeze(0),
        #      tgt_padding_mask=None,  # [n_scene, n_map, n_tgt_knn]
        #      rpe=_rpeap,  # [n_scene, n_map, n_tgt_knn, d_rpe]
        #      )
        #
        #
        # return agentwo


class Simpl(nn.Module):
    # Initialization
    def __init__(self, cfg, device):
        super(Simpl, self).__init__()
        self.device = device

        self.actor_net = ActorNet(n_in=cfg['in_actor'],
                                  hidden_size=cfg['d_actor'],
                                  n_fpn_scale=cfg['n_fpn_scale'])

        self.lane_net = LaneNet(device=self.device,
                                in_size=cfg['in_lane'],
                                hidden_size=cfg['d_lane'],
                                dropout=cfg['dropout'])
        self.knn_net = SceneRealitivePose(device=self.device,config=cfg,n_layer_map=1,n_layer_agent=1)

        self.fusion_net = FusionNet(device=self.device,
                                    config=cfg)

        # self.pred_net = MLPDecoder(device=self.device,
        #                            config=cfg)
        self.adjust_pred = Trajectory_Decoder(device=self.device,
                                    config=cfg)

        if cfg["init_weights"]:
            self.apply(init_weights)

    def forward(self, data):
        actors, actor_idcs, lanes, lane_idcs, rpe,rel_pose = data

        # * actors/lanes encoding
        actors = self.actor_net(actors)  # output: [N_{actor}, 128]
        lanes = self.lane_net(lanes)  # output: [N_{lane}, 128]
        actors, lanes = self.knn_net(actors, actor_idcs, lanes, lane_idcs, rpe, rel_pose)
        # * fusion
        actors, lanes, _ = self.fusion_net(actors, actor_idcs, lanes, lane_idcs, rpe)
        # * decoding
        out = self.adjust_pred(actors, actor_idcs)
        #out = self.pred_net(actors, actor_idcs)

        return out

    def pre_process(self, data):
        '''
            Send to device
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'TRAJS_FUT', 'PAD_OBS', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS',
            'LANE_GRAPH',
            'RPE',
            'ACTORS', 'ACTOR_IDCS', 'LANES', 'LANE_IDCS'
        '''
        actors = gpu(data['ACTORS'], self.device)
        actor_idcs = gpu(data['ACTOR_IDCS'], self.device)
        lanes = gpu(data['LANES'], self.device)
        lane_idcs = gpu(data['LANE_IDCS'], self.device)
        rpe = gpu(data['RPE'], self.device)
        # RPE_XY = gpu(data['RPE_XY'], self.device)
        # total_yaws = gpu( data['TOTAL_YAW'],self.device)
        rel_pose =gpu(data['REL_POSE'], self.device)

        return actors, actor_idcs, lanes, lane_idcs, rpe,rel_pose

    def post_process(self, out):
        post_out = dict()
        res_cls = out[0]
        res_reg = out[1]

        # get prediction results for target vehicles only
        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0)
        cls = torch.stack([probs[0] for probs in res_cls], dim=0)

        post_out['out_raw'] = out
        post_out['traj_pred'] = reg  # batch x n_mod x pred_len x 2
        post_out['prob_pred'] = cls  # batch x n_mod

        return post_out
