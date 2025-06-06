# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Tuple
from torch import Tensor, nn
from torch.nn import functional as F
from .attention import AttentionRPE


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerBlock(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        d_model: int,
        n_head: int = 4,
        d_feedforward: int = 2048,
        dropout_p: float = 0.1,
        activation: str = "relu",
        n_layer: int = 1,
        norm_first: bool = True,
        decoder_self_attn: bool = False,
        bias: bool = True,
        d_rpe: int = -1,
        apply_q_rpe: bool = False,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerCrossAttention(
                    d_model=d_model,
                    n_head=n_head,
                    d_feedforward=d_feedforward,
                    dropout_p=dropout_p,
                    activation=activation,
                    norm_first=norm_first,
                    decoder_self_attn=decoder_self_attn,
                    bias=bias,
                    d_rpe=d_rpe,
                    apply_q_rpe=apply_q_rpe,
                )
                for _ in range(n_layer)
            ]
        )

        # self.layers = _get_clones(encoder_layer, n_layer)
        # self.n_layer = n_layer
        # self.norm = nn.LayerNorm(d_model) if norm_first else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        decoder_rpe: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            src_padding_mask: [n_batch, n_src], bool, if True, src is invalid.
            tgt: [n_batch, (n_src), n_tgt, d_model], None for self attention, (n_src) if using rpe.
            tgt_padding_mask: [n_batch, (n_src), n_tgt], bool, if True, tgt is invalid, (n_src) if using rpe.
            rpe: [n_batch, n_src, n_tgt, d_rpe]
            decoder_tgt: [n_batch, (n_src), n_tgt_decoder, d_model], (n_src) if using rpe.
            decoder_tgt_padding_mask: [n_batch, (n_src), n_tgt_decoder], (n_src) if using rpe.
            decoder_rpe: [n_batch, n_src, n_tgt_decoder, d_rpe]
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.

        Returns:
            src: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None

        Remarks:
            absoulte_pe should be already added to src/tgt.
        """
        attn_weights = None
        for mod in self.layers:
            src, attn_weights = mod(
                src=src,#(3,1024,256)
                src_padding_mask=src_padding_mask,#(3,1024)
                tgt=tgt,#(3,1024,36,256)
                tgt_padding_mask=tgt_padding_mask,
                rpe=rpe,#(3,1024,36,256)
                decoder_tgt=decoder_tgt, #None
                decoder_tgt_padding_mask=decoder_tgt_padding_mask, #None
                decoder_rpe=decoder_rpe,#None
                attn_mask=attn_mask, #None
                need_weights=need_weights,
            )
        # if self.norm is not None:
        #     src = self.norm(src)
        return src, attn_weights

## 模块实现了一个Transformer层，包含交叉注意力（cross-attention）
# 和可选的解码器自注意力（decoder self-attention），并支持相对位置编码（Relative Positional Encoding，RPE）-----就是说实现了一个crossattention和可选择decoder-selfattention（两个都包含AttentionRPE）
class TransformerCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int,
        dropout_p: float,
        activation: str,#前馈网络中使用的激活函数
        norm_first: bool,#是否在子层之前应用LayerNorm（前置归一化）
        decoder_self_attn: bool,#是否使用解码器自注意力
        bias: bool, #注意力投影中是否使用偏置
        d_rpe: int = -1,#相对位置编码的维度。如果小于等于0，则不使用RPE
        apply_q_rpe: bool = False,
    ) -> None:
        super(TransformerCrossAttention, self).__init__()
        self.norm_first = norm_first
        self.d_feedforward = d_feedforward
        self.decoder_self_attn = decoder_self_attn
        inplace = False

        self.dropout = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
        self.activation = _get_activation_fn(activation)
        self.norm1 = nn.LayerNorm(d_model)
        ##为True时候，定义解码器自注意力层，使用AttentionRPE模块
        if self.decoder_self_attn:
            self.attn_src = AttentionRPE(
                d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias, d_rpe=d_rpe, apply_q_rpe=apply_q_rpe
            )
            ##用于用于解码器自注意力归一化层
            self.norm_src = nn.LayerNorm(d_model)
            self.dropout_src = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

        if self.norm_first:
        ## 交叉注意力的归一化层。
            self.norm_tgt = nn.LayerNorm(d_model)
        ## 定义交叉注意力模块，使用AttentionRPE层
        self.attn = AttentionRPE(
            d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias, d_rpe=d_rpe, apply_q_rpe=apply_q_rpe
        )
        ##self.linear1、self.linear2：两层线性变换，构成前馈网络
        if self.d_feedforward > 0:
            self.linear1 = nn.Linear(d_model, d_feedforward)
            self.linear2 = nn.Linear(d_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
            self.dropout2 = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        decoder_rpe: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            src: [n_batch, n_src, d_model]
            src_padding_mask: [n_batch, n_src], bool, if True, src is invalid.
            tgt: [n_batch, (n_src), n_tgt, d_model], None for self attention, (n_src) if using rpe.
            tgt_padding_mask: [n_batch, (n_src), n_tgt], bool, if True, tgt is invalid, (n_src) if using rpe.
            rpe: [n_batch, n_src, n_tgt, d_rpe]
            decoder_tgt: [n_batch, n_src, n_tgt_decoder, d_model], when use decoder_rpe
            decoder_tgt_padding_mask: [n_batch, n_src, n_tgt_decoder], when use decoder_rpe
            decoder_rpe: [n_batch, n_src, n_tgt_decoder, d_rpe]
            attn_mask: [n_batch, n_src, n_tgt], bool, if True, attn is disabled for that pair of src/tgt.

        Returns:
            out: [n_batch, n_src, d_model]
            attn_weights: [n_batch, n_src, n_tgt] if need_weights else None

        Remarks:
            absoulte_pe should be already added to src/tgt.
         前向传播逻辑
解码器自注意力（可选）：

如果decoder_self_attn为True，首先对src输入进行自注意力计算。
根据norm_first决定归一化和残差连接的顺序。
使用self.attn_src模块计算注意力，并可能应用RPE。
交叉注意力：

如果tgt为None，则将tgt设为src（即自注意力）。
对src和tgt进行归一化（如果norm_first为True）。
使用self.attn模块计算注意力，可能应用RPE。
将注意力输出与src进行残差连接。
前馈网络（可选）：

如果d_feedforward > 0，则通过前馈网络进一步处理。
激活函数由activation参数指定。
归一化和残差连接的顺序同样取决于norm_first。
输出：

返回处理后的src张量，以及可选的注意力权重
        """
        if self.decoder_self_attn:
            # transformer decoder
            if self.norm_first:
                _s = self.norm_src(src)
                if decoder_tgt is None:
                    _s = self.attn_src(_s, tgt_padding_mask=src_padding_mask)[0]
                else:
                    decoder_tgt = self.norm_src(decoder_tgt)
                    _s = self.attn_src(_s, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask, rpe=decoder_rpe)[0]

                if self.dropout_src is None:
                    src = src + _s
                else:
                    src = src + self.dropout_src(_s)
            else:
                if decoder_tgt is None:
                    _s = self.attn_src(src, tgt_padding_mask=src_padding_mask)[0]
                else:
                    _s = self.attn_src(src, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask, rpe=decoder_rpe)[0]

                if self.dropout_src is None:
                    src = self.norm_src(src + _s)
                else:
                    src = self.norm_src(src + self.dropout_src(_s))

        if tgt is None:
            tgt_padding_mask = src_padding_mask

        if self.norm_first:
            src2 = self.norm1(src)
            if tgt is not None:
                tgt = self.norm_tgt(tgt)
        else:
            src2 = src

        # [n_batch, n_src, d_model]
        src2, attn_weights = self.attn(
            src=src2,
            tgt=tgt,
            tgt_padding_mask=tgt_padding_mask,
            attn_mask=attn_mask,
            rpe=rpe,
            need_weights=need_weights,
        )

        if self.d_feedforward > 0:
            if self.dropout1 is None:
                src = src + src2
            else:
                src = src + self.dropout1(src2)

            if self.norm_first:
                src2 = self.norm2(src)
            else:
                src = self.norm1(src)
                src2 = src

            src2 = self.activation(self.linear1(src2))
            if self.dropout is None:
                src2 = self.linear2(src2)
            else:
                src2 = self.linear2(self.dropout(src2))

            if self.dropout2 is None:
                src = src + src2
            else:
                src = src + self.dropout2(src2)

            if not self.norm_first:
                src = self.norm2(src)
        else:
            # densetnt vectornet
            src2 = self.activation(src2)
            if self.dropout is None:
                src = src + src2
            else:
                src = src + self.dropout(src2)
            if not self.norm_first:
                src = self.norm1(src)

        if src_padding_mask is not None:
            src.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
            if need_weights:
                attn_weights.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
        return src, attn_weights
