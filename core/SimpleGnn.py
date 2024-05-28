import math
from typing import Union, Tuple, Optional
from torch import Tensor, cat
from torch.nn import init, Parameter, Linear, LayerNorm
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn import aggr
import torch
def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class HypergraphTransformer(MessagePassing):
    r"""Hypergraph Conv containing relation transform、edge fusion(including time fusion)、
    self attention and gated residual connection(or skip connection).

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        attn_heads: int = 4,
        residual_beta: Optional[float] = None,
        learn_beta: bool = False,
        dropout: float = 0.,
        negative_slope: float = 0.2,
        bias: bool = True,
        trans_method: str = 'add',
        edge_fusion_mode: str = 'add',
        time_fusion_mode: str = None,
        head_fusion_mode: str = 'concat',
        residual_fusion_mode: str = None,
        edge_dim: int = None,
        rel_embed_dim: int = None,
        time_embed_dim: int = 0,
        dist_embed_dim: int = 0,
        normalize: bool = True,
        message_mode: str = 'node_edge',
        have_query_feature: bool = False,
        **kwargs
       
    ):
        super(HypergraphTransformer, self).__init__(aggr=aggr.AttentionalAggregation(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, 1),
                torch.nn.Sigmoid()
            ), 
            torch.nn.Sequential(
                torch.nn.Linear(in_channels,out_channels),
                # torch.nn.Dropout(0.5),
                torch.nn.Sigmoid())
        ), node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)

        self.attention_aggr = aggr.AttentionalAggregation(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, 1),
                torch.nn.Sigmoid()
            ), 
            torch.nn.Sequential(
                torch.nn.Linear(in_channels,out_channels),
                # torch.nn.Dropout(0.5),
                torch.nn.Sigmoid())
        )

        self.reset_parameters()

    def reset_parameters(self):
       pass

    # the edge_type are stored as edge_index value
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_time_embed: Tensor,
        edge_dist_embed: Tensor,
        edge_type_embed: Tensor,
        edge_attr_embed: Tensor,
    ):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if isinstance(edge_index, SparseTensor):
            out = self.propagate(
                edge_index,
                x=(x[0][edge_index.storage.col()], x[1][edge_index.storage.row()]),
                size=None
            )
        else:
            out = self.propagate(
                edge_index,
                x=(x[0][edge_index[0]], x[1][edge_index[1]]),
                size=None
            )
        return out

    def message(self, x: OptPairTensor):      # 这里理论上是需要计算 中心节点和对应的关系的运算
        x_j, x_i = x
        return x_j


    # def aggregate(self, inputs, index):
    #     return self.softmax_aggr(inputs, index,dim=0)