import math
from typing import Union, Tuple, Optional
from torch import Tensor, cat
from torch.nn import init, Parameter, Linear, LayerNorm
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch
import torch_geometric

def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class MulScoreGnn(MessagePassing):

    def __init__(
        self,
        **kwargs
    ):
        super(MulScoreGnn, self).__init__(node_dim=0, **kwargs)
        self.mul_aggr = torch_geometric.nn.aggr.MulAggregation()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
    ):
        out = self.propagate(
                edge_index,
                x=(x[0][edge_index.storage.col()], x[1][edge_index.storage.row()]),
            )
        return out

    def aggregate(self, inputs, index):
        return self.mul_aggr(inputs, index.storage.row(),dim=0)


class MeanScoreGnn(MessagePassing):

    def __init__(
        self,
        **kwargs
    ):
        super(MulScoreGnn, self).__init__(node_dim=0, **kwargs)
        self.mul_aggr = torch_geometric.nn.aggr.M()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
    ):
        out = self.propagate(
                edge_index,
                x=(x[0][edge_index.storage.col()], x[1][edge_index.storage.row()]),
            )
        return out

    def aggregate(self, inputs, index):
        return self.mul_aggr(inputs, index.storage.row(),dim=0)

class AttentionScoreGnn(MessagePassing):

    def __init__(
        self,
        dim,
        **kwargs
    ):
        super(AttentionScoreGnn, self).__init__(node_dim=0, **kwargs)
        self.mul_aggr = torch_geometric.nn.aggr.AttentionalAggregation(
            torch.nn.Sequential(
                torch.nn.Linear(dim, 1),
                torch.nn.Sigmoid()
            ), 
            torch.nn.Sequential(
                torch.nn.Linear(dim,dim),
                torch.nn.Sigmoid())
        )

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
    ):
        out = self.propagate(
                edge_index,
                x=(x[0][edge_index.storage.col()], x[1][edge_index.storage.row()]),
            )
        return out

    def aggregate(self, inputs, index):
        return self.mul_aggr(inputs, index.storage.row(),dim=0)
