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

def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class MulScoreGnn(MessagePassing):
    r"""Hypergraph Conv containing relation transform、edge fusion(including time fusion)、
    self attention and gated residual connection(or skip connection).

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed via
    """
    def __init__(
        self,
        **kwargs
    ):
        super(MulScoreGnn, self).__init__(aggr='add', node_dim=0, **kwargs)

    # the edge_type are stored as edge_index value
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        e_emb: Tensor
    ):
        out = self.propagate(
                edge_index,
                x=(x[0][edge_index.storage.col()], x[1][edge_index.storage.row()]),
                e_emb = e_emb
            )
        return out

    def message(
        self,
        x: OptPairTensor,
        e_emb: Tensor, 
        index: Tensor,
    ) -> Tensor:
        x_j, x_i = x
        # e_emb = e_num * dim
        x_j = x_j @ e_emb.transpose(0,1)  # x_j * e_num 
        return x_j
    

    def aggregate(self, inputs: Tensor, index: Tensor,
              ptr: Optional[Tensor] = None,
              dim_size: Optional[int] = None, final_node_num=None) -> Tensor:

        if final_node_num == None:
            final_node_num = torch.max(index)
        
        result_tensor = torch.zeros(final_node_num+1, inputs.shape[-1],device = inputs.device)
        for i in range(final_node_num+1):
            emb = inputs[index==i]
            assert len(emb) != 0
            result_tensor[i] = torch.sum(torch.softmax(emb,dim=0) * emb,dim=0)  # number * e_num
        
        return result_tensor
