import datetime
import math
import numpy as np
import time
from scipy.sparse import csr_matrix
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from core.HyperCEBrenda import HyperCE
from loss import *

from core.BaseLayer import MulScoreGnn
from core.HypergraphTransformer import HypergraphTransformer

from torch_geometric.data import Batch
from core.SmilesGnn import *

from core.SmileTransformer import *
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, sigmoid_last_layer=False):
        super(MLPModel, self).__init__()

        # construct layers
        layers = [torch.nn.Linear(input_dim, hidden_dim),
                  torch.nn.ReLU(),
                  torch.nn.Dropout(dropout),
                  torch.nn.Linear(hidden_dim, output_dim)]
        if sigmoid_last_layer:
            layers.append(torch.nn.Sigmoid())

        # construct model
        self.predictor = torch.nn.Sequential(*layers)

    def forward(self, X):
        X = self.predictor(X)
        return X

# class Collater:
#     @staticmethod
#     def package( batch):
#         elem = batch[0]
#         if isinstance(elem, BaseData):
#             return Batch.from_data_list(
#                 batch
#             )
#         elif isinstance(elem, torch.Tensor):
#             return default_collate(batch)
#         elif isinstance(elem, TensorFrame):
#             return torch_frame.cat(batch, dim=0)
#         elif isinstance(elem, float):
#             return torch.tensor(batch, dtype=torch.float)
#         elif isinstance(elem, int):
#             return torch.tensor(batch)
#         elif isinstance(elem, str):
#             return batch
#         elif isinstance(elem, Mapping):
#             return {key: Collater.package([data[key] for data in batch]) for key in elem}
#         elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
#             return type(elem)(*(Collater.package(s) for s in zip(*batch)))
#         elif isinstance(elem, Sequence) and not isinstance(elem, str):
#             return [Collater.package(s) for s in zip(*batch)]
#         raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")
    
class HyperGraphV3(Module):
    def __init__(self, hyperkgeConfig=None,n_node=0,n_hyper_edge=0,e_num=100,graph_info=None,config=None,NodeGnnDataset=None,clDataset=None):
        super(HyperGraphV3, self).__init__()

        self.hyperkgeConfig = hyperkgeConfig
        self.encoder = HyperCE(hyperkgeConfig,n_node,n_hyper_edge,e_num,graph_info)
        hidden_dim = hyperkgeConfig.embedding_dim

        self.entity_dim = hyperkgeConfig.embedding_dim
      
        self.c_num = graph_info["c_num"]
        self.e_num = graph_info["e_num"]


        self.node_emb = nn.Embedding(self.c_num+self.e_num, hidden_dim)
        init_range =  6.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.node_emb.weight, -init_range, init_range)

        self.dropout = torch.nn.Dropout(p=0.5)
        self.node_encoder = GNN(
            num_layer=config["node_gnn_layer"],
            emb_dim=hidden_dim,
            gnn_type='gin',
            drop_ratio=config["node_gnn_dropout"],
            JK='last',
        )
       
        self.NodeGnnDataset= NodeGnnDataset
        self.clDataset = clDataset

        self.text_proj = nn.Linear(768,  hyperkgeConfig.embedding_dim)

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim * 2, hyperkgeConfig.embedding_dim),
            torch.nn.ReLU()
        )
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, 1),
            torch.nn.Sigmoid()
        )
        bert_name = 'allenai/scibert_scivocab_uncased'
        encoder_config = BertConfig.from_pretrained(bert_name)
        encoder_config.encoder_width = hidden_dim
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        # encoder_config.query_length = num_query_token
        self.Qformer = BertLMHeadModel.from_pretrained(
            bert_name, config=encoder_config
        )
        self.loss_funcation = nn.BCELoss()

        self.W  = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.entity_dim, self.entity_dim,self.entity_dim)),dtype=torch.float))

        self.input_dropout = torch.nn.Dropout(0.5)
        self.hidden_dropout1 = torch.nn.Dropout(0.5)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

        self.bn0 = torch.nn.BatchNorm1d(self.entity_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.entity_dim)
    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
   

    def reg_l2(self):
        return torch.mean(torch.norm(self.node_emb.weight,dim=-1))



    def get_base_emb(self, nids):
        batch = []
        for i in range(len(nids)):
            batch.append(self.NodeGnnDataset.get(nids[i]))
        batch = Collater.package(batch)
        batch_node = self.node_encoder(batch)
        return batch_node

    def train_cl(self):
        graph_data,text_data,txt_mask = next(self.clDataset)
        batch_node = self.node_encoder(graph_data)
        text_data=text_data.cuda()
        txt_mask=txt_mask.cuda()

        text_output = self.Qformer.bert(text_data, attention_mask=txt_mask, return_dict=True) 
        text_emb = self.text_proj(text_output.last_hidden_state[:, 0, :])

        text_feats, graph_feats = F.normalize(text_emb, p=2, dim=-1), F.normalize(batch_node, p=2, dim=-1)

        return self.caculate_cl_loss(text_feats, graph_feats)

    def caculate_cl_loss(self, single_emb, double_emb):

        score = single_emb @ double_emb.transpose(0,1)
        batch_size = len(single_emb)
        pos_score = score[range(batch_size), range(batch_size)]
        neg_score = (score.sum(dim=1) - pos_score)
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos_score))-torch.log(1e-8 + (1 - torch.sigmoid(neg_score))))
        return con_loss

    def single_emb(self, data):
        n_id, adjs, split_idx = data
        n_id = n_id.cuda()
        # n_id = n_id[split_idx:]
        x = self.node_emb(n_id[split_idx:])
        # x = self.get_base_emb(n_id)
        hyper_edge_emb = self.encoder(n_id,x, adjs,split_idx, True)
        return hyper_edge_emb

    def full_score(self, head_emb, relation, tail_emb):
        relation_emb = self.node_emb(relation+self.e_num)

        if len(relation_emb.shape) == 2:
            relation_emb = relation_emb.unsqueeze(1)

        if len(head_emb.shape) == 2:
            head_emb = head_emb.unsqueeze(1)

        if len(tail_emb.shape) == 2:
            tail_emb = tail_emb.unsqueeze(1)
        
        return self.tucker_score(head_emb, relation_emb, tail_emb)

    def complex_score(self, head, relation, tail):
        head_re, head_im = head.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)
        relation_re, relation_im = relation.chunk(2, -1)   # (batch,1,dim)
        tail_re, tail_im = tail.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)

        score_re = head_re * relation_re - head_im * relation_im
        score_im = head_re * relation_im + head_im * relation_re 
        result = score_re * tail_re + score_im * tail_im
        score = torch.sum(result,dim=-1)
        return score

    def tucker_score(self, head, relation, tail):
        batch_size = relation.shape[0]
        
        head = head.reshape(-1, self.entity_dim)
        x = self.bn0(head)
        x = self.input_dropout(x)
        x = x.reshape(batch_size, -1, self.entity_dim)

        print(x.shape)
        # 核心张量与关系做x2 乘积
        relation = relation.reshape(-1, self.entity_dim)
        print(relation.shape)

        W_mat = torch.mm(relation, self.W.reshape(self.entity_dim, -1))
        W_mat = W_mat.reshape(-1, self.entity_dim, self.entity_dim)
        W_mat = self.hidden_dropout1(W_mat)                     # shape = (batch_size, e_dim, e_dim)
        
        x = torch.bmm(x, W_mat)                                 # shape = (batch_size, n, e_dim)

        x = x.reshape(-1, self.entity_dim)      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)                             # shape = (batch_size*n, e_dim)

        # 然后根据tail的形状进行计算: (batch_size, n, e_dim) or （1 , n_entity, e_dim)
        if tail.shape[0] == batch_size:
            x = x.reshape(batch_size, -1, self.entity_dim) # shape = (batch_size, n, e_dim)
            x = torch.bmm(x,tail.permute(0,2,1)) # result(batch_size, n, 1)
        else:
            tail = tail.reshape(-1,self.entity_dim)
            x = torch.mm(x, tail.permute(1,0))
        if len(x.shape) > 2:
            x = torch.squeeze(x)
        return x
    #  2.1.0
    @staticmethod
    def train_step(model,optimizer,data,loss_funcation, config=None):
        optimizer.zero_grad()
        model.train()

        head,relation,tail,negative_sample,head_out,tail_out = data

        head_emb = model.single_emb(head_out)
        tail_emb = model.single_emb(tail_out)

        relation = relation.cuda()
        negative_sample = negative_sample.cuda()

        pos_score = model.full_score(head_emb, relation, tail_emb)
        neg_score = model.full_score(head_emb, negative_sample, tail_emb)

        loss = loss_funcation(pos_score, neg_score)
        logs = {    
            "loss": loss.item(),
            # "ec_loss": loss_ec.item(),
            # "ko_loss": loss_enzyme_ko.item()
        }

        # cl loss
        # cl_loss = model.train_cl()
        # logs["cl_loss"] = cl_loss.item()
        # loss += cl_loss

        if config["reg_weight"] != 0.0:
            reg = model.reg_l2()
            logs["reg"] = reg * config["reg_weight"]
            loss += reg * config["reg_weight"]
        loss.backward()
        optimizer.step()
        
        return logs





