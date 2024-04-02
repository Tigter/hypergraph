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
    
class HyperGraphV3(Module):
    def __init__(self, hyperkgeConfig=None,n_node=0,n_hyper_edge=0,e_num=100,graph_info=None):
        super(HyperGraphV3, self).__init__()

        self.hyperkgeConfig = hyperkgeConfig
        self.encoder = HyperCE(hyperkgeConfig,n_node,n_hyper_edge,e_num,graph_info)
        hidden_dim = hyperkgeConfig.embedding_dim
      
        self.c_num = graph_info["c_num"]
        self.e_num = graph_info["e_num"]


        self.node_emb = nn.Embedding(self.c_num+self.e_num, hidden_dim)

        init_range =  6.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.node_emb.weight, -init_range, init_range)

        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim * 2, hyperkgeConfig.embedding_dim),
            torch.nn.ReLU()
        )
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, 1),
            torch.nn.Sigmoid()
        )
        

        self.loss_funcation = nn.BCELoss()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

  

   

    def reg_l2(self):
        return torch.mean(torch.norm(self.node_emb.weight,dim=-1))


    def single_emb(self, data):
        n_id, adjs, split_idx = data
        n_id = n_id.cuda()
        x = self.node_emb(n_id[split_idx:])
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
        return self.complex_score(head_emb, relation_emb, tail_emb)

    def complex_score(self, head, relation, tail):
        head_re, head_im = head.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)
        relation_re, relation_im = relation.chunk(2, -1)   # (batch,1,dim)
        tail_re, tail_im = tail.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)

        score_re = head_re * relation_re - head_im * relation_im
        score_im = head_re * relation_im + head_im * relation_re 
        result = score_re * tail_re + score_im * tail_im
        score = torch.sum(result,dim=-1)
        return score
    
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

        if config["reg_weight"] != 0.0:
            reg = model.reg_l2()
            logs["reg"] = reg * config["reg_weight"]
            loss += reg * config["reg_weight"]
        loss.backward()
        optimizer.step()
        
        return logs





