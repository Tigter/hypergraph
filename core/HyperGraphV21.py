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
from core.HyperCEV2 import HyperCE
from loss import *

from core.HypergraphTransformer import HypergraphTransformer

class HyperGraphV3(Module):
    def __init__(self, hyperkgeConfig=None,n_node=0,n_hyper_edge=0,e_num=100,graph_info=None):
        super(HyperGraphV3, self).__init__()

        self.hyperkgeConfig = hyperkgeConfig
        self.encoder = HyperCE(hyperkgeConfig,n_node,n_hyper_edge,e_num,graph_info)

        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, e_num),
            torch.nn.Sigmoid()
        )
        self.cl_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, hyperkgeConfig.embedding_dim),
            torch.nn.Sigmoid()
        )
        self.cl_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, hyperkgeConfig.embedding_dim),
            torch.nn.Sigmoid()
        )
       
        self.loss_funcation = nn.CrossEntropyLoss()


    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

  
    
    def train_base_pre_relation(self,hyper_node_embeddings,base,base_edge_index,loss_function,ground_truth):
    
        base_edge_index = torch.squeeze(base_edge_index,dim=-1) - self.n_node
        ht_embedding = torch.index_select(hyper_node_embeddings, dim=0, index=base_edge_index) # batch_size * dim
        
        
        rel_emb = self.embedding(base)
        base_batch_size = base.shape[0]
        rel_emb = rel_emb.reshape(base_batch_size, -1, self.emb_size) # batch_size * n *dim

        # score = torch.cosine_similarity(rel_emb,ht_embedding)
        rel_emb = F.normalize(rel_emb, p=2, dim=-1)
        ht_embedding = F.normalize(ht_embedding, p=2, dim=-1)
        ht_embedding = ht_embedding.unsqueeze(2) # batch_size * 1 * dim
        score = rel_emb @ ht_embedding
        score = score.squeeze(2)
        loss =  torch.mean(F.softplus( - score * ground_truth))
        return loss
    
    def forward(self,n_id, x , adjs, lable, cuda = True):
        return  self.model(n_id, x, adjs, lable, cuda)

    def predict(self,ht,relation):
        ht_embedding = torch.index_select(self.hyper_node_embeddings, dim=0, index=ht) # batch_size * dim
        ht_embedding = ht_embedding.unsqueeze(2) # batch_size * 1 * dim
        rel_emb = self.embedding(relation)
        base_batch_size = relation.shape[0]
        rel_emb = rel_emb.reshape(base_batch_size, -1, self.emb_size) # batch_size * n *dim
        return torch.cosine_similarity(ht, relation)

    def rel_cat(self, relation_base, relation_attr):
        rel_emb = torch.cat([relation_base,relation_attr],dim=-1)
        rel_emb = self.rel_merge(rel_emb)
        return rel_emb

    def cons_loss(self, rel_base, rel_attr):
        sim_matrix = rel_base @ rel_attr.transpose(0,1)
        batch_size = len(sim_matrix)
        sim_matrix = torch.sigmoid(sim_matrix)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    
    def mul_score(self, edge, rel):
        return torch.sigmoid(torch.norm(edge*rel, dim=-1))
    
    def mul_pre_score(self, edge, rel):
        return self.ce_predictor(edge*rel)

    def cat_pre_score(self, edge, rel):
        if edge.shape[1] != rel.shape[1]:
            edge = edge.repeat(1,rel.shape[1],1)
        combine = torch.cat([edge, rel],dim=-1)
        score = self.ce2_predictor(combine)    
        return score
    
    def combine_score(self, edge, rel):
        sim_emb = edge*rel
        if edge.shape[1] != rel.shape[1]:
            edge = edge.repeat(1,rel.shape[1],1)
        combine = torch.cat([edge, rel],dim=-1)
        cat_emb = self.ce_half(combine)    
        total_emb = torch.cat([sim_emb, cat_emb], dim=-1)
        score = self.ce2_predictor(total_emb)
        return score

    def lable_predict(self, data, mode="train"):
        pos_data = data
        n_id, x, adjs, lable,split_idx = pos_data
        hyper_edge_emb = self.encoder(n_id,x, adjs, lable,split_idx, True)
        score  = self.ce_predictor(hyper_edge_emb)
        
        return score, lable

    def lable_train(self, pos_data, mode="train"):
        n_id, x, adjs, lable,split_idx = pos_data
        hyper_edge_emb = self.encoder(n_id,x, adjs, lable,split_idx, True)
        score  = self.ce_predictor(hyper_edge_emb)
        return score, lable, hyper_edge_emb

    def double_train(self, data, mode="double"):
        n_id, x, adjs, split_idx = data
        hyper_edge_emb = self.encoder(n_id,x, adjs, None,split_idx, True,mode=mode)
        return hyper_edge_emb

    def caculate_cl_loss(self, single_emb, double_emb):
        single_emb = self.cl_mlp1(single_emb)
        single_emb = self.cl_mlp2(single_emb)
        score = single_emb @ double_emb.transpose(0,1)

        batch_size = len(single_emb)
        pos_score = score[range(batch_size), range(batch_size)]
        neg_score = (score.sum(dim=1) - pos_score)
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos_score))-torch.log(1e-8 + (1 - torch.sigmoid(neg_score))))
        return con_loss

    @staticmethod
    def train_step(model,optimizer,data,loss_funcation, margin=0.2, rel_samper=None, config=None):
        optimizer.zero_grad()
        model.train()

        single_data, double_data = data
        score, label, single_emb = model.lable_train(single_data)
        double_emb =  model.double_train(double_data)

        label = label.cuda()    
        loss = model.loss_funcation(score, label)

        cl_loss = model.caculate_cl_loss(single_emb, double_emb)

        loss = loss + config["cl_weight"] * cl_loss
        loss.backward()
        optimizer.step()
        logs = {    
            "loss": loss.item(),
            "cl_loss": cl_loss.item() * config["cl_weight"]
        }
        return logs





