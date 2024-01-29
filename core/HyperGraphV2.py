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
from core.HyperCE import HyperCE
from loss import *


from core.HypergraphTransformer import HypergraphTransformer

class HyperGraphV3(Module):
    def __init__(self, hyperkgeConfig=None,n_node=0,n_hyper_edge=0,e_num=100):
        super(HyperGraphV3, self).__init__()

        self.hyperkgeConfig = hyperkgeConfig
        self.encoder = HyperCE(hyperkgeConfig,n_node,n_hyper_edge)

        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, e_num),
            torch.nn.Sigmoid()
        )
        self.loss_funcation = nn.CrossEntropyLoss()
        # self.loss_funcation = nn.BCELoss()
        # self.loss_funcation = MRL(hyperkgeConfig.gamma)

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

    def score(self, data):
        pos_data, rel_pos,rel_neg = data

        n_id, x, adjs, lable,split_idx = pos_data

        hyper_edge_emb = self.encoder(n_id,x, adjs, lable,split_idx, True)
        # hyper_edge_emb = hyper_edge_emb.unsqueeze(1)
        batch_size = len(hyper_edge_emb)


        score = self.ce_predictor(hyper_edge_emb)

        # r_n_id, r_x, r_adjs,split_idx = rel_pos
        # relation_emb = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)
        # rel_emb  = relation_emb.unsqueeze(1)
        # r_n_id, r_x, r_adjs,split_idx = rel_neg
        # relation_emb_neg = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)
        # relation_emb_neg  = relation_emb_neg.reshape(batch_size,-1, self.hyperkgeConfig.embedding_dim)
        return score,lable # ,rel_emb, relation_emb_neg

    @staticmethod
    def train_step(model,optimizer,data,loss_funcation, margin=0.2, rel_samper=None):
        optimizer.zero_grad()
        model.train()

        # hyper_edge_emb, rel_emb, relation_emb_neg = model.score(data)
        # p_score =  torch.norm(hyper_edge_emb * rel_emb,p=2,dim=-1)
        # n_score =  torch.norm(hyper_edge_emb * relation_emb_neg, p=2,dim=-1)

        # p_score = torch.cosine_similarity(hyper_edge_emb, rel_emb)
        # n_score =  torch.cosine_similarity(hyper_edge_emb, relation_emb_neg)
        score, label = model.score(data)
        label = label.cuda()
        loss = model.loss_funcation(score, label)

        loss.backward()
        optimizer.step()
        logs = {    
            "loss": loss.item()
        }
        return logs
    

        
            




