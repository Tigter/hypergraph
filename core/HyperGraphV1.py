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
from core.HyperKGE import HyperKGE
from loss import *


from core.HypergraphTransformer import HypergraphTransformer

class HyperGraphV2(Module):
    def __init__(self, hyperkgeConfig=None,n_node=0,n_hyper_edge=0):
        super(HyperGraphV2, self).__init__()

        self.hyperkgeConfig = hyperkgeConfig
        self.encoder = HyperKGE(hyperkgeConfig,n_node,n_hyper_edge)

        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, 1),
            torch.nn.Sigmoid()
        )
        # self.loss_funcation = nn.CrossEntropyLoss()
        # self.loss_funcation = nn.BCELoss()
        self.loss_funcation = MRL(hyperkgeConfig.gamma)


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


    @staticmethod
    def train_step(model,optimizer,data,rel_emb,loss_funcation, margin=0.2, rel_samper=None):
        optimizer.zero_grad()
        model.train()
       
        pos_data, neg_data,rel_out = data

        n_id, x, adjs, lable,split_idx = pos_data
        hyper_edge_emb = model.encoder(n_id,x, adjs, lable,split_idx, True)
        batch_size = len(hyper_edge_emb)

        neg_n_id, neg_x, neg_adjs, neg_lable,split_idx = neg_data
        neg_hyper_edge_emb = model.encoder(neg_n_id, neg_x, neg_adjs, neg_lable,split_idx, True,mode="neg")
        neg_hyper_edge_emb = neg_hyper_edge_emb.reshape(batch_size,-1,model.hyperkgeConfig.embedding_dim)

        r_n_id, r_x, r_adjs,split_idx = rel_out
        relation_emb = model.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)
        rel_emb  = relation_emb.unsqueeze(1)

        hyper_edge_emb = hyper_edge_emb.unsqueeze(1)
        p_score =  torch.norm(hyper_edge_emb * rel_emb,p=2,dim=-1)
        n_score =  torch.norm(neg_hyper_edge_emb * rel_emb, p=2,dim=-1)

        loss = model.loss_funcation(p_score, n_score)

        loss.backward()
        optimizer.step()
        logs = {    
            "loss": loss.item()
        }
        return logs
    

        
            




