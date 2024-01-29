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
from core.HyperNode import HyperNode
from loss import *


from core.HypergraphTransformer import HypergraphTransformer

class HyperGraphV4(Module):
    def __init__(self, hyperkgeConfig=None,n_node=0,n_hyper_edge=0):
        super(HyperGraphV4, self).__init__()

        self.hyperkgeConfig = hyperkgeConfig
        self.encoder = HyperNode(hyperkgeConfig,n_node,n_hyper_edge)

        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, 1),
            torch.nn.Sigmoid()
        )
        # self.loss_funcation = nn.CrossEntropyLoss()
        self.loss_funcation = nn.BCELoss()
        # self.loss_funcation = NSSAL(hyperkgeConfig.gamma)
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

    def score(self, rel_out, ent_out,batch_size,n_size):
       
        n_id, input_x, adjs, split_idx = rel_out
        
        rel_emb = self.encoder(n_id,input_x, adjs,cuda = True, mode="rel")
        rel_emb = rel_emb.view(batch_size,1,-1)

        n_id, input_x, adjs, split_idx = ent_out
        ent_emb = self.encoder(n_id,input_x, adjs,cuda = True, mode="ent")

        ent_emb = ent_emb.view(batch_size, n_size+2, -1)
        head,tail,neg_emb = torch.split(ent_emb,[1,1,n_size],dim=1)
        return head,tail,neg_emb , rel_emb
    

    @staticmethod
    def train_step(model,optimizer,data, batch_size, n_size):
        optimizer.zero_grad()
        model.train()
        rel_out, ent_out, subsampling_weight, mode = data
 
        head,tail,neg_emb ,rel_emb = model.score(rel_out, ent_out,batch_size,n_size)

        # pos_score = torch.norm(head * rel_emb * tail, p=2,dim=-1)
        pos_emb = head * rel_emb * tail
        p_score = model.ce_predictor(pos_emb)
        # p_score = p_score.squeeze(-1)

        if mode == "hr_t":
            neg_emb = head*rel_emb*neg_emb
            # neg_score = torch.norm(head*rel_emb*neg_emb, p=2, dim=-1)
        else:
            neg_emb = tail*rel_emb*neg_emb
            # neg_score = torch.norm(tail*rel_emb*neg_emb, p=2, dim=-1)
        
        neg_score = model.ce_predictor(neg_emb)
        # neg_score = neg_score.squeeze(-1)
        # neg_score = neg_score.view(-1,1)

        true_label = torch.ones_like(p_score)
        neg_label = torch.zeros_like(neg_score)



        p_loss = model.loss_funcation(p_score, true_label)
        n_loss = model.loss_funcation(neg_score, neg_label)

        # loss = model.loss_funcation(pos_score, neg_score, subsampling_weight)
        loss = p_loss + n_loss

        loss.backward()
        optimizer.step()
        logs = {    
            "loss": loss.item()
        }
        return logs
    

        
            




