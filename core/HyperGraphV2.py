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

        # self.ce_predictor = torch.nn.Sequential(
        #     torch.nn.Linear(hyperkgeConfig.embedding_dim, e_num),
        #     torch.nn.Sigmoid()
        # )
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, 1),
            torch.nn.Sigmoid()
        )
        self.ce2_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim*2, 1),
            torch.nn.Sigmoid()
        )
        self.ce_half = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim*2, hyperkgeConfig.embedding_dim),
            torch.nn.Sigmoid()
        )

        self.rel_merge = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim*2, hyperkgeConfig.embedding_dim),
            torch.nn.Sigmoid()
        )
        # self.loss_funcation = nn.CrossEntropyLoss()
        self.loss_funcation = nn.BCELoss()
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

    def rel_cat(self, relation_base, relation_attr):
        rel_emb = torch.cat([relation_base,relation_attr],dim=-1)
        rel_emb = self.rel_merge(rel_emb)
        return rel_emb

    def cons_loss(self, rel_base, rel_attr):
        sim_matrix = rel_base @ rel_attr.transpose(0,1)
        batch_size = len(sim_matrix)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    # def get_rel_emb(self, rel_data,mode):
    #     rel_base, rel_attr  = rel_data
    #     r_n_id, r_x, r_adjs,split_idx = rel_base
    #     relation_base = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)

    #     r_n_id, r_x, r_adjs,split_idx = rel_attr
    #     relation_attr = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True, mode="rel_attr")
    #     if mode == "train":
    #         return self.rel_cat(relation_base, relation_attr), self.cons_loss(rel_base, rel_attr)
    #     else:
    #         return self.rel_cat(relation_base, relation_attr)

    def get_rel_emb(self, rel_data,mode):
        rel_base, rel_attr  = rel_data
        r_n_id, r_x, r_adjs,split_idx = rel_base
        relation_base = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)

        r_n_id, r_x, r_adjs,split_idx = rel_attr
        relation_attr = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True, mode="rel_attr")
        if mode == "train":
            return  relation_attr + relation_base, self.cons_loss(relation_base, relation_attr)
        else:
            return  relation_attr + relation_base, None

    # def get_rel_emb(self, rel_data):
    #     rel_base, rel_attr  = rel_data
    #     r_n_id, r_x, r_adjs,split_idx = rel_base
    #     relation_base = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)

    #     r_n_id, r_x, r_adjs,split_idx = rel_attr
    #     relation_attr = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True, mode="rel_attr")

    #     return relation_attr + relation_base
    
    def mul_score(self, edge, rel):
        return torch.norm(edge*rel, dim=-1)
    
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

    def score(self, data, mode="train"):
        pos_data, rel_pos,rel_neg = data

        n_id, x, adjs, lable,split_idx = pos_data

        hyper_edge_emb = self.encoder(n_id,x, adjs, lable,split_idx, True)
        hyper_edge_emb = hyper_edge_emb.unsqueeze(1)
        batch_size = len(hyper_edge_emb)
        score = self.ce_predictor(hyper_edge_emb)

        pos_rel,con_loss1 = self.get_rel_emb(rel_pos,mode)
        pos_rel = pos_rel.unsqueeze(1)

        neg_rel,con_loss2 = self.get_rel_emb(rel_neg,mode)
        neg_rel = neg_rel.reshape(batch_size,-1, self.hyperkgeConfig.embedding_dim)


        pos_score = self.mul_pre_score(hyper_edge_emb, pos_rel)
        neg_score = self.mul_pre_score(hyper_edge_emb, neg_rel)

        score = torch.cat([pos_score,neg_score ], dim=1)

        binery_label = torch.zeros_like(score)
        binery_label[:,0] = 1

        # relation_emb = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)
        # rel_emb  = relation_emb.unsqueeze(1)
        # r_n_id, r_x, r_adjs,split_idx = rel_neg
        # relation_emb_neg = self.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)
        # relation_emb_neg  = relation_emb_neg.reshape(batch_size,-1, self.hyperkgeConfig.embedding_dim)
        if mode == "train":
            return score,lable,binery_label,con_loss1+con_loss2 # ,rel_emb, relation_emb_neg
        else:
            return score,lable,binery_label # ,rel_emb, relation_emb_neg
    @staticmethod
    def train_step(model,optimizer,data,loss_funcation, margin=0.2, rel_samper=None, config=None):
        optimizer.zero_grad()
        model.train()

        # hyper_edge_emb, rel_emb, relation_emb_neg = model.score(data)
        # p_score =  torch.norm(hyper_edge_emb * rel_emb,p=2,dim=-1)
        # n_score =  torch.norm(hyper_edge_emb * relation_emb_neg, p=2,dim=-1)

        # p_score = torch.cosine_similarity(hyper_edge_emb, rel_emb)
        # n_score =  torch.cosine_similarity(hyper_edge_emb, relation_emb_neg)
        score, label,binery_label ,conLoss= model.score(data)
        binery_label = binery_label.cuda()
        loss = model.loss_funcation(score, binery_label) + conLoss*config['cl_weight']

        loss.backward()
        optimizer.step()
        logs = {    
            "loss": loss.item()
        }
        return logs
    




