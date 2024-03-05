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
from core.HyperCEV4 import HyperCE
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

        # self.ce_predictor = torch.nn.Sequential(
        #     torch.nn.Linear(hyperkgeConfig.embedding_dim, e_num),
        #     torch.nn.Sigmoid()
        # )
        self.c_num = graph_info["c_num"]
        self.e_num = graph_info["e_num"]


        self.cl_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, hyperkgeConfig.embedding_dim),
            torch.nn.Sigmoid()
        )
        self.cl_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim, hyperkgeConfig.embedding_dim),
            torch.nn.Sigmoid()
        )
        self.dropout = torch.nn.Dropout(p=0.5)

        self.MF_Embedding_Compound = torch.nn.Embedding(graph_info["c_num"], hyperkgeConfig.embedding_dim)
        self.MF_Embedding_Enzyme = torch.nn.Embedding(graph_info["e_num"], hyperkgeConfig.embedding_dim)

        self.MLP_Embedding_Compound = torch.nn.Embedding(graph_info["c_num"], hyperkgeConfig.embedding_dim)
        self.MLP_Embedding_Enzyme = torch.nn.Embedding(graph_info["e_num"], hyperkgeConfig.embedding_dim)

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim * 2, hyperkgeConfig.embedding_dim),
            torch.nn.ReLU()
        )
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hyperkgeConfig.embedding_dim*2, 1),
            torch.nn.Sigmoid()
        )
        hidden_dim = hyperkgeConfig.embedding_dim


        self.loss_funcation = nn.BCELoss()

        self.baseGnn = MulScoreGnn()
        self.ec_predictor = []
        for ec_dim in [13, 86, 312]:
            self.ec_predictor.append(MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=ec_dim, dropout=0.5, sigmoid_last_layer=False))

        self.ko_predictor = MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=6903, dropout=0.5, sigmoid_last_layer=True)

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

    def lable_predict_base(self, data, mode="train"):
        pos_data,baseData = data

        n_id, x, adjs, lable,split_idx = baseData

        hyper_edge_emb = torch.zeros(
            split_idx,
            self.hyperkgeConfig.embedding_dim,
            device=x.device
        )
        x = torch.cat([hyper_edge_emb,x], dim=0)
        adj_t, edge_attr,  edge_type, e_id, size = adjs[0]
        x_target = x[:adj_t.size(0)]
        adj_t = adj_t.cuda()

        score = self.baseGnn(
            x= (x, x_target),
            e_emb= self.encoder.edge_type_embedding_layer.edge_type_embedding,
            edge_index = adj_t
        )
        return score,lable

    def reg_l2(self, x):
        return torch.mean(torch.norm(x,dim=-1))


    def train_base_score(self, base_out):
        n_id, x, adjs, lable, split_idx = base_out
        hyper_edge_emb = torch.zeros(
            split_idx,
            self.hyperkgeConfig.embedding_dim,
            device=x.device
        )
        x = torch.cat([hyper_edge_emb,x], dim=0)
        adj_t, edge_attr,  edge_type, e_id, size = adjs[0]
        x_target = x[:adj_t.size(0)]
        adj_t = adj_t.cuda()

        score = self.baseGnn(
            x= (x, x_target),
            e_emb= self.encoder.edge_type_embedding_layer.edge_type_embedding,
            edge_index = adj_t
        )
        return score

    def lable_train(self, pos_data,base_out, mode="train"):
        
        n_id, adjs, lable,split_idx = pos_data
        x = self.node_emb(n_id[split_idx:])

        hyper_edge_emb = self.encoder(x, adjs,split_idx, True)
        # score  = self.ce_predictor(hyper_edge_emb)
        score = self.train_base_score(base_out)
        if mode == 'train':
            return score, lable, hyper_edge_emb, self.reg_l2(x)
        else:
            return score, lable, hyper_edge_emb

    def cl_train(self, single_data, double_data):
        n_id, adjs, lable,split_idx = single_data
        n_id = n_id.cuda()
        x = self.node_emb(n_id[split_idx:])
        single_emb = self.encoder(x, adjs,split_idx, True)

        n_id,  adjs, split_idx = double_data
        n_id = n_id.cuda()
        x = self.node_emb(n_id[split_idx:])
        double_emb = self.encoder(x, adjs,split_idx, True)

        return single_emb, double_emb

    def double_train(self, data):
        n_id,  adjs, split_idx = data
        n_id = n_id.cuda()
        x = self.node_emb(n_id[split_idx:])

        hyper_edge_emb = self.encoder(x, adjs,split_idx, True)
        return hyper_edge_emb

    def caculate_cl_loss(self, single_emb, double_emb):
        # single_emb = self.cl_mlp1(single_emb)
        # single_emb = self.cl_mlp2(single_emb)
        score = single_emb @ double_emb.transpose(0,1)

        batch_size = len(single_emb)
        pos_score = score[range(batch_size), range(batch_size)]
        neg_score = (score.sum(dim=1) - pos_score)
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos_score))-torch.log(1e-8 + (1 - torch.sigmoid(neg_score))))
        return con_loss

    def base_score(self, compound_ids, enzyme_ids,toNodeData, mode="valid"):

        n_id, adjs, split_idx = toNodeData
        n_id = n_id.cuda()
        
        input_x =  self.node_emb(n_id[split_idx:])
        node_embedding = self.encoder(input_x, adjs,split_idx, True,mode='node')


        mf_embedding_compound = node_embedding
        mf_embedding_enzyme = self.node_emb(enzyme_ids+self.c_num)

        if mode =="train":
            mf_embedding_compound = mf_embedding_compound.unsqueeze(1)
        # node_embedding = node_embedding.unsqueenze(1)
        

        mf_vector = mf_embedding_enzyme * mf_embedding_compound

        if mode == "train":
            mf_embedding_compound = mf_embedding_compound.repeat(1,mf_embedding_enzyme.shape[1],1)
        else:
            mf_embedding_compound = mf_embedding_compound.repeat(mf_embedding_enzyme.shape[0],1)
        # mlp_embedding_compound = self.MLP_Embedding_Compound(compound_ids)
        # mlp_embedding_enzyme = self.MLP_Embedding_Enzyme(enzyme_ids)

        # if mode == "train":
        #     mlp_embedding_compound = mlp_embedding_compound.unsqueeze(1)
        #     mlp_embedding_compound = mlp_embedding_compound.repeat(1,mlp_embedding_enzyme.shape[1],1)
        # else:
        #     mlp_embedding_compound = mlp_embedding_compound.repeat(mlp_embedding_enzyme.shape[0],1)

        mlp_vector = torch.cat([mf_embedding_enzyme, mf_embedding_compound], dim=-1)
        mlp_vector = self.fc1(mlp_vector)
        
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        predict_vector = self.dropout(predict_vector)
        # predict_vector = self.dropout(mf_vector)
        predict_vector = self.ce_predictor(predict_vector)
        return predict_vector,self.reg_l2(input_x)

    # def base_score(self, compound_ids, enzyme_ids,toNodeData, mode="valid"):

    #     n_id, adjs, split_idx = toNodeData
    #     n_id = n_id.cuda()
        
    #     input_x =  self.node_emb(n_id[split_idx:])
    #     node_embedding = self.encoder(input_x, adjs,split_idx, True,mode='node')


    #     mf_embedding_compound = node_embedding
    #     mf_embedding_enzyme = self.node_emb(enzyme_ids+self.c_num)

    #     if mode =="train":
    #         mf_embedding_compound = mf_embedding_compound.unsqueeze(1)
    #         # node_embedding = node_embedding.unsqueenze(1)
    #         mf_embedding_compound = mf_embedding_compound.repeat(1,mf_embedding_enzyme.shape[1],1)

    #     mf_vector = mf_embedding_enzyme * mf_embedding_compound

    #     # mlp_embedding_compound = self.MLP_Embedding_Compound(compound_ids)
    #     # mlp_embedding_enzyme = self.MLP_Embedding_Enzyme(enzyme_ids)

    #     # if mode == "train":
    #     #     mlp_embedding_compound = mlp_embedding_compound.unsqueeze(1)
    #     #     mlp_embedding_compound = mlp_embedding_compound.repeat(1,mlp_embedding_enzyme.shape[1],1)
    #     # else:
    #     #     mlp_embedding_compound = mlp_embedding_compound.repeat(mlp_embedding_enzyme.shape[0],1)

    #     mlp_vector = torch.cat([mlp_embedding_enzyme, mlp_embedding_compound], dim=-1)
    #     mlp_vector = self.fc1(mlp_vector)
        
    #     predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
    #     predict_vector = self.dropout(predict_vector)
    #     # predict_vector = self.dropout(mf_vector)
    #     predict_vector = self.ce_predictor(predict_vector)
    #     return predict_vector,self.reg_l2(input_x)
    
    @staticmethod
    def train_step(model,optimizer,data,loss_funcation, margin=0.2, rel_samper=None, config=None,sampler=None,help_data=None):
        optimizer.zero_grad()
        model.train()

        sampler_h, sampler_t,label, toNodeData = data
        sampler_h = sampler_h.cuda()
        sampler_t = sampler_t.cuda()
        label = label.cuda()

        score,regvalie = model.base_score(sampler_h, sampler_t,toNodeData,mode="train")
        score = score.squeeze()
        loss = model.loss_funcation(score, label)
        loss = loss 
        
        add_cl = False
        if add_cl :
            single_data, double_data = next(sampler)
            single_emb, double_emb = model.cl_train(single_data,double_data)
            cl_loss = model.caculate_cl_loss(single_emb, double_emb)
            cl_loss_weighted = config["cl_weight"] * cl_loss 
            loss = loss  +  cl_loss_weighted

        reg_loss_weighted = config["reg_weight"] * regvalie
        loss += reg_loss_weighted
        loss.backward()
        optimizer.step()
        logs = {    
            "loss": loss.item(),
            # "cl_loss": cl_loss_weighted.item(),
            "reg_loss": reg_loss_weighted.item(),
        }
        return logs





