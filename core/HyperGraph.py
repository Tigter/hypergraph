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



class HyperConv(Module):
    def __init__(self, layers, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers

    def forward(self, adjacency, embedding):
        node_embeddings = embedding
        node_embedding_layer0 = node_embeddings
        final = [node_embedding_layer0]
        for i in range(self.layers):
            node_embeddings = torch.sparse.mm((adjacency), node_embeddings)
            final.append(node_embeddings)
        result = torch.stack(final,dim=0)
        node_embeddings = torch.sum(result, dim=0) / (self.layers + 1)
        return node_embeddings

class TransE(Module):
    def __init__(self, emb_size=100):
        super(TransE, self).__init__()
        self.emb_size = emb_size

    def forward(self, batch_pos_emb, batch_neg_emb):
        pos_score = batch_pos_emb[0] + batch_pos_emb[1] - batch_pos_emb[2]
        neg_score = batch_neg_emb[0] + batch_neg_emb[1] - batch_neg_emb[2]
        return pos_score, neg_score


class OurModel(Module):
    def __init__(self, v2e, shared_e, shared_r, n_node, layers, emb_size=100, batch_size=100, hyperkgeConfig=None):
        super(OurModel, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.v2e = v2e
        self.shared_e = shared_e
        self.shared_r = shared_r

        self.v2e[1] = self.v2e[1] - self.n_node
        self.shared_e = self.shared_e - self.n_node

        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.layers = layers

        decoder = 'complex'
        # triple_emb_method = 'contact' # mlp complex mean
        self.encoder_name = 'new_graph' # new_graph old
        self.regu_type = 'l3'  # l3, n3
        if decoder == 'MLP':
            self.decoder = nn.Linear(self.emb_size,1)
        else:
            self.decoder = self.complex_decoder_score
        
        if self.encoder_name == 'old':
            self.encoder = HyperConv(self.layers)   
        elif self.encoder_name == 'new_graph':
            self.encoder  = HyperKGE(hyperkgeConfig, n_node)
            self.encoder.reset_parameters()

    def complex_decoder_score(self,head, relation, tail, triple=False):
        head_re, head_im = head.chunk(2, -1)              
        relation_re, relation_im = relation.chunk(2, -1)  
        tail_re, tail_im = tail.chunk(2, -1)               
        score_re = head_re * relation_re - head_im * relation_im
        score_im = head_re * relation_im + head_im * relation_re 

        result = score_re * tail_re + score_im * tail_im
        if triple:
            return result
        score = torch.sum(result,dim=-1)
        return score

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def l3_reg(self):
        return (torch.norm(self.embedding.weight, p=3) ** 3) 

    
    def network_reg(self):
        # mlp_regu = torch.norm(self.triple_mlp.weight) + torch.norm(self.triple_mlp.bias)
        decoder_regu = torch.norm(self.decoder.weight)

        return decoder_regu
        
    
    def n3_reg(self,head,relation,tail):
        head_re, head_im = head.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)
        relation_re, relation_im = relation.chunk(2, -1)   # (batch,1,dim)
        tail_re, tail_im = tail.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)
        
        factor1 = torch.sqrt(head_re**2 + head_im**2)
        factor2 = torch.sqrt(relation_re**2 + relation_im**2)
        factor3 =  torch.sqrt(tail_re**2 + tail_im**2)
        norm = torch.sum(factor1**3) + torch.sum(factor2**3) + torch.sum(factor3**3)
        data = [norm,factor1,head]
        return norm/(factor1.shape[0]), data
    
    # triple-embedding的方法
    def mean_triple(self,con_emb, con_shape):
        con_emb = con_emb.reshape(con_shape[0],con_shape[1],-1,self.emb_size)
        con_emb = torch.mean(con_emb,dim=-2)
        return con_emb

    def mlp_triple(self,con_emb, con_shape):
        con_emb = con_emb.reshape(con_shape[0],con_shape[1],-1)
        con_emb = self.triple_mlp(con_emb)
        return con_emb

    def complex_triple(self, conf_emb,con_shape):
        conf_emb = conf_emb.reshape(con_shape[0],con_shape[1],-1,self.emb_size)
        conf_head, conf_rel,conf_tail = torch.chunk(conf_emb,3,dim=-2)
        conf_head = conf_head.squeeze(dim=-2)
        conf_rel = conf_rel.squeeze(dim=-2)
        conf_tail = conf_tail.squeeze(dim=-2)
        return self.complex_decoder_score(conf_head,conf_rel,conf_tail,True)

    def contact_triple(self, conf_emb,con_shape):
        conf_emb = conf_emb.reshape(con_shape[0],con_shape[1],-1)
        return conf_emb
    # 
    def train_base(self,hyper_node_embeddings,base,loss_function,ground_truth):
        
        head,relation,tail = torch.chunk(base,3,dim=-1)
        base_batch_size = head.shape[0]
       
        head = head.reshape(-1)
        relation = relation.reshape(-1)
        tail = tail.reshape(-1)

        head_emb = torch.index_select(hyper_node_embeddings, dim=0, index=head)  
        # rel_emb = torch.index_select(raw_embedding,dim=0,index=relation)   
        rel_emb = self.embedding(relation)
        tail_emb = torch.index_select(hyper_node_embeddings,dim=0,index=tail)   

        head_emb = head_emb.reshape(base_batch_size, -1, self.emb_size)
        rel_emb = rel_emb.reshape(base_batch_size, -1, self.emb_size)
        tail_emb = tail_emb.reshape(base_batch_size, -1, self.emb_size)

        score = self.complex_decoder_score(head_emb, rel_emb,tail_emb)
        score = torch.sigmoid(score)

        if ground_truth != None:
            loss1 = loss_function(score, ground_truth)
            # loss1 = torch.mean(torch.sum(F.softplus(-score * ground_truth),dim=-1)) #
        else:  # 这里使用
            positive_score = score[...,0].unsqueeze(1)
            negative_score = score[...,1:]
            loss1 = loss_function(positive_score,negative_score)
        return loss1,head_emb, rel_emb, tail_emb
    
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
    
    def train_base_with_edge(self,edg_embedding,base,edge_index,loss_function,ground_truth,base_true_id):
        base_batch_size = base.shape[0]
        edge_index = edge_index.reshape(-1)

        po_emb = torch.index_select(edg_embedding, dim=0, index=edge_index)  
        po_score = self.decoder(po_emb.reshape(base_batch_size,-1,self.emb_size))
        
        ne_emb = self.embedding(base)
        ne_emb = self.triple_mlp(ne_emb.reshape(base_batch_size, -1, self.emb_size*3))
        ne_score = self.decoder(ne_emb)

        po_mlp = self.embedding(base_true_id)
        po_mlp = self.triple_mlp(po_mlp.reshape(base_batch_size, -1, self.emb_size*3))
        p_mlp_score = self.decoder(po_mlp)
        p_mlp_score = torch.sigmoid(p_mlp_score).reshape(base_batch_size)
        lable = torch.ones(size= p_mlp_score.shape)

        score = torch.cat([po_score,ne_score],dim=1).squeeze(dim=-1)
        score = torch.sigmoid(score)
        
        if ground_truth != None:
            # loss1 = torch.mean(torch.sum(F.softplus(-score * ground_truth),dim=-1)) #
            loss1 = loss_function(score,ground_truth) + loss_function(p_mlp_score,lable)
        else:  # 这里使用
            positive_score = score[...,0].unsqueeze(1)
            negative_score = score[...,1:]
            loss1 = loss_function(positive_score,negative_score)
        return loss1
    
    def forward(self, base,loss_function, regu_weight,ground_truth=None,
                base_edge_index=None,base_true_id=None):
        if self.encoder_name == 'old':
            hyper_node_embeddings = self.encoder(self.adjacency, self.embedding.weight)
        elif self.encoder_name =='new_graph':
            hyper_edge_embedding, hyper_node_embeddings = self.encoder(self.v2e, self.shared_e, self.shared_r, self.embedding)
        else:
            hyper_node_embeddings = self.embedding.weight
        
        # hyper_edge_embedding= None
        # hyper_edge_embedding,hyper_node_embeddings=self.encoder(self.adjacency, self.embedding)
        if  torch.isnan(hyper_node_embeddings).any(): # torch.isnan(hyper_node_embeddings).any() or
            raise(ValueError("Node embedding hava nan"))

        if  base_edge_index != None:
            loss1 = self.train_base_pre_relation(hyper_node_embeddings,base,base_edge_index,loss_function,ground_truth)
        else:
            loss1,head_emb, rel_emb, tail_emb = self.train_base(hyper_node_embeddings,base,loss_function,ground_truth)
      
        # if self.regu_type == 'n3':
        #     reug,data = self.n3_reg(head_emb,rel_emb,tail_emb)
        #     reug = reug*regu_weight
        # elif self.regu_type == 'l3':
        #     reug = regu_weight * (self.l3_reg())
        loss  = loss1 # + reug 
        logs = {
            'loss_1':loss1.item(),
            # 'loss_r':reug,
        }
        return loss,logs

        
    def buil_embedding_for_test(self, v2e, shared_e, shared_r):
        if self.encoder_name == 'old':
            print("Build Embedding of Old Graph For Test ")
            self.hyper_node_embeddings = self.encoder(self.adjacency, self.embedding.weight)
        elif self.encoder_name == 'new_graph':
            print("Build Embedding of New Graph For Test ")
            self.hyper_edge_embedding,self.hyper_node_embeddings = self.encoder(v2e, shared_e, shared_r, self.embedding)
        else:
            print("Use the raw embedding for Test")
            self.hyper_node_embeddings = self.embedding.weight
            
    def clean_hyper_node_embd(self):
        if self.encoder_name == 'old':
            del self.hyper_node_embeddings 
        elif self.encoder_name == 'new_graph':
            del self.hyper_edge_embedding
            del self.hyper_node_embeddings 

    # def predict(self,head,relation, tail):
        
    #     base_batch_size = head.shape[0]

    #     head = head.reshape(-1)
    #     relation = relation.reshape(-1)
    #     tail = tail.reshape(-1)

    #     head_emb = torch.index_select(self.hyper_node_embeddings, dim=0, index=head)  
    #     rel_emb = torch.index_select(self.embedding.weight,dim=0,index=relation)   
    #     tail_emb = torch.index_select(self.hyper_node_embeddings,dim=0,index=tail)  

    #     head_emb = head_emb.reshape(base_batch_size, -1, self.emb_size)
    #     rel_emb = rel_emb.reshape(base_batch_size, -1, self.emb_size)
    #     tail_emb = tail_emb.reshape(base_batch_size, -1, self.emb_size)
    #     score = self.decoder(head_emb, rel_emb,tail_emb)

    #     return score

    def predict(self,ht,relation):
        ht_embedding = torch.index_select(self.hyper_node_embeddings, dim=0, index=ht) # batch_size * dim
        ht_embedding = ht_embedding.unsqueeze(2) # batch_size * 1 * dim
        rel_emb = self.embedding(relation)
        base_batch_size = relation.shape[0]
        rel_emb = rel_emb.reshape(base_batch_size, -1, self.emb_size) # batch_size * n *dim
        return torch.cosine_similarity(ht, relation)


    @staticmethod
    def train_step(model,optimizer, base_data, base_loss_function,cuda=True,regu_weight=0,step=0,dataset=None):
        optimizer.zero_grad()
        model.train()
        edge_index = False

        # 基础模型的数据集
        h,r,t, ground_truth,base_edge_index,true_id  = next(base_data)
        # base = torch.stack([h,r,t],dim=-1)
        # base = base.cuda()
        base = r.cuda()
        base_edge_index = base_edge_index.cuda()
        ground_truth = ground_truth.cuda()
        
        loss,logs = model(base,base_loss_function,regu_weight,ground_truth,base_edge_index=base_edge_index,base_true_id=true_id)

        loss.backward()

        optimizer.step()
      
        return logs
    

        
            




