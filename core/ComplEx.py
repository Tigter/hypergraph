import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean
from torch_geometric.nn import aggr
import numpy as np  
class ComplEx(nn.Module):
    def __init__(self, n_entity, n_relation, dim, gamma=12,p_norm=1):
        super(ComplEx,self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.epsilon = 2
        self.entity_dim = dim
        self.relation_dim = dim*2
        self.entity_embedding = nn.Embedding(n_entity, self.entity_dim)
        self.relation_embedding = nn.Embedding(n_relation,self.relation_dim)

        # self.W  = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.entity_dim, self.relation_dim,self.entity_dim)),dtype=torch.float))
        # self.input_dropout = torch.nn.Dropout(0.5)
        # self.hidden_dropout1 = torch.nn.Dropout(0.5)
        # self.hidden_dropout2 = torch.nn.Dropout(0.5)

        # self.bn0 = torch.nn.BatchNorm1d(self.entity_dim)
        # self.bn1 = torch.nn.BatchNorm1d(self.entity_dim)

        self.aggr = aggr.AttentionalAggregation(
            torch.nn.Sequential(
                torch.nn.Linear(self.entity_dim, 1),
                torch.nn.Sigmoid()
            ), 
            torch.nn.Sequential(
                torch.nn.Linear(self.entity_dim,self.entity_dim),
                torch.nn.Sigmoid())
        )
        if gamma != None:
            self.embedding_range = (gamma + 2)/dim
            nn.init.uniform_(
                tensor=self.entity_embedding.weight,
                a=-self.embedding_range,
                b=self.embedding_range
                )
            nn.init.uniform_(
                tensor=self.relation_embedding.weight,
                a=-self.embedding_range,
                b=self.embedding_range
                )
        else:
            nn.init.xavier_uniform_(self.entity_embedding.weight)
            nn.init.xavier_uniform_(self.relation_embedding.weight)
       
    def init_model(self):
        pass
    
   
    def forward(self, h,r,t, mode='hrt'):
        head = None
        tail = None
        if len(r.shape) == 1:
            relation = self.relation_embedding(r).unsqueeze(1)
        else:
            relation = self.relation_embedding(r)
        
        if len(h.shape) == 1:
            head = self.entity_embedding(h).unsqueeze(1)
        else:
            head = self.entity_embedding(h)

        if len(t.shape) == 1:
            tail = self.entity_embedding(t).unsqueeze(1)
        else:
            tail = self.entity_embedding(t)

        return self.score_function(head, relation,tail)

    def full_score(self, head, head_index,tail, tail_index, relation, mode="hrt"):
        head_emb = self.entity_embedding(head)
        head_index = head_index.squeeze(1)
        batch_size = relation.shape[0]

        # head_emb = scatter_mean(head_emb,head_index, dim=0,)
        head_emb = self.aggr(head_emb,head_index, dim=0,)

        if mode == "h_rt":
            head_emb = head_emb.reshape(batch_size,-1,self.entity_dim)


        tail_emb = self.entity_embedding(tail)
        tail_index = tail_index.squeeze(1)
        # tail_emb = scatter_mean(tail_emb,tail_index, dim=0)
        tail_emb = self.aggr(tail_emb,tail_index, dim=0)
        if mode == "hr_t":
            tail_emb = tail_emb.reshape(batch_size,-1,self.entity_dim)

        relation = self.relation_embedding(relation)

        if len(relation.shape) == 2:
            relation = relation.unsqueeze(1)
        
        if len(head_emb.shape) == 2:
            head_emb = head_emb.unsqueeze(1)

        if len(tail_emb.shape) == 2:
            tail_emb = tail_emb.unsqueeze(1)
        
        return self.paire_score(head_emb, relation, tail_emb)


    def paire_score(self, head, relation, tail):
        re_head, re_tail= torch.chunk(relation, 2, dim=-1)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
       
        score = head * re_head - tail * re_tail
        score = - torch.norm(score, p=1, dim=2)
        return score
    def complex_score(self, head, relation, tail):
        head_re, head_im = head.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)
        relation_re, relation_im = relation.chunk(2, -1)   # (batch,1,dim)
        tail_re, tail_im = tail.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)

        score_re = head_re * relation_re - head_im * relation_im
        score_im = head_re * relation_im + head_im * relation_re 
        result = score_re * tail_re + score_im * tail_im
        score = torch.sum(result,dim=-1)
        return score

    def dist_mult_score(self, head, relation, tail):
        socre = head *  tail * relation
        score = torch.sum(socre,dim=-1)
        return score

    def rotate_function(self, head, relation, tail):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # CreateMake phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range/pi)
       
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if  tail.shape[1] == relation.shape[1]:
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score =  - score.sum(dim = 2)
        return score

    def tucker_score(self, head, relation, tail):
        batch_size = head.shape[0]
        
        head = head.reshape(-1, self.entity_dim)
        x = self.bn0(head)
        x = self.input_dropout(x)
        x = x.reshape(batch_size, -1, self.entity_dim)

        # 核心张量与关系做x2 乘积
        tail = tail.reshape(-1, self.entity_dim)
        W_mat = torch.mm(tail, self.W.reshape(self.entity_dim, -1))
        W_mat = W_mat.reshape(-1, self.entity_dim, self.entity_dim)
        W_mat = self.hidden_dropout1(W_mat)                     # shape = (batch_size, e_dim, e_dim)
        x = torch.bmm(x, W_mat)                                 # shape = (batch_size, n, e_dim)

        x = x.reshape(-1, self.entity_dim)      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)                             # shape = (batch_size*n, e_dim)

        # 然后根据tail的形状进行计算: (batch_size, n, e_dim) or （1 , n_entity, e_dim)
        if relation.shape[0] == batch_size:
            x = x.reshape(batch_size, -1, self.entity_dim) # shape = (batch_size, n, e_dim)
            x = torch.bmm(x,relation.permute(0,2,1)) # result(batch_size, n, 1)
        else:
            tail = tail.reshape(-1,self.entity_dim)
            x = torch.mm(x, relation.permute(1,0))
        if len(x.shape) > 2:
            x = torch.squeeze(x)
        return x