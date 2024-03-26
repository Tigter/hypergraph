import torch
import torch.nn as nn

from torch_scatter import scatter_mean

class ComplEx(nn.Module):
    def __init__(self, n_entity, n_relation, dim, gamma=12,p_norm=1):
        super(ComplEx,self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.epsilon = 2
        self.entity_dim = dim
        self.relation_dim = dim
        self.entity_embedding = nn.Embedding(n_entity, self.entity_dim)
        self.relation_embedding = nn.Embedding(n_relation,self.relation_dim)

        # self.aggr = torch_geometric.nn.aggr.AttentionalAggregation(
        #     torch.nn.Sequential(
        #         torch.nn.Linear(dim, 1),
        #         torch.nn.Sigmoid()
        #     ), 
        #     torch.nn.Sequential(
        #         torch.nn.Linear(dim,dim),
        #         torch.nn.Sigmoid())
        # )
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

        head_emb = scatter_mean(head_emb,head_index, dim=0,)
        if mode == "h_rt":
            head_emb = head_emb.reshape(batch_size,-1,self.entity_dim)


        tail_emb = self.entity_embedding(tail)
        tail_index = tail_index.squeeze(1)
        tail_emb = scatter_mean(tail_emb,tail_index, dim=0)
        if mode == "hr_t":
            tail_emb = tail_emb.reshape(batch_size,-1,self.entity_dim)

        relation = self.relation_embedding(relation)

        if len(relation.shape) == 2:
            relation = relation.unsqueeze(1)
        
        if len(head_emb.shape) == 2:
            head_emb = head_emb.unsqueeze(1)

        if len(tail_emb.shape) == 2:
            tail_emb = tail_emb.unsqueeze(1)
        
        return self.complex_score(head_emb, relation, tail_emb)

       


    def complex_score(self, head, relation, tail):
        head_re, head_im = head.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)
        relation_re, relation_im = relation.chunk(2, -1)   # (batch,1,dim)
        tail_re, tail_im = tail.chunk(2, -1)               # (batch,1,dim), (batch,n,dim),  (1,n_e,dim)

        score_re = head_re * relation_re - head_im * relation_im
        score_im = head_re * relation_im + head_im * relation_re 
        result = score_re * tail_re + score_im * tail_im
        score = torch.sum(result,dim=-1)
        return score