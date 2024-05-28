import torch
import torch.nn as nn
import torch.nn.functional as F
from box_embeddings.parameterizations import BoxTensor,MinDeltaBoxTensor,SigmoidBoxTensor
from box_embeddings.modules.volume import SoftVolume,HardVolume
from box_embeddings.modules.regularization import L2SideBoxRegularizer
from box_embeddings.modules.intersection import hard_intersection, gumbel_intersection
import numpy as np
from itertools import product


class BoxLevel(nn.Module):
    def __init__(self,n_entity, n_relation, dim,
               init_interval_center: float = 0.1,
               p_norm=1,regu_weight=0,soft_temp=0.5
               ):
        super(BoxLevel,self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.p_norm = p_norm

        # array = np.random.uniform(-1,1,size=[n_entity,2,dim]) 
        # 初始化方法 1:随机初始化
        init_tensor = torch.randn((n_entity,2,dim),requires_grad=True)
        self.init_tensor = torch.nn.Parameter(torch.tensor(init_tensor))
        range_value = (20 + 2)/500
        self.trans_emb = torch.nn.Embedding(n_entity, dim)

        nn.init.uniform_(
            tensor=self.trans_emb.weight,
            a=-range_value,
            b=range_value
        )
        nn.init.uniform_(
            tensor=self.init_tensor[0:,0,0:],
            a=-range_value,
            b=0
        )
        nn.init.uniform_(
            tensor=self.init_tensor[0:,1,0:],
            a=0.001,
            b=0.001+range_value
        )
        
        # self.box_regularizer = L2SideBoxRegularizer(regu_weight)
        self.entity_embedding = MinDeltaBoxTensor(self.init_tensor,threshold=10)
        self.soft_volume = SoftVolume(volume_temperature=1.0)
        self.hard_volume = HardVolume()
        
        self.criterion = nn.Softplus()
        self.max_value = self.dim


    # 这里可以实现从其他格式的文件读取data
    def init_model(self):
        pass

    def value_of_index(self, index):
        return self.soft_volume(self.entity_embedding[index])
    
    def self_soft_volum(self,box):
        return torch.sum(torch.log(F.softplus(box.Z-box.z) + 1),dim=-1)
    
    def box_embedding_score(self, head, relation, tail, head_pos, tail_pos, loss=False):

        # head_center = head.centre 
        # tail_center = tail.centre

        head_center = head.centre + tail_pos
        tail_center = tail.centre + head_pos
        head_width = head.Z - head.z # > 0
        tail_width = tail.Z - tail.z # > 0
        
        val_score = torch.norm( torch.relu(head_width - 0.1*tail_width),dim=-1) # 体积的差距

        head_width = head_width.detach()
        tail_width = tail_width.detach()

        insert_score =  torch.relu((torch.abs(head_center - tail_center)+ head_width - 0.4*(tail_width))) # 位置大小的约束
        insert_score = torch.sum(insert_score,dim=-1)
        
        return -val_score, -insert_score

    def box_center_score(self, boxA, boxB):
        w = boxB.Z - boxB.z + 1
        k = 0.5*(w - 1) * (w - 1/w)

        z_score = torch.where(torch.logical_and(torch.ge(boxA.z, boxB.z), torch.le(boxA.Z, boxB.Z)),
                     (torch.abs(boxA.centre - boxB.centre)) / w,
                     (torch.abs(boxA.centre - boxB.centre)) * w - k)
        return z_score
        
    
    def another_score(self, head, relation, tail,head_pos, tail_pos, loss=False):
       
        val, pos = self.box_embedding_score(head, relation, tail,head_pos, tail_pos)
        score =100*val + pos
        return score
  
    def loss(self, score, batch_y):
        return torch.mean(torch.sum(self.criterion(-score * batch_y),dim=-1))#

    def regu_loss(self,values):
        # 筛选
        less_1_index = values  <  5
        bigger_200_index = values > self.max_value
        less_value = values[less_1_index]
        bigger_value = values[bigger_200_index]
        return -torch.norm(less_value) + torch.norm(bigger_value)


    def set_max(self, value):
        self.max_value = value

    def predict(self,h,r,t, mode='hrt'):
        head = None
        tail = None
        relation = None
        
        head_pos = self.trans_emb(h)
        tail_pos = self.trans_emb(t)

        if mode=='hr_t':
            negative_size = t.shape[1]
            batch_size  = t.shape[0]
            t = t.reshape(-1,1)
            head = self.entity_embedding[h]
            head = head.box_reshape(target_shape=(batch_size, 1, self.dim))
            tail = self.entity_embedding[t]
            tail = tail.box_reshape(target_shape=(batch_size, negative_size, self.dim))

            head_pos = head_pos.unsqueeze(1)

        elif mode == 'h_rt':
            negative_size = h.shape[1]
            batch_size  = h.shape[0]
            h = h.reshape(-1,1)
            tail = self.entity_embedding[t]
            tail = tail.box_reshape(target_shape=(batch_size, 1, self.dim))
            head = self.entity_embedding[h]
            head = head.box_reshape(target_shape=(batch_size, negative_size, self.dim))  
            
            tail_pos = tail_pos.unsqueeze(1)

        elif mode == 'hrt':
            batch_size  = h.shape[0]
            tail = self.entity_embedding[t]
            tail = tail.box_reshape(target_shape=(batch_size, 1, self.dim))
            head = self.entity_embedding[h]
            head = head.box_reshape(target_shape=(batch_size, 1, self.dim))     

            tail_pos = tail_pos.unsqueeze(1)
            head_pos = head_pos.unsqueeze(1)

        elif mode == 'hr_all':
            batch_size  = h.shape[0]
            head = self.entity_embedding[h]
            head = head.box_reshape(target_shape=(batch_size, 1, self.dim))    
            tail = self.entity_embedding
            tail = tail.box_reshape(target_shape=(1, -1, self.dim))    
     
        return self.another_score(head, relation, tail, head_pos, tail_pos, False)
    
    def forward(self, h,r,t, mode='hrt'):
        head = None
        tail = None
        relation = None

        head_pos = self.trans_emb(h)
        tail_pos = self.trans_emb(t)

        if mode=='hr_t':
            negative_size = t.shape[1]
            batch_size  = t.shape[0]
            t = t.reshape(-1,1)
            head = self.entity_embedding[h]
            head = head.box_reshape(target_shape=(batch_size, 1, self.dim))
            tail = self.entity_embedding[t]
            tail = tail.box_reshape(target_shape=(batch_size, negative_size, self.dim))

            head_pos = head_pos.unsqueeze(1)

        elif mode == 'h_rt':
            negative_size = h.shape[1]
            batch_size  = h.shape[0]
            h = h.reshape(-1,1)
            tail = self.entity_embedding[t]
            tail = tail.box_reshape(target_shape=(batch_size, 1, self.dim))
            head = self.entity_embedding[h]
            head = head.box_reshape(target_shape=(batch_size, negative_size, self.dim))   

            tail_pos = tail_pos.unsqueeze(1)

        elif mode == 'hrt':
            batch_size  = h.shape[0]
            tail = self.entity_embedding[t]
            tail = tail.box_reshape(target_shape=(batch_size, -1, self.dim))
            head = self.entity_embedding[h]
            head = head.box_reshape(target_shape=(batch_size, -1, self.dim))   

            tail_pos = tail_pos.unsqueeze(1)
            head_pos = head_pos.unsqueeze(1)

        elif mode == 'hr_all':
            batch_size  = h.shape[0]
            head = self.entity_embedding[h]
            head = head.box_reshape(target_shape=(batch_size, 1, self.dim))    
            tail = self.entity_embedding
            tail = tail.box_reshape(target_shape=(1, -1, self.dim))    
        return self.box_embedding_score(head, relation,tail,head_pos,tail_pos,True)