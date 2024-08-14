import torch
import torch.nn as nn

def modify_grad(x, inds):
    x[inds] = 0 
    return x

class HousE(nn.Module):

    def __init__(self,nentity, nrelation, dim, house_dim=20, house_num=20, housd_num=6, gamma=5.0):    #fb15k-237
       
        super(HousE,self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = int(dim / house_dim)
        self.house_dim = house_dim
        self.house_num = house_num
        self.epsilon = 2.0
        self.housd_num = housd_num
        self.house_num = house_num + (2*self.housd_num)
       
        self.thred = 0.6   # fb

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / (self.hidden_dim * (self.house_dim ** 0.5))]),
            requires_grad=False
        )
        
        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim, self.house_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.house_dim*self.house_num))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.k_dir_head = nn.Parameter(torch.zeros(nrelation, 1, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_dir_head,
            a=-0.01,
            b=+0.01
        )

        self.k_dir_tail = nn.Parameter(torch.zeros(nrelation, 1, self.housd_num))
        with torch.no_grad():
            self.k_dir_tail.data = - self.k_dir_head.data
        
        self.k_scale_head = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_scale_head,
            a=-1,
            b=+1
        )

        self.k_scale_tail = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_scale_tail,
            a=-1,
            b=+1
        )

        self.relation_weight = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.house_dim))
        nn.init.uniform_(
            tensor=self.relation_weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def score_function(self, head, relation, tail, k_head, k_tail,mode):
        r_list = torch.chunk(relation, self.house_num, 3)  # self.house_num 个关系
        epsilon =  0.05
        pi = 3.14159265358979323846
        if mode == 'h_rt':
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

          

            for i in range(self.housd_num, self.house_num-self.housd_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]

            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
           

            for i in range(self.housd_num, self.house_num-self.housd_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]

            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        score = - cos_score
        return score

    def norm_embedding(self):
        entity_embedding = self.entity_embedding
        r_list = torch.chunk(self.relation_embedding, self.house_num, 2)
        normed_r_list = []
        for i in range(self.house_num):
            r_i = torch.nn.functional.normalize(r_list[i], dim=2, p=2)
            normed_r_list.append(r_i)
        r = torch.cat(normed_r_list, dim=2)
        self.k_head = self.k_dir_head * torch.abs(self.k_scale_head)
        self.k_head[self.k_head>self.thred] = self.thred
        self.k_tail = self.k_dir_tail * torch.abs(self.k_scale_tail)
        self.k_tail[self.k_tail>self.thred] = self.thred
        return entity_embedding, r
    
    def forward(self, h,r,t, mode='hrt',trap_para=None,add_trap=False):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        entity_embedding, r_emb = self.norm_embedding()
        batch_size = r.shape[0]

        if mode == 'h_rt':
            # print('h.shape:',h.shape)
            negative_sample_size = h.shape[1]
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=h.reshape(-1)
            ).reshape(batch_size, negative_sample_size, self.entity_dim, -1)

            k_head = torch.index_select(
                self.k_head,
                dim=0,
                index=r
            ).unsqueeze(1)

            k_tail = torch.index_select(
                self.k_tail,
                dim=0,
                index=r
            ).unsqueeze(1)

            re_weight = torch.index_select(
                self.relation_weight,
                dim=0,
                index=r
            ).unsqueeze(1)

            relation = torch.index_select(
                r_emb,
                dim=0,
                index=r
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=t
            ).unsqueeze(1)

        elif mode == 'hr_t':
            # print('t.shape:',t.shape)
            negative_sample_size = t.shape[1]
           

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=h
            ).unsqueeze(1)

            k_head = torch.index_select(
                self.k_head,
                dim=0,
                index=r
            ).unsqueeze(1)

            k_tail = torch.index_select(
                self.k_tail,
                dim=0,
                index=r
            ).unsqueeze(1)

            re_weight = torch.index_select(
                self.relation_weight,
                dim=0,
                index=r
            ).unsqueeze(1)

            relation = torch.index_select(
                r_emb,
                dim=0,
                index=r
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=t.reshape(-1)
            ).reshape(batch_size, negative_sample_size, self.entity_dim, -1)

        elif mode == 'hrt':
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=h
            ).unsqueeze(1)

            k_head = torch.index_select(
                self.k_head,
                dim=0,
                index=r
            ).unsqueeze(1)

            k_tail = torch.index_select(
                self.k_tail,
                dim=0,
                index=r
            ).unsqueeze(1)

            re_weight = torch.index_select(
                self.relation_weight,
                dim=0,
                index=r
            ).unsqueeze(1)

            relation = torch.index_select(
                r_emb,
                dim=0,
                index=r
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=t.reshape(-1)
            ).reshape(batch_size, 1, self.entity_dim, -1)

        else:
            raise ValueError('mode %s not supported' % mode)
        
        if add_trap:
            return self.score_function(head, relation, tail, k_head, k_tail, mode, trap_para, add_trap)

        score = self.score_function(head, relation, tail, k_head, k_tail, mode)

        return score
