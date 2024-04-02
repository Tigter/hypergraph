import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.data import NeighborSampler,Data
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
def load_data():

    graph_info = torch.load("/home/tengwei/hypergraph/pre_handle_data/brenda_dataset_reaction_graph_info.pkl")
    train_info = torch.load("/home/tengwei/hypergraph/pre_handle_data/brenda_dataset_reaction_train_info.pkl")
    return graph_info, train_info

    sing_graph = {
        "train_edge_index": edge_index_train,
        "train_edge_type": edge_type_train,

        "valid_edge_index": new_edge_index,
        "valid_edge_type": new_edge_type,

        "v2e_all_index": v2e_all_index,
        "v2e_train_index": v2e_train_index,
        "e2v_all_index": e2v_all_index,
        "max_edge_id":edge_start_id + hyedge_num,
        "base_node_num":edge_start_id,
        "clist2edgeId":clist2edgeId
    }
    train_info = {
        "train_id_list": train_id_list,
        "valid_id_list": valid_id_list,
        "test_id_list": test_id_list,

        "train_triple": train_triples,
        "valid_triple": valid_triples,
        "test_triple": test_triples,     
        # "edgeid2label": Hy2E,
    }
def build_graph_sampler(config):
    graph_info, train_info = load_data()
    base_node_num = graph_info["base_node_num"]

    c_num, e_num = graph_info["c_num"],graph_info["e_num"]
    e2id, e_num = graph_info["e2id"],graph_info["e_num"]


    n_hyedge = len(graph_info["clist2edgeId"])


    sampler = CEGraphSampler(graph_info, train_info, 
            batch_size=config["batch_size"],
            size=config["neibor_size"],mode="train",
            config=config)
    
    valid_sampler = CEGraphSampler(graph_info, train_info, 
            batch_size=16,
            size=config["test_neibor_size"],
            mode="valid"
    )
    all_true_triples = train_info["train_triple"] + train_info["valid_triple"] + train_info["test_triple"]
    train_dataset = DataLoader(
        NagativeRelationSampleDataset(train_info["train_triple"], n_hyedge, e_num, config["n_size"],sampler,c_num), 
        batch_size=config["batch_size"],
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=NagativeRelationSampleDataset.collate_fn
    )
   
    valid_dataset = DataLoader(
        TestRelationDataset(train_info["train_triple"],all_true_triples, n_hyedge, e_num, valid_sampler,c_num), 
        batch_size=16,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=TestRelationDataset.collate_fn
    )
    test_dataset = DataLoader(
        TestRelationDataset(train_info["train_triple"],all_true_triples, n_hyedge, e_num ,valid_sampler,c_num), 
        batch_size=16,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=TestRelationDataset.collate_fn
    )

    return train_dataset,valid_dataset,test_dataset,graph_info,train_info


class NagativeRelationSampleDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, graph_sampler, e_num):

        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.true_realtion = self.get_true_head_and_tail(self.triple_set, e_num)
        self.graph_sampler = graph_sampler
        self.e_num = e_num

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(0, self.nrelation,size=self.negative_sample_size*2)
            mask = np.in1d(
                    negative_sample, 
                    self.true_realtion[(head, tail)], 
                    assume_unique=True, 
                    invert=True
            )
            negative_sample = negative_sample[mask] # filter true triples
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        head = torch.LongTensor([head])
        relation = torch.LongTensor([relation]) - self.e_num
        tail = torch.LongTensor([tail])
        
        return head,relation,tail, negative_sample, self.graph_sampler
    
    @staticmethod
    def collate_fn(data):
        head = torch.cat([_[0] for _ in data], dim=0)
        relation = torch.cat([_[1] for _ in data], dim=0)
        tail = torch.cat([_[2] for _ in data], dim=0)
        negative_sample = torch.stack([_[3] for _ in data], dim=0)
        sampler = data[0][4]
        head_out = sampler.sample(head)
        tail_out = sampler.sample(tail)

        return head,relation,tail,negative_sample,head_out,tail_out
    
    @staticmethod
    def get_true_head_and_tail(triples,e_num):
      
        true_relation = {}
        for head, relation, tail in triples:
            if (head, tail) not in true_relation:
                true_relation[(head, tail)] = []
            true_relation[(head, tail)].append(relation-e_num)
           
        for head, tail in true_relation:
            true_relation[(head, tail)] = np.array(list(set(true_relation[(head, tail)])))
        return true_relation


class TestRelationDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, sampler, e_number):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.sampler = sampler
        self.e_num = e_number
        

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        relation -= self.e_num

        tmp = [(0, rand_r) if (head, rand_r+self.e_num, tail) not in self.triple_set
                   else (-1, relation) for rand_r in range(self.nrelation)]
        
        if relation >= len(tmp): print("Error")
        tmp[relation] = (0, relation)


        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]
        head = torch.LongTensor([head])
        relation = torch.LongTensor([relation])
        tail = torch.LongTensor([tail])
        return head, relation, tail, negative_sample, filter_bias, self.sampler
    
    @staticmethod
    def collate_fn(data):
        head = torch.cat([_[0] for _ in data], dim=0)
        relation = torch.cat([_[1] for _ in data], dim=0)
        tail = torch.cat([_[2] for _ in data], dim=0)

        negative_sample = torch.stack([_[3] for _ in data], dim=0)
        filter_bias = torch.stack([_[4] for _ in data], dim=0)
        sampler = data[0][5]
        head_out = sampler.sample(head)
        tail_out = sampler.sample(tail)
        return head,relation,tail,negative_sample,filter_bias,head_out,tail_out

# 负责根据超边进行采样
class CEGraphSampler(torch.utils.data.DataLoader):
   
    def __init__(self, graph_info,train_info,batch_size=128,size=[2,2], mode="train",config=None,**kwargs):
        
        self.batch_size = batch_size
        self.graph_info = graph_info
        self.train_info = train_info
        self.sizes = size
        self.mode = mode
        
        self.all_node_num = self.graph_info["max_edge_id"]
        self.base_node_num = self.graph_info["base_node_num"]
        self.e_num = self.graph_info["e_num"]
        self.c_num = self.graph_info["c_num"]

        if mode == 'train':
            self.edge_index = self.graph_info["train_edge_index"]
            self.traj2traj_edge_type = self.graph_info["train_edge_type"]
        else:
            self.edge_index = self.graph_info["valid_edge_index"]
            self.traj2traj_edge_type = self.graph_info["valid_edge_type"]

        # 超边之间的连接
        self.traj2traj_adj_t = SparseTensor(
            row= self.edge_index[0],
            col= self.edge_index[1],
            value=torch.arange(self.edge_index.size(1)),  # 超边之间
            sparse_sizes=(self.all_node_num, self.all_node_num)
        ).t()

        self.e2v_index = self.graph_info["v2e_all_index"]

        # 实体和超边之间的连接
        self.ci2traj_adj_t = SparseTensor(
            row=self.e2v_index[0],
            col=self.e2v_index[1],
            value=torch.arange(self.e2v_index.size(1)),
            sparse_sizes=(self.all_node_num, self.all_node_num)
        ).t()

        # 需要构建一个新的sampler， 增加了负采样的sample
        node_idx = torch.tensor([0])
        super(CEGraphSampler, self).__init__(node_idx.view(-1).tolist(), collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch):
        # n_id = torch.tensor(batch, dtype=torch.long)   # 但是采样中心还是使用原来的 id，因为在整个图结构当中是这样的，不然采样会不正确
        n_id = batch.contiguous()
        adjs = [] 
        for i, size in enumerate(self.sizes):
            if i == len(self.sizes) - 1:
                # Sample ci2traj one-hop checkin relation
                adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = None
            else:
                # Sample traj2traj multi-hop relation
                adj_t, n_id = self.traj2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.traj2traj_edge_type[e_id]
                split_idx = len(n_id)
            size = adj_t.sparse_sizes()[::-1]
            adjs.append((adj_t, edge_attr,  edge_type, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (n_id, adjs, split_idx)
        
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

