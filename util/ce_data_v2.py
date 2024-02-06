import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.data import NeighborSampler,Data
import math

def load_data():

    graph_info = torch.load("/home/skl/yl/ce_project/relation_cl/pre_handle_data/ce_data_double_graph_info.pkl")
    train_info = torch.load("/home/skl/yl/ce_project/relation_cl/pre_handle_data/ce_data_double_train_info.pkl")
    return graph_info, train_info


def build_graph_sampler(config):
    graph_info, train_info = load_data()
    base_node_num = graph_info["base_node_num"]

    node_emb = nn.Embedding(base_node_num, config["dim"])
    graph_info["node_emb"] = node_emb.weight

    init_range =  6.0 / math.sqrt(config["dim"])
    nn.init.uniform_(node_emb.weight, -init_range, init_range)

    sampler = CEGraphSampler(graph_info, train_info, 
        train_info["train_id_list"],
            batch_size=config["batch_size"],
            size=config["neibor_size"],
            n_size=config["n_size"])
    
    valid_sampler = CEGraphSampler(graph_info, train_info, 
        train_info["valid_id_list"],
            batch_size=16,
            size=config["neibor_size"],
             mode="valid"
            )
    test_sampler = CEGraphSampler(graph_info, train_info, 
        train_info["test_id_list"],
            batch_size=16,
            size=config["neibor_size"],
             mode="valid"
            )
    return sampler,valid_sampler, test_sampler,graph_info,train_info,node_emb
# graph_info = {
#     "train_edge_index": edge_index_train,
#     "train_edge_type": edge_type_train,
#     "valid_edge_index": new_edge_index,
#     "valid_edge_type": new_edge_type,
#     "node2edge_index": v2e_index,
#     "edge2rel_index": e2r_index,
#     "c_num": c_num,
#     "e_num": e_num,
#     "max_train_id": max_train_num,
#     "max_edge_id":edge_id,
# }

# train_info = {
#     "train_id_list": train_id_list,
#     "valid_id_list": valid_id_list,
#     "test_id_list": test_id_list,
#     "edgeid2label": edgeid2label,
#     "edgeid2true_train": edgeid2true_train,
#     "edgeid2true_all": edgeid2true_train,
# }


class CEGraphSampler(torch.utils.data.DataLoader):
   
    def __init__(self, graph_info,train_info,node_idx,batch_size=128,size=[2,2],n_size=10,mode="train",**kwargs):
        
        self.batch_size = batch_size
        self.graph_info = graph_info
        self.train_info = train_info

        self.sizes = size
        self.node_idx = node_idx
        self.mode = mode
        self.node_emb = self.graph_info["node_emb"]

        edgeid2label_list = []
        for key in self.train_info["edgeid2label"]:
            edgeid2label_list.append([key,self.train_info["edgeid2label"][key]])
        sorted(edgeid2label_list,key=lambda x:x[0])
        labels = [x[1] for x in edgeid2label_list]
        self.label = torch.LongTensor(labels) # - self.graph_info["c_num"]
        self.n_size = n_size
        
        if mode == "train":
            self.mask_true = self.train_info["edgeid2true_train"]
        else:
            self.mask_true = self.train_info["edgeid2true_all"]
            self.n_size = 5000

        self.num_nodes = self.graph_info["max_edge_id"]

        self.n_node = self.graph_info["base_node_num"]
        self.e_num = self.graph_info["e_num"]
        self.c_num = self.graph_info["c_num"]

        if mode == 'train':
            print("train graph")
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
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        self.e2v_index = self.graph_info["node2edge_index"]

        # 实体和超边之间的连接
        self.ci2traj_adj_t = SparseTensor(
            row=self.e2v_index[0],
            col=self.e2v_index[1],
            value=torch.arange(self.e2v_index.size(1)),
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        # 需要构建一个新的sampler， 增加了负采样的sample
        node_idx = torch.tensor(node_idx)
        print("sampler range:")
        print(torch.min(node_idx))
        print(torch.max(node_idx))
        print("*********************")

        if self.mode == "train":
            self.doubleSampler = DoubleGraphSampler(
                graph_info,train_info,size=size
            )
        single2double = self.graph_info["single2double"]

        self.single2double = torch.LongTensor([x[1] for x in single2double])
      
        super(CEGraphSampler, self).__init__(node_idx.view(-1).tolist(), collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch):
        sample_idx = [i - self.n_node for i in batch]  # 输入是超边的id，然后转为从0开始的index，这样才能获取到正确的label
        

        lable = self.label[sample_idx] 
        sample_idx = torch.LongTensor(sample_idx)
        n_id = torch.tensor(batch, dtype=torch.long)   # 但是采样中心还是使用原来的 id，因为在整个图结构当中是这样的，不然采样会不正确
        

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

        input_x =  self.node_emb[n_id[split_idx:]]

        out = (n_id, input_x, adjs, lable - self.c_num,split_idx)

        
        if self.mode == "train":
            double_id = self.single2double[sample_idx]
            double_out = self.doubleSampler.sample(double_id)
            return out,double_out
        else:
            return out
    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


class DoubleGraphSampler(torch.utils.data.DataLoader):
   
    def __init__(self, graph_info,train_info,batch_size=128,size=[2,2],n_size=10,mode="train",**kwargs):
        
        self.batch_size = batch_size
        self.graph_info = graph_info
        self.train_info = train_info

        self.sizes = size
        self.mode = mode
        self.node_emb = self.graph_info["node_emb"]


        edgeid2label_list = []
        
        self.n_size = n_size
        
        self.num_nodes = self.graph_info["double_max_edge_id"]
        self.n_node = self.graph_info["base_node_num"]
        self.e_num = self.graph_info["e_num"]
        self.c_num = self.graph_info["c_num"]

        self.edge_index = self.graph_info["double_train_edge_index"]
        self.traj2traj_edge_type = self.graph_info["double_train_edge_type"]

        # 超边之间的连接
        self.traj2traj_adj_t = SparseTensor(
            row= self.edge_index[0],
            col= self.edge_index[1],
            value=torch.arange(self.edge_index.size(1)),  # 超边之间
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()

        self.e2v_index = self.graph_info["double_v2e"]

        # 实体和超边之间的连接
        print(torch.max(self.e2v_index[0]))
        print(torch.max(self.e2v_index[1]))
        print(self.num_nodes)

        self.ci2traj_adj_t = SparseTensor(
            row=self.e2v_index[0],
            col=self.e2v_index[1],
            value=torch.arange(self.e2v_index.size(1)),
            sparse_sizes=(self.num_nodes, self.num_nodes)
        ).t()
       
        # 需要构建一个新的sampler， 增加了负采样的sample
        node_idx = torch.tensor([0])
        super(DoubleGraphSampler, self).__init__(node_idx.view(-1).tolist(), collate_fn=self.sample,batch_size=batch_size,**kwargs)

    def sample(self, batch):

        n_id = batch   # 但是采样中心还是使用原来的 id，因为在整个图结构当中是这样的，不然采样会不正确
        adjs = [] 
        for i, size in enumerate(self.sizes):
            if i == len(self.sizes) - 1:
                adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = None
            else:
                adj_t, n_id = self.traj2traj_adj_t.sample_adj(n_id, size, replace=False)
                row, col, e_id = adj_t.coo()
                edge_attr = None
                edge_type = self.traj2traj_edge_type[e_id]
                split_idx = len(n_id)
            
            size = adj_t.sparse_sizes()[::-1]
            adjs.append((adj_t, edge_attr,  edge_type, e_id, size))
        
        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]

        input_x = self.node_emb[n_id[split_idx:]]

        out = (n_id, input_x, adjs, split_idx)
      
        return out
    
    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

