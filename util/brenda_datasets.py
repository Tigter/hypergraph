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
import os
import json
from torch_geometric.loader.dataloader import Collater
from transformers import BertTokenizer
from rdkit import Chem

from util.smiles.transormer_util import *

def load_data():
    # graph_info = torch.load("./pre_handle_data/brenda_small_dataset_reaction_filter_lowf_graph_info.pkl")
    # train_info = torch.load("./pre_handle_data/brenda_small_dataset_reaction_filter_lowf_train_info.pkl")
    # graph_info = torch.load("./pre_handle_data/brenda_small_no_e_add_edge_type_graph_info.pkl")
    # train_info = torch.load("./pre_handle_data/brenda_small_no_e_add_edge_type_train_info.pkl")
    # 大数据集过滤高频和低频
    # graph_info = torch.load("./pre_handle_data/brenda_07/brenda_bigger_lhf_no_e_add_edge_type_graph_info.pkl")
    # train_info = torch.load("./pre_handle_data/brenda_07/brenda_bigger_lhf_no_e_add_edge_type_train_info.pkl")
    # 大数据集全量数据
    # graph_info = torch.load("./pre_handle_data/brenda_07/brenda_bigger_no_e_add_edge_type_graph_info.pkl")
    # train_info = torch.load("./pre_handle_data/brenda_07/brenda_bigger_no_e_add_edge_type_train_info.pkl")
    # 大数据集加入了 没有酶的数据
    graph_info = torch.load("./pre_handle_data/brenda_07/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_graph_info.pkl")
    train_info = torch.load("./pre_handle_data/brenda_07/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_train_info.pkl")
    return graph_info, train_info

def build_graph_sampler(config):
    graph_info, train_info = load_data()
    base_node_num = graph_info["base_node_num"]

    c_num, e_num = graph_info["c_num"],graph_info["e_num"]
    e2id, e_num = graph_info["e2id"],graph_info["e_num"]

    n_hyedge = len(graph_info["clist2edgeId"])
    clist2edgeId = graph_info["clist2edgeId"]
    edge2id2cList = {
        clist2edgeId[key] :key  for key in clist2edgeId.keys()
    }

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
        TestRelationDataset(train_info["valid_triple"],all_true_triples, n_hyedge, e_num, valid_sampler,c_num), 
        batch_size=16,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=TestRelationDataset.collate_fn
    )
    test_dataset = DataLoader(
        TestRelationDataset(train_info["test_triple"],all_true_triples, n_hyedge, e_num ,valid_sampler,c_num), 
        batch_size=16,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=TestRelationDataset.collate_fn
    )
    train_test = DataLoader(
        TestRelationDataset(train_info["train_triple"],all_true_triples, n_hyedge, e_num ,valid_sampler,c_num), 
        batch_size=16,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=TestRelationDataset.collate_fn
    )

    # cl_dataset = OneShotIterator(DataLoader(
    #     HyperEdgeClDataset(train_info["train_triple"],edge2id2cList,sampler), 
    #     batch_size=config["batch_size"],
    #     shuffle=True, 
    #     num_workers=max(1, 4//2),
    #     collate_fn=HyperEdgeClDataset.collate_fn
    # ))
    # dataset = SmilesDataset()
    cl_dataset = None   
    return train_dataset,valid_dataset,test_dataset,graph_info,train_info, None, None,train_test,cl_dataset#smileGraphDataset,clDataset

def build_graph_sampler_ne_predict(config):
    graph_info, train_info = load_data()
    base_node_num = graph_info["base_node_num"]

    c_num, e_num = graph_info["c_num"],graph_info["e_num"]
    e2id, c2id = graph_info["e2id"],graph_info["c2id"]

    index2e = {
        e2id[key] - c_num:key  for key in e2id.keys()
    }

    n_hyedge = len(graph_info["clist2edgeId"])
    clist2edgeId = graph_info["clist2edgeId"]
    edge2id2cList = {
        clist2edgeId[key] :key  for key in clist2edgeId.keys()
    }
    
    valid_sampler = CEGraphSampler(graph_info, train_info, 
            batch_size=16,
            size=config["test_neibor_size"],
            mode="valid"
    )
    
    return graph_info,train_info,clist2edgeId,valid_sampler,index2e,e_num

class NagativeRelationSampleDataset(Dataset):

    def __init__(self, triples, nentity, nrelation, negative_sample_size, graph_sampler, c_num):

        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)

        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.true_realtion = self.get_true_head_and_tail(self.triple_set, c_num)
        self.graph_sampler = graph_sampler
        self.c_num = c_num

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
        relation = torch.LongTensor([relation]) - self.c_num
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
    def get_true_head_and_tail(triples,c_num):
      
        true_relation = {}
        for head, relation, tail in triples:
            if (head, tail) not in true_relation:
                true_relation[(head, tail)] = []
            true_relation[(head, tail)].append(relation-c_num)
           
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

    def cl_sampler(self, batch):
        # n_id = torch.tensor(batch, dtype=torch.long)   # 但是采样中心还是使用原来的 id，因为在整个图结构当中是这样的，不然采样会不正确
        # n_id = batch.contiguous()  # 超边的id
        n_id = batch
        adjs = [] 

        n_id_list = []
        index = []
        for i in range(len(n_id)):
            adj_t, n_id = self.traj2traj_adj_t.sample_adj(n_id[i:i+1], 5, replace=False)
            if len(n_id) > 1:
                n_id_list.append(n_id[1:2])
                index.append(i)
        
        n_id = torch.cat(n_id_list)
        split_idx = len(n_id)
        adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, self.sizes[-1], replace=False)
        row, col, e_id = adj_t.coo()
        edge_attr = None
        edge_type = None
        size = adj_t.sparse_sizes()[::-1]
        adjs.append((adj_t, edge_attr,  edge_type, e_id, size))
        index = torch.LongTensor(index)
        out = (n_id, adjs, split_idx),index
        return out

    def sampler_v(self, batch):
        adjs = []
        n_id = batch.contiguous()
        split_idx = len(n_id)
        adj_t, n_id = self.ci2traj_adj_t.sample_adj(n_id, 20, replace=False)
        row, col, e_id = adj_t.coo()
        edge_attr = None
        edge_type = None        
        size = adj_t.sparse_sizes()[::-1]
        adjs.append((adj_t, edge_attr,  edge_type, e_id, size))

        out = (n_id, adjs, split_idx)
        return out

    def sample(self, batch):
        # n_id = torch.tensor(batch, dtype=torch.long)   # 但是采样中心还是使用原来的 id，因为在整个图结构当中是这样的，不然采样会不正确
        n_id = batch.contiguous()  # 超边的id
        if self.mode == "train":
            # cl_out = self.cl_sampler(n_id)
            cl_out = None
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
        if self.mode=='train':
            return out,cl_out
        else:
            return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


class GINPretrainDataset(Dataset):
    def __init__(self, c_num ):
        super(GINPretrainDataset, self).__init__()
        self.c_num = c_num

        path = "/home/skl/yl/ce_project/relation_cl/brenda_data/filter_data/graph/"
        self.path_list = []
        for i in range(c_num):
            self.path_list.append(os.path.join(path,"graph_"+str(i)+".pt"))
            
    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return self.c_num

    def __getitem__(self, index):
        graph_name = self.path_list[index]
        # load and process graph
        data_graph = torch.load(graph_name)
      
        return data_graph

class NodeEmbeddingClDataset(Dataset):
    def __init__(self, c_num ):
        super(NodeEmbeddingClDataset, self).__init__()
        self.c_num = c_num
        path = "/home/skl/yl/ce_project/relation_cl/brenda_data/filter_data/id2cid.json"
        with open(path) as f:
            id2cid = json.load(f)
        self.graph_path = "/home/skl/yl/ce_project/relation_cl/brenda_data/filter_data/graph/"
        self.text_path  = "/home/skl/yl/ce_project/relation_cl/brenda_data/filter_data/text/"

        self.datalist = []
        for id,cid in id2cid.items():
            text_file = os.path.join(self.text_path,"text_%s.txt"% str(cid))
            if os.path.exists(text_file):
                self.datalist.append((id,cid))

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self.datalist)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        graph_name =os.path.join(self.graph_path,"graph_%s.pt"% str(self.datalist[index][0]))
        data_graph = torch.load(graph_name)

        text_file = os.path.join(self.text_path,"text_%s.txt"% str(self.datalist[index][1]))
        with open(text_file,"r") as f:
            lines = f.readlines()
            line = "\n".join(lines)

        return data_graph, line




class TrainCollater(object):
    def __init__(self, tokenizer, text_max_len):
        self.tokenizer = tokenizer
        self.graph_collector = Collater([], [])
        self.text_max_len = text_max_len
    
    def __call__(self, batch):
        graph_list = [_[0] for _ in batch]
        text_list = [_[1] for _ in batch]
        graph_batch = self.graph_collector(graph_list)       
        text_batch = self.tokenizer(text_list, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
        return graph_batch, text_batch.input_ids, text_batch.attention_mask


class OneShotIterator(object):
    def __init__(self, dataloader):
        self.dataloader = self.one_shot_iterator(dataloader)
 
    def __next__(self):
        data = next(self.dataloader)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data



class HyperEdgeClDataset(Dataset):
    def __init__(self, triples,edge2id2cList,sampler):
        super(HyperEdgeClDataset, self).__init__()
        self.triples = triples

        print(len(self.triples))
        # 目标： 计算不同超边的相似度
        # 然后 left 的edgeId 之间判断相似性，如果相似，将他们的 
        self.samples = []
        self.weights = []
        with open("/home/skl/yl/ce_project/relation_cl/sh/head_dict.json","r") as f:
            self.head_dict = json.load(f)
        with open("/home/skl/yl/ce_project/relation_cl/sh/tail_dict.json","r") as f:
            self.tail_dict = json.load(f)

        # self.head_dict, self.tail_dict = self.find_intersections(triples, edge2id2cList)
        
        # with open("/home/skl/yl/ce_project/relation_cl/sh/head_dict.json","w") as f:
        #     json.dump( self.head_dict,f)
        # with open("/home/skl/yl/ce_project/relation_cl/sh/tail_dict.json","w") as f:
        #     json.dump( self.tail_dict,f)

        for head_id_1, similarity_dict in self.head_dict.items():
            for head_id_2, weight in similarity_dict.items():
                if head_id_1 != head_id_2:  # 排除自身匹配
                    self.samples.append((head_id_1, head_id_2))
                    self.weights.append(weight)

        for head_id_1, similarity_dict in self.tail_dict.items():
            for head_id_2, weight in similarity_dict.items():
                if head_id_1 != head_id_2:  # 排除自身匹配
                    self.samples.append((head_id_1, head_id_2))
                    self.weights.append(weight)
        print(len(self.samples))
        self.sampler = sampler

    def find_intersections(self,lst, edge2id2cList):
        head_dict = {}
        tail_dict = {}
        for i in range(len(lst)):
            head_id, r, tail_id = lst[i]
            # print(head_id)
            # if head_id not in edge2id2cList or tail_id not in edge2id2cList: continue
            head_list = edge2id2cList[head_id]
            tail_list = edge2id2cList[tail_id]
            for j in range(i + 1, len(lst)):
                other_head_id, _, other_tail_id = lst[j]
                # if other_head_id not in edge2id2cList or other_tail_id not in edge2id2cList: continue
                other_head_list = edge2id2cList[other_head_id]
                other_tail_list = edge2id2cList[other_tail_id]
                tail_intersection = set(tail_list) & set(other_tail_list)
                head_intersection = set(head_list) & set(other_head_list)
                if tail_intersection:
                    if head_id not in head_dict:
                        head_dict[head_id] = {}
                    if other_head_id not in head_dict[head_id]:
                        head_dict[head_id][other_head_id] = len(tail_intersection)
                    else:
                        head_dict[head_id][other_head_id] += len(tail_intersection)
                        
                    if other_head_id not in head_dict:
                        head_dict[other_head_id] = {}
                    if head_id not in head_dict[other_head_id]:
                        head_dict[other_head_id][head_id] = len(tail_intersection)
                    else:
                        head_dict[other_head_id][head_id] += len(tail_intersection)
                        
                if head_intersection:
                    if tail_id not in tail_dict:
                        tail_dict[tail_id] = {}
                    if other_tail_id not in tail_dict[tail_id]:
                        tail_dict[tail_id][other_tail_id] = len(head_intersection)
                    else:
                        tail_dict[tail_id][other_tail_id] += len(head_intersection)
                        
                    if other_tail_id not in tail_dict:
                        tail_dict[other_tail_id] = {}
                    if tail_id not in tail_dict[other_tail_id]:
                        tail_dict[other_tail_id][tail_id] = len(head_intersection)
                    else:
                        tail_dict[other_tail_id][tail_id] += len(head_intersection)     
        return head_dict, tail_dict
    
    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        base,pos = self.samples[index]

        weight = int(self.weights[index])
        return torch.LongTensor([int(base)]),torch.LongTensor([int(pos)]),torch.LongTensor([weight]),self.sampler

    @staticmethod
    def collate_fn(data):
        base = torch.cat([_[0] for _ in data], dim=0)
        pos = torch.cat([_[1] for _ in data], dim=0)
        weight = torch.cat([_[2] for _ in data], dim=0)
        sampler = data[0][3]
        base_out = sampler.sampler_v(base)
        pos_out = sampler.sampler_v(pos)
        return base,pos,weight,base_out,pos_out