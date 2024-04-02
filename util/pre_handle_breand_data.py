
import pickle
from collections import defaultdict
# from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix
import torch
import numpy as np
import json 

from data_pre_handle_utils import *

# 处理酶和化合物的数据

def load_brenda_data():
    with open("/home/tengwei/hypergraph/brenda_data/filter_data/train_reaction.json", "r") as f:
        train_data = json.load(f)
    with open("/home/tengwei/hypergraph/brenda_data/filter_data/valid_reaction.json", "r") as f:
        valid_data = json.load(f)
    with open("/home/tengwei/hypergraph/brenda_data/filter_data/test_reaction.json", "r") as f:
        test_data = json.load(f)
    return train_data, valid_data, test_data

train_data, valid_data, test_data = load_brenda_data()

cset,eset, e2id,c2id = build_dict_for_double_data(train_data)


c_num = len(cset)
e_num = len(eset)



# 开始构建超图
base_node_num = c_num + e_num
edge_id = 0



sing_graph, train_info =  build_single_graph(train_data,valid_data,test_data, c2id, e2id, base_node_num)


# 重新构建一个超图数据
# 首先需要将ko 的 id 和 层次lable 进行混合编码：
# 构造一个空白的embedding 作为 0（这个地方存疑）

graph_info = {
    "c_num": c_num,
    "e_num": e_num,
    "base_node_num":base_node_num,
    "train": train_data,
    "valid": valid_data,
    "test": test_data,
    "c2id": c2id,
    "e2id": e2id,
}
graph_info.update(sing_graph)

# torch.save(graph_info,"../pre_handle_data/brenda_dataset_reaction_graph_info.pkl")
# torch.save(train_info,"../pre_handle_data/brenda_dataset_reaction_train_info.pkl")
