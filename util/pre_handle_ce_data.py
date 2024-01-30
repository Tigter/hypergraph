
import pickle
from collections import defaultdict
# from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix
import torch
import numpy as np
import json 

# 处理酶和化合物的数据
print("begin")
with open("./base_data.pkl",'rb') as f:
    data = pickle.load(f)
    f.close()

print("load finised")
print(data.keys())

def build_id_dict(data):
    cset = set()
    eset = set()
    for c_list, e in data["train"]:
        for c in c_list:
            cset.add(c)
        eset.add(e)
    cset = list(cset)
    eset = list(eset)

    c2id = {
        cset[i]:i for i in range(len(cset))
    }

    e2id = {
        eset[i]: i+ len(cset) for i in range(len(eset))
    }
    return cset,eset, e2id,c2id

cset,eset, e2id,c2id = build_id_dict(data)

c_num = len(cset)
e_num = len(eset)


with open("./aux_data.json", 'r') as f:
    aux_data = json.load(f)
    # "ec_label_dict": lalbel_dict,
    # "e2label": e2lable,
    # "ko2id": ko2id,
    # "e2ko_list": e2ko_list

ec_label_dict = aux_data["ec_label_dict"]
e2label = aux_data["e2label"]
e2ko_list = aux_data["e2ko_list"]
ko2id = aux_data["ko2id"]

# 首先将两个东西所有的label 和 ko label 统一编码
l1_dict = ec_label_dict["lable1"]
l2_dict = ec_label_dict["lable2"]
l3_dict = ec_label_dict["lable3"]


uni_dict = {}
uni_id = c_num + e_num

for key in sorted(list(l1_dict.keys())):
    uni_dict[key] = uni_id
    uni_id += 1

for key in sorted(list(l2_dict.keys())):
    uni_dict[key] = uni_id
    uni_id += 1

for key in sorted(list(l3_dict.keys())):
    uni_dict[key] = uni_id
    uni_id += 1

for key in sorted(list(ko2id.keys())):
    uni_dict[key] = uni_id
    uni_id += 1

att_id = []
e_id_list = []

for key in e2id.keys():

    l1,l2,l3 = e2label[key]
    att_id.append(uni_dict[l1])
    e_id_list.append(e2id[key])

    att_id.append(uni_dict[l2])
    e_id_list.append(e2id[key])
    
    att_id.append(uni_dict[l3])
    e_id_list.append(e2id[key])

    if key in e2ko_list:
        for ko in e2ko_list[key]:
            att_id.append(uni_dict[ko])
            e_id_list.append(e2id[key])    

attr2e_index = torch.stack([torch.LongTensor(att_id), torch.LongTensor(e_id_list)])


edge_id = uni_id


reaction2edge_id = {}
edgeid2label = {}
edgeid2true_train = defaultdict(set)
edgeid2true_all = defaultdict(set)

cl2e = defaultdict(list)

train_id_list = []

# 关系到超边的连接
rr_edge = []
re_node = []

# 实体到超边的连接
c_node = []
r_edge = []

print("train data number: %d" % (len(data["train"])))
print("valid data number: %d" % (len(data["valid"])))
print("test data number : %d" % (len(data["test"])))



for c_list, e in data["train"]:
    for c in c_list:
        c_node.append(c2id[c])
        r_edge.append(edge_id)

    re_node.append(e2id[e])
    rr_edge.append(edge_id)

    edgeid2label[edge_id]= e2id[e]
    edgeid2true_train[edge_id].add(e2id[e])
    edgeid2true_all[edge_id].add(e2id[e])
    train_id_list.append(edge_id)

    c_list = sorted(c_list)
    cl2e[tuple(c_list)].append(e)
    edge_id += 1

train_hyper_edge_num = edge_id - (c_num + e_num)
print("train hyper edge number: %d" % train_hyper_edge_num)

max_train_num = edge_id

# 测试和验证的超边
val_c_node = []
val_r_edge = []

val_e_node = []
val_re_edge = []

valid_id_list = []

for c_list, e in data["valid"]:
    for c in c_list:
        val_c_node.append(c2id[c])
        val_r_edge.append(edge_id)
    
    val_e_node.append(e2id[e])
    val_re_edge.append(edge_id)

    edgeid2label[edge_id]=e2id[e]
    edgeid2true_all[edge_id].add(e2id[e])
    c_list = sorted(c_list)
    valid_id_list.append(edge_id)
    cl2e[tuple(c_list)].append(e)
    edge_id += 1


test_id_list = []
for c_list, e in data["test"]:
    for c in c_list:
        val_c_node.append(c2id[c])
        val_r_edge.append(edge_id)

    val_e_node.append(e2id[e])
    val_re_edge.append(edge_id)
    edgeid2label[edge_id]=e2id[e]
    edgeid2true_all[edge_id].add(e2id[e])
    c_list = sorted(c_list)
    test_id_list.append(edge_id)
    cl2e[tuple(c_list)].append(e)
    edge_id += 1

valid_test_edge_num  = edge_id - train_hyper_edge_num - (c_num + e_num)

print("valid hyperedge num: %d" % valid_test_edge_num)


# 构建实体和超边之间的连接
entity_edge = r_edge + val_r_edge
entity = c_node + val_c_node
# v2e = SparseTensor(
#             row=torch.as_tensor(entity_edge, dtype=torch.long) ,
#             col=torch.as_tensor(entity, dtype=torch.long),
#             value=torch.as_tensor(range(0, len(entity)), dtype=torch.long)
# )
# v2e_index = torch.stack([v2e.storage.col(), v2e.storage.row()])
v2e_index = torch.stack([torch.as_tensor(entity, dtype=torch.long), torch.as_tensor(entity_edge, dtype=torch.long)])

# 建立酶和超边之间的连接
e2r_index = torch.stack([torch.LongTensor(rr_edge), torch.LongTensor(re_node)])


print("Relation and Edge: %s" % (str(e2r_index.shape)))
print("Entity and Edge: %s" % (str(v2e_index.shape)))

# 能够完成训练集当中，化合物和超边之间的连接
# 构建训练集当中节点和超边之间的关联
r_edge_temp = [c - c_num - e_num for c in r_edge]
edge2entity_train = coo_matrix((
    np.ones(len(c_node)),
    (np.array(c_node), np.array(r_edge_temp))), shape=(c_num, train_hyper_edge_num)).tocsr()

edge2edge_train = edge2entity_train.T * edge2entity_train
share_entity = edge2edge_train.tocoo()

share_entity_row_entity = share_entity.row 
share_entity_col_entity = share_entity.col 
edge_type_node = np.zeros_like(share_entity_row_entity)


r_edge_temp = [c - c_num - e_num for c in rr_edge]
re_node_temp = [c - c_num  for c in re_node]
edge2rel_train = coo_matrix((
    np.ones(len(re_node_temp)),
    (np.array(re_node_temp), np.array(r_edge_temp))), shape=(e_num, train_hyper_edge_num)).tocsr()

edge2edge_rel_train = edge2rel_train.T * edge2rel_train
shared_rel = edge2edge_rel_train.tocoo()

share_entity_row_rel = shared_rel.row 
share_entity_col_rel = shared_rel.col 
edge_type_node_rel = np.ones_like(share_entity_row_rel)

# train 数据中超边和超边之间的连接
edge_index_row = torch.LongTensor(np.concatenate([share_entity_row_entity,share_entity_row_rel]))
edge_index_col = torch.LongTensor(np.concatenate([share_entity_col_entity,share_entity_col_rel]))
edge_type_train = torch.LongTensor(np.concatenate([edge_type_node, edge_type_node_rel]))
edge_index_train = torch.stack([
    edge_index_row + c_num + e_num,
    edge_index_col +  c_num + e_num
])

# 计算超边和超边之间的平均情况
# edge2number = {}
# for i in range(0,train_hyper_edge_num):
#     edge2number[i] = 0

# for i in range(len(share_entity_row)):
#     edge2number[share_entity_row[i]] += 1
#     # edge2number[share_entity_col[i]] += 1
# number_list = []
# for key in edge2number.keys():
#     number_list.append(edge2number[key])

# print("train graph hyperedge connect number min: %d" % (np.min(number_list)))
# print("train graph hyperedge connect number max: %d" % (np.max(number_list)))
# print("train graph hyperedge connect number mean: %d" % (np.mean(number_list)))
# print("train graph hyperedge connect number mean: %d" % (np.median(number_list)))
# print("train graph hyperedge connect number 1: %d" % (number_list.count(1)))


# 测试和验证数据，超边和实体之间的连接，稀疏矩阵，下标从0开始

val_c_node_temp = val_c_node
val_r_edge_temp = [c - max_train_num for c in val_r_edge]

edge2entity_valid = coo_matrix((
    np.ones(len(val_c_node_temp)),
    (np.array(val_c_node_temp), np.array(val_r_edge_temp))), shape=(c_num, valid_test_edge_num )).tocsr()

valid2train_edge = edge2entity_valid.T * edge2entity_train
share_entity = valid2train_edge.tocoo()
print(share_entity.shape)
share_entity_row = share_entity.row 
share_entity_col = share_entity.col 
edge_type_node = np.zeros_like(share_entity_row)


val_re_edge_temp =[c - max_train_num for c in val_re_edge]
val_re_node_temp =[c - c_num for c in val_e_node]


edge2rel_valid = coo_matrix((
    np.ones(len(val_re_node_temp)),
    (np.array(val_re_node_temp), np.array(val_re_edge_temp))), shape=(e_num, valid_test_edge_num )).tocsr()

valid2train_edge_share_rel = edge2rel_valid.T * edge2rel_train
share_rel = valid2train_edge_share_rel.tocoo()
share_rel_row = share_rel.row 
share_rel_col = share_rel.col 
edge_type_node_rel = np.ones_like(share_rel_row)


edge_type_valid2train = torch.LongTensor(np.concatenate([edge_type_node,edge_type_node_rel]))
edge_index_row = torch.LongTensor(np.concatenate([share_entity_row, share_rel_row]))
edge_index_col = torch.LongTensor(np.concatenate([share_entity_col,share_rel_col]))


# validedge2number = {}

# for i in range(0,valid_test_edge_num):
#     validedge2number[i] = 0

# for i in range(len(share_entity_row)):
#     validedge2number[share_entity_row[i]] += 1

# number_list = []
# for key in validedge2number.keys():
#     number_list.append(validedge2number[key])

# print("valid graph hyperedge connect number min: %d" % (np.min(number_list)))
# print("valid graph hyperedge connect number max: %d" % (np.max(number_list)))
# print("valid graph hyperedge connect number mean: %d" % (np.mean(number_list)))
# print("valid graph hyperedge connect number median: %d" % (np.median(number_list)))
# print("valid graph hyperedge connect number 1: %d" % (number_list.count(1)))

edge_index_valid = torch.stack([
    edge_index_row + max_train_num,  # valid 
    edge_index_col +  c_num + e_num  # train 
])

new_edge_index = torch.cat((edge_index_train, edge_index_valid),dim=-1)
new_edge_type = torch.cat((edge_type_train, edge_type_valid2train),dim=-1)


# 重新构建一个超图数据
# 首先需要将ko 的 id 和 层次lable 进行混合编码：
# 构造一个空白的embedding 作为 0 （这个地方存疑）

graph_info = {
    "train_edge_index": edge_index_train,
    "train_edge_type": edge_type_train,
    "valid_edge_index": new_edge_index,
    "valid_edge_type": new_edge_type,
    "node2edge_index": v2e_index,
    "edge2rel_index": e2r_index,
    "attr2e_index":attr2e_index,
    "c_num": c_num,
    "e_num": e_num,
    "base_node_num":uni_id,
    "max_train_id": max_train_num,
    "max_edge_id":edge_id,
}

train_info = {
    "train_id_list": train_id_list,
    "valid_id_list": valid_id_list,
    "test_id_list": test_id_list,
    "edgeid2label": edgeid2label,
    "edgeid2true_train": edgeid2true_train,
    "edgeid2true_all": edgeid2true_all,
}

torch.save(graph_info,"../pre_handle_data/ce_data_new_graph_info.pkl")
torch.save(train_info,"../pre_handle_data/ce_data_new_train_info.pkl")

# 构建一个单项图
# 读取数据集，确定酶的数量和化合物的数量， 将两个统一编码
# 读取训练集，将其中的每个反应的化合物和超边进行关联
# 记录每个超边和酶之间的关系

# 记录每个反应的酶的lable

# 计算训练图当中超边之间的关系

# 读取测试数据

# 