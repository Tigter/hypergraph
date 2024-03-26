
import pickle
from collections import defaultdict
# from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix
import torch
import numpy as np
import json 

# 处理酶和化合物的数据
print("begin")
with open("./bkms_new.pkl",'rb') as f:
    data = pickle.load(f)
    f.close()

print("load finised")
print(data.keys())

train_data = data["train"]
valid_data = data["valid"]
test_data = data["test"]

def filter_train(train_set, valid_data):
    new_valid_data = []
    for c_list, e in valid_data:
        new_clist = []
        for c in c_list:
            if (c,e) in train_set: continue
            new_clist.append(c)
        if len(new_clist) != 0:
            new_valid_data.append((new_clist, e))
    return new_valid_data

train_set = set()
for c_list, e in train_data:
    for c in c_list:
        train_set.add((c,e))

valid_data = filter_train(train_set, valid_data)
test_data = filter_train(train_set, test_data)



def build_id_dict(data):
    cset = set()
    eset = set()
    for c_list, e in data["double_train"]:
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

def add_aux_data():

    with open("./aux_data.json", 'r') as f:
        aux_data = json.load(f)

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
    return uni_dict, uni_id,e2label,e2ko_list

uni_dict, uni_id,e2label,e2ko_list = add_aux_data()

def add_attr_id_to_e():
    att_id = []
    e_id_list = []

    eid2label = []

    for key in e2id.keys():

        l1,l2,l3 = e2label[key]
        att_id.append(uni_dict[l1])
        e_id_list.append(e2id[key])

        att_id.append(uni_dict[l2])
        e_id_list.append(e2id[key])
        
        att_id.append(uni_dict[l3])
        e_id_list.append(e2id[key])
        eid2label.append([ e2id[key], uni_dict[l1], uni_dict[l2], uni_dict[l3]])

        if key in e2ko_list:
            for ko in e2ko_list[key]:
                att_id.append(uni_dict[ko])
                e_id_list.append(e2id[key])    

    attr2e_index = torch.stack([torch.LongTensor(att_id), torch.LongTensor(e_id_list)])
    return attr2e_index,eid2label

attr2e_index,eid2label = add_attr_id_to_e()



# 开始构建超图
base_node_num = uni_id
edge_id = uni_id

print("base node num: %d" % base_node_num)


# 这里修改共享酶的超边之间的连接：
def share_e(e2edgeId, c_num):
    node1 = []
    node2 = []
    edge_type = []
   
    count = 0
    for e in e2edgeId.keys():
        edge_list = list(e2edgeId[e])
        for i in range(len(edge_list)):
            for j in range(i+1, len(edge_list)):
                edge1 = edge_list[i]
                edge2 = edge_list[j]

                node1.append(edge1)
                node2.append(edge2)

                node1.append(edge2)
                node2.append(edge1)
                edge_type.append(e - c_num)
                edge_type.append(e - c_num)
                count +=1 
                if count % 10000000 == 0: 
                    print("count : %d" % count)

    node1, node2, edge_type  = torch.LongTensor(node1),torch.LongTensor(node2),torch.LongTensor(edge_type)
    return node1, node2, edge_type


def build_single_graph(data, c2id, e2id, base_node_num):
    edge_id = base_node_num

    e2edgeId = defaultdict(set)
    edgeid2label = {}
    edgeid2true_train = defaultdict(set)
    edgeid2true_all = defaultdict(set)
    cl2e = defaultdict(list)
    cl2edge_id = {}

    rr_edge = []
    re_node = []

    # 实体到超边的连接
    c_node = []
    r_edge = []
    train_id_list = []

    c2singleId = defaultdict(list) 

    for c_list, e in data["sing_train"]:
        for c in c_list:
            c_node.append(c2id[c])
            r_edge.append(edge_id)
            c2singleId[c2id[c]].append(edge_id)

        re_node.append(e2id[e])
        rr_edge.append(edge_id)

        e2edgeId[e2id[e]].add(edge_id)

        edgeid2label[edge_id]= e2id[e]
        edgeid2true_train[edge_id].add(e2id[e])
        edgeid2true_all[edge_id].add(e2id[e])
        train_id_list.append(edge_id)

        c_list = sorted(c_list)
        cl2e[tuple(c_list)].append(e)
        
        cl2edge_id[tuple(c_list)] = edge_id
        edge_id += 1

    train_hyper_edge_num = edge_id - base_node_num
    print("train hyper edge number: %d" % train_hyper_edge_num)

    max_train_num = edge_id

    # 测试和验证的超边
    val_c_node = []
    val_r_edge = []

    val_e_node = []
    val_re_edge = []

    valid_id_list = []

    for c_list, e in data["sing_valid"]:
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
    for c_list, e in data["sing_test"]:
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

    valid_test_edge_num  = edge_id - train_hyper_edge_num - base_node_num

    print("valid hyperedge num: %d" % valid_test_edge_num)

    # 构建实体到超边的连接:
    entity_edge = r_edge + val_r_edge
    entity = c_node + val_c_node
    v2e_index = torch.stack([torch.as_tensor(entity, dtype=torch.long), torch.as_tensor(entity_edge, dtype=torch.long)])

    entity_edge = r_edge 
    entity = c_node 
    v2e_train_index = torch.stack([torch.as_tensor(entity_edge, dtype=torch.long), torch.as_tensor(entity, dtype=torch.long)])

    entity_edge = r_edge + val_r_edge
    entity = c_node + val_c_node
    v2e_valid_index = torch.stack([torch.as_tensor(entity_edge, dtype=torch.long), torch.as_tensor(entity, dtype=torch.long)])



    # 建立酶和超边之间的连接: 只在训练图当中, 当前无用
    e2r_index = torch.stack([torch.LongTensor(rr_edge), torch.LongTensor(re_node)])

    print("Relation and Edge: %s" % (str(e2r_index.shape)))
    print("Entity and Edge: %s" % (str(v2e_index.shape)))

    # 共享实体的超边连接
    
    r_edge_temp = [c - base_node_num for c in r_edge]

    edge2entity_train = coo_matrix((
        np.ones(len(c_node)),
        (np.array(c_node), np.array(r_edge_temp))), shape=(c_num, train_hyper_edge_num)).tocsr()

    edge2edge_train = edge2entity_train.T * edge2entity_train
    share_entity = edge2edge_train.tocoo()

    share_entity_row_entity = torch.LongTensor(share_entity.row) + base_node_num
    share_entity_col_entity = torch.LongTensor(share_entity.col) + base_node_num

    edge_type_node = torch.LongTensor([e_num for i in range(len(share_entity_col_entity))])

    # 共享酶之间的超边
    shareE_node1 , shareE_node, shareE_edge_type = share_e(e2edgeId, c_num)
    edge_index_row = torch.cat([share_entity_row_entity, shareE_node1],dim=-1)
    edge_index_col = torch.cat([share_entity_col_entity, shareE_node],dim=-1)
    edge_type_train = torch.cat([edge_type_node, shareE_edge_type],dim=-1)
    edge_index_train = torch.stack([
        edge_index_row ,
        edge_index_col
    ])

    # 把测试集和训练图连接起来
    val_c_node_temp = val_c_node
    val_r_edge_temp = [c - max_train_num for c in val_r_edge]

    edge2entity_valid = coo_matrix((
        np.ones(len(val_c_node_temp)),
        (np.array(val_c_node_temp), np.array(val_r_edge_temp))), shape=(c_num, valid_test_edge_num )).tocsr()

    valid2train_edge = edge2entity_valid.T * edge2entity_train
    share_entity = valid2train_edge.tocoo()
    share_entity_row = torch.LongTensor(share_entity.row)
    share_entity_col = torch.LongTensor(share_entity.col)

    edge_type_node_valid = torch.LongTensor(np.zeros_like(share_entity_row)) + e_num

    edge_index_valid = torch.stack([
        share_entity_row + max_train_num,  # valid 
        share_entity_col + base_node_num  # train 
    ])

    new_edge_index = torch.cat((edge_index_train, edge_index_valid),dim=-1)
    new_edge_type = torch.cat((edge_type_train, edge_type_node_valid),dim=-1)

    sing_graph = {
        "train_edge_index": edge_index_train,
        "train_edge_type": edge_type_train,
        "valid_edge_index": new_edge_index,
        "valid_edge_type": new_edge_type,
        "node2edge_index": v2e_index,
        "v2e_train_index": v2e_train_index,
        "v2e_valid_index": v2e_valid_index,
        "edge2rel_index": e2r_index,
        "max_train_id": max_train_num,
        "max_edge_id":edge_id,
        "base_node_num":uni_id,
        "cl2edge_id": cl2edge_id
    }
    train_info = {
        "train_id_list": train_id_list,
        "valid_id_list": valid_id_list,
        "test_id_list": test_id_list,
        "edgeid2label": edgeid2label,
        "edgeid2true_train": edgeid2true_train,
        "edgeid2true_all": edgeid2true_all,
        "c2singleId":c2singleId
    }
    return sing_graph, train_info

sing_graph, train_info =  build_single_graph(data, c2id, e2id, base_node_num)


def build_doube_graph(data, c2id,e2id, clist2id, base_node_num):

    edge_id = base_node_num
    rr_edge = []
    re_node = []

    # 实体到超边的连接
    c_node = []
    r_edge = []
    e2edgeId = defaultdict(set)
    train_id_list = []
    single2double = []
    for left, right, e in data["train"]:
        c_list = left + right
        for c in c_list:
            c_node.append(c2id[c])
            r_edge.append(edge_id)

        re_node.append(e2id[e])
        rr_edge.append(edge_id)
        train_id_list.append(edge_id)
        e2edgeId[e2id[e]].add(edge_id)

        c_list = sorted(c_list)
        if len(left) > 0:
            single2double.append([clist2id[tuple(sorted(left))] - base_node_num, edge_id]) 
        if len(right) > 0:
            single2double.append([clist2id[tuple(sorted(right))] - base_node_num, edge_id]) 

        edge_id += 1

    entity_edge = r_edge
    entity = c_node
    v2e_index = torch.stack([torch.as_tensor(entity, dtype=torch.long), torch.as_tensor(entity_edge, dtype=torch.long)])

    r_edge_temp = [c - base_node_num for c in r_edge]

    train_hyper_edge_num = edge_id - base_node_num
    edge2entity_train = coo_matrix((
        np.ones(len(c_node)),
        (np.array(c_node), np.array(r_edge_temp))), shape=(c_num, train_hyper_edge_num)).tocsr()


    edge2edge_train = edge2entity_train.T * edge2entity_train
    share_entity = edge2edge_train.tocoo()

    share_entity_row_entity = torch.LongTensor(share_entity.row) + base_node_num
    share_entity_col_entity = torch.LongTensor(share_entity.col) + base_node_num

    edge_type_node = torch.LongTensor([e_num for i in range(len(share_entity_col_entity))])


    shareE_node1 , shareE_node, shareE_edge_type = share_e(e2edgeId, c_num)

    edge_index_row = torch.cat([share_entity_row_entity, shareE_node1],dim=-1)
    edge_index_col = torch.cat([share_entity_col_entity, shareE_node],dim=-1)
    edge_type_train = torch.cat([edge_type_node, shareE_edge_type],dim=-1)
    
    edge_index_train = torch.stack([
        edge_index_row ,
        edge_index_col
    ])


    sorted(single2double, key=lambda x:x[0])

    double_data = {
        "double_v2e":v2e_index,
        "double_train_edge_type": edge_type_train,
        "double_train_edge_index": edge_index_train,
        "double_train_id_list": train_id_list,
        "single2double": single2double,
        "double_max_edge_id": edge_id
    }
    return double_data


double_graph = build_doube_graph(data, c2id, e2id, sing_graph["cl2edge_id"], base_node_num)

# 重新构建一个超图数据
# 首先需要将ko 的 id 和 层次lable 进行混合编码：
# 构造一个空白的embedding 作为 0（这个地方存疑）

graph_info = {
    "c_num": c_num,
    "e_num": e_num,
    "base_node_num":uni_id,
    "eid2label":eid2label,
    "attr2e_index":attr2e_index,
    "single_train":data["sing_train"],
    "single_valid":data["sing_valid"],
    "single_test":data["sing_test"],
    "c2id": c2id,
    "e2id": e2id,
}
graph_info.update(sing_graph)
graph_info.update(double_graph)

# torch.save(graph_info,"../pre_handle_data/ce_data_single_base_v4_graph_info.pkl")
# torch.save(train_info,"../pre_handle_data/ce_data_single_base_v4_train_info.pkl")
