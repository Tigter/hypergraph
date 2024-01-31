# 读取模型，对embedding进行可视化分析（主要是分析层次结构的表达）
# 读取日志，对loss之类进行可视化分析
# 对数据集进行分析和可视化

from collections import defaultdict
from util.data_process import DataProcesser as DP
import os
from core.TransER import TransER
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging
import torch
import random
import numpy as np
import networkx as nx

def read_dic(file):
    with open(file,encoding='utf-8') as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    return entity2id

def read_cross_data(data_path):
    ins_entity_dic = read_dic(os.path.join(data_path,"in_entity2id.dic"))
    on_entity_dic = read_dic(os.path.join(data_path,"on_entity2id.dic"))
    relation_dic = read_dic(os.path.join(data_path,"cross_relation2id.dic"))
 
    train = read_triples(os.path.join(data_path,"cross/train.txt"),ins_entity_dic,relation_dic,on_entity_dic)
    test = read_triples(os.path.join(data_path,"cross/test.txt"),ins_entity_dic,relation_dic,on_entity_dic)
    return train,test

def read_triples(file_path, h_dic,r_dic,t_dic):
    triples = []
    with open(file_path, encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')  
            triples.append((h_dic[h],r_dic[r],t_dic[t]))
    return triples  

def one_hop_relation(triples,  rel_num,entity_num,relation_num):
    pad_id = relation_num
    entity_r = defaultdict(set)
    for h,r,t in triples:
        entity_r[h].add(r)
    unique_1hop_relations = [
        random.sample(entity_r[i], k=min(rel_num, len(entity_r[i]))) + [pad_id] * (rel_num-min(len(entity_r[i]), rel_num))
        for i in range(entity_num)
    ]
    return unique_1hop_relations

def draw_pca(data):
    pca = PCA(n_components=2)
    result = pca.fit(data)
    print(np.sum(pca.explained_variance_ratio_))
    return result


def any_levelGraph(G):
    # 平均的深度
    # find all leaf node
    leaf_ndoes = set()
    for node in G.nodes():
        if len(list(G.predecessors(node))) == 0:
    
            leaf_ndoes.add(node)
    print("All leaf Node: %d" % len(leaf_ndoes))

    distance = []
    for node in leaf_ndoes:
        print(node)
        length = path_from_leaf(G,node)
        distance.append(length)
    distance = np.array(distance)
    mean_root2leaf = np.mean(distance)

    # 分析一下是不是存在传递属性数据
    count = 0
    for node in G.nodes():
        node_count = n_hop_trans(G, node)
        count += node_count
    print("max depth: %d \n mean depth: %d \n" %(np.max(distance),mean_root2leaf))
    print("Transitive Count: %d " % count)

def type_any(type_data, G):
    # 首先统计有多少个类型
    # 每个实体的类型数量（均值，中位数，最大值，最小值）
    entity2type = {}
    for h,r,t in type_data:
        if h not in entity2type:
            entity2type[h] = [t]
        else:
            entity2type[h].append(t)
    
    count_list = []
    type_with_level = 0
    for keys in entity2type.keys():
        count = len(entity2type[keys])
        count_list.append(count)
        if count > 1:
            for node1 in entity2type[keys]:
                for node2 in entity2type[keys]:
                    if node1 != node2:
                        path =nx.lowest_common_ancestor(G,node1,node2)
                        if path == node1:
                            type_with_level += 1

    count_list = np.arrasy(count_list)

    print("Entity with Type: %d" % len(entity2type))
    print("Type count max: %d \nType count mean: %.2f \n" % (np.max(count_list),np.mean(count_list)))
    print("Type count median: %d \nType count percentile: %.2f \n" % (np.median(count_list),np.percentile(count_list,75)))
    print("type with level: %d" % type_with_level)


def n_hop_trans(G, node):
    count = 0
    for node1 in list(G.successors(node)):
        for node2 in list(G.successors(node1)):
            if (node, node2) in G.edges():
                count += 1
    return count

def path_from_leaf(G,node):

    nodes = []
    nodes.append(node)
    result = 1
    now = 0
    distance = []
    distance.append(1)
    while len(nodes) != 0 :
        node = nodes.pop()
        now = distance.pop()
        for next in list(G.successors(node)):
            nodes.append(next)
            distance.append(now+1)
        if list(G.successors(node)) == 0:
            result = max(result,now)
    return result
        

def dataset_any(cross_train,on,level):
    entity_dic = on.entity2id
    relation_dic = on.relation2id
    print("Ons entity num: %d\nOns relation num: %d" % (len(entity_dic), len(relation_dic)))

    level_triples = []
    equal = 0
    inverse = 0
    for h,r,t in level.all_true_triples:
        if h == t :
            equal += 1
        elif (t,r,h) in level.all_true_triples:
            inverse += 1
        else:
            level_triples.append((h,r,t))
            
    print("Level a is a: %d" % equal) 
    print("Level a is b: %d" % inverse)

    level_entity = set()
    edgeList = []
    for h,r,t in level_triples:
        level_entity.add(h)
        level_entity.add(t)
        edgeList.append((h,t))
    print("level entity num: %d" % len(level_entity))

    # 分析一下层次数据的层次结构
    G = nx.DiGraph(edgeList)
    any_levelGraph(G)
    # type_any(cross_train,G)


if __name__=="__main__":
    data_path = "/home/skl/yl/yago/"
    mode_path = "/home/skl/yl/models/Self_CT_iter_upe_01/futune/"
    rel_num = 1

    ins = DP(os.path.join(data_path,"ins"),idDict=True, reverse=False)
    on = DP(os.path.join(data_path,"on"),idDict=True, reverse=False)

    cross_train,cross_test = read_cross_data(data_path)
    level = DP(os.path.join(data_path,"level"),idDict=True, reverse=False)
    ins_e2r = one_hop_relation(ins.all_true_triples, rel_num, ins.nentity, ins.nrelation)
    ons_e2r = one_hop_relation(on.all_true_triples, rel_num, on.nentity, on.nrelation)
    # logging.info('init: %s' % mode_path)
    # checkpoint = torch.load(os.path.join(mode_path, 'checkpoint'))
    # print(checkpoint.keys())
    # ins_dim = 300
    # ons_dim =  200
    # model = TransER(on.nentity,on.nrelation+1,ons_dim,ins.nentity,ins.nrelation+1,ins_dim,ins_e2r,ons_e2r)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # ons_entity = model.on_entity_embedding.data
    # ons_relation = model.on_relation_embedding.data
    # ons_e_r = model.on_entity_r.data
    # ins_entity = model.in_entity_embedding.data
    # ins_relation = model.in_relation_embedding.data

    # ons_entity_pca = draw_pca(ons_entity)
    # ons_relation_pca = draw_pca(ons_relation)
    # ins_entity_pca = draw_pca(ins_entity)
    # ins_relation_pca = draw_pca(ins_relation)

    cross = cross_train + cross_test

    dataset_any(cross,on,level)

   



   
   
   