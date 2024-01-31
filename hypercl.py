from collections import defaultdict
from curses import init_pair
from torch.utils.tensorboard import SummaryWriter   
import time
from core.HyperKGE import HyperKGEConfig 
from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP
import os
from core.ComplEx import ComplEx
from core.TripleCL import MLPTripleEncoder,ContrastiveLoss,InfoNCELoss
import numpy as np
from scipy.sparse import csr_matrix
import yaml 
import math
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,MultiStepLR
from util.dataloader import OneShotIterator
import torch.nn as nn
from util.cl_dataloader import NewOne2OneDataset,CLDataset,OGOneToOneTestDataset,OgOne2OneDataset
from loss import NSSAL,MRL,NSSAL_aug,MRL_plus,NSSAL_sub
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR
from core.HyperCL import OurModel
from torchstat import stat
import argparse
import random

#!!!!!!!!!!!!!!!!!!!!!!!!#
# 设置是否使用原始的超图
old_hyper = True
og_vio = True

# old version hyper graph adj
def get_incidence_matrix(all_triples, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    triple2edge_id = {}
    idset = set()
    for j in range(len(all_triples)):
        triple = np.unique(all_triples[j])
        length = len(triple)
        s = indptr[-1]
        indptr.append((s + length))
        triple2edge_id[all_triples[j]] = j
        for i in range(length):
            idset.add(triple[i])
            indices.append(triple[i])  # 默认id 是从1开始的？
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_triples), n_node),dtype=np.int32)
    # check
    print(len(idset))
    for i in range(n_node):
        if i not in idset:
            print(i)
    return matrix,triple2edge_id


def cal_adj(all_triples, n_node):
    print("Train Triple Num:%d" % len(all_triples))

    H_T,triple2edge_id = get_incidence_matrix(all_triples, n_node) # shape = (edge, node)
    
    BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))  # shape = (node, edge)
    BH_T = BH_T.T        # shape = (edge,node)

    DH = H_T.multiply(1.0 / (H_T.T).sum(axis=1).reshape(1, -1))
    DH = DH.T  # node,edge

    DHBH_T = DH.dot(BH_T)

    # DHBH_T = DHBH_T.dot(BH_T)
    adjacency = DHBH_T.tocoo()
    print(adjacency.shape)
    print("Caculat Adj Over")
    return adjacency,triple2edge_id



# new hyper graph graph infor
def buil_graph(trin_triples, n_entity, n_node):
    node = []
    edge = []
    triple2edge_id = {}

    n_hyper_edge = 0
    for h,r,t in trin_triples:
        node.append(h)
        edge_idx = n_hyper_edge + n_node
        edge.append(edge_idx)
        node.append(r)
        edge.append(edge_idx)
        node.append(t)
        edge.append(edge_idx)
        triple2edge_id[(h,r,t)] = edge_idx - n_node
        n_hyper_edge += 1
    
    edge_index_v2e = torch.tensor([node, edge])
    _, sorted_idx = torch.sort(edge_index_v2e[0])
    edge_index_v2e = edge_index_v2e[:, sorted_idx].type(torch.LongTensor)
    return edge_index_v2e, n_hyper_edge,triple2edge_id

# 构建超图的方式：关系的头构建为一个超边，关系的尾构建为一个超边
# 设定超图的图的构建方式，共享实体或者共享边
def build_relation_graph(triin_triples, n_entity, n_node):
    node = []
    edge = []

    r2h = defaultdict(set)
    r2t = defaultdict(set)
    for h,r,t in trin_triples:
        r2h[r].add(h)
        r2t[r].add(t)
    r_list = set(list(r2h.keys())+list(r2t.keys()))
    n_hyper_edge = 0
    rh2id = {}
    rt2id = {}
    for r in r_list: # 所有的关系
        # 关系的头是一个超边：每个头和超边
        edge_idx = n_hyper_edge + n_node
        rh2id[r]= n_hyper_edge
        for h in r2h[r]:
            node.append(h)
            edge.append(edge_idx)
        n_hyper_edge += 1
        # 关系的尾是一个超边：每个尾实体和超边连接
        edge_idx = n_hyper_edge + n_node
        rt2id[r] = n_hyper_edge
        for t in r2t[r]:
            node.append(t)
            edge.append(edge_idx)
         n_hyper_edge += 1
    edge_index_v2e = torch.tensor([node, edge])
    _, sorted_idx = torch.sort(edge_index_v2e[0])
    edge_index_v2e = edge_index_v2e[:, sorted_idx].type(torch.LongTensor)

    # 共享节点构建超图的连接
    hyper_graph_node1 = []
    hyper_graph_node2 = []
    for r1 in r_list:
        for r2 in r_list:
            if r1 == r2: continue
            if r2h[r1] & r2h[r2]:
                hyper_graph_node1.append(rh2id[r1])
                hyper_graph_node2.append(rh2id[r2])
            if r2t[r1] & r2t[r2]:
                hyper_graph_node1.append(rt2id[r1])
                hyper_graph_node2.append(rt2id[r2])

    hyper_graph_share_entity = torch.tensor([hyper_graph_node1, hyper_graph_node1])
     _, sorted_idx = torch.sort(hyper_graph_share_entity[0])
    hyper_graph_share_entity = hyper_graph_share_entity[:, sorted_idx].type(torch.LongTensor)

    # 共享关系构建超图的连接
    hyper_graph_node1 = []
    hyper_graph_node2 = []
    for r1 in r_list:
        hyper_graph_node1.append(rh2id[r1])
        hyper_graph_node2.append(rt2id[r1])

    hyper_graph_share_relation = torch.tensor([hyper_graph_node1, hyper_graph_node2])
     _, sorted_idx = torch.sort(hyper_graph_share_relation[0])
    hyper_graph_share_relation = hyper_graph_share_relation[:, sorted_idx].type(torch.LongTensor)

    return edge_index_v2e, n_hyper_edge, hyper_graph_share_entity, hyper_graph_share_relation


def logging_log(step, logs,writer):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        writer.add_scalar(metric, metrics[metric], global_step=step, walltime=None)
    logset.log_metrics('Training average', step, metrics)



def ndcg_at_k(idx):
    idcg_k = 0
    dcg_k = 0
    n_k = 1
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)
    dcg_k += 1 / math.log(idx + 2, 2)
    return float(dcg_k / idcg_k)   

def train_double(train_iterator,model,cuda, args,loss_funcation=None):
    model.train()
    h,r,t, value = next(train_iterator)
    if cuda:
        h = h.cuda()
        r = r.cuda()
        t = t.cuda()
        value = value.cuda()
    score,regu = model(h, r, t)

    regu_loss  = regu*args.loss_weight

    if loss_funcation == None:
        score_loss = model.loss(score, value)
        loss = score_loss + regu_loss
        log = {
            '_loss': loss.item(),
            '_regu_loss': regu_loss.item(),
            '_socre_loss': score_loss.item(),
        }
    else:
        positive_score = score[...,0].unsqueeze(1)
        negative_score = score[...,1:]
        loss1 = loss_funcation(positive_score,negative_score)
        loss = regu_loss + loss1
        log = {
            '_loss': loss.item(),
            '_regu': regu_loss.item()
        }   
    return log, loss


def train_cl(train_iterator,model,cuda, triple_encoder,loss_funcation=None):
    model.train()
    triple_encoder.train()
    base, pos,neg = next(train_iterator)
    if cuda:
        base = base.cuda()
        pos = pos.cuda()
        neg = neg.cuda()
        
    pos_h,pos_r,pos_t = torch.chunk(pos, 3,dim=-1)
    neg_h,neg_r,neg_t = torch.chunk(neg, 3,dim=-1)

    true_triple_emb = model.get_triple_embedding(base[...,0], base[...,1], base[...,2])
    pos_triple_emb = model.get_triple_embedding( pos_h,pos_r,pos_t)
    neg_triple_emb = model.get_triple_embedding( neg_h,neg_r,neg_t)

    # print("*************************************Top*****************************************")
    # print(torch.sum(true_triple_emb,dim=-1))
    # print(torch.sum(pos_triple_emb,dim=-1))
    # print(torch.sum(neg_triple_emb,dim=-1))
    # loss = loss_funcation(true_triple_emb,pos_triple_emb,neg_triple_emb)
 
    true_emb = triple_encoder(true_triple_emb)
    pos_emb = triple_encoder(pos_triple_emb)
    neg_emb = triple_encoder(neg_triple_emb)

    # # print("*************************************split-line*****************************************")
    loss = loss_funcation(true_emb,pos_emb,neg_emb)
    # print("*************************************Buttion*****************************************")
    
    log = {
        'loss':loss.item()
    }
    return log, loss

def test_model_conv(model, test_triples, all_true_triples,nentity,nrelation,cuda=True, inverse=False, onType=None,head_num=None,tail_num=None,level=None,vio_test=False):
    '''
    Evaluate the model on test or valid datasets
    '''
    test_batch_size = 1
    model.eval()
    test_dataloader_tail = DataLoader(
        OGOneToOneTestDataset(
            test_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'hr_t'
        ), 
        batch_size=test_batch_size,
        num_workers=1, 
        collate_fn=OGOneToOneTestDataset.collate_fn
    )
    test_dataloader_head = DataLoader(
        OGOneToOneTestDataset(
            test_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'h_rt',
        ), 
        batch_size=test_batch_size,
        num_workers=1, 
        collate_fn=OGOneToOneTestDataset.collate_fn
    )
    if not onType is None:
        if onType == 'head':
            test_dataset_list = [test_dataloader_head]
        else:
            test_dataset_list = [test_dataloader_tail]
    else:
        if not inverse:
            test_dataset_list = [test_dataloader_tail,test_dataloader_head]
        else:
            test_dataset_list = [test_dataloader_tail]
    logs = []
    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])
    count = 0
    ndcg_1 = 0
    torch.cuda.empty_cache()
    model.buil_embedding_for_test()
    # print(total_steps)
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for postive, samples, filter_bias, mode in test_dataset:
                batch_size = postive.shape[0]
                if cuda:
                    postive = postive.cuda()
                    samples = samples.cuda()
                    filter_bias = filter_bias.cuda()
                
                h,r,t = torch.chunk(postive,3,dim=-1)
                test_h, test_r,test_t =  torch.chunk(samples,3,dim=-1)
                negative_score = model.predict(test_h, test_r,test_t)
                if mode == 'hr_t':
                    positive_arg = t
                else:
                    positive_arg = h
            
                score = negative_score + filter_bias
                argsort = torch.argsort(score, dim = 1, descending=True)
                
                for i in range(batch_size):
                    count = count + 1
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1
                    ranking = 1 + ranking.item()

                    if vio_test:
                        ndcg_1 += ndcg_at_k(ranking)
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })
                step += 1
    model.clean_hyper_node_embd()
    torch.cuda.empty_cache()
    
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    metrics["Test Count"] = count
    if vio_test:
        return ndcg_1/count
    else:
        return metrics
    
def test_true_score(model, test_triples, triple2edge=None):
    '''
    Evaluate the model on test or valid datasets
    '''
    test_batch_size = 1
    model.eval()
    logs = []
    step = 0
    count = 0
    ndcg_1 = 0
    torch.cuda.empty_cache()
    model.buil_embedding_for_test()
    # print(total_steps)
    result = []
    count = 0
    with torch.no_grad():
        for h,r,t in test_triples:
            count += 1
            if count % 1000 == 0:
                break
            h_t = torch.LongTensor([h])
            r_t = torch.LongTensor([r])
            t_t = torch.LongTensor([t])
            if triple2edge != None:
                index = triple2edge[(h,r,t)]
                index = torch.LongTensor([index])
                index = index.cuda()
            else:
                index = None
            index=None
            if cuda:
                h_t = h_t.cuda()
                r_t = r_t.cuda()
                t_t = t_t.cuda()

            score = model.predict(h_t, r_t, t_t,index)
            result.append((h,r,t,score.item()))

           
    model.clean_hyper_node_embd()
    torch.cuda.empty_cache()
    with open("result2.txt",'w',encoding='utf-8') as f:
        for h,r,t,score in result:
            f.write("(%d\t%d\t%d):%.5f\n" %(h,r,t,score))


def set_config(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--valid', action='store_true', help='valid model')
    
    parser.add_argument('--max_step', type=int,default=200001, help='最大的训练step')
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_step", type=int, default=10000)
    parser.add_argument("--neg_size",type=int, default=256)
    parser.add_argument("--gamma", type=float, default=20)
    parser.add_argument("--adversial_temp", type=float, default=0.5)

    parser.add_argument("--dim", type=int, default=200)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--decay", type=float)
    parser.add_argument("--warm_up_step", type=int, default=50000)

    parser.add_argument("--loss_function", type=str)

    # HAKE 模型的混合权重
    parser.add_argument("--mode_weight",type=float,default=0.5)
    parser.add_argument("--phase_weight",type=float,default=0.5)

    parser.add_argument("--g_type",type=int,default=5)
    parser.add_argument("--g_level",type=int,default=5)

    parser.add_argument("--model",type=str)
    parser.add_argument("--init",type=str)
    parser.add_argument("--configName",type=str)

    parser.add_argument("--g_mode",type=float,default=0.5)
    parser.add_argument("--g_phase",type=float,default=0.5)

    # RotPro 约束参数配置
    parser.add_argument("--gamma_m",type=float,default=0.000001)
    parser.add_argument("--alpha",type=float,default=0.0005)
    parser.add_argument("--beta",type=float,default=1.5)
    parser.add_argument("--train_pr_prop",type=float,default=1)
    parser.add_argument("--loss_weight",type=float,default=1)

    # 选择数据集
    parser.add_argument("--level",type=str,default='ins')
    parser.add_argument("--data_reverse",action='store_true')

    return parser.parse_args(args)

def save_embedding(emb_map,file):
    for key in emb_map.keys():
        np.save(
            os.path.join(file,key),emb_map[key].detach().cpu().numpy()
        )


def build_dataset_for_cl(data_path, args, cl_nsize,og_vio=False):
    # data_path: train.txt valid.txt test.txt 
    # 关系的id > 实体的id
    print(data_path)
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_reverse,unify_code=True)
    
    on.splitKG_by_relationType()
    on.data_aug_relation_pattern()

    if old_hyper:
        graph,triple2edge = cal_adj(on.train,on.nentity+on.nrelation)
        edge_index_v2e = torch.sparse.FloatTensor(torch.LongTensor([graph.row.tolist(), graph.col.tolist()]),
                                torch.FloatTensor(graph.data.astype(np.float64)))
        edge_index_v2e.detach_()  # 定点到超边的
    else:
        logging.info("Start build Graph")
        edge_index_v2e, n_hyper_edge, hyper_graph_share_entity, hyper_graph_share_relation =  buil_graph(on.trained, on.nentity, on.nentity+on.nrelation)
        print('Edge_Num: ',n_hyper_edge)
        logging.info("build Graph Over")
        edge_index_v2e.detach_()
    
    # 当使用RotateE loss 时 random 参数需要设置为False，这样默认第一个是gt 的index
    # 这里采用生成triple +  0-1 lable 形式的数据，triple 可能替换头or尾，不需要双向训练
    base_dataset =NewOne2OneDataset(on.train,on.nentity,on.nrelation,init_value=-1,n_size=n_size,random=True,triple2Edge=None)
    train_dataloader = DataLoader(
        base_dataset,
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=NewOne2OneDataset.collate_fn
    )
    on_train_iterator = OneShotIterator(train_dataloader)

    conv_dataloader = DataLoader(
        CLDataset(on.conf_kg,on.nentity,on.nrelation,p_size=1,n_size=cl_nsize,po_dict=on.conf2pos,relation_p=base_dataset.pofHead,
        funct_idset=on.func_id,asy_idset=on.asy_id,po_type='conf',triple2edge=triple2edge),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=CLDataset.collate_fn
    )
    conv_iterator = OneShotIterator(conv_dataloader)
  
    if not og_vio:
        func_dataloader = DataLoader(
            CLDataset(on.vio_kg,on.nentity,on.nrelation,p_size=cl_nsize,n_size=cl_nsize,po_dict=None,
                funct_idset=on.func_id,asy_idset=on.asy_id, po_type='Vio',asyDict=on.asymR2paire,relation_p=base_dataset.pofHead,
                funcDict=on.funcR2triples,triple2edge=triple2edge),
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=CLDataset.collate_fn
        )
        func_iterator = OneShotIterator(func_dataloader)
        data_iterator={
            'base':on_train_iterator,
            'conf': conv_iterator,
            'vio':func_iterator,
        }
    else:
        func_dataloader = DataLoader(
            OgOne2OneDataset(on,on.nentity,on.nrelation,n_size=n_size, Datatype='Func',relation_p=base_dataset.pofHead),
            batch_size=args.batch_size//2,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=OgOne2OneDataset.collate_fn
        )
        func_iterator = OneShotIterator(func_dataloader)

        asys = DataLoader(
            OgOne2OneDataset(on,on.nentity,on.nrelation,n_size=n_size, Datatype='Asy',relation_p=base_dataset.pofHead),
            batch_size=args.batch_size//2,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=OgOne2OneDataset.collate_fn
        )
        asys_iterator = OneShotIterator(asys)
    
        data_iterator={
            'base':on_train_iterator,
            'conf': conv_iterator,
            'vio_func':func_iterator,
            'vio_asy': asys_iterator
        }
 
    return on, data_iterator,edge_index_v2e,triple2edge

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__=="__main__":
    # 读取4个数据集
    args = set_config()
    with open('./config/cl.yml','r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        baseConfig = config['baseConfig']
        modelConfig = config[args.configName]

    cuda = baseConfig['cuda']
    init_step  = 0
    save_steps = baseConfig['save_step']
    n_size = modelConfig['n_size']
    log_steps = 1000
    root_path = os.path.join("/home/skl/yl/models/",args.save_path)
    args.save_path = root_path
    args.loss_weight = modelConfig['reg_weight']
    args.batch_size = modelConfig['batch_size']
    init_path = args.init
    max_step   = modelConfig['max_step']
    batch_size = args.batch_size
    test_step = modelConfig['test_step']
    dim = modelConfig['dim']
    lr = modelConfig['lr']
    decay = modelConfig['decay']

    args.data_reverse = modelConfig['data_reverse']

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(os.path.join(root_path,'log')):
        os.makedirs(os.path.join(root_path,'log'))
    
    writer = SummaryWriter(os.path.join(root_path,'log'))


    if args.train:
        logset.set_logger(root_path,"init_train.log")
    else:
        logset.set_logger(root_path,'test.log')
    
    # 读取数据集
    if modelConfig['dataset'] == 'YAGO':
        instance_data_path ="/home/skl/yl/data/YAGO3-1668k/yago_insnet/"
    elif modelConfig['dataset'] == 'FB15-237':
        
        instance_data_path ="/home/skl/yl/data/FB15k-237"
    elif modelConfig['dataset'] == 'NELL-995':
        instance_data_path ="/home/skl/yl/data/NELL-995"
    elif modelConfig['dataset'] == 'test':
        instance_data_path ='/home/skl/yl/data/YAGO3-1668k/yago_new_ontonet/'
    og_vio = True
    instance_dataset, instance_ite,edge_index_v2e,triple2edge = build_dataset_for_cl(instance_data_path, args, cl_nsize=modelConfig['cl_nsize'],og_vio=og_vio)

    conf_cl_loss = InfoNCELoss(modelConfig['cl_temp'])
    # base_loss_funcation = NSSAL(modelConfig['gamma'])
    base_loss_funcation = nn.BCELoss()

    if og_vio:
        vio_cl_loss =None
    else:
        vio_cl_loss = InfoNCELoss(modelConfig['cl_temp'])

    if cuda:
        edge_index_v2e = edge_index_v2e.cuda()

    # graph_tensor = graph_tensor.cuda()
    #  ！！！！！！！！！！！！！！这里采用不用的方法，需要输入不同的构图方式！！！！！！！

    hyperConfig = HyperKGEConfig()
    hyperConfig.embedding_dim = modelConfig['dim']
    model = OurModel(
        adjacency=edge_index_v2e,n_node=instance_dataset.nentity+instance_dataset.nrelation,layers=modelConfig['layer'],
        emb_size = dim,batch_size=args.batch_size,hyperkgeConfig=hyperConfig
    )

    if cuda:
        model = model.cuda()


    optimizer = torch.optim.Adam([
        {'params':filter(lambda p: p.requires_grad, model.parameters())},
        ], lr=lr
    )
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    result = get_parameter_number(model)
    print("模型总大小为：",result["Total"])
    # 如果-有保-存模-型则，读取-模型,进行-测试
    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        init_step = checkpoint['step']

    logging.info('Model: %s' % modelConfig['name'])
    logging.info('Instance nentity: %s' % instance_dataset.nentity)
    logging.info('Instance nrelatidataset. %s' % instance_dataset.nrelation)
    logging.info('max step: %s' % max_step)
    logging.info('init step: %s' % init_step)
    logging.info('lr: %s' % lr)

    # 设置学习率更新策略
    lr_scheduler = MultiStepLR(optimizer,milestones=[50000,100000,200000], gamma=decay)
    logsInstance = []
    logsTypeOf= []
    logsSubOf = []
    logAll = []
   
    stepW = 0
    bestModel = {
        "MRR":0,
        "MR":1000000000,
        "HITS@1":0,
        "HITS@3":0,
        "HITS@10":0
    }
    baselog = []
    conf_cllog = []
    vio_cllog = []
    numIter =int(len(instance_dataset.train)/batch_size)
    if args.train :
        for step in range(init_step, max_step):
            stepW = (step//10000) % 2
            if og_vio:
                log = OurModel.train_step(model=model,optimizer=optimizer,
                    base_data=instance_ite['base'],con_data=instance_ite['conf'],vio_data=None,
                    base_loss_function=base_loss_funcation,cl_loss_function=conf_cl_loss,con_psize=1, vio_psize=modelConfig['cl_nsize']
                    ,cuda=baseConfig['cuda'],regu_weight=modelConfig['reg_weight'],og_vio=og_vio,og_vio_data=instance_ite,step=step,dataset=instance_dataset)
            else:
                log = OurModel.train_step(model=model,optimizer=optimizer,
                    base_data=instance_ite['base'],con_data=instance_ite['conf'],vio_data=instance_ite['vio'],
                    base_loss_function=base_loss_funcation,cl_loss_function=conf_cl_loss,con_psize=1, vio_psize=modelConfig['cl_nsize']
                    ,cuda=baseConfig['cuda'],regu_weight=modelConfig['reg_weight'],og_vio=og_vio,step=step,dataset=instance_dataset)
                    
            lr_scheduler.step()
            baselog.append(log)
            if step % log_steps == 0:
                logging_log(step,baselog,writer)
                baselog=[]
                conf_cllog=[]
                vio_cllog=[]
            
            if step % 10000 == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,'ConfigName':args.configName
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

            if step % test_step == 0 and step != 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,'ConfigName':args.configName
                }
                logging.info('Valid InstanceOf at step: %d' % step)
                metrics = test_model_conv(model, instance_dataset.valid, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'])
                logset.log_metrics('Valid ',max_step, metrics)
                ModelUtil.save_best_model(metrics=metrics,best_metrics=bestModel,model=model,optimizer=optimizer,save_variable_list=save_variable_list,args=args)
        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":max_step,'ConfigName':args.configName
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

        logging.info('Valid InstanceOf at step: %d' % max_step)
        metrics = test_model_conv(model, instance_dataset.test, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'])
        logset.log_metrics('Valid ',max_step, metrics)
       
    if args.test :
        # logging.info('Test InstanceOf at step: %d' % init_step)
        # metrics = test_true_score(model, instance_dataset.test, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'])
        # logset.log_metrics('Valid ',init_step, metrics)
        metrics = test_true_score(model, instance_dataset.train, triple2edge)
