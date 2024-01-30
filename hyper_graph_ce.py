from collections import defaultdict
from curses import init_pair
from torch.utils.tensorboard import SummaryWriter   
import time
from core.HyperCE import HyperKGEConfig 
from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP
import os
import numpy as np
from scipy.sparse import csr_matrix
import yaml 
import math
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,MultiStepLR
from util.dataloader import OneShotIterator
import torch.nn as nn
from util.cl_dataloader import NewOne2OneDataset,CLDataset,OGOneToOneTestDataset,OgOne2OneDataset,RelationPredictDataset,RelationTestDataset
from loss import NSSAL,MRL,NSSAL_aug,MRL_plus,NSSAL_sub
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
from util.generate_hyper_graph import *
import logging
from torch.optim.lr_scheduler import StepLR
from torchstat import stat
import argparse
import torch.nn.functional as F

from util.ce_data import *
import random
from core.HyperGraphV2 import HyperGraphV3

#!!!!!!!!!!!!!!!!!!!!!!!!#
# 设置是否使用原始的超图
old_hyper = False
og_vio = False

def logging_log(step, logs,writer):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        writer.add_scalar(metric, metrics[metric], global_step=step, walltime=None)
    logset.log_metrics('Training average', step, metrics)


def set_config(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='test model')
    parser.add_argument('--valid', action='store_true', help='valid model')
    parser.add_argument('--debug', action='store_true', help='valid model')

    
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


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def test_inductive(model, sampler):
   
    logs = []
    count = 0
    for data in sampler:
        count += 1
        if count % 50 == 0:
            print("test step count: %d" % count)

        # pos_data, rel_pos,rel_neg = data

        # n_id, x, adjs, lable,split_idx = pos_data

        # hyper_edge_emb = model.encoder(n_id,x, adjs, lable,split_idx, True)
        # # hyper_edge_emb = hyper_edge_emb.unsqueeze(1)
        # batch_size = len(hyper_edge_emb)
        # score  = model.ce_predictor(hyper_edge_emb)
        # r_n_id, r_x, r_adjs,split_idx = rel_pos
        # relation_emb = model.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)
        # rel_emb  = relation_emb.unsqueeze(1)
        # r_n_id, r_x, r_adjs,split_idx = rel_neg
        # relation_emb_neg = model.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)
        # relation_emb_neg  = relation_emb_neg.reshape(batch_size,-1, model.hyperkgeConfig.embedding_dim)
        
        # # p_score = torch.cosine_similarity(hyper_edge_emb, rel_emb)
        # # n_score =  torch.cosine_similarity(hyper_edge_emb, relation_emb_neg)
        # p_score =  torch.norm(hyper_edge_emb * rel_emb,p=2,dim=-1)
        # n_score =  torch.norm(hyper_edge_emb * relation_emb_neg, p=2,dim=-1)
        # score = n_score
        score, label,binery_label = model.score(data)
        score = score[:,1:]
        score = score.squeeze(-1)
        argsort = torch.argsort(score, dim = 1, descending=True)
        for i in range(score.shape[0]):
 

            ranking = (argsort[i, :] == label[i]).nonzero()
            assert ranking.size(0) == 1
            ranking = 1 + ranking.item()
            logs.append({
                'MRR': 1.0/ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0,
            })

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    return metrics

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__=="__main__":
    # 读取4个数据集
    setup_seed(20)
    args = set_config()
    with open('./config/hyper_graph_ce.yml','r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        baseConfig = config['baseConfig']
        modelConfig = config[args.configName]

    cuda = baseConfig['cuda']
    # cuda = False
    init_step  = 0
    save_steps = baseConfig['save_step']
    n_size = modelConfig['n_size']
    log_steps = 100
    root_path = os.path.join("./models/",args.save_path)
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

    torch.seed

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(os.path.join(root_path,'log')):
        os.makedirs(os.path.join(root_path,'log'))
    
    writer = SummaryWriter(os.path.join(root_path,'log'))

    if args.train:
        logset.set_logger(root_path,"train.log")
    else:
        logset.set_logger(root_path,'test.log')
    
    # 读取数据集
    sampler,valid_sampler, test_sampler,graph_info,train_info,node_emb = build_graph_sampler(modelConfig)
   
    logging.info('build trainning dataset....')
    base_loss_funcation = nn.CosineEmbeddingLoss(margin=modelConfig['margin'])
    hyperConfig = HyperKGEConfig()
    hyperConfig.embedding_dim = modelConfig['dim']
    hyperConfig.gamma = modelConfig['gamma']
    n_node = graph_info['base_node_num']

    model = HyperGraphV3(hyperkgeConfig=hyperConfig,n_node=n_node, n_hyper_edge=graph_info["max_edge_id"]-n_node,e_num=graph_info['e_num'])
    model.node_emb = node_emb
    if cuda:
        model = model.cuda()
       
    optimizer = torch.optim.Adam([
        {
            'params':filter(lambda p: p.requires_grad, model.parameters())
        },
        # {
        #     'params':node_emb.weight,
        # },
        ], lr=lr,
    )
    result = get_parameter_number(model)
    logging.info("模型总大小为：%s" % str(result["Total"]))
    # 如果-有保-存模-型则，读取-模型,进行-测试
    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        init_step = checkpoint['step']

    logging.info('Model: %s' % modelConfig['name'])
    logging.info('Instance nentity: %s' % graph_info['c_num'])
    logging.info('Instance nrelatidataset. %s' % graph_info['e_num'])
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

    if args.train :
        logging.info('beging trainning')
        for step in range(init_step, max_step):
            begin_time = time.time()
            if step % 10 == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,'ConfigName':args.configName
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

            if step % test_step == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,'ConfigName':args.configName
                }
                logging.info('Valid InstanceOf at step: %d' % step)
                metrics = test_inductive(model,valid_sampler)
                logset.log_metrics('Valid ',step, metrics)
                ModelUtil.save_best_model(metrics=metrics,best_metrics=bestModel,model=model,optimizer=optimizer,save_variable_list=save_variable_list,args=args)
            for data in sampler:
                log = HyperGraphV3.train_step(model=model,optimizer=optimizer,data=data,loss_funcation=base_loss_funcation)
                baselog.append(log)
            if step % 10 == 0:
                logging_log(step, baselog, writer)
                baselog=[]
                time_used = time.time() - begin_time
                begin_time = time.time()
                logging.info("epoch %d used time: %.3f" % (step, time_used))
            lr_scheduler.step()
        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":max_step,'ConfigName':args.configName
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

        logging.info('Test InstanceOf at step: %d' % max_step)
        metrics = test_inductive(model,test_sampler)
        logset.log_metrics('Test ',max_step, metrics)
       
    # if args.test :
    # 模型embedding debug 分析工具
    if args.debug :
        entity_embdding = model.embedding.weight[0:14541]
        relation_embedding = model.embedding.weight[14541:]
        with open('./entity_embdding.txt','w') as f:
            e_mean = torch.mean(entity_embdding,dim=-1)
            f.write(str(list(e_mean.cpu().detach().numpy().tolist())))
        with open('./relation_embdding.txt','w') as f:
            e_mean = torch.mean(relation_embedding,dim=-1)
            f.write(str(e_mean.cpu().detach().numpy().tolist()))
