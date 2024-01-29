from collections import defaultdict
from curses import init_pair
from torch.utils.tensorboard import SummaryWriter   
import time
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
from loss import NSSAL,MRL,NSSAL_aug,MRL_plus,NSSAL_sub
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR

from torchstat import stat
import argparse
import torch.nn.functional as F
from util.generate_simple_graph import *

import random
from core.HyperNode import *
from core.HyperGraphV3 import HyperGraphV4


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


def test_inductive(model, datasets,valid_data,n_id_list,test_and_valid_neg_info,mode="valid"):
    print("begin valid")
    if mode != "train":
        new_edge_index, new_edge_type = valid_data
        datasets["edge_index"] =  [datasets["edge_index"][0], new_edge_index]
        datasets["hyper_edge_type"] =  new_edge_type
        if mode == "valid":
            datasets["neg_info"] =  test_and_valid_neg_info["valid"]
        else:
            datasets["neg_info"] =  test_and_valid_neg_info["test"]
       
    dataloader = GraphSampler(
        datasets,n_id_list,
        batch_size = 8,
        size = [10,2],
        mode=mode,
        n_size=49
    )

    logs = []
    count = 0
    for data in dataloader:
        count += 1
        if count % 10 == 0:
            print("test step count: %d" % count)

        pos_data, neg_data,rel_out = data
        n_id, x, adjs, lable,split_idx = pos_data
        hyper_edge_emb = model.encoder(n_id,x, adjs, lable,split_idx, True)
        batch_size = len(hyper_edge_emb)

        neg_n_id, neg_x, neg_adjs, neg_lable,split_idx = neg_data
        neg_hyper_edge_emb = model.encoder(neg_n_id, neg_x, neg_adjs, neg_lable,split_idx, True,mode="neg")
        neg_hyper_edge_emb = neg_hyper_edge_emb.reshape(batch_size,-1,model.hyperkgeConfig.embedding_dim)

        r_n_id, r_x, r_adjs,split_idx = rel_out
        relation_emb = model.encoder(r_n_id, r_x, r_adjs , None,split_idx, True)
        
        rel_emb  = relation_emb.unsqueeze(1)
        hyper_edge_emb = hyper_edge_emb.unsqueeze(1)

        p_score =  torch.norm(hyper_edge_emb * rel_emb,p=2,dim=-1)
        n_score =  torch.norm(neg_hyper_edge_emb * rel_emb, p=2,dim=-1)
        score = torch.cat([p_score,n_score],dim=1)
        argsort = torch.argsort(score, dim = 1, descending=True)
        for i in range(batch_size):
            ranking = (argsort[i, :] == 0).nonzero()
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

def test_entity_predict(model, datasets,test_triples,sampler):
    batch_size = 8
    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples, 
            datasets.all_true_triples, 
            datasets.nentity, 
            datasets.nrelation, 
            'hr_t'), 
        batch_size=batch_size,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples, 
            datasets.all_true_triples, 
            datasets.nentity, 
            datasets.nrelation, 
            'h_rt',
         
        ), 
        batch_size=batch_size,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )
    test_dataset_list = [test_dataloader_tail, test_dataloader_head]
    torch.cuda.empty_cache()
    logs = []
    count = 0
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                batch_size = len(positive_sample)
                rel_out, ent_out = sampler.sampler_helper(positive_sample, negative_sample)
                head,tail, neg_emb ,rel_emb = model.score(rel_out, ent_out,positive_sample.shape[0],negative_sample.shape[1])
                if mode == "hr_t":
                    score_emb = head*rel_emb*neg_emb
                    neg_score = model.ce_predictor(score_emb)
                    # neg_score = torch.norm(head*rel_emb*neg_emb, p=2, dim=-1)
                    positive_arg = positive_sample[:,2]
                else:
                    score_emb = tail*rel_emb*neg_emb
                    neg_score = model.ce_predictor(score_emb)
                    # neg_score = torch.norm(tail*rel_emb*neg_emb, p=2, dim=-1)
                    positive_arg = positive_sample[:,0]
                neg_score = neg_score.squeeze(-1)
                filter_bias = filter_bias.cuda()
                positive_arg = positive_arg.cuda()
                score = neg_score + filter_bias
                argsort = torch.argsort(score, dim = 1, descending=True)

                for i in range(batch_size):
                    count = count + 1
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()

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
    with open('./config/hyper_graph.yml','r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        baseConfig = config['baseConfig']
        modelConfig = config[args.configName]

    cuda = baseConfig['cuda']
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
    if modelConfig['dataset'] == 'YAGO':
        instance_data_path ="/home/skl/yl/data/YAGO3-1668k/yago_insnet/"
    elif modelConfig['dataset'] == 'FB15k-237':
        instance_data_path ="/home/skl/yl/data/FB15k-237"
    elif modelConfig['dataset'] == 'NELL-995':
        instance_data_path ="/home/skl/yl/data/NELL-995"
    elif modelConfig['dataset'] == '237-v1':
        instance_data_path ="/home/skl/yl/data/FB15k-237-v1"
    elif modelConfig['dataset'] == 'wn18rr':
        instance_data_path ="/home/skl/yl/data/wn18rr"
    elif modelConfig['dataset'] == 'test':
        instance_data_path ='/home/skl/yl/data/YAGO3-1668k/yago_new_ontonet/'
   
    logging.info('DataPath: %s' % instance_data_path)
    on = DP(os.path.join(instance_data_path),idDict=True, reverse=args.data_reverse,unify_code=True)
    logging.info('build trainning dataset....')
    sampler, embedding = build_dataset(on, modelConfig)

  
    hyperConfig = HyperKGEConfig()
    hyperConfig.embedding_dim = modelConfig['dim']
    hyperConfig.gamma = modelConfig['gamma']

    model = HyperGraphV4(hyperkgeConfig=hyperConfig,n_node=on.nentity+on.nrelation,n_hyper_edge=0)
    if cuda:
        model = model.cuda()
    model.embedding = embedding

    optimizer = torch.optim.Adam([
        {
            'params':filter(lambda p: p.requires_grad, model.parameters())
        },
        # {
        #     'params':embedding.weight,
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
    logging.info('Instance nentity: %s' % on.nentity)
    logging.info('Instance nrelatidataset. %s' % on.nrelation)
    logging.info('max step: %s' % max_step)
    logging.info('init step: %s' % init_step)
    logging.info('lr: %s' % lr)

    # 设置学习率更新策略
    lr_scheduler = MultiStepLR(optimizer,milestones=[500,100000,200000], gamma=decay)
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
    numIter =int(len(on.train)/batch_size)

    if args.train :
        logging.info('beging trainning')
        begin_time =  time.time()
        for step in range(init_step, max_step):
            
            data = sampler.sample(1)
            log = HyperGraphV4.train_step(model=model,optimizer=optimizer,data=data,batch_size=batch_size,n_size = n_size)
            lr_scheduler.step()
            baselog.append(log)

            if step % 100 == 0:
                logging_log(step, baselog, writer)
                baselog=[]
                time_used = time.time() - begin_time
                begin_time = time.time()
                logging.info("step %d used time: %.3f" % (step, time_used))

            if step % 100 == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,'ConfigName':args.configName
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

            if step % test_step == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,'ConfigName':args.configName
                }
                logging.info('Valid at step: %d' % step)
                metrics = test_entity_predict(model,on,on.valid,sampler)
                logset.log_metrics('Valid ',max_step, metrics)
                ModelUtil.save_best_model(metrics=metrics,best_metrics=bestModel,model=model,optimizer=optimizer,save_variable_list=save_variable_list,args=args)

        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":max_step,'ConfigName':args.configName
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

        logging.info('Valid InstanceOf at step: %d' % max_step)
        metrics = test_entity_predict(model,on,on.test,sampler)
        logset.log_metrics('Valid ',max_step, metrics)
       
    # if args.test :
    # 模型embedding debug 分析工具
    if args.debug :
        # logging.info('Test InstanceOf at step: %d' % init_step)
        # metrics = test_true_score(model, instance_dataset.test, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'])
        # logset.log_metrics('Valid ',init_step, metrics)
        entity_embdding = model.embedding.weight[0:14541]
        relation_embedding = model.embedding.weight[14541:]

        with open('./entity_embdding.txt','w') as f:
            e_mean = torch.mean(entity_embdding,dim=-1)
            f.write(str(list(e_mean.cpu().detach().numpy().tolist())))
        with open('./relation_embdding.txt','w') as f:
            e_mean = torch.mean(relation_embedding,dim=-1)
            f.write(str(e_mean.cpu().detach().numpy().tolist()))

        # metrics = test_true_score(model, instance_dataset.test)
        # metrics = test_model_conv(model, instance_dataset.test, instance_dataset.all_true_triples,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'])
        # logset.log_metrics('Test Train ',max_step, metrics)