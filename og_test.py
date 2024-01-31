from collections import defaultdict
from curses import init_pair

import time
from torch.utils.tensorboard import SummaryWriter   

from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP
import os
from core.HAKE import HAKE
from core.RotPro import RotPro
from core.TransE import TransE

from core.TuckER import TuckER
from core.DistMult import DistMult
from core.RotatE import RotatE
from core.TransC import TransC
from core.ComplEx import ComplEx
from core.SimplE import SimplE
from core.ConvKB import ConvKB
from core.PairRE import PairRE
import yaml 
import numpy as np
import math
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,MultiStepLR
from util.dataloader import NagativeSampleDataset,BidirectionalOneShotIterator,OneShotIterator,OGNagativeSampleDataset,OGOneToOneTestDataset
from util.dataloader import NewOne2OneDataset,OgOne2OneDataset
from loss import NSSAL,MRL,NSSAL_aug,MRL_plus,NSSAL_sub
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR

import argparse
import random


def logging_log(step, logs,writer,pre=""):
    metrics = {}
    for metric in logs[0].keys():
        metrics[pre+metric] = sum([log[metric] for log in logs])/len(logs)
        writer.add_scalar(metric+pre, metrics[metric], global_step=step, walltime=None)
    logset.log_metrics('Training average', step, metrics)

def train_double(train_iterator,model,cuda, args,number_iter,model_name,loss_funcation=None):
    h,r,t, value = next(train_iterator)
    if cuda:
        h = h.cuda()
        r = r.cuda()
        t = t.cuda()
        value = value.cuda()
    score,regu = model(h, r, t)

    if model_name in ['SimplE','ComplEx','DistMult']:
        regu_loss  = regu*args.loss_weight / number_iter
    elif model_name in ['ConvKB']:
        regu_loss  = regu*args.loss_weight
    else:
        regu_loss = 0
    
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
        }   
    return log, loss


def test_model_conv(model, test_triples, all_true_triples,nentity,nrelation,cuda=True, inverse=False, onType=None,head_num=None,tail_num=None,level=None,vio_test=False):
    '''
    Evaluate the model on test or valid datasets
    '''
    test_batch_size = 2
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
                negative_score = model.predict(test_h, test_r,test_t,mode=mode)
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
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    metrics["Test Count"] = count
    if vio_test:
        return ndcg_1/count
    else:
        return metrics

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
    parser.add_argument("--data_inverse",action='store_true')

    return parser.parse_args(args)

def save_embedding(emb_map,file):
    for key in emb_map.keys():
        np.save(
            os.path.join(file,key),emb_map[key].detach().cpu().numpy()
        )


def build_dataset_for_double(data_path,args, random=True):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
    on.data_aug_relation_pattern()
    on.data_aug_test_relation_patter()
    train_dataloader = DataLoader(
        NewOne2OneDataset(on.train,on.nentity,on.nrelation,init_value=-1,n_size=n_size,random=random),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=NewOne2OneDataset.collate_fn
    )
    on_train_iterator = OneShotIterator(train_dataloader)
    data_set = {
        'base':on_train_iterator
    }
    return on, data_set


def build_og_dataset_for_double(data_path, args):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
    on.data_aug_relation_pattern()
    on.data_aug_test_relation_patter()

    # 构造信息的时候需要用全部的训练集，构造conf的数据集合 

    train_dataloader = DataLoader(
        NewOne2OneDataset(on.train,on.nentity,on.nrelation,init_value=-1,n_size=n_size),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=NewOne2OneDataset.collate_fn
    )
    on_train_iterator = OneShotIterator(train_dataloader)
 # dp, nentity, nrelation, n_size=100,Datatype='Conf',pos_rat=1
    conv_dataloader = DataLoader(
        OgOne2OneDataset(on,on.nentity,on.nrelation,n_size=n_size, Datatype='Conf'),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=OgOne2OneDataset.collate_fn
    )
    conv_iterator = OneShotIterator(conv_dataloader)

    func_dataloader = DataLoader(
        OgOne2OneDataset(on,on.nentity,on.nrelation,n_size=n_size, Datatype='Func'),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=OgOne2OneDataset.collate_fn
    )
    func_iterator = OneShotIterator(func_dataloader)

    asys = DataLoader(
        OgOne2OneDataset(on,on.nentity,on.nrelation,n_size=n_size, Datatype='Asy'),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=OgOne2OneDataset.collate_fn
    )
    asys_iterator = OneShotIterator(asys)
    on_iterator={
        'base':on_train_iterator,
        'conf':conv_iterator,
        'func':func_iterator,
        'Asy':asys_iterator
    }
    return on, on_iterator


def build_dataset(data_path,args,trans_rate=1, head_num=None, tail_num=None):

    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_inverse)
    on.data_aug_relation_pattern()
    on.data_aug_test_relation_patter()

    on_train_t = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'hr_t',head_num,tail_num,all_traied=on.trained),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    if(not args.data_inverse):
        on_train_h = DataLoader(NagativeSampleDataset(on.train, on.nentity, on.nrelation, n_size, 'h_rt',head_num,tail_num,all_traied=on.trained),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
        )
        on_train_iterator = BidirectionalOneShotIterator(on_train_h, on_train_t)
    else:
        on_train_iterator = OneShotIterator(on_train_t)

    # build Conf 和 
    on_conf_t = DataLoader(OGNagativeSampleDataset(on,on.trained, on.nentity, on.nrelation, n_size, 'hr_t',pos_rat=trans_rate,Datatype='Conf'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )

    if(not args.data_inverse):
        on_conf_h = DataLoader(OGNagativeSampleDataset(on,on.trained, on.nentity, on.nrelation, n_size, 'h_rt',pos_rat=trans_rate,Datatype='Conf'),
                batch_size=batch_size,
                shuffle=True, 
                num_workers=max(1, 4//2),
                collate_fn=NagativeSampleDataset.collate_fn
        )
        on_conf_iterater = BidirectionalOneShotIterator(on_conf_t, on_conf_h)
    else:
        on_conf_iterater = OneShotIterator(on_conf_t)


    on_void_t = DataLoader(OGNagativeSampleDataset(on,on.trained, on.nentity, on.nrelation, n_size, 'hr_t',Datatype='Func'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    on_void_func_iterater = OneShotIterator(on_void_t)

    on_asy_t = DataLoader(OGNagativeSampleDataset(on,on.trained, on.nentity, on.nrelation, n_size, 'hrt',Datatype='Asy'),
            batch_size=batch_size,
            shuffle=True, 
            num_workers=max(1, 4//2),
            collate_fn=NagativeSampleDataset.collate_fn
    )
    on_void_asy_iterater = OneShotIterator(on_asy_t)

    on_ite = {
        "base": on_train_iterator,
        "conf": on_conf_iterater,
        "func": on_void_func_iterater,
        "Asy" : on_void_asy_iterater
    }
    return on,on_ite


if __name__=="__main__":
    # 读取4个数据集
    args = set_config()
    with open('./config/new_og.yml','r', encoding='utf-8') as f:
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
    warm_up_steps = args.warm_up_step
    args.data_inverse = modelConfig['data_inverse']

    g_base  = modelConfig['g_base']
    g_conf = modelConfig['g_conf']
    g_voi = modelConfig['g_voi']

    writer = SummaryWriter(os.path.join(root_path,'log'))

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
        
    if args.test:
        logset.set_logger(root_path,'test.log')
    else:
        logset.set_logger(root_path,"init_train.log")
    
    # 读取数据集
    if modelConfig['dataset'] == 'YAGO':
        instance_data_path ="/home/skl/yl/data/YAGO3-1668k/yago_insnet/"
    elif modelConfig['dataset'] == 'FB15-237':
        instance_data_path ="/home/skl/yl/data/FB15k-237"
    elif modelConfig['dataset'] == 'NELL-995':
        instance_data_path ="/home/skl/yl/data/NELL-995"


    if modelConfig['name'] in ['ComplEx','SimplE','ConvKB']:
        instance_dataset, instance_ite = build_dataset_for_double(instance_data_path,args) # 使用 one2one softplus 损失
    elif modelConfig['name'] in ['TransE','DistMult','rotpro']:             ## 使用RotatE的margin loss
        instance_dataset, instance_ite = build_dataset_for_double(instance_data_path,args, random=False)


    if modelConfig['name'] == 'ConvKB':
        model = ConvKB(instance_dataset.nentity,instance_dataset.nrelation,dim=dim,
                out_channels=modelConfig['out_channels'],
                dropout=modelConfig['dropout'],
                kernel_size=modelConfig['kernel_size']
                )
    elif modelConfig['name'] == 'ComplEx':
        print("ComplEx")
        model = ComplEx(
            instance_dataset.nentity,
            instance_dataset.nrelation,
            dim=dim,
        )
    elif modelConfig['name'] == 'SimplE':
        print("SimpLE")
        model = SimplE(
            instance_dataset.nentity,
            instance_dataset.nrelation,
            dim=dim,
        )
    elif modelConfig['name'] == 'TransE':
        print("TransE")
        model = TransE(
            instance_dataset.nentity,
            instance_dataset.nrelation,
            dim=dim,
        )   
    elif modelConfig['name'] == 'DistMult':
        print("TransE")
        model = DistMult(
            instance_dataset.nentity,
            instance_dataset.nrelation,
            dim=dim,
        )      
    if cuda:
        model = model.cuda()

    if modelConfig['name'] in ['TransE','DistMult']:
        loss_base = MRL(g_base)
        loss_conf = MRL(g_conf)
        loss_voi = MRL(g_voi)
    else:
        loss_base = None
        loss_conf = None
        loss_voi = None

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    # 如果有保存模型则，读取模型,进行测试
    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        init_step = checkpoint['step']
        # save_embedding(model.save_model_embedding(),root_path)

    logging.info('Model: %s' % modelConfig['name'])
    logging.info('Instance nentity: %s' % instance_dataset.nentity)
    logging.info('Instance nrelatidataset. %s' % instance_dataset.nrelation)

    logging.info('max step: %s' % max_step)
    logging.info('Instance gamma: %s' % g_base)
    logging.info('Conf gamma: %s' % g_conf)
    logging.info('Voi gamma: %s' % g_voi)
    logging.info('lr: %s' % lr)

    # 设置学习率更新策略
    lr_scheduler = MultiStepLR(optimizer,milestones=[50000], gamma=decay)
    # lr_scheduler = MultiStepLR(optimizer,milestones=[30000,70000,110000,160000], gamma=decay)

    logAll = []
   
    log1s = []
    log2s = []
    log3s = []
    log4s = []


    stepW = 0
    bestModel = {
        "MRR":0,
        "MR":1000000000,
        "HITS@1":0,
        "HITS@3":0,
        "HITS@10":0
    }
    numIter =int(len(instance_dataset.train)/batch_size)
    if args.train :
        for step in range(init_step, max_step):
            stepW = (step//10000) % 2
            optimizer.zero_grad()
            loss = 0
            if modelConfig['name'] in ['ComplEx','SimplE','ConvKB','TransE','DistMult']:
                log1,loss1 = train_double(instance_ite['base'],model,cuda,args,model_name=modelConfig['name'],number_iter=numIter,loss_funcation=loss_base)
                if modelConfig['use_og']:
                    log2,loss2 = train_double(instance_ite['conf'],model,cuda,args,model_name=modelConfig['name'],number_iter=numIter,loss_funcation=loss_conf)
                    log3,loss3 = train_double(instance_ite['func'],model,cuda,args,model_name=modelConfig['name'],number_iter=numIter,loss_funcation=loss_voi)
                    log4,loss4 = train_double(instance_ite['Asy'],model,cuda,args,model_name=modelConfig['name'],number_iter=numIter,loss_funcation=loss_voi)
                    loss = loss1 + loss2 + loss3 + loss4
                    log1s.append(log1)
                    log2s.append(log2)
                    log3s.append(log3)
                    log4s.append(log4)

                else:
                    loss = loss1 

            logAll.append(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if step % log_steps == 0 :
                logging_log(step,logAll,writer)
                logAll=[]
                if modelConfig['use_og']:
                    logging_log(step,log1s,writer)
                    logging_log(step,log2s,writer)
                    logging_log(step,log3s,writer)
                    logging_log(step,log4s,writer)

                log1s = []
                log2s = []
                log3s = []
                log4s = []
            if step % test_step == 0 :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step,"ModelConfig":args.configName,
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)
                if modelConfig['name'] in ['ConvKB','SimplE','ComplEx','TransE','DistMult']:
                    logging.info('Valid InstanceOf at step: %d' % step)
                  #  metrics = test_model_conv(model, instance_dataset.valid, instance_dataset.all_true_triples,instance_dataset.nentity,instance_dataset.nrelation,cuda,baseConfig['data_reverse'])
                    metrics = test_model_conv(model, instance_dataset.valid, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,baseConfig['data_reverse'])
                    logset.log_metrics('Valid ',step, metrics)
                ModelUtil.save_best_model(metrics=metrics,best_metrics=bestModel,model=model,optimizer=optimizer,save_variable_list=save_variable_list,args=args)
        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":max_step,
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

        logging.info('Test InstanceOf at step: %d' % max_step)
        metrics = test_model_conv(model, instance_dataset.test, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,baseConfig['data_reverse'])
        logset.log_metrics('Valid ',max_step, metrics)
       
    if args.test :
        logging.info('Test InstanceOf at step: %d' % init_step)
        metrics = test_model_conv(model, instance_dataset.test, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,baseConfig['data_reverse'])
        logset.log_metrics('Test ',init_step, metrics)

        conf_aug,vio= instance_dataset.get_self_test()
        logging.info('Test Positive at step: %d' % init_step)
        metrics = test_model_conv(model, conf_aug, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda)
        logset.log_metrics('Test ',init_step, metrics)

        logging.info('Test Vio at step: %d' % init_step)
        metrics = test_model_conv(model, vio, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,vio_test=True)
        print("Asymmetric NDCG@1 %.4f" % metrics)


                

        