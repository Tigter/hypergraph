from collections import defaultdict
from curses import init_pair

import time

from pickle import FALSE
from tkinter.tix import Tree
from util.data_process import DataProcesser as DP

import os
from core.ComplEx import ComplEx
from core.TripleCL import MLPTripleEncoder,ContrastiveLoss,InfoNCELoss

import yaml 
import numpy as np
import math
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,MultiStepLR
from util.dataloader import OneShotIterator
from util.cl_dataloader import NewOne2OneDataset,CLDataset,OGOneToOneTestDataset,OgOne2OneDataset
from loss import NSSAL,MRL,NSSAL_aug,MRL_plus,NSSAL_sub
from util.tools import logset
from util.model_util import ModelUtil,ModelTester
from torch.utils.data import DataLoader
from util.dataloader import TestDataset
import logging
from torch.optim.lr_scheduler import StepLR

import argparse
import random

def logging_log(step, logs):
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    logset.log_metrics('Training average', step, metrics)

def ndcg_at_k(idx):
    idcg_k = 0
    dcg_k = 0
    n_k = 1
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)
    dcg_k += 1 / math.log(idx + 2, 2)
    return float(dcg_k / idcg_k)   

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

def train_step_old(train_iterator,model,cuda,args, loss_funcation ):
    positive_sample,negative_sample, subsampling_weight, mode = next(train_iterator)
    if cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()
   
    h = positive_sample[:,0]
    r = positive_sample[:,1]
    t = positive_sample[:,2]
    if mode =='hr_t':
        negative_score = model(h,r, negative_sample,mode)
    else:
        negative_score = model(negative_sample,r, t,mode)
    positive_score = model(h,r,t)
   
    loss = loss_funcation(positive_score, negative_score,subsampling_weight)
   
    reg = args.loss_weight *model.base_regu()
    loss_1 = loss + reg

    log = {
        '_loss': loss_1.item(),
        'regul': reg.item()
    }
    return log, loss_1
    
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
            # '_loss': loss.item(),
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
    if False:
        base = base.reshape(2,-1,3)
        pos = pos.reshape(2,-1,3)
        neg = neg.reshape(2,-1,3)
        index = torch.cat([base,pos,neg],dim=1) 
        return log,loss, index
    return log, loss

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

def build_dataset_for_cl(data_path, args, cl_nsize):
    on = DP(os.path.join(data_path),idDict=True, reverse=args.data_reverse)
    on.splitKG_by_relationType()
    on.data_aug_relation_pattern()

    base_dataset =  NewOne2OneDataset(on.train,on.nentity,on.nrelation,init_value=-1,n_size=n_size,random=False)
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
        funct_idset=on.func_id,asy_idset=on.asy_id,po_type='conf'),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=CLDataset.collate_fn
    )
    conv_iterator = OneShotIterator(conv_dataloader)

    # func_dataloader = DataLoader(
    #     CLDataset(on.vio_kg,on.nentity,on.nrelation,p_size=cl_nsize,n_size=cl_nsize,po_dict=None,
    #         funct_idset=on.func_id,asy_idset=on.asy_id, po_type='Vio',asyDict=on.asymR2paire,relation_p=base_dataset.pofHead,
    #         funcDict=on.funcR2triples),
    #     batch_size=args.batch_size,
    #     shuffle=True, 
    #     num_workers=max(1, 4//2),
    #     collate_fn=CLDataset.collate_fn
    # )
    # func_iterator = OneShotIterator(func_dataloader)

    func_dataloader = DataLoader(
        OgOne2OneDataset(on,on.nentity,on.nrelation,n_size=n_size, Datatype='Func'),
        batch_size=args.batch_size//2,
        shuffle=True, 
        num_workers=max(1, 4//2),
        collate_fn=OgOne2OneDataset.collate_fn
    )
    func_iterator = OneShotIterator(func_dataloader)

    asys = DataLoader(
        OgOne2OneDataset(on,on.nentity,on.nrelation,n_size=n_size, Datatype='Asy'),
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
    return on, data_iterator


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

    instance_dataset, instance_ite = build_dataset_for_cl(instance_data_path,args,cl_nsize=modelConfig['cl_nsize'])

    conf_cl_loss = InfoNCELoss(modelConfig['cl_temp'])

    # base_loss_funcation = NSSAL(modelConfig['gamma'])
    # vio_cl_loss = InfoNCELoss(modelConfig['cl_temp'])

    base_loss_funcation = None
    vio_cl_loss =None

    model = ComplEx(
        instance_dataset.nentity,
        instance_dataset.nrelation,
        dim=dim,
    )
    tripleEncoder =  MLPTripleEncoder(input_dim=dim, hiddle_dim1=dim*2, out_dim=dim)

    if cuda:
        model = model.cuda()
        tripleEncoder = tripleEncoder.cuda()
    
    optimizer = torch.optim.Adam([
        {'params':filter(lambda p: p.requires_grad, model.parameters())},
        {'params': filter(lambda p: p.requires_grad, tripleEncoder.parameters())}
    ]
        , 
        lr=lr
    )
    # 如果有保存模型则，读取模型,进行测试
    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
        tripleEncoder.load_state_dict(checkpoint['tripleEncoder'])
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
    lr_scheduler = MultiStepLR(optimizer,milestones=[50000], gamma=decay)
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
    asy_loss = []
    numIter =int(len(instance_dataset.train)/batch_size)
    if args.train :
        for step in range(init_step, max_step):
            stepW = (step//10000) % 2
            optimizer.zero_grad()
            log1,base_loss = train_double(instance_ite['base'],model,cuda,args,loss_funcation=base_loss_funcation)
            log3,loss3 = train_double(instance_ite['vio_func'],model,cuda,args,loss_funcation=vio_cl_loss)
            log4,loss4 = train_double(instance_ite['vio_asy'],model,cuda,args,loss_funcation=vio_cl_loss)
            log2,conf_loss = train_cl(instance_ite['conf'],model,cuda,tripleEncoder,conf_cl_loss)


           # log3,vio_loss = train_cl(instance_ite['vio'],model,cuda,tripleEncoder,vio_cl_loss)
            loss =  conf_loss + base_loss + loss3 + loss4
            vio_log = {
                '_regu' : loss3['_regu_loss'] + loss4['_regu_loss'],
                '_score_loss': loss3['_socre_loss'] + loss4['_socre_loss'],
            }
            baselog.append(log1)
            conf_cllog.append(log2) 
            vio_cllog.append(log3)
            asy_loss.append(log4)

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if step % log_steps == 0:
                logging_log(step,baselog)
                logging_log(step,conf_cllog)
                logging_log(step,vio_cllog)
                logging_log(step,asy_loss)
                baselog=[]
                conf_cllog=[]
                vio_cllog=[]
                asy_loss=[]

            if step % test_step == 0  :
                save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":step, 'tripleEncoder':tripleEncoder.state_dict()
                }
                ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)
                logging.info('Valid InstanceOf at step: %d' % step)
                metrics = test_model_conv(model, instance_dataset.valid, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'])
                logset.log_metrics('Valid ',max_step, metrics)
                ModelUtil.save_best_model(metrics=metrics,best_metrics=bestModel,model=model,optimizer=optimizer,save_variable_list=save_variable_list,args=args)
        save_variable_list = {"lr":lr_scheduler.get_last_lr(),"step":max_step,'tripleEncoder':tripleEncoder.state_dict()
        }
        ModelUtil.save_model(model,optimizer,save_variable_list=save_variable_list,path=root_path,args=args)

        logging.info('Valid InstanceOf at step: %d' % max_step)
        metrics = test_model_conv(model, instance_dataset.test, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'])
        logset.log_metrics('Valid ',max_step, metrics)
       
    if args.test :
        logging.info('Test InstanceOf at step: %d' % init_step)
        metrics = test_model_conv(model, instance_dataset.test, instance_dataset.test_filter,instance_dataset.nentity,instance_dataset.nrelation,cuda,modelConfig['data_reverse'])
        logset.log_metrics('Valid ',init_step, metrics)




        