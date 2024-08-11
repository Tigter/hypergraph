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
from torch.optim.lr_scheduler import MultiStepLR

from loss import NSSAL
from util.tools import logset
from util.model_util import ModelUtil
from util.generate_hyper_graph import *
import logging
from torchstat import stat
import argparse

from util.brenda_datasets import *
import random
from core.HyperGraphBrenda20 import HyperGraphV3
from core.HyperCEBrenda import HyperKGEConfig 


def cosine_annealing(epoch, num_epochs, start_lr, end_lr):
    import math
    cos_val = math.cos(math.pi * epoch / num_epochs)
    return end_lr + (start_lr - end_lr) * 0.5 * (1 + cos_val)

def get_noise_sigma(step, max_step,config):
    noise_sigma = config["noise_sigma"]
    noise_sigma = noise_sigma * cosine_annealing(step, max_step, 0, 1)
    return noise_sigma

def get_entity_noise_sigma(weight, step, max_step):
    _3sigma_percent = 0.9973
    noise_sigma = float(torch.topk(weight.reshape(-1).abs(), int(weight.shape[0]*weight.shape[1] * _3sigma_percent), largest=False)[0][-1]) / 3 * 0.1
    noise_sigma = noise_sigma * cosine_annealing(step, max_step, 0, 1)
    return noise_sigma

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


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def test_step_function(model, test_dataset, args, clist2edgeid, sampler, index2e,e_num):
        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        logs = []
        step = 0
        total_steps = len(test_dataset)
        reaction2result = {}
        relation = torch.LongTensor([i for i in range(e_num)]).cuda()
        with torch.no_grad():
            for key,value in test_dataset.items():
                reaction_add_score = []
                for reaction in value:
                    left, right = reaction[0], reaction[1]
                    left_id = clist2edgeid[tuple(sorted(left))]
                    right_id = clist2edgeid[tuple(sorted(right))]

                    left_id = torch.LongTensor([left_id])
                    right_id = torch.LongTensor([right_id])
                    left_out = sampler.sample(left_id)
                    right_out = sampler.sample(right_id)
                    
                    head_emb = model.single_emb(left_out)
                    tail_emb = model.single_emb(right_out)
                    score = model.full_score(head_emb, relation,tail_emb)
                    score = score.squeeze(0)
                    # top_values, top_indices = torch.topk(score, k=10)
                    reaction_result = []
                    sub2score = {}
                    for i in range(len(score)):
                        sub2score[index2c[i]] = score[i].item()
                    reaction_add_score.append(sub2score)
                reaction2result[key] = reaction_add_score
                if step % 1000 == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                step += 1
        # 保存为pickle文件
        import pickle
        with open("./enzrank_model/e2noereaction_pairre.pkl",'wb') as f:
            pickle.dump(reaction2result,f)
        # with open("./enzrank_model/e2noereaction.json",'w') as f:
        # with open("./enzrank_model/noereaction2score.json",'w') as f:
        #     json.dump(reaction2result,f)
    
if __name__=="__main__":
    # 读取4个数据集
    setup_seed(20)
    args = set_config()
    with open('./config/hypergraph_brenda_07_all_data.yml','r', encoding='utf-8') as f:
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
    logging.info('Model: %s begining load data' % modelConfig['name'])
    graph_info,train_info,clist2edgeid, sampler,index2c,e_num = build_graph_sampler_ne_predict(modelConfig)
    logging.info('build trainning dataset....')

    base_loss_funcation = NSSAL(gamma=modelConfig["gamma"], plus_gamma=False)
    sub_loss_function = NSSAL(gamma=modelConfig["gamma_s"], plus_gamma=True)
    type_loss_function = NSSAL(gamma=modelConfig["gamma_t"], plus_gamma=True)


    hyperConfig = HyperKGEConfig()
    hyperConfig.embedding_dim = modelConfig['dim']
    hyperConfig.conv_args_conv_dropout_rate = modelConfig['dropout']
    hyperConfig.gamma = modelConfig['gamma']
    n_node = graph_info['base_node_num']
    help_data = None

    model = HyperGraphV3(hyperkgeConfig=hyperConfig,n_node=n_node, n_hyper_edge=graph_info["max_edge_id"]-n_node,e_num=graph_info['e_num'],graph_info=graph_info,config=modelConfig,NodeGnnDataset=None,clDataset=None)
    
    if cuda:
        model = model.cuda()
   
    typeEmb = [param for name,param in model.named_parameters() if name == 'box.init_tensor' or  name =='box.trans_emb.weight']
    otherEmb = [param for name,param in model.named_parameters() if name != 'box.init_tensor' and name != 'box.trans_emb.weight']
   
    optimizer = torch.optim.Adam([
       {
         'params':filter(lambda p: p.requires_grad , typeEmb), 
         'lr': modelConfig['box_lr']
        },
        {
         'params':filter(lambda p: p.requires_grad , otherEmb), 
         'lr': lr
        }
        ], lr=lr,
    )
    result = get_parameter_number(model)
    logging.info("模型总大小为：%s" % str(result["Total"]))
    # 如果-有保-存模-型则，读取-模型,进行-测试
    # init_path = "/home/skl/yl/ce_project/relation_cl/models/models/breada_hyperedge_complex_transformer_clean_01/hit10"
    init_path ="/home/skl/yl/ce_project/relation_cl/models/models/202406/brenda_bigger_lhf_ne_new_edge_predict_2/hit1"
    init_path ="/home/skl/yl/ce_project/relation_cl/models/models/202406/brenda_bigger_all_ne_new_edge_pairre_4/hit10"
    if init_path != None:
        logging.info('init: %s' % init_path)
        checkpoint = torch.load(os.path.join(init_path, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
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
    lr_scheduler = MultiStepLR(optimizer,milestones=[3000], gamma=decay)
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
        "HITS@10":0,
        "Mix":0,
    }
    baselog = []
    conf_cllog = []
    vio_cllog = []

    with open("/home/skl/yl/ce_project/relation_cl/enzrank_model/e2noereaction.json",'r') as f:
        json_data = json.load(f)
    logging.info('Test InstanceOf at step: %d' % checkpoint['step'])
    metrics = test_step_function(model, json_data,modelConfig,clist2edgeid,sampler,index2c,e_num)
    # logset.log_metrics('Test ',checkpoint['step'], metrics)