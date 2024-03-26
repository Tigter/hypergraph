# 该文件要能够通过参数来配置所有项目

#!/usr/bin/python3


import json
import logging
import os

import torch
from torch.optim.lr_scheduler import StepLR,MultiStepLR
from torch.utils.data import DataLoader
import warnings
# 关闭警告

from util.model_util import ModelTester,ReactionTrainer

from util.data_process import DataProcesser as Data
from util.tools import logset
from core import *
from core.ComplEx import ComplEx

from loss import NSSAL,MRL
import numpy as np

from util.dataloader import TestReactionRelationDataset,OneToNDataset,BidirectionalOneShotIterator
from util.dataloader import NagativeRelationSampleDataset,OneShotIterator,NagativeReactionRelationSampleDataset,NagativeReactionSampleDataset

from config.config import parse_args

import json
def trans2id(train_data, entity_dict,relation_dict ):
    train_id_data = []
    for left,right,e in train_data:
        left_id = []
        right_id = []
        for c in left:
            left_id.append(entity_dict[c])
        for c in right:
            right_id.append(entity_dict[c])
        left_id = tuple(sorted(left_id))
        right_id = tuple(sorted(right_id))
        e = e.strip()
        train_id_data.append((left_id, right_id, relation_dict[e]))
    return train_id_data
def load_data():

    with open("/home/skl/yl/ce_project/relation_cl/brenda_data/train_reaction.json") as f:
        train_data = json.load(f)
    with open("/home/skl/yl/ce_project/relation_cl/brenda_data/valid_reaction.json") as f:
        valid_data = json.load(f)
    with open("/home/skl/yl/ce_project/relation_cl/brenda_data/test_reaction.json") as f:
        test_data = json.load(f)
    
    with open("/home/skl/yl/ce_project/relation_cl/brenda_data/reaction_entity.dict") as f:
        datas = f.readlines()
        entity_dict = {}
        for line in datas:
            key,value = line.split("\t")
            key = key.strip()
            value = int(value.strip())
            entity_dict[key] = value

    with open("/home/skl/yl/ce_project/relation_cl/brenda_data/reaction_relation.dict") as f:
        datas = f.readlines()
        relation_dict = {}
        for line in datas:
            key,value = line.split("\t")
            key = key.strip()
            value = int(value.strip())
            relation_dict[key] = value
           
    
    train_id_data = trans2id(train_data, entity_dict, relation_dict)
    valid_id_data = trans2id(valid_data, entity_dict, relation_dict)
    test_id_data = trans2id(test_data, entity_dict, relation_dict)

    datas = {
        "train":train_id_data,
        "valid":valid_id_data,
        "test": test_id_data,
        "nentity": len(entity_dict),
        "nrelation": len(relation_dict),
        "all_true_triples" : train_id_data + valid_id_data + test_id_data
    }
    return datas


def read_dataset(args):
    path  = "/home/skl/yl/ce_project/relation_cl/brenda_data"
    data = Data(path, reverse=False)
    classTest_data = None
    # if args.do_test_class:
    #     classTest_data = Data(args.data_path, dataType="ClassTest")
    return data, classTest_data

def bi_nagativeSampleDataset(train_triples,nentity,nrelation, args):
    train_dataloader_head = DataLoader(
        NagativeReactionRelationSampleDataset(train_triples, nentity, nrelation, args.negative_sample_size), 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=NagativeReactionRelationSampleDataset.collate_fn
    )
    train_iterator = OneShotIterator(train_dataloader_head)

    entity_dataloader_head = DataLoader(
        NagativeReactionSampleDataset(train_triples, nentity, nrelation, args.negative_sample_size,"hr_t"), 
        batch_size=512,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=NagativeReactionSampleDataset.collate_fn
    )
    entity_dataloader_tail = DataLoader(
        NagativeReactionSampleDataset(train_triples, nentity, nrelation, args.negative_sample_size,"h_rt"), 
        batch_size=512,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
        collate_fn=NagativeReactionSampleDataset.collate_fn
    )
    entity_iter = BidirectionalOneShotIterator(entity_dataloader_head,entity_dataloader_tail)




    
    return train_iterator,entity_iter

def OneToN_data_iterator(train_triples,nentity,nrelation, args):
    train_dataloader = DataLoader(
        OneToNDataset(train_triples,nentity,nrelation),
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=max(1, args.cpu_num//2),
    )
    return OneShotIterator(train_dataloader)


def buildModel(args,data):
    model = None
    name_to_model = {
        # "TransE":TransE,
        # "TransH":TransH,
        # "TransR":TransR, 
        # "PairRE":PairRE,
        # "DistMult":DistMult,
        # "HoLE":HoLE,
        "ComplEx":ComplEx,
        # "RotatE":RotatE,
        # "RotPro":RotPro,
        # "TuckER":TuckER,
    }
    return ComplEx(n_entity=data["nentity"],
            n_relation=data["nrelation"],
            dim=args.hidden_dim)
    model_class = name_to_model[args.model]

    if args.model in ["TransE","TransH","DistMult","HoLE","ComplEx","PairRE","RotatE"]:
        model =  model_class(
            n_entity=data["nentity"],
            n_relation=data["nrelation"],
            dim=args.hidden_dim
        )
    elif args.model in  ["TransR"]:
        model = model_class(
            n_entity=data["nentity"],
            n_relation=data["nrelation"],
            entity_dim=args.hidden_dim,
            relation_dim=args.relation_dim
        )
       
    elif args.model in ["TuckER"]:
        model = model_class(
            n_entity=data["nentity"],
            n_relation=data["nrelation"],
            entity_dim=args.hidden_dim,
            relation_dim = args.relation_dim,
            dropout1 = args.dropout1,
            dropout2 = args.dropout2,
            dropout3 = args.dropout3
        ) 
    elif args.model in ["RotPro"]:
        model = model_class(
            n_entity=data["nentity"],
            n_relation=data["nrelation"],
            entity_dim=args.hidden_dim,
            relation_dim = args.relation_dim,
            gamma = args.gamma
        ) 
    else:
        raise ValueError("Unknown model")

    return model


def override_config(args):
    '''
    Override model and data configuration
    '''
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.lr_step = argparse_dict['lr_step']
    args.relation_dim = argparse_dict['relation_dim']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.loss_function = argparse_dict['loss_function']
    args.optimizer = argparse_dict['optimizer']
    args.train_type = argparse_dict['train_type']
    args.test_batch_size = argparse_dict['test_batch_size']

def buildLoss(name,args):
    name_to_loss={
        "NSSAL":NSSAL,
        "MRL": MRL,
        "BCE": torch.nn.BCELoss,
        "CrossEntropyLoss":torch.nn.CrossEntropyLoss
    }
    if name not in name_to_loss:
        raise ValueError("Sorry! Unknown Loss Name, you can implement it by yourself")
    if name in ["NSSAL","MRL"]:
        loss = name_to_loss[name](gamma=args.gamma)
    else:
        loss = name_to_loss[name]()
    return loss


def buildOptimizer(name):
    name_to_optimizer ={
        "Adam":torch.optim.Adam
    }
    return name_to_optimizer[name]

def main(args):
   
    if args.train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logset.set_logger(args.save_path)

    if args.init_checkpoint:
        override_config(args)

    data = load_data()
    model = buildModel(args,data)
    if args.cuda:
        model =model.cuda()

    loss = buildLoss(args.loss_function, args)

    current_learning_rate = args.learning_rate
    optimizer_class = buildOptimizer(args.optimizer)
    optimizer = optimizer_class( 
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate)
    
    args.nentity = data["nentity"]
    args.nrelation = data["nrelation"]

    if args.init_checkpoint:
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        args.init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.train:
            current_learning_rate = checkpoint['current_learning_rate']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        args.init_step = 0
    
    step = args.init_step
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % data["nentity"])
    logging.info('#relation: %d' % data["nrelation"])
    logging.info('#train: %d' % len(data["train"]))
    logging.info('#valid: %d' % len(data["valid"]))
    logging.info('#test: %d' % len(data["test"]))
    if args.cuda:
        logging.info('Device: %s' % ("CUDA"))
    else:
        logging.info('Device: %s' % ("CPU"))

    if args.train:
        lr_scheduler = MultiStepLR(optimizer,milestones=[50000,100000,150000], gamma=0.1)
        
        train_iterator,entity_iter = bi_nagativeSampleDataset(data["train"], data["nentity"], data["nrelation"],args)
        
        trainer = ReactionTrainer(
                data=data,
                train_iterator=train_iterator,
                model=model,
                optimizer=optimizer,
                loss_function=loss,
                args=args,
                lr_scheduler=lr_scheduler,
                logging=logging,
                train_type=args.train_type,entity_iter=entity_iter)
        trainer.logging_traing_info()
        trainer.train_model_()
        step = args.max_steps

    if args.test:
        metrics = ModelTester.test_step(model, data["test"], data["all_true_triples"], args)
        logset.log_metrics('Test',step, metrics)
    

if __name__ == '__main__':
    main(parse_args())
