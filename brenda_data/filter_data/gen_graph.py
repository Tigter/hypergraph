import os.path as osp
from func_timeout import func_set_timeout
import func_timeout
import torch
from torch_geometric.data import Dataset, Data

from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import pubchempy as pcp
import re
import os
import json

def smiles2graph(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    
    atom_features_list = []

    if not mol:
        return None

    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)
    
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x

    return graph 

name2id = {}
with open("./reaction_entity.dict") as f:
    lines = f.readlines()
    for line in lines:
        cname,cid = line.strip().split("\t")
        name2id[cname] = cid

with open("./name2smiles.json", "r") as f:
    name2smiles = json.load(f)

@func_set_timeout(20)
def f(smiles):
    graph = smiles2graph(smiles)
    return graph
count = 0
for name in tqdm(list(name2id.keys())):
    
    graph = None
    try:
        graph = f(name2smiles[name])
    except func_timeout.exceptions.FunctionTimedOut:
        print('timeout!')

    pth = './graph'
    if not graph:
        count += 1
        print(name2id[name])
        continue

    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    cid = name2id[name]
    torch.save(data, osp.join(pth, f'graph_{cid}.pt'))
