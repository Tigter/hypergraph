# 数据的构建
# 加载数据

import pandas as pd
import numpy as np
import json
from collections import defaultdict

def extract_rows_with_label(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 筛选label=1的行
    extracted_rows = df[df['Label'] == 1 ]
    extracted_rows =df
    # 将每行数据转换为元组的列表
    tuples = extracted_rows[['Protein_ID', 'Compound_ID', 'Label']].to_records(index=False).tolist()
    return tuples

def build_smiles_dict(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 构建字典
    smiles_dict = df.set_index('Compound_ID')['smiles'].to_dict()
    return smiles_dict

train = build_smiles_dict("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/training_dataset/training_compound.csv")
valid = build_smiles_dict("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/validation_dataset/validation_compound.csv")
test = build_smiles_dict("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/test_dataset/test_compound.csv")


train_dict = {}
train_dict.update(train)
train_dict.update(valid)
train_dict.update(test) 

train = extract_rows_with_label("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/training_dataset/training_act.csv")
valid = extract_rows_with_label("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/validation_dataset/validation_act.csv")
test = extract_rows_with_label("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/test_dataset/test_act.csv")


all_triple = train + valid + test

new_data = []
e_set = set()
c_set = set()


with open("/home/skl/yl/ce_project/relation_cl/brenda_07/all/train.json","r") as f:
    train_reaction = json.load(f)

with open("/home/skl/yl/ce_project/relation_cl/brenda_07/all/valid.json","r") as f:
    valid_reaction = json.load(f)

with open("/home/skl/yl/ce_project/relation_cl/brenda_07/all/test.json","r") as f:
    test_reaction = json.load(f)

all_reaction = train_reaction + valid_reaction + test_reaction

with open("./name2smiles_all.json","r") as f:
    name2smiles = json.load(f)

smile2name = {
    smiles: name for name,smiles in name2smiles.items()
}

leftc2reaction = defaultdict(set)
c_set = set()
for left, right, e in all_reaction:
    for c in left:
        c_set.add(c)
        if c in name2smiles:
            leftc2reaction[name2smiles[c]].add((tuple(left),tuple(right),e))
    for c in right:
        c_set.add(c)

with open("/home/skl/yl/ce_project/relation_cl/enzrank_model/no_e_react.json","r") as f:
    noereaction = json.load(f)
leftc2reaction_new = defaultdict(set)
clean_reaction = []
for left, right in noereaction:
    new_left = []
    new_right = []
    for c in left:
        if c in c_set:
            new_left.append(c)
    for c in right:
        if c in c_set:
            new_right.append(c)
    if len(new_left) == 0 or len(new_right) == 0:
        continue
    clean_reaction.append((tuple(new_left),tuple(new_right)))

for left, right in clean_reaction:
    for c in left:
        if c in name2smiles:
            leftc2reaction_new[name2smiles[c]].add((tuple(left),tuple(right)))

e2reaction = defaultdict(set)
for e,c,l in all_triple:
    if train_dict[c] in  leftc2reaction:
        e2reaction[c].update(leftc2reaction[train_dict[c]])

e2noereaction = defaultdict(set)
metch_data = []
no_match_data = []

for e,c,l in all_triple:
    if train_dict[c] in  leftc2reaction_new:
        metch_data.append((e,c,l))
        e2noereaction[c].update(leftc2reaction_new[train_dict[c]])
    else:
        no_match_data.append((e,c,l))
# 统计match data中正样本的数量
positives = 0
for e,c,l in metch_data:
    if l == 1:
        positives += 1


print("match data positive number: ", positives)
print("match data negative number: ", len(metch_data)-positives)

# 统计no match data中正样本的数量
positives = 0
for e,c,l in no_match_data:
    if l == 1:
        positives += 1


print("no match data positive number: ", positives)
print("no match data negative number: ", len(no_match_data)-positives)

# 将match data 正负样本分开 随机打乱然后各保留2200条，多余的合并到no match data中
import random
random.shuffle(metch_data)
random.shuffle(no_match_data)

match_pos_data = [(e,c,l) for e,c,l in metch_data if l == 1]
match_neg_data = [(e,c,l) for e,c,l in metch_data if l == 0]

valid_pos_data = match_pos_data[:1100]
valid_neg_data = match_neg_data[:1100]

test_pos_data = match_pos_data[1100:2200]
test_neg_data = match_neg_data[1100:2200]

valid_data = valid_pos_data + valid_neg_data
test_data = test_pos_data + test_neg_data

# 打乱valid 和 test 的顺序
random.shuffle(valid_data)
random.shuffle(test_data)

train_data =match_pos_data[2200:] + match_neg_data[2200:] + no_match_data

# 保存数据，为 csv 文件 列名称为：'Protein_ID', 'Compound_ID', 'Label'
# import csv

# with open("./training_act.csv","w",newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Protein_ID', 'Compound_ID', 'Label'])
#     for e,c,l in train_data:
#         writer.writerow([e,c,l])

# with open("./validatation_act.csv","w",newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Protein_ID', 'Compound_ID', 'Label'])
#     for e,c,l in valid_data:
#         writer.writerow([e,c,l])

# with open("./test_act.csv","w",newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Protein_ID', 'Compound_ID', 'Label'])
#     for e,c,l in test_data:
#         writer.writerow([e,c,l])

# # 将e2reaction 的 set转为list
for e in e2reaction:
    e2reaction[e] = list(e2reaction[e])

for e in e2noereaction:
    e2noereaction[e] = list(e2noereaction[e])

with open("./e2reaction.json","w") as f:
    json.dump(e2reaction,f)

with open("./e2noereaction.json","w") as f:
    json.dump(e2noereaction,f)

# print("valid reaction match number: ", len(e2reaction))
# print("noe reaction match number: ", len(e2noereaction))
# print("all reaction match number: ", len(set(list(e2reaction.keys())+list(e2noereaction.keys()))))
# print("all triple number: ", len(all_triple))
