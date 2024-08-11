import pandas as pd

# 读取全部的化合物
entity_list = []
with open("/home/skl/yl/ce_project/relation_cl/brenda_07/all/reaction_entity.dict") as f:
    lines = f.readlines()
    for line in lines:
        entity, id = line.strip().split("\t")

        entity_list.append(entity)

import json

with open("./name2smiles_all.json", "r") as f:
    name2smile = json.load(f)

def extract_rows_with_label(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 筛选label=1的行
    # extracted_rows = df[df['Label'] == 1]
    
    # 将每行数据转换为元组的列表
    tuples = df[['Protein_ID', 'Compound_ID', 'Label']].to_records(index=False).tolist()
    
    return tuples

def build_smiles_dict(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 构建字典
    smiles_dict = df.set_index('Compound_ID')['smiles'].to_dict()
    return smiles_dict


train_dict = build_smiles_dict("../training_compound.csv")
valid_dict = build_smiles_dict("../validation_compound.csv")
test_dict = build_smiles_dict("../test_compound.csv")

train_paire = extract_rows_with_label("../training_act.csv")
valid_paire = extract_rows_with_label("../validation_act.csv")
test_paire = extract_rows_with_label("../test_act.csv")

# 合并3个dict
train_dict.update(valid_dict)
train_dict.update(test_dict)

# 反转dict 的 key和value
train_dict = {v: k for k, v in train_dict.items()}

# 遍历train_paire 统计 Compound_ID 出现的次数


def count_compound_frequency(pairs):
    compound_frequency = {}
    for pair in pairs:
        compound_id = pair[1]
        if compound_id in compound_frequency:
            compound_frequency[compound_id] += 1
        else:
            compound_frequency[compound_id] = 1
    return compound_frequency


train_compound_frequency = count_compound_frequency(train_paire)
valid_compound_frequency = count_compound_frequency(valid_paire)
test_compound_frequency = count_compound_frequency(test_paire)

# 合并3个dict
train_compound_frequency.update(valid_compound_frequency)
train_compound_frequency.update(test_compound_frequency)

# 将频次统保存为csv 文件
df = pd.DataFrame(train_compound_frequency.items(), columns=['Compound_ID', 'Frequency'])
df.to_csv("./compound_frequency.csv", index=False)

# 统计完整方程式里面的化合物频次

with open("/home/skl/yl/ce_project/relation_cl/brenda_07/all/train.json") as f:
    data = json.load(f)

with open("/home/skl/yl/ce_project/relation_cl/brenda_07/all/valid.json") as f:
    data.extend(json.load(f))

with open("/home/skl/yl/ce_project/relation_cl/brenda_07/all/test.json") as f:
    data.extend(json.load(f))

compound_frequency = {}
for left,right, e in data:
    for c in left:
        if c in compound_frequency:
            compound_frequency[c] += 1
        else:
            compound_frequency[c] = 1
    for c in right:
        if c in compound_frequency:
            compound_frequency[c] += 1
        else:
            compound_frequency[c] = 1

df = pd.DataFrame(compound_frequency.items(), columns=['Compound_ID', 'Frequency'])
df.to_csv("./full_compound_frequency.csv", index=False)

# 统计没有酶的化合物频次

with open("../e2noereaction.json", "r") as f:
    noedata = json.load(f)

no_enzyme_compound_frequency = {}
for key,value in noedata.items():
    for left,right in value:
    
        for c in left:
            if c in no_enzyme_compound_frequency:
                no_enzyme_compound_frequency[c] += 1
            else:
                no_enzyme_compound_frequency[c] = 1
        for c in right:
            if c in no_enzyme_compound_frequency:
                no_enzyme_compound_frequency[c] += 1
            else:
                no_enzyme_compound_frequency[c] = 1

df = pd.DataFrame(no_enzyme_compound_frequency.items(), columns=['Compound_ID', 'Frequency'])
df.to_csv("./no_enzyme_compound_frequency.csv", index=False)

noe_c_set = set(no_enzyme_compound_frequency.keys())
full_c_set = set(compound_frequency.keys())


c_list = list(noe_c_set.union(full_c_set))
full_count = []
nonlocal_count = []
raw_count = []
find_train_c = set()
count   = 0
for c in c_list:
    if c in noe_c_set:
        nonlocal_count.append(no_enzyme_compound_frequency[c])
    else:
        nonlocal_count.append(0)
    if c in full_c_set:
        full_count.append(compound_frequency[c])
    else:
        full_count.append(0)
    
    if c in name2smile and  name2smile[c] in train_dict:
        count += 1
        find_train_c.add(name2smile[c])
        raw_count.append(train_compound_frequency[train_dict[name2smile[c]]])
    else:
        raw_count.append(0)

for c_smile in train_dict.keys():
    if c_smile  in find_train_c: continue
    c_list.append(train_dict[c_smile])
    full_count.append(0)
    nonlocal_count.append(0)
    raw_count.append(train_compound_frequency[train_dict[c_smile]])

df = pd.DataFrame({'Compound_ID': c_list, 'Full_Frequency': full_count, 'Nonlocal_Frequency': nonlocal_count, 'Raw_Frequency': raw_count})
df.to_csv("./compound_frequency_with_train.csv", index=False)
print(len(train_dict))
print(count)