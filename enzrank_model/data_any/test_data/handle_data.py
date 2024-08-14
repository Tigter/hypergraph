import pandas as pd

def extract_rows_with_label(csv_file, tag):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 筛选label=1的行
    extracted_rows = df[df['Label'] == tag]
    
    # 将每行数据转换为元组的列表
    tuples = extracted_rows[['Protein_ID', 'Compound_ID', 'Label']].to_records(index=False).tolist()
    
    return tuples

def build_smiles_dict(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 构建字典
    smiles_dict = df.set_index('Compound_ID')['smiles'].to_dict()
    return smiles_dict


# 读取测试和验证集文件：
train_dict = build_smiles_dict("../../train_data/training_compound.csv")
valid_dict = build_smiles_dict("../../train_data/validation_compound.csv")
test_dict = build_smiles_dict("../../train_data/test_compound.csv")

train_paire = extract_rows_with_label("../../train_data/training_act.csv",1)
valid_paire = extract_rows_with_label("../../train_data/validation_act.csv",1)
test_paire = extract_rows_with_label("../../train_data/test_act.csv",1)


all_true_list = train_paire + valid_paire + test_paire
all_true_set = set()

for p,c,l in all_true_list:
    all_true_set.add((p,c))


valid_paire_neg = extract_rows_with_label("../../train_data/validation_act.csv",0)
train_paire_neg = extract_rows_with_label("../../train_data/training_act.csv",0)
test_paire_neg = extract_rows_with_label("../../train_data/test_act.csv",0)

all_false_list = train_paire_neg + valid_paire_neg + test_paire_neg

true_c_list = []
all_p_list = []
for p,c,l in valid_paire:
    all_p_list.append(p)
    true_c_list.append(c)

for p,c,l in all_false_list:
    all_p_list.append(p)

print(len(all_p_list))

c2neg = {}
for c in true_c_list:
    for p in all_p_list:
        if (p,c) not in all_true_set:
            if c not in c2neg:
                c2neg[c] = []
            c2neg[c].append(p)

# import json
# with open("neg_pairs.json","w") as f:
#     json.dump(c2neg,f)