
import pandas as pd

def get_rows(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 构建字典
    return df

def extract_rows_with_label(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 筛选label=1的行
    extracted_rows =df
    # 将每行数据转换为元组的列表
    tuples = extracted_rows[['Protein_ID', 'Compound_ID', 'Label']].to_records(index=False).tolist()
    return tuples

train = get_rows("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/training_dataset/training_compound.csv")
valid = get_rows("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/validation_dataset/validation_compound.csv")
test = get_rows("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/test_dataset/test_compound.csv")

# 将train，valid，test的行合并
all_rows = pd.concat([train, valid, test])
# 去除重复的行
all_rows = all_rows.drop_duplicates()


train_tuples = extract_rows_with_label("./training_act.csv")
valid_tuples = extract_rows_with_label("./validation_act.csv")
test_tuples = extract_rows_with_label("./test_act.csv")

train_e_set = set([e for e,c,l in train_tuples])
valid_e_set = set([e for e,c,l in valid_tuples])
test_e_set = set([e for e,c,l in test_tuples])

train_c_set = set([c for e,c,l in train_tuples])
valid_c_set = set([c for e,c,l in valid_tuples])
test_c_set = set([c for e,c,l in test_tuples])

print(len(train_e_set), len(valid_e_set), len(test_e_set))
print(len(train_c_set), len(valid_c_set), len(test_c_set))
# 筛选all_rows中含有train_c_set的行
train_rows = all_rows[all_rows['Compound_ID'].isin(train_c_set)]
# 筛选all_rows中含有valid_c_set的行
valid_rows = all_rows[all_rows['Compound_ID'].isin(valid_c_set)]
# 筛选all_rows中含有test_c_set的行
test_rows = all_rows[all_rows['Compound_ID'].isin(test_c_set)]

# 保存到文件中
train_rows.to_csv("./training_compound.csv", index=False)
valid_rows.to_csv("./validation_compound.csv", index=False)
test_rows.to_csv("./test_compound.csv", index=False)


train = get_rows("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/training_dataset/training_protein.csv")
valid = get_rows("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/validation_dataset/validation_protein.csv")
test = get_rows("/home/skl/yl/ce_project/EnzRank-main/CNN_data_split/CNN_data_1/test_dataset/test_protein.csv")

# 将train，valid，test的行合并
all_rows = pd.concat([train, valid, test])
# 去除重复的行
all_rows = all_rows.drop_duplicates()

# 筛选all_rows中含有train_e_set的行
train_rows = all_rows[all_rows['Protein_ID'].isin(train_e_set)]
# 筛选all_rows中含有valid_e_set的行
valid_rows = all_rows[all_rows['Protein_ID'].isin(valid_e_set)]
# 筛选all_rows中含有test_e_set的行
test_rows = all_rows[all_rows['Protein_ID'].isin(test_e_set)]


# 保存到文件中
train_rows.to_csv("./training_protein.csv", index=False)
valid_rows.to_csv("./validation_protein.csv", index=False)
test_rows.to_csv("./test_protein.csv", index=False)