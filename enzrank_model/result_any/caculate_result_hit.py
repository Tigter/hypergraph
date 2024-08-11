# 对于测试样本的metric计算


# 读取csv 文件
import pandas as pd
import numpy as np
import os
import json
# 计算 sigmoid值
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 读取csv文件   

# df = pd.read_csv("./validation_act.csv")

# predict = pd.read_csv("./valid_result.json.json")
with open("./valid_result.json.json", "r") as f:
    predict = json.load(f)

# 计算hit 指标
for data in predict:
    score = data['prediction']
    sub_list= [data['Protein_ID'], *data['neg_protein']]
    # score 排序，返回排序的score 和 index

# # 加载 e2reaction json文件
# e2reaction = {}
# with open("./e2reaction.json", "r") as f:
#     e2reaction = json.load(f)

# # 加载noereaction2score.json 文件
# with open("./noereaction2score.json", "r") as f:
#     noereaction2score = json.load(f)

# 将json文件转为pkl保存
    
import pickle
# with open("e2reaction.pkl", "wb") as f:
#     pickle.dump(e2reaction, f)


# with open("noereaction2score.pkl", "wb") as f:
#     pickle.dump(noereaction2score, f)

# 加载pkl文件
with open("e2reaction.pkl", "rb") as f:
    e2reaction = pickle.load(f)

# with open("noereaction2score.pkl", "rb") as f:
#     noereaction2score = pickle.load(f)
with open("e2noereaction_pairre.pkl", "rb") as f:
    noereaction2score = pickle.load(f)

print("pkl文件保存成功")
prediction_list = []
lable_list = []


# 遍历df中的每一行, 用 tqdm 显示进度条
from tqdm import tqdm
for index, row in tqdm(predict.iterrows()):
    # 获取测试样本的enz
    c_id = row['Compound_ID']
    e_id = row['Protein_ID']
    # label = row['Label']

    # 判断是否在e2reaction中
    find_score = False
    # if c_id in e2reaction :
    #     reaction_list = e2reaction[c_id]
    #     e_set = set([e for left, right, e in reaction_list])
    #     if e_id in e_set:
    #         find_score = True
    #         prediction_list.append(0.9999)

    if not find_score:
        no_e_scores = []
        if c_id in noereaction2score:
            for result in noereaction2score[c_id]:
                if e_id in result:
                    no_e_scores.append(result[e_id])
            if len(no_e_scores) > 0:
                mean_score = np.mean(no_e_scores)
                predict_value = predict.loc[index]['predicted']
                weight_score = 0.1 * sigmoid(mean_score)+ 0.9* predict_value
                prediction_list.append(weight_score)
                find_score = True

    if not find_score:
        # 获取 predict 文件中对应行的预测值
        predict_value = predict.loc[index]['predicted']
        prediction_list.append(predict_value)

    # 获取测试样本的标签
    lable_list.append(predict.loc[index]['label'])


lable_list = np.array(lable_list, dtype=int)
prediction_list = np.array(prediction_list, dtype=float)

# 计算测试样本的metric：auc,aupr
from sklearn.metrics import roc_curve, precision_recall_curve,auc

fpr, tpr, thresholds_AUC = roc_curve(lable_list, prediction_list)
AUC = auc(fpr, tpr)
precision, recall, thresholds = precision_recall_curve(lable_list, prediction_list)
AUPR = auc(recall, precision)

distance = (1 - fpr) ** 2 + (1 - tpr) ** 2
EERs = (1 - recall) / (1 - precision)
positive = sum(lable_list)
negative = len(lable_list) - positive
ratio = negative / positive
opt_t_AUC = thresholds_AUC[np.argmin(distance)]
opt_t_AUPR = thresholds[np.argmin(np.abs(EERs - ratio))]

print(f"\tArea Under ROC Curve(AUC): {AUC:.3f}")
print(f"\tArea Under PR Curve(AUPR): {AUPR:.3f}")
print(f"\tOptimal threshold(AUC)   : {opt_t_AUC:.3f}")
print(f"\tOptimal threshold(AUPR)  : {opt_t_AUPR:.3f}")
print("=================================================")