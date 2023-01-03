# coding: utf-8
# 2021/4/24 @ zengxiaonan
import logging
import numpy as np
import torch
import torch.utils.data as Data

from DKT_C import DKT

NUM_QUESTIONS = 31
BATCH_SIZE = 64
HIDDEN_SIZE = 10
NUM_LAYERS = 1


def get_data_loader(data_path, batch_size, shuffle=False):
    data = torch.FloatTensor(np.load(data_path))
    data_loader = Data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader


train_loader = get_data_loader('D:\Files\experiment\\20220305\\train_data.npy', BATCH_SIZE, True)
test_loader = get_data_loader('D:\Files\experiment\\20220305\\test_data.npy', BATCH_SIZE, False)
logging.getLogger().setLevel(logging.INFO)

# 输出训练集和测试集学生数目
train_data = np.load('D:\Files\experiment\\20220305\\train_data.npy')

test_data = np.load('D:\Files\experiment\\20220305\\test_data.npy')
print("\n=====================================================")
print("train_data students num is:", train_data.shape[0])
print("train_data students num is:", test_data.shape[0])
print("=====================================================")


dkt = DKT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS)
train_skills_all, train_pred_all, train_truth_all = dkt.train(train_loader, epoch=1000)
dkt.save("dkt.params")
dkt.load("dkt.params")
print("test_loader:", test_loader)
auc, eval_pred_all, eval_truth_all, eval_skills_all = dkt.eval(test_loader)
print("auc: %.6f" % auc)

train_skills_data_path = 'D:\Files\experiment\\20220305\\train_skills_all.txt'
with open(train_skills_data_path, "w") as f_skills:
    f_skills.write(str(train_skills_all))


train_pred_data_path = 'D:\Files\experiment\\20220305\\train_pred_all.txt'
with open(train_pred_data_path, "w") as f_pred:
    f_pred.write(str(train_pred_all))

train_truth_data_path = 'D:\Files\experiment\\20220305\\train_truth_all.txt'
with open(train_truth_data_path, "w") as f_pred:
    f_pred.write(str(train_truth_all))

# print('train_skills_all:', train_skills_all)
# print('train_pred_all:', train_pred_all)
# print('train_truth_all:', train_truth_all)
# print('eval_pred_all:', eval_pred_all)
# print('eval_truth_all:', eval_truth_all)
# print('eval_truth_all:', eval_skills_all)
