import logging
import numpy as np
import torch
import torch.utils.data as Data
import random

from DKT_C import DKT_Module

NUM_QUESTIONS = 31
BATCH_SIZE = 64
HIDDEN_SIZE = 10
NUM_LAYERS = 1


def get_data_loader(data_path, batch_size, shuffle=False):
    data = torch.FloatTensor(np.load(data_path))
    data_loader = Data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader

data = torch.FloatTensor(np.load('../dataset/data.npy'))
print(data.size())

def Train_Test_Fold(data, batch_size, train_size=.7, shuffle=True):
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    train_data = data[: boundary]
    test_data = data[boundary:]
    print(train_data)
    print("train_data students num is:", train_data.shape[0])
    print("train_data students num is:", test_data.shape[0])
    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader
n_splits = 5
aucs = []
for split in range(n_splits):
    train_loader, test_loader = Train_Test_Fold(data, batch_size=1)
    logging.getLogger().setLevel(logging.INFO)
    dkt = DKT_Module.DKT(NUM_QUESTIONS, HIDDEN_SIZE, NUM_LAYERS)
    train_skills_all, train_pred_all, train_truth_all = dkt.train(train_loader, epoch=200)
    auc, eval_pred_all, eval_truth_all, eval_skills_all = dkt.eval(test_loader)
    print("split:%d, auc: %.6f" % (split, auc))
    aucs.append(auc)

dkt.save("dkt.params")
dkt.load("dkt.params")
print(aucs)

train_skills_data_path = '../dataset/train_skills_all.txt'
with open(train_skills_data_path, "w") as f_skills:
    f_skills.write(str(train_skills_all))

train_pred_data_path = '../dataset/train_pred_all.txt'
with open(train_pred_data_path, "w") as f_pred:
    f_pred.write(str(train_pred_all))

train_truth_data_path = '../dataset/train_truth_all.txt'
with open(train_truth_data_path, "w") as f_pred:
    f_pred.write(str(train_truth_all))
