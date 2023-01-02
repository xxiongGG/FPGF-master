import pandas as pd
from numpy import *
import numpy as np
import os


def dp_pred(pred_path, output_path):
    with open(pred_path, 'r') as f:
        lines = f.readlines()
        data = ''
        for line in lines:
            data += line.strip().replace(', tensor([', '\ntensor([')

        data = data.replace('[tensor([', '')
        data = data.replace('tensor([', '')
        data = data.replace('], grad_fn=<ViewBackward>)]', '')
        data = data.replace('], grad_fn=<ViewBackward>)', '')
        data = data.replace(', grad_fn = < ViewBackward >)', '')
        data = data.replace('],grad_fn=<ViewBackward>)', '')
        data = data.replace(']', '')

    with open(output_path + 'pred_data.txt', 'w') as f:
        f.write(data)

    dp_del_row(output_path + 'pred_data.txt')
    print('============pred finish!============')

def dp_truth(truth_path, output_path):
    with open(truth_path, 'r') as f:
        lines = f.readlines()
        data = ''
        for line in lines:
            data += line.strip().replace(', tensor([', '\ntensor([')

        data = data.replace('[tensor([', '')
        data = data.replace('tensor([', '')
        data = data.replace('], grad_fn=<ViewBackward>)]', '')
        data = data.replace('], grad_fn=<ViewBackward>)', '')
        data = data.replace(', grad_fn = < ViewBackward >)', '')
        data = data.replace('],grad_fn=<ViewBackward>)', '')
        data = data.replace('])', '')
        data = data.replace('], dtype=torch.int64)', '')
        data = data.replace(']', '')
    with open(output_path + 'truth_data.txt', 'w') as f:
        f.write(data)

    dp_del_row(output_path + 'truth_data.txt')
    print('============truth finish!============')

def dp_skill(skill_path, output_path):
    with open(skill_path, 'r') as f:
        lines = f.readlines()
        data = ''
        for line in lines:
            data += line.strip().replace(', tensor([', '\ntensor([')
        data = data.replace('[tensor([', '')
        data = data.replace('tensor([', '')
        data = data.replace('], grad_fn=<ViewBackward>)]', '')
        data = data.replace('], grad_fn=<ViewBackward>)', '')
        data = data.replace(', grad_fn = < ViewBackward >)', '')
        data = data.replace('],grad_fn=<ViewBackward>)', '')
        data = data.replace('])', '')
        data = data.replace(']', '')

    # with open(output_path + 'skill_data.txt', 'w') as f:
    #     f.write(data)

    data_temp = ''
    for line in data.split('\n'):
        line = line.partition(',')[2]
        data_temp += line + '\n'

    with open(output_path + 'skill_data.txt', 'w') as f:
        f.write(data_temp)

    dp_del_row(output_path + 'skill_data.txt')
    print('============skill finish!============')

def dp_skill4others(skill_path, output_path):
    with open(skill_path, 'r') as f:
        lines = f.readlines()
        data = ''
        for line in lines:
            data += line.strip().replace(', tensor([', '\ntensor([')
        data = data.replace('[tensor([', '')
        data = data.replace('tensor([', '')
        data = data.replace('], grad_fn=<ViewBackward>)]', '')
        data = data.replace('], grad_fn=<ViewBackward>)', '')
        data = data.replace(', grad_fn = < ViewBackward >)', '')
        data = data.replace('],grad_fn=<ViewBackward>)', '')
        data = data.replace('])', '')
        data = data.replace(']', '')

    with open(output_path + 'skill_data.txt', 'w') as f:
        f.write(data)

    dp_del_row(output_path + 'skill_data.txt')
    print('============skill finish!============')

def dp_tolist(path):
    file_pred = open(path)
    mat_data = []
    for data in file_pred.readlines():
        if ' ' in data:
            data = data.replace(' ', '')
        data = eval(data)
        mat_data.append(data)
    print('============dp_tolist finish!============')
    return mat_data


def dp_del_row(path):
    data_temp = ''
    with open(path, 'r', encoding='utf-8') as fr:
        for text in fr.readlines():
            if text.split():
                data_temp += text
    with open(path, 'w', encoding='utf-8') as fd:
        fd.write(data_temp)


def dp_tsne(path, mat_skill, mat_pred):
    DKTResult_data = np.zeros((len(mat_pred), 124))
    # len(mat_pred)是学生数
    print("stu_num:", len(mat_pred))

    for i in range(len(mat_pred)):
        if type(mat_skill[i]) == int:
            mat_skill[i] = [mat_skill[i]]

        if type(mat_pred[i]) == float:
            mat_pred[i] = [mat_pred[i]]

    for i in range(len(mat_pred)):
        # 当前学生的答题和正确率
        skill_nums = mat_skill[i]
        # print(skill_nums)
        preds = mat_pred[i]
        # print(preds)
        for j in range(len(skill_nums)):
            # skill_nun表示当前题目,pred表示当前题目的正确率
            skill_num = skill_nums[j]
            pred = preds[j]
            # 如果为空，直接填入，如果不为空，取平均数
            if DKTResult_data[i][skill_num] == 0:
                DKTResult_data[i][skill_num] = pred
            else:
                DKTResult_data[i][skill_num] = (DKTResult_data[i][skill_num] + pred) / 2
    DKTResult_data = DKTResult_data.T
    DKTResult_data = pd.DataFrame(DKTResult_data)
    if os.path.exists(path + 'DKTResult_data.npy'):
        os.remove(path + 'DKTResult_data.npy')
    np.save(path + 'DKTResult_data.npy', DKTResult_data)
    writer = pd.ExcelWriter(path + 'DKTResult_data.xlsx')
    DKTResult_data.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer.close()
    # np.save("D:/experiment/e1/DKTResult_data", DKTResult_data)
    print('============dp_tsne finish!============')

#
# if __name__ == '__main__':
#     pred_path = 'D:\experiment\C_experiment\C_DKT\data\\train_pred_all.txt'
#     skill_path = 'D:\experiment\C_experiment\C_DKT\data\\train_skills_all.txt'
#     truth_path = 'D:\experiment\C_experiment\C_DKT\data\\train_truth_all.txt'
#     output_path = 'D:\experiment\C_experiment\C_DKT\data\\'
#
#     dp_pred(pred_path, output_path)
#     dp_skill(skill_path, output_path)
#     dp_truth(truth_path, output_path)
#     mat_skill = dp_tolist(output_path + 'skill_data.txt')
#     mat_pred = dp_tolist(output_path + 'pred_data.txt')
#     mat_truth = dp_tolist(output_path + 'truth_data.txt')
#     dp_tsne(output_path, mat_skill, mat_pred)
