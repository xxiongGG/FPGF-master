import math
import numpy as np
import pandas as pd
import time
from sklearn import preprocessing

data_sequence = 'D:/Files/experiment/20220305/dataset/skill_data.txt'
data_edge = 'D:/Files/experiment/20220305/fpgk/edge_csv_FPGK.csv'
data_record = 'D:/Files/experiment/20220305/dataset/truth_data.txt'
concept_map = ['穷举与递推', '基本类型变量作函数参数', '分类统计', '一维数组作函数参数', '排序查找算法', '字符串', '字符数组', '矩阵运算', '二维数组', '循环控制结构', '最值计算',
               '字符数组作函数参数', '指针', '数值计算', '累加累乘', '选择控制结构', '日期转换', '键盘输入和屏幕输出', '函数', '数据类型、运算符与表达式', '递归函数', '输出图形',
               '一维数组', '数组', '结构体', '动态数据结构', '变量的作用域和存储类型', '链表', '共用体', '流程转移控制', '字符指针']
print(len(concept_map))
with open(data_sequence, 'r') as f:
    lines = f.readlines()
    sequences = []
    for line in lines:
        line = eval(line)
        sequences.append(line)
print(sequences)

with open(data_record, 'r') as f:
    lines = f.readlines()
    records = []
    for line in lines:
        line = eval(line)
        records.append(line)
print(records)

edges = pd.read_csv(data_edge, usecols=[0, 1])
edges = edges.values.tolist()
print(edges)
edge_against = []
for edge in edges:
    edge_against += [[edge[1], edge[0]]]
edges += edge_against
print(edges)
edge_directed = []
count_num = 22572
edge_count = []  # [e1,e2,count_r,count_W]
for edge in edges:
    print("The current edge:", edge)
    count_all = 0
    count_r = 0
    count_w = 0
    # count_p_1 = 0
    # count_p_0 = 0

    for i in range(len(sequences)):
        sequence = sequences[i]
        print(sequence)
        record = records[i]
        if len(sequence) == len(record):
            record_oe = []
            sequence_oe = []

            record_f = []
            sequence_f = []

            for i in range(len(sequence)):
                # if sequence[i] == edge[0] and record[i] == 1:
                #     count_p_1 += 1
                # elif sequence[i] == edge[0] and record[i] == 0:
                #     count_p_0 += 1
                # print("count_p_1:", count_p_1)
                # print("count_p_0:", count_p_0)
                if sequence[i] in edge:
                    sequence_oe.append(sequence[i])
                    record_oe.append(record[i])
            print("sequence_oe:", sequence_oe)
            print("record_oe:", record_oe)
            for i in range(len(sequence_oe) - 1):
                if sequence_oe[i] != sequence_oe[i + 1]:
                    sequence_f.append(sequence_oe[i])
                    record_f.append(record_oe[i])
                    if len(sequence_f) > 0 and sequence_f[0] != edge[0]:
                        sequence_f.remove(sequence_f[0])
                        record_f.remove(record_f[0])
            if len(sequence_oe) > 0:
                sequence_f.append(sequence_oe[-1])
                record_f.append(record_oe[-1])
                print("sequence_f:", sequence_f)
                print("record_f:", record_f)
                count_temp = math.floor(len(sequence_f) / 2)
                count_all += count_temp
                print("The total number of edges:", count_all)

                if len(sequence_f) > 2:
                    for i in range(0, count_temp * 2, 2):
                        if sequence_f[i] == edge[0] and sequence_f[i + 1] == edge[1] and record_f[i] > 0.5 and record_f[
                            i + 1] > 0.5:
                            count_r += 1
                        elif sequence_f[i] == edge[0] and sequence_f[i + 1] == edge[1] and record_f[i] < 0.5 and \
                                record_f[
                                    i + 1] < 0.5:
                            count_w += 1
    edge_count.append([edge[0], edge[1], count_r, count_w])
    print("Answer the right number:", count_r)
    print("Answer the wrong number:", count_w)

print("edge_count:", edge_count)

edge_final = []
edge_final_map = []
for i in range(int(len(edge_count) / 2)):
    rate_1 = (edge_count[i][2] / count_num) * (edge_count[int(len(edge_count) / 2) + i][3] / count_num)
    rate_2 = (edge_count[int(len(edge_count) / 2) + i][2] / count_num) * (edge_count[i][3] / count_num)
    edge_final.append(
        [edge_count[i][0], edge_count[i][1], rate_1] if rate_1 > rate_2 else [
            edge_count[int(len(edge_count) / 2) + i][0], edge_count[int(len(edge_count) / 2) + i][1], rate_2])
for edge in edge_final:
    edge_final_map.append([concept_map[edge[0]], concept_map[edge[1]], edge[2]])
edge_final_map = pd.DataFrame(edge_final_map)
edge_final_map = pd.DataFrame(edge_final_map)
edge_final = pd.DataFrame(edge_final)
print(edge_final)
print(edge_final_map)
date = time.strftime("%Y%m%d%H%M")
edge_final.to_csv('D:/Files/experiment/20220305/ri/edge_final' + '_' + date + '.csv')
edge_final_map.to_csv('D:/Files/experiment/20220305/ri/edge_final_map' + '_' + date + '.csv', encoding='GB2312')
