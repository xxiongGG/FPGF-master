import copy
import math
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def end_conditon(U, U_old):
    Epsilon = 0.0000001
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def fuzzy(data, cluster_number, membership):
    # 初始化隶属度矩阵U
    MAX = 10000.0
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    # print_matrix(U)
    # 循环更新U
    while (True):
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    # 分子
                    dummy_sum_num += (U[k][j] ** membership) * data[k][i]
                    # 分母
                    dummy_sum_dum += (U[k][j] ** membership)
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)

        # 创建一个距离向量, 用于计算U矩阵。
        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                if len(data[i]) != len(C[j]):
                    return -1
                dummy = 0.0
                for k in range(0, len(data[i])):
                    dummy += abs(data[i][k] - C[j][k]) ** 2
                current.append(math.sqrt(dummy))
            distance_matrix.append(current)

        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    # 分母
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (membership - 1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            break
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    C = np.array(C)
    return U, C


def Fuzzy_C_Means(Data_PATH, cluster_number, membership, show_sate=False):
    data = np.load(Data_PATH)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    coordinates = tsne.fit_transform(data)
    # 调用模糊C均值函数
    res_U, res_C = fuzzy(coordinates, cluster_number, membership)
    labels = []
    for u in res_U:
        for i in range(len(u)):
            if u[i] == 1:
                labels.append(i)
    labels = np.array(labels)
    colors = ['#f68200', '#00b7cc', '#98ce9b', '#b2b4ff', '#4574C6', '#FDC100', '#BAD0C4']
    if show_sate == True:
        for i in range(len(labels)):
            plt.scatter(coordinates[i, 0], coordinates[i, 1], c=colors[labels[i]], s=300)
        # plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels)
        # plt.plot(res_C[:, 0], res_C[:, 1], 'ro')
        for i in range(coordinates.shape[0]):
            plt.text(coordinates[i, 0]-3, coordinates[i, 1]-3, str(i), fontdict={'weight': 'bold', 'size': 8})
        plt.title('FC-means')
        plt.show()
    return labels, res_C, coordinates

# if __name__ == '__main__':
#     Data_PATH = 'D:\Files\experiment\\20220305\dataset\DKTResult_data.npy'
#     # Data_edges = 'D:/experiment/e1/Task_Apriori/Task3/association_rules.csv'
#     labels, res_C = Fuzzy_C_Means(Data_PATH, 4, 6, True)
#     print(labels)
#     # DKT_Kmeans3(Data_PATH, Data_edges)
#     # DKT_Kmeans4(Data_PATH, Data_edges)


