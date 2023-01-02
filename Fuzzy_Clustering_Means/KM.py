import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np


def K_means(Data_PATH, n_clusters, show_sate):
    data = np.load(Data_PATH)
    print(data)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    coordinates = tsne.fit_transform(data)
    print(coordinates)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coordinates)
    labels = kmeans.labels_
    colors = ['#e41a1c', '#3d8a9c', '#9d435c', '#999999', '#4574C6', '#FDC100', '#BAD0C4']
    kmeans.predict(coordinates)  # kmeans.predict(X)== kmeans.labels_
    # 模型cluster_centers_ 属性保存了聚类结果的中⼼点
    centers = kmeans.cluster_centers_

    if show_sate == True:
        for i in range(len(labels)):
            plt.scatter(coordinates[i, 0], coordinates[i, 1], c=colors[labels[i]], s=300)
        # plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, s=100)
        # plt.plot(centers[:, 0], centers[:, 1], 'ro', c='#48a365')
        for i in range(coordinates.shape[0]):
            plt.text(coordinates[i, 0] - 3, coordinates[i, 1] - 3, str(i), fontdict={'weight': 'bold', 'size': 8})
        plt.title('K-means')
        plt.show()
    return labels, coordinates


# if __name__ == '__main__':
#     Data_PATH = 'D:\Files\experiment\\20220305\dataset\DKTResult_data.npy'
#     # Data_edges = 'D:/experiment/e1/Task_Apriori/Task3/association_rules.csv'
#     labels, res_C = K_means(Data_PATH, 4, True)
#     print(labels)
#     # DKT_Kmeans3(Data_PATH, Data_edges)
#     # DKT_Kmeans4(Data_PATH, Data_edges)
