import numpy
import pandas as pd
from FCM import Fuzzy_C_Means
from KM import K_means

Data_PATH = 'D:/Files/experiment/20220305/dataset/DKTResult_data.npy'
Out_PATH = '../results'
labels_FCM, res_C_FCM, coordinates_FCM = Fuzzy_C_Means(Data_PATH, 4, 4, True)
group_mapping_FCM = []
for i in range(len(labels_FCM)):
    group_mapping_FCM.append([i, labels_FCM[i]])
group_mapping_FCM = pd.DataFrame(group_mapping_FCM, columns=['sample', 'group'])
coordinates_mapping_FCM = pd.DataFrame(coordinates_FCM)

labels_KM, coordinates_KM = K_means(Data_PATH, 4)
group_mapping_KM = []
for i in range(len(labels_KM)):
    group_mapping_KM.append([i, labels_KM[i]])
group_mapping_KM = pd.DataFrame(group_mapping_KM, columns=['sample', 'group'])
coordinates_mapping_KM = pd.DataFrame(coordinates_KM)

numpy.savetxt('../results/labels_KM.txt', labels_KM)
numpy.savetxt('../results/labels_FCM.txt', labels_FCM)

