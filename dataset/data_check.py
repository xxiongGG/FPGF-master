import numpy as np

Data_PATH_1 = 'D:/Files/experiment/20220305/dkt/test_data.npy'
Data_PATH_2 = 'D:/Files/experiment/20220305/dkt/train_data.npy'
data_1 = np.load(Data_PATH_1)
data_2 = np.load(Data_PATH_2)
print(data_1.shape)
print(data_2.shape)

