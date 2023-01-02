import numpy as np
import pandas as pd

# Data_PATH = 'D:/Files/experiment/20220305/dataset/DKTResult_data.npy'
Data_PATH1 = 'D:/Files/experiment/20220305/dkt/test_data.npy'
Data_PATH2 = 'D:/Files/experiment/20220305/dkt/train_data.npy'
data1 = np.load(Data_PATH1)
data2 = np.load(Data_PATH2)
print(data1.shape)
print(data2.shape)

