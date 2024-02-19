import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os


current_file_path = os.path.realpath(__file__)
current_directory = os.path.dirname(current_file_path)

test_dataset_path= os.path.join(current_directory,"spaceship-titanic", "test.csv")
train_dataset_path= os.path.join(current_directory,"spaceship-titanic", "train.csv")
test_dataset= pd.read_csv(test_dataset_path)
train_dataset= pd.read_csv(train_dataset_path)
# 提取第一行
row = train_dataset.iloc[[0]]
print(row)
print(train_dataset.loc[10].shape)
# 将这一行添加到DataFrame的末尾，并将结果赋值回train_dataset
train_dataset = pd.concat([train_dataset, row], ignore_index=False)
print(train_dataset.duplicated().sum())