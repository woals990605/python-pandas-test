import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


bream_length = pd.read_csv(
    'fish_csv/bream_length.csv', header=None)

bream_weight = pd.read_csv(
    'fish_csv/bream_weight.csv', header=None)

smelt_length = pd.read_csv(
    'fish_csv/smelt_length.csv', header=None)
smelt_weight = pd.read_csv(
    'fish_csv/smelt_weight.csv', header=None)

bream_length = bream_length.to_numpy()
bream_weight = bream_weight.to_numpy()
smelt_length = smelt_length.to_numpy()
smelt_weight = smelt_weight.to_numpy()

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length = np.concatenate([bream_length, smelt_length])
weight = np.concatenate([bream_weight, smelt_weight])

fish_data = [[l, w] for l, w in zip(length, weight)]
# print(fish_data)

fish_target = np.array([1]*35+[0]*14)

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

index = np.arange(49)
np.random.shuffle(index)
# print(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])

plt.xlabel('length')
plt.ylabel('weight')
plt.show()
