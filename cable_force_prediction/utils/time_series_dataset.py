"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: time_series_dataset.py
@time: 2024/10/23 17:06
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDatasetTraining(Dataset):
    def __init__(self, data_file_root, window_size, step_size):
        self.file_list = os.listdir(data_file_root)

        self.data_file_root = data_file_root
        self.window_size = window_size

        self.horizon = step_size  # 滑窗之间的步长与需要预存未来多长时间的输出是一致的

        self.samples = self.create_samples()

        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.samples[idx]

    def create_samples(self):
        samples = []

        # count = 0
        for filename in self.file_list:
            # count += 1
            df = pd.read_csv(self.data_file_root + filename)
            # 除去第一列
            data_uni = df.iloc[:, 1:]
            # 将数据转换为 NumPy 数组
            data_uni = data_uni.to_numpy()

            cable_force = data_uni[:, 0:1]
            env_temp = data_uni[:, 1:]

            for index_data in range(self.window_size+self.horizon-1, len(data_uni) - self.horizon-1, self.horizon):  # 如果每个窗口预测两个点数据，那就要滑动两步，这样可以保证预测数据不重叠，当然也可以不这么做，那就要均化；在这里先简单一些
                end_index = index_data-self.horizon+1

                cable_force_sample = cable_force[(end_index-self.window_size):end_index]
                env_temp_sample = env_temp[(end_index-self.window_size):end_index]

                cable_force_output_sample = cable_force[end_index:end_index+self.horizon]

                samples.append((torch.tensor(cable_force_sample, dtype=torch.float32),
                                torch.tensor(env_temp_sample, dtype=torch.float32),
                                torch.tensor(cable_force_output_sample, dtype=torch.float32)))

            # if count == 5:  # 先在小数据集上确定模型的可训练性
            #     break

        return samples


class TimeSeriesDatasetTrainingMissingInputTrain(Dataset):
    def __init__(self, data_file_root, window_size, step_size, missing_input_index_list):
        self.file_list = os.listdir(data_file_root)

        self.data_file_root = data_file_root
        self.window_size = window_size

        self.horizon = step_size  # 滑窗之间的步长与需要预存未来多长时间的输出是一致的

        self.missing_input_index_list = missing_input_index_list
        self.using_input_index_list = self.get_final_input_index_list()  # 将不需要的输入的index移除

        self.samples = self.create_samples()

        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_final_input_index_list(self):
        # 初始列表
        original_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # 需要去掉的元素
        elements_to_remove = self.missing_input_index_list

        # 使用列表推导去除元素
        filtered_list = [element for element in original_list if element not in elements_to_remove]

        return filtered_list

    def create_samples(self):
        samples = []

        # count = 0
        for filename in self.file_list:
            # count += 1
            df = pd.read_csv(self.data_file_root + filename)
            # 除去第一列，第一列为时间戳
            data_uni = df.iloc[:, 1:]
            # 将数据转换为 NumPy 数组
            data_uni = data_uni.to_numpy()

            cable_force = data_uni[:, 0:1]
            env_temp_total = data_uni[:, 1:]
            env_temp = env_temp_total[:, self.using_input_index_list]

            for index_data in range(self.window_size+self.horizon-1, len(data_uni) - self.horizon-1, self.horizon):  # 如果每个窗口预测两个点数据，那就要滑动两步，这样可以保证预测数据不重叠，当然也可以不这么做，那就要均化；在这里先简单一些
                end_index = index_data-self.horizon+1

                cable_force_sample = cable_force[(end_index-self.window_size):end_index]
                env_temp_sample = env_temp[(end_index-self.window_size):end_index]

                cable_force_output_sample = cable_force[end_index:end_index+self.horizon]

                samples.append((torch.tensor(cable_force_sample, dtype=torch.float32),
                                torch.tensor(env_temp_sample, dtype=torch.float32),
                                torch.tensor(cable_force_output_sample, dtype=torch.float32)))

            # if count == 5:  # 先在小数据集上确定模型的可训练性
            #     break

        return samples


class TimeSeriesDatasetTrainingMissingInputTest(Dataset):
    def __init__(self, data_file_root, window_size, step_size, missing_input_index_list_train, missing_input_index_list_true):
        self.file_list = os.listdir(data_file_root)

        self.data_file_root = data_file_root
        self.window_size = window_size

        self.horizon = step_size  # 滑窗之间的步长与需要预存未来多长时间的输出是一致的

        self.missing_input_index_list_train = missing_input_index_list_train
        self.missing_input_index_list_true = missing_input_index_list_true
        self.using_input_index_list = self.get_final_input_index_list()  # 将不需要的输入的index移除

        self.samples = self.create_samples()

        self.len = len(self.samples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_final_input_index_list(self):
        # 初始列表
        original_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        A = self.missing_input_index_list_train
        B = self.missing_input_index_list_true

        # 创建一个副本以避免修改原始列表
        modified_list = original_list.copy()

        # 找出A和B中的相同元素
        common_elements = list(set(A) & set(B))

        # 查找A和B中元素的索引
        indices_A = {a: original_list.index(a) for a in A}
        indices_B = {b: original_list.index(b) for b in B}

        # 处理相同元素
        for element in common_elements:
            modified_list[indices_B[element]] = original_list[indices_A[element]]

        # 处理其余元素
        remaining_A = [a for a in A if a not in common_elements]
        remaining_B = [b for b in B if b not in common_elements]

        for a, b in zip(remaining_A, remaining_B):
            modified_list[indices_A[a]] = original_list[indices_B[b]]
            modified_list[indices_B[b]] = original_list[indices_A[a]]

        # 需要去掉的元素
        elements_to_remove = self.missing_input_index_list_true

        # 使用列表推导去除元素
        filtered_list = [element for element in modified_list if element not in elements_to_remove]

        return filtered_list

    def create_samples(self):
        samples = []

        # count = 0
        for filename in self.file_list:
            # count += 1
            df = pd.read_csv(self.data_file_root + filename)
            # 除去第一列，第一列为时间戳
            data_uni = df.iloc[:, 1:]
            # 将数据转换为 NumPy 数组
            data_uni = data_uni.to_numpy()

            cable_force = data_uni[:, 0:1]
            env_temp_total = data_uni[:, 1:]
            env_temp = env_temp_total[:, self.using_input_index_list]

            for index_data in range(self.window_size+self.horizon-1, len(data_uni) - self.horizon-1, self.horizon):  # 如果每个窗口预测两个点数据，那就要滑动两步，这样可以保证预测数据不重叠，当然也可以不这么做，那就要均化；在这里先简单一些
                end_index = index_data-self.horizon+1

                cable_force_sample = cable_force[(end_index-self.window_size):end_index]
                env_temp_sample = env_temp[(end_index-self.window_size):end_index]

                cable_force_output_sample = cable_force[end_index:end_index+self.horizon]

                samples.append((torch.tensor(cable_force_sample, dtype=torch.float32),
                                torch.tensor(env_temp_sample, dtype=torch.float32),
                                torch.tensor(cable_force_output_sample, dtype=torch.float32)))

            # if count == 5:  # 先在小数据集上确定模型的可训练性
            #     break

        return samples