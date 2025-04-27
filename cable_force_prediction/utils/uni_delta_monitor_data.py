"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: uni_delta_monitor_data.py
@time: 2024/10/23 17:16
"""
import csv
import os

import numpy as np
import pandas as pd


def gen_all_monitor_data(ori_data_file_root):
    filename_list = os.listdir(ori_data_file_root)

    total_data_array = None
    first_option = True
    for filename in filename_list:
        ori_data_filepath = ori_data_file_root + filename

        df = pd.read_csv(ori_data_filepath)

        # 除去第一列
        data = df.iloc[:, 1:]
        # 将数据转换为 NumPy 数组
        numpy_array = data.to_numpy()

        if first_option:
            total_data_array = numpy_array
            first_option = False
        else:
            total_data_array = np.concatenate((total_data_array, numpy_array), axis=0)

    return total_data_array


def generate_uni_monitor_data(ori_data_file_root, uni_data_file_root, initial_value_array, max_value_array, min_value_array):
    filename_list = os.listdir(ori_data_file_root)

    for filename in filename_list:
        ori_data_filepath = ori_data_file_root + filename

        ori_df = pd.read_csv(ori_data_filepath)

        # 将第一列设为索引
        ori_df.set_index('Unnamed: 0', inplace=True)

        # 处理为增量
        delta_df = ori_df - initial_value_array

        # 归一化处理
        del_df = (2.0*(delta_df-min_value_array)/(max_value_array-min_value_array)) - 1.0

        del_df.to_csv(uni_data_file_root + filename, index=True)


def main():
    ori_data_file_root_root = "E:/liujiming2/KG_paper/cable_force_prediction/source_data/monitor_data_daily/"

    uni_data_file_root_root = "E:/liujiming2/KG_paper/cable_force_prediction/source_data/uni_delta_monitor_data/"

    ini_cable_force_list = [1777.525, 1722.005, 1741.021, 1674.005, 1728.351, 1718.727, 1576.611]

    cable_index = 0
    max_delta_value_record = np.zeros((7, 10))
    min_delta_value_record = np.zeros((7, 10))
    for cable_name in ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]:
        ori_data_file_root = ori_data_file_root_root + cable_name + "/"
        uni_data_file_root = uni_data_file_root_root + cable_name + "/"

        cable_force = ini_cable_force_list[cable_index]
        initial_value_array = np.array([cable_force, 16.5, 15.6, 11.5, 16.5, 15.3, 11.5, 18.0, 16.8, 22.0])

        all_monitor_data = gen_all_monitor_data(ori_data_file_root)

        delta_monitor_data = all_monitor_data - initial_value_array

        max_value_array = np.max(delta_monitor_data, axis=0)
        min_value_array = np.min(delta_monitor_data, axis=0)
        max_delta_value_record[cable_index, :] = max_value_array[:]
        min_delta_value_record[cable_index, :] = min_value_array[:]

        generate_uni_monitor_data(ori_data_file_root, uni_data_file_root, initial_value_array, max_value_array, min_value_array)

        cable_index += 1

    with open(uni_data_file_root_root + 'max_delta_value_record.csv', mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(max_delta_value_record)

    with open(uni_data_file_root_root + 'min_delta_value_record.csv', mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(min_delta_value_record)


if __name__ == '__main__':
    main()