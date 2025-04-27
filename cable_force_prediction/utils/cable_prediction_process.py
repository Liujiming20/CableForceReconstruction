"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: cable_prediction_process.py
@time: 2024/10/24 15:51
"""
import numpy as np


def back_process(args, cable_data_norm):
    max_array = np.genfromtxt(args.sample_root+"max_delta_value_record.csv", delimiter=",")
    min_array = np.genfromtxt(args.sample_root+"min_delta_value_record.csv", delimiter=",")

    max_delta_cable_force = max_array[args.max_min_value_index, 0]
    min_delta_cable_force = min_array[args.max_min_value_index, 0]

    delta_cable_force = ((cable_data_norm + 1.0) * (max_delta_cable_force - min_delta_cable_force) / 2.0) + min_delta_cable_force

    cable_force_real = delta_cable_force + args.target_initial_value

    return cable_force_real