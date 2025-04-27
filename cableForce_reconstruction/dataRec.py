import json
import os

import numpy as np
import pandas as pd
import torch

from utils.SM_utils import create_required_SM


def read_json_to_dict(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # 确保字典的 value 是字符串列表
    for key, value in data.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            continue
        else:
            raise ValueError(f"The value for key '{key}' is not a list of strings.")

    return data


def reorder_missing_list(missing_input_list_true):
    # 初始列表
    original_list = ["W 8","W 9","W 10","W 11","W 12","W 13","W 14","W 15","W 16"]

    if len(missing_input_list_true) == 1:
        missing_input_list_train = ["W 8"]
    elif len(missing_input_list_true) == 2:
        missing_input_list_train = ["W 8", "W 9"]
    elif len(missing_input_list_true) == 3:
        missing_input_list_train = ["W 8", "W 9", "W 10"]
    else:
        missing_input_list_train = None
        SystemExit("数据丢失模式异常，请核查！")

    A = missing_input_list_train
    B = missing_input_list_true

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
    elements_to_remove = missing_input_list_true

    # 使用列表推导去除元素
    filtered_list = [element for element in modified_list if element not in elements_to_remove]

    return filtered_list


def cal_delta_uni_fun(initial_monitoring_data):
    if initial_monitoring_data.ndim == 1:
        monitoring_data_array = initial_monitoring_data.reshape(-1,1)
    elif initial_monitoring_data.ndim == 2:
        monitoring_data_array = initial_monitoring_data
    else:
        monitoring_data_array = initial_monitoring_data
        SystemExit("输入监测数据有误，请检查！")

    initial_monitoring_value = np.array([1728.351, 1718.727, 1576.611, 16.5, 15.6, 11.5, 16.5, 15.3, 11.5, 18.0, 16.8, 22.0])

    monitoring_delta_array = monitoring_data_array - initial_monitoring_value  # 获得增量

    max_delta_value = np.array([44.71525184999973, 42.044381999999814, 57.427787599999874,29.246666666666663,30.0,28.0,29.299999999999997,29.999999999999996,27.1,30.1,29.8,32.8])
    min_delta_value = np.array([-86.80822200000011, -87.99946599999998, -159.45266300000003,-19.3,-18.0,-15.3,-21.9,-20.8,-15.6,-21.6,-20.0,-27.6])

    uni_array = (2.0 * (monitoring_delta_array - min_delta_value) / (max_delta_value - min_delta_value)) - 1.0

    return uni_array


def cal_ori_data_fun(rec_target, uni_data):
    if rec_target == "Cable 16":
        use_index = 0
    elif rec_target == "Cable 17":
        use_index = 1
    else:
        use_index = 2

    initial_monitoring_value = np.array([1728.351, 1718.727, 1576.611])
    max_delta_value = np.array([44.71525184999973, 42.044381999999814, 57.427787599999874])
    min_delta_value = np.array([-86.80822200000011, -87.99946599999998, -159.45266300000003])

    ori_delta = (uni_data + 1.0) / 2.0 * (max_delta_value[use_index] - min_delta_value[use_index]) + min_delta_value[use_index]
    ori_data = ori_delta + initial_monitoring_value[use_index]

    return ori_data


def load_his_data(filepath, rec_target, missing_input_list, missing_option):
    if missing_option:
        # 自定义排序键函数
        def sort_key(item):
            return int(item.split()[1])

        # 按数字从小到大排序
        sorted_missing_input_list_re = sorted(missing_input_list, key=sort_key)

        used_list = reorder_missing_list(sorted_missing_input_list_re)
    else:
        used_list = ["W 8", "W 9", "W 10", "W 11", "W 12", "W 13", "W 14", "W 15", "W 16"]

    # 读取CSV文件
    df_his_data = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=[0],
                     date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M:%S'))

    # 先对pd数据归一化
    df_his_data.set_index('Time', inplace=True)
    # 将 null 值替换为 0
    df_his_data.fillna(0, inplace=True)

    # 提取除时间戳外的数据到 NumPy 数组
    numpy_data = df_his_data.values

    uni_data = cal_delta_uni_fun(numpy_data)

    df_his_data.loc[:, :] = uni_data

    cable_force_his = df_his_data[rec_target].values.reshape(1,-1,1)  # batch_size, window_size, feature_num
    env_temp_his = df_his_data[used_list].values.reshape(1,-1, len(used_list))   # batch_size, window_size, feature_num

    return env_temp_his, cable_force_his


def rec_abnormal_data(target_filepath):
    model_option = "RWKV_TS"
    mapping_dict = {"cableTensionObs16": "Cable 16",
                    "cableTensionObs17": "Cable 17",
                    "cableTensionObs18": "Cable 18",
                    "environmentalTemperatureObs8": "W 8",
                    "environmentalTemperatureObs9": "W 9",
                    "environmentalTemperatureObs10": "W 10",
                    "environmentalTemperatureObs11": "W 11",
                    "environmentalTemperatureObs12": "W 12",
                    "environmentalTemperatureObs13": "W 13",
                    "environmentalTemperatureObs14": "W 14",
                    "environmentalTemperatureObs15": "W 15",
                    "environmentalTemperatureObs16": "W 16"}

    # 1. 解析重构报告，确定重构模式
    rec_mode_info_dict = read_json_to_dict(target_filepath)
    obs_need_rec = rec_mode_info_dict.keys()

    info_dict = {}
    target_write_file = "G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData/recovered_cable_force.json"
    # 判断文件是否存在
    if os.path.exists(target_write_file):
        # 删除文件
        os.remove(target_write_file)
    for current_rec_cable_force_whole in obs_need_rec:
        current_rec_cable_force = mapping_dict[current_rec_cable_force_whole.replace("http://www.semanticweb.org/16648/ontologies/2023/2/instance/", "")]  # 与数据库一致

        rec_mode_info = rec_mode_info_dict[current_rec_cable_force_whole]
        rec_mode = rec_mode_info[-1]

        missing_num = 0
        missing_option = False
        his_abnormal_obs_list = []
        if "FirstReconstructionMode" in rec_mode:
            pass
        elif "SecondReconstructionMode" in rec_mode:
            missing_num = 1
            missing_option = True
        elif "ThirdReconstructionModeShape" in rec_mode:
            missing_num = 2
            missing_option = True
        elif "FourthReconstructionModeShape" in rec_mode:
            missing_num = 3
            missing_option = True
            pass
        elif "DisabledReconstructionModeShape" in rec_mode:  # 直接返回json文件，并提示无法重构
            # 创建一个字典
            info_dict["DisabledReconstructionModeShape"] = rec_mode_info[:-1]

            with open(target_write_file, 'w') as json_file:
                json.dump(info_dict, json_file, indent=4)
            return
        else:
            SystemExit("请检查cableForce_rec_info.json内容，核查异常文本")

        if missing_option:
            for index in range(missing_num):
                obs_str = rec_mode_info[index].replace("http://www.semanticweb.org/16648/ontologies/2023/2/instance/", "")
                his_abnormal_obs_list.append(mapping_dict[obs_str])

        # 2. 加载历史数据，移除缺失数据
        filepath = "G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData/last_50_rows.csv"
        env_temp_his, cable_force_his = load_his_data(filepath, current_rec_cable_force, his_abnormal_obs_list, missing_option)
        env_temp_his_torch = torch.tensor(env_temp_his, dtype=torch.float32)
        cable_force_his_torch = torch.tensor(cable_force_his, dtype=torch.float32)

        # 3. 加载重构模式对应的代理模型，恢复异常数据
        sm_for_data_rec, device = create_required_SM(current_rec_cable_force, missing_num)
        x_1 = env_temp_his_torch.to(device)
        x_2 = cable_force_his_torch.to(device)

        if model_option == "LSTNet":
            input_train = torch.cat((x_1, x_2), -1)
            norm_prediction_train = sm_for_data_rec(input_train)
        elif model_option == "DA_RNN":
            norm_prediction_train = sm_for_data_rec(x_1, x_2)
        else:
            input_train = torch.cat((x_1, x_2), -1)
            norm_prediction_train = sm_for_data_rec(input_train, None, input_train, None)

        # 4. 反归一化预测结果，编码重构数据
        recovered_data = cal_ori_data_fun(current_rec_cable_force, norm_prediction_train.item())
        info_dict[current_rec_cable_force] = recovered_data

    with open(target_write_file, 'w') as json_file:
        json.dump(info_dict, json_file, indent=4)


# def main():
#     target_filepath = "./config_files/cableForce_rec_info.json"
#     rec_abnormal_data(target_filepath)
#
#
# if __name__ == '__main__':
#     main()