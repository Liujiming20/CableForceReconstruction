"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: det_Net_paras.py
@time: 2024/10/24 9:41
"""
import torch
from torch.utils.data import Subset

import train_model_main
import model_additional_paras
from utils.time_series_dataset import TimeSeriesDatasetTraining
from utils.train_tools import train_model, test_model, test_model_rolling, test_model_rolling_repeat


def main():
    # 初始化基础参数
    parser = train_model_main.define_paras()
    parser = train_model_main.add_prediction_option(parser, prediction_option=5)  # ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]各元素下标即为选项

    model_option = "LSTNet"  # LSTNet、DA_RNN、RWKV_TS

    if model_option == "LSTNet":
        parser = model_additional_paras.add_LSTNet_paras(parser)
    elif model_option == "DA_RNN":
        parser = model_additional_paras.add_DARNN_paras(parser)
    else:
        parser = model_additional_paras.add_RWKV_paras(parser)

    args = parser.parse_args()
    args.model_option = model_option

    model_additional_paras.modified_paras_by_model_option(args)

    args.step_size = 1  # 设置未来提前预测的时段长度
    args.input_feature_num = 10
    args.window_size = 50

    # args.epochs = 2

    if args.model_option == "DA_RNN":
        args.seq_len = args.window_size
        args.input_feature_num = args.input_feature_num - 1  # 历史数据被设为了y_hist，所以输入维度要少一维
    elif args.model_option == "RWKV_TS":
        args = model_additional_paras.modified_RWKV_args(args)

    torch.manual_seed(args.seed_num)  # 定义随机种子，开始准备代理模型训练

    # 获得训练集
    my_dataset = TimeSeriesDatasetTraining(args.sample_root + args.target_option + "/", args.window_size, args.step_size)

    # 获取数据集长度
    dataset_length = len(my_dataset)

    # 第一次划分，计算前 90% 作为训练+验证集，后 10% 作为测试集
    split_point = int(dataset_length * args.ratio_train_test)
    # 创建训练+验证集和测试集
    train_val_dataset = Subset(my_dataset, range(0, split_point))
    test_dataset = Subset(my_dataset, range(split_point, dataset_length))

    # train_model(args, train_val_dataset)

    test_model(args, test_dataset)

    # test_model_rolling(args, test_dataset)

    # test_model_rolling_repeat(args, test_dataset, repeat_num=5)


if __name__ == '__main__':
    main()