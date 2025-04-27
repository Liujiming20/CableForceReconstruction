"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: extract_rolling_result.py
@time: 2024/11/5 9:07
"""
import csv

import torch
from torch.utils.data import Subset

import model_additional_paras
import train_model_main
from utils.time_series_dataset import TimeSeriesDatasetTraining
from utils.train_tools import test_model_rolling_repeat_record


def main():
    model_option = "DA_RNN"  # LSTNet、DA_RNN、RWKV_TS

    with open("./result/{}.csv".format(model_option), mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["case_name", "segment01", "segment02", "segment03", "segment04", "segment05",
                         "segment06", "segment07", "segment08", "segment09", "segment10"])

    for s_index in [4, 5, 6]:
        # 初始化基础参数
        parser = train_model_main.define_paras()
        parser = train_model_main.add_prediction_option(parser, prediction_option=s_index)  # ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]各元素下标即为选项

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

        # args.seed_num = 43  # 41、42、43
        for seed_num_value in [41, 42, 43]:
            args.seed_num = seed_num_value
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

            if args.model_option == "LSTNet":
                args.lr = 0.0008

                args.skip = 7

                args.highway_window = 10

                args.hid_CNN = 256

                args.CNN_kernel = 4

                args.hid_RNN = 128

                args.hidSkip = 64
            elif args.model_option == "DA_RNN":
                args.lr = 0.001

                args.hidden_size_encoder = 256

                args.hidden_size_decoder = 256
            else:
                args.lr = 0.001

                args.n_heads = 4

                args.gpt_layers = 3

                args.d_model = 512

                args.d_ff = 128

                args.dropout = 0.2

            test_model_rolling_repeat_record(args, test_dataset, repeat_num=10)


if __name__ == '__main__':
    main()