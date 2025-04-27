"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: det_RWKVTS_paras.py
@time: 2024/10/31 9:04
"""
import time

import torch
from torch.utils.data import Subset

import model_additional_paras
import train_model_main
from utils.time_series_dataset import TimeSeriesDatasetTraining
from utils.train_tools import train_model, test_model, test_model_rolling, test_model_rolling_repeat


def main():
    # 初始化基础参数
    parser = train_model_main.define_paras()
    parser = train_model_main.add_prediction_option(parser, prediction_option=5)  # ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]各元素下标即为选项

    model_option = "RWKV_TS"  # LSTNet、DA_RNN、RWKV_TS

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

    args.lr = 0.001
    # lr_list = [0.01, 0.001, 0.0001]
    # lr_list = [0.002, 0.0005]
    # lr_list = [0.0006, 0.0007, 0.0008, 0.0009, 0.0011, 0.0012]

    args.n_heads = 4
    # n_heads_list = [2,4,8]

    args.gpt_layers = 3
    # gpt_layers_list = range(2,7)

    args.d_model = 512
    # d_model_list = [64, 128, 256, 512]

    args.d_ff = 128  # 必须小于等于d_model
    # d_ff_list = [64, 128, 256, 512]

    args.dropout = 0.2
    # dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # for lr_value in lr_list:
    #     args.lr = lr_value
    #     print("执行lr为{}的训练测试".format(str(lr_value)))
    #
    #     train_model(args, train_val_dataset)

    # for n_heads_value in n_heads_list:
    #     args.n_heads = n_heads_value
    #     print("执行n_heads为{}的训练测试".format(str(n_heads_value)))
    #
    #     train_model(args, train_val_dataset)

    # for gpt_layers_value in gpt_layers_list:
    #     args.gpt_layers = gpt_layers_value
    #     print("执行gpt_layers为{}的训练测试".format(str(gpt_layers_value)))
    #
    #     train_model(args, train_val_dataset)

    # for d_model_value in d_model_list:
    #     args.d_model = d_model_value
    #     print("执行d_model为{}的训练测试".format(str(d_model_value)))
    #
    #     for d_ff_value in d_ff_list:
    #         if d_ff_value > d_model_value:
    #             continue
    #
    #         args.d_ff = d_ff_value
    #         print("执行d_ff为{}的训练测试".format(str(d_ff_value)))
    #
    #         train_model(args, train_val_dataset)

    # for dropout_value in dropout_list:
    #     args.dropout = dropout_value
    #     print("执行dropout为{}的训练测试".format(str(dropout_value)))
    #
    #     train_model(args, train_val_dataset)

    # # 记录整个训练过程的开始时间
    # total_start_time = time.time()
    # train_model(args, train_val_dataset)
    # # 记录整个训练过程的结束时间
    # total_end_time = time.time()
    # total_duration = total_end_time - total_start_time
    # print("训练持续时间为：{}秒".format(str(total_duration)))

    # test_model(args, test_dataset)

    # test_model_rolling(args, test_dataset)

    test_model_rolling_repeat(args, test_dataset, repeat_num=10)


if __name__ == '__main__':
    main()
