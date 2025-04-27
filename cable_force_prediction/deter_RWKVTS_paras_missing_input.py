"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: deter_RWKVTS_paras_missing_input.py
@time: 2024/11/29 18:19
"""
import time

import torch
from torch.utils.data import Subset

import model_additional_paras
import train_model_main
from utils.time_series_dataset import TimeSeriesDatasetTrainingMissingInputTrain
from utils.train_tools import train_model, test_model, test_model_rolling, test_model_rolling_repeat


def modify_args_paras_partial_input(missing_option, args):
    # 修改网络存储位置
    if missing_option == 1:
        root_str = "./result/{}/partial_input/one_input_missing/"
    elif missing_option == 2:
        root_str = "./result/{}/partial_input/two_input_missing/"
    elif missing_option == 3:
        root_str = "./result/{}/partial_input/three_input_missing/"
    else:
        root_str = None
        SystemExit("缺失选项有误！")

    args.trained_model_root = root_str.format("networks")
    args.R2_root = root_str.format("R2")
    args.trained_model_loss_root = root_str.format("train_loss")

    args.input_feature_num = 10 - missing_option  # 输入特征减少


def main():
    # 定义缺失的输入列的下标list
    missing_option = 2
    if missing_option == 1:
        missing_input_index_list_training = [0]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性
    elif missing_option == 2:
        missing_input_index_list_training = [0,1]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性
    elif missing_option == 3:
        missing_input_index_list_training = [0,1,2] # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性
    else:
        missing_input_index_list_training = None
        SystemExit("缺失选项有误！")


    # 初始化基础参数
    parser = train_model_main.define_paras()
    parser = train_model_main.add_prediction_option(parser, prediction_option=5)  # ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]各元素下标即为选项

    model_option = "RWKV_TS"

    parser = model_additional_paras.add_RWKV_paras(parser)

    args = parser.parse_args()
    args.model_option = model_option

    model_additional_paras.modified_paras_by_model_option(args)

    args.step_size = 1  # 设置未来提前预测的时段长度
    args.window_size = 50
    modify_args_paras_partial_input(missing_option, args)  # 根据缺失输入再次修改args的参数

    # args.epochs = 2

    args = model_additional_paras.modified_RWKV_args(args)

    # args.seed_num = 43  # 41、42、43
    torch.manual_seed(args.seed_num)  # 定义随机种子，开始准备代理模型训练

    # 获得训练集
    my_dataset = TimeSeriesDatasetTrainingMissingInputTrain(args.sample_root + args.target_option + "/", args.window_size, args.step_size, missing_input_index_list_training)

    # 获取数据集长度
    dataset_length = len(my_dataset)

    # 第一次划分，计算前 90% 作为训练+验证集，后 10% 作为测试集
    split_point = int(dataset_length * args.ratio_train_test)
    # 创建训练+验证集和测试集
    train_val_dataset = Subset(my_dataset, range(0, split_point))
    test_dataset = Subset(my_dataset, range(split_point, dataset_length))

    if missing_option == 1:
        args.lr = 0.001
        # lr_list = [0.01, 0.001, 0.0001]
        # lr_list = [0.002, 0.0005]
        # lr_list = [0.0006, 0.0007, 0.0008, 0.0009, 0.0011, 0.0012]

        args.n_heads = 4
        # n_heads_list = [2,4,8]
        # n_heads_list = [2, 8]

        args.gpt_layers = 3
        # gpt_layers_list = range(2,7)

        args.d_model = 256
        # d_model_list = [64, 128, 256, 512]

        args.d_ff = 128  # 必须小于等于d_model
        # d_ff_list = [64, 128, 256, 512]

        args.dropout = 0.2
        # dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    elif missing_option == 2:
        args.lr = 0.0008
        # lr_list = [0.01, 0.001, 0.0001]
        # lr_list = [0.002, 0.0005]
        # lr_list = [0.0006, 0.0007, 0.0008, 0.0009, 0.0011, 0.0012]

        args.n_heads = 8
        # n_heads_list = [2,4,8]
        # n_heads_list = [2, 8]

        args.gpt_layers = 3
        # gpt_layers_list = range(2,7)

        args.d_model = 256
        # d_model_list = [64, 128, 256, 512]

        args.d_ff = 128  # 必须小于等于d_model
        # d_ff_list = [64, 128, 256, 512]

        args.dropout = 0.2
        # dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        args.lr = 0.0011
        # lr_list = [0.01, 0.001, 0.0001]
        # lr_list = [0.002, 0.0005]
        # lr_list = [0.0006, 0.0007, 0.0008, 0.0009, 0.0011, 0.0012]

        args.n_heads = 2
        # n_heads_list = [2,4,8]
        # n_heads_list = [2, 8]

        args.gpt_layers = 4
        # gpt_layers_list = range(2,7)

        args.d_model = 256
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

    # 记录整个训练过程的开始时间
    # total_start_time = time.time()
    train_model(args, train_val_dataset)
    # 记录整个训练过程的结束时间
    # total_end_time = time.time()
    # total_duration = total_end_time - total_start_time
    # print("训练持续时间为：{}秒".format(str(total_duration)))

    # test_model(args, test_dataset)

    # test_model_rolling(args, test_dataset)

    # test_model_rolling_repeat(args, test_dataset, repeat_num=10)


if __name__ == '__main__':
    main()