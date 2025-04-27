"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: test_RWKVTS_missing_data.py
@time: 2024/11/29 19:26
"""
import torch
from torch.utils.data import Subset

import model_additional_paras
import train_model_main
from deter_RWKVTS_paras_missing_input import modify_args_paras_partial_input

import itertools

from utils.time_series_dataset import TimeSeriesDatasetTrainingMissingInputTest
from utils.train_tools import test_model


def main():
    # 定义缺失的输入列的下标list
    missing_option = 3
    if missing_option == 1:
        missing_input_index_list_training = [0]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性

        combinations = [[a] for a in range(0, 9)]
    elif missing_option == 2:
        missing_input_index_list_training = [0,1]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性

        combinations = [[a, b] for a, b in itertools.product(range(0, 8), range(1, 9)) if a < b]
    elif missing_option == 3:
        missing_input_index_list_training = [0,1,2] # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性

        combinations = [[a, b, c] for a, b, c in itertools.product(range(0, 7), range(1, 8), range(2, 9)) if a < b and b < c]
    else:
        missing_input_index_list_training = None
        combinations = None
        SystemExit("缺失选项有误！")

    # 初始化基础参数
    parser = train_model_main.define_paras()
    parser = train_model_main.add_prediction_option(parser, prediction_option=5)  # ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]各元素下标即为选项

    model_option = "RWKV_TS"  # LSTNet、DA_RNN、RWKV_TS

    parser = model_additional_paras.add_RWKV_paras(parser)

    args = parser.parse_args()
    args.model_option = model_option

    model_additional_paras.modified_paras_by_model_option(args)

    args.step_size = 1  # 设置未来提前预测的时段长度
    args.window_size = 50
    modify_args_paras_partial_input(missing_option, args)

    # args.epochs = 2

    args = model_additional_paras.modified_RWKV_args(args)

    # args.seed_num = 43  # 41、42、43
    torch.manual_seed(args.seed_num)  # 定义随机种子，开始准备代理模型训练

    for com_index in range(len(combinations)):
        # if com_index == 0:
        #     continue

        missing_input_index_list_real = combinations[com_index]

        my_dataset = TimeSeriesDatasetTrainingMissingInputTest(args.sample_root + args.target_option + "/",
                                                                 args.window_size, args.step_size,
                                                                 missing_input_index_list_training,
                                                                 missing_input_index_list_real)

        # 获取数据集长度
        dataset_length = len(my_dataset)

        # 第一次划分，计算前 90% 作为训练+验证集，后 10% 作为测试集
        split_point = int(dataset_length * args.ratio_train_test)
        # 创建训练+验证集和测试集
        train_val_dataset = Subset(my_dataset, range(0, split_point))
        test_dataset = Subset(my_dataset, range(split_point, dataset_length))

        args.lr = 0.0011
        args.n_heads = 2
        args.gpt_layers = 4
        args.d_model = 256
        args.d_ff = 128  # 必须小于等于d_model
        args.dropout = 0.2

        test_model(args, test_dataset)


if __name__ == '__main__':
    main()