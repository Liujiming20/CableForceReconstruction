"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: deter_RWKVTS_paras_missing_input_MU_MODEL.py
@time: 2024/11/30 12:43
"""
import torch
from torch.utils.data import Subset

import model_additional_paras
import train_model_main
from deter_RWKVTS_paras_missing_input import modify_args_paras_partial_input
from utils.time_series_dataset import TimeSeriesDatasetTrainingMissingInputTrain
from utils.train_tools import train_model


def main():
    for missing_option in range(1, 4):
        if missing_option == 1:
            missing_input_index_list_training = [0]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性
        elif missing_option == 2:
            missing_input_index_list_training = [0, 1]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性
        elif missing_option == 3:
            missing_input_index_list_training = [0, 1, 2]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性
        else:
            missing_input_index_list_training = None
            SystemExit("缺失选项有误！")

        for s_index in [4, 5, 6]:
            # 初始化基础参数
            parser = train_model_main.define_paras()
            parser = train_model_main.add_prediction_option(parser, prediction_option=s_index)  # ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]各元素下标即为选项

            parser.add_argument('-missing_option', type=str, default=missing_option, help="输入缺失的选项")

            model_option = "RWKV_TS"

            parser = model_additional_paras.add_RWKV_paras(parser)

            args = parser.parse_args()
            args.model_option = model_option

            model_additional_paras.modified_paras_by_model_option(args)

            args.step_size = 1  # 设置未来提前预测的时段长度
            args.window_size = 50
            modify_args_paras_partial_input(missing_option, args)

            args.epochs = 2

            args = model_additional_paras.modified_RWKV_args(args)

            for seed_num_value in [41, 42, 43]:
                args.seed_num = seed_num_value
                torch.manual_seed(args.seed_num)  # 定义随机种子，开始准备代理模型训练

                # 获得训练集
                my_dataset = TimeSeriesDatasetTrainingMissingInputTrain(args.sample_root + args.target_option + "/",
                                                                        args.window_size, args.step_size,
                                                                        missing_input_index_list_training)

                # 获取数据集长度
                dataset_length = len(my_dataset)

                # 第一次划分，计算前 90% 作为训练+验证集，后 10% 作为测试集
                split_point = int(dataset_length * args.ratio_train_test)
                # 创建训练+验证集和测试集
                train_val_dataset = Subset(my_dataset, range(0, split_point))
                test_dataset = Subset(my_dataset, range(split_point, dataset_length))

                # 设置最优超参数组合
                if missing_option == 1:
                    args.lr = 0.001
                    args.n_heads = 4
                    args.gpt_layers = 3
                    args.d_model = 256
                    args.d_ff = 128  # 必须小于等于d_model
                    args.dropout = 0.2
                elif missing_option == 2:
                    args.lr = 0.0008
                    args.n_heads = 8
                    args.gpt_layers = 3
                    args.d_model = 256
                    args.d_ff = 128  # 必须小于等于d_model
                    args.dropout = 0.2
                else:
                    args.lr = 0.0011
                    args.n_heads = 2
                    args.gpt_layers = 4
                    args.d_model = 256
                    args.d_ff = 128  # 必须小于等于d_model
                    args.dropout = 0.2

                train_model(args, train_val_dataset)


if __name__ == '__main__':
    main()