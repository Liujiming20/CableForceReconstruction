"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: test_model_R2_missing_input.py
@time: 2024/12/2 9:01
"""
import csv
import itertools

import torch
from torch.utils.data import Subset, DataLoader

from torcheval.metrics.functional import r2_score

from utils.RWKV_TS import Model

import model_additional_paras
import train_model_main
from deter_RWKVTS_paras_missing_input import modify_args_paras_partial_input
from utils.time_series_dataset import TimeSeriesDatasetTrainingMissingInputTest


def cal_test_R2_missing_input(args, test_dataset, R2_list):
    torch.manual_seed(args.seed_num)

    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # 定义GPU
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")

    rnn_model = Model(args, device)

    trained_model_path = args.trained_model_root + "/{}".format(args.target_option) + "/seed_{}/".format(args.seed_num) + args.model_option + ".pth"
    rnn_model.load_state_dict(torch.load(trained_model_path))
    rnn_model.to(device)

    # cal test set
    rnn_model.eval()
    with torch.no_grad():
        for _, (cable_force_test, env_temp_test, cable_force_output_test) in enumerate(dataloader_test):
            x_1_test = env_temp_test.to(device)
            x_2_test = cable_force_test.to(device)

            norm_output_test = cable_force_output_test.to(device)

            if args.model_option == "LSTNet":
                input_test = torch.cat((x_1_test, x_2_test), -1)
                norm_prediction_test = rnn_model(input_test)
            elif args.model_option == "DA_RNN":
                norm_prediction_test = rnn_model(x_1_test, x_2_test)
            else:
                input_test = torch.cat((x_1_test, x_2_test), -1)
                norm_prediction_test = rnn_model(input_test, None, input_test, None)

            # 以下是索力增量的分析程序
            R2_test = r2_score(norm_prediction_test[:,-1,:], norm_output_test[:,-1,:], multioutput="raw_values").clone().detach().to("cpu").numpy()
            R2_list.append(R2_test[0])

    return R2_list


def main():
    for missing_option in range(1,4):
        # 定义缺失的输入列的下标list
        if missing_option == 1:
            missing_input_index_list_training = [0]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性

            combinations = [[a] for a in range(0, 9)]
        elif missing_option == 2:
            missing_input_index_list_training = [0, 1]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性

            combinations = [[a, b] for a, b in itertools.product(range(0, 8), range(1, 9)) if a < b]
        elif missing_option == 3:
            missing_input_index_list_training = [0, 1, 2]  # 在这里就讨论丢[0]，[0,1],[0,1,2]各自训练模型的鲁棒性

            combinations = [[a, b, c] for a, b, c in itertools.product(range(0, 7), range(1, 8), range(2, 9)) if
                            a < b and b < c]
        else:
            missing_input_index_list_training = None
            combinations = None
            SystemExit("缺失选项有误！")

        with open("./result/RWKV_TS-missing{}_R2.csv".format(missing_option), mode='w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Cable", "Seed"])

        for s_index in [4, 5, 6]:
            # 初始化基础参数
            parser = train_model_main.define_paras()
            parser = train_model_main.add_prediction_option(parser, prediction_option=s_index)  # ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]各元素下标即为选项

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

            # 分别记录三个随机种子的测试结果
            for seed_num_value in [41, 42, 43]:
                args.seed_num = seed_num_value
                torch.manual_seed(args.seed_num)  # 定义随机种子，开始准备代理模型训练

                R2_list = [args.target_option, "Seed-{}".format(str(seed_num_value))]

                for com_index in range(len(combinations)):  # 每一个缺失组合算一个结果
                    # if com_index == 0:  # 对于训练时拟定的缺失列的结果不验证
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

                    R2_list = cal_test_R2_missing_input(args, test_dataset, R2_list)

                with open("./result/RWKV_TS-missing{}_R2.csv".format(missing_option), mode='a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(R2_list)

            # break
        # break


if __name__ == '__main__':
    main()