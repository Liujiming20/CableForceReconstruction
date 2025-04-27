"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: test_model_R2.py
@time: 2024/11/6 8:48
"""
import csv

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from torcheval.metrics.functional import r2_score, mean_squared_error

import model_additional_paras
import train_model_main
from utils import cable_prediction_process
from utils.DA_RNN import AutoEncForecast
from utils.LSTNet import LSTNet
from utils.RWKV_TS import Model
from utils.time_series_dataset import TimeSeriesDatasetTraining
from utils.train_tools import test_model_rolling_repeat_record


def cal_training_test_R2(args, train_val_dataset, test_dataset, R2_list):
    torch.manual_seed(args.seed_num)

    # 定义加载器
    dataloader_train = DataLoader(train_val_dataset, batch_size=len(train_val_dataset)//9, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # 定义GPU
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")

    if args.model_option == "LSTNet":
        rnn_model = LSTNet(args)
    elif args.model_option == "DA_RNN":
        rnn_model = AutoEncForecast(args, device)
    else:
        rnn_model = Model(args, device)

    trained_model_path = args.trained_model_root + "/{}".format(args.target_option) + "/seed_{}/".format(args.seed_num) + args.model_option + ".pth"
    rnn_model.load_state_dict(torch.load(trained_model_path))
    rnn_model.to(device)

    # cal training set
    all_predictions = []
    all_targets = []
    rnn_model.eval()
    with torch.no_grad():
        for _, (cable_force_train, env_temp_train, cable_force_output_train) in enumerate(dataloader_train):
            x_1_train = env_temp_train.to(device)
            x_2_train = cable_force_train.to(device)

            norm_output_train = cable_force_output_train.to(device)

            if args.model_option == "LSTNet":
                input_train = torch.cat((x_1_train, x_2_train), -1)
                norm_prediction_train = rnn_model(input_train)
            elif args.model_option == "DA_RNN":
                norm_prediction_train = rnn_model(x_1_train, x_2_train)
            else:
                input_train = torch.cat((x_1_train, x_2_train), -1)
                norm_prediction_train = rnn_model(input_train, None, input_train, None)

            # 累积预测值和真实值
            all_predictions.append(norm_prediction_train[:, -1, :].clone().detach().cpu())
            all_targets.append(norm_output_train[:, -1, :].clone().detach().cpu())

            # 释放GPU缓存
        torch.cuda.empty_cache()

    # 将所有累积的结果拼接成一个完整的张量
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 最终计算整体R²
    R2_train_final = r2_score(all_predictions, all_targets, multioutput="raw_values").clone().detach().to("cpu").numpy()
    R2_list.append(R2_train_final[0])

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


def cal_training_test_R2_RMSE(args, train_val_dataset, test_dataset, R2_list, RMSE_list):
    torch.manual_seed(args.seed_num)

    # 定义加载器
    dataloader_train = DataLoader(train_val_dataset, batch_size=len(train_val_dataset)//9, shuffle=False)
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # 定义GPU
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")

    if args.model_option == "LSTNet":
        rnn_model = LSTNet(args)
    elif args.model_option == "DA_RNN":
        rnn_model = AutoEncForecast(args, device)
    else:
        rnn_model = Model(args, device)

    trained_model_path = args.trained_model_root + "/{}".format(args.target_option) + "/seed_{}/".format(args.seed_num) + args.model_option + ".pth"
    rnn_model.load_state_dict(torch.load(trained_model_path))
    rnn_model.to(device)

    # cal training set
    all_predictions = []
    all_targets = []
    rnn_model.eval()
    with torch.no_grad():
        for _, (cable_force_train, env_temp_train, cable_force_output_train) in enumerate(dataloader_train):
            x_1_train = env_temp_train.to(device)
            x_2_train = cable_force_train.to(device)

            norm_output_train = cable_force_output_train.to(device)

            if args.model_option == "LSTNet":
                input_train = torch.cat((x_1_train, x_2_train), -1)
                norm_prediction_train = rnn_model(input_train)
            elif args.model_option == "DA_RNN":
                norm_prediction_train = rnn_model(x_1_train, x_2_train)
            else:
                input_train = torch.cat((x_1_train, x_2_train), -1)
                norm_prediction_train = rnn_model(input_train, None, input_train, None)

            # 累积预测值和真实值
            all_predictions.append(norm_prediction_train[:, -1, :].clone().detach().cpu())
            all_targets.append(norm_output_train[:, -1, :].clone().detach().cpu())

            # 释放GPU缓存
        torch.cuda.empty_cache()

    # 将所有累积的结果拼接成一个完整的张量
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 最终计算整体R²
    R2_train_final = r2_score(all_predictions, all_targets, multioutput="raw_values").clone().detach().to("cpu").numpy()
    R2_list.append(R2_train_final[0])
    # 计算整体的RMSE
    all_predictions_real = cable_prediction_process.back_process(args, all_predictions)
    all_targets_real = cable_prediction_process.back_process(args, all_targets)
    mse_train_final = mean_squared_error(all_predictions_real, all_targets_real, multioutput="raw_values").clone().detach().to("cpu").numpy()
    rmse_train_final = np.sqrt(mse_train_final)
    RMSE_list.append(rmse_train_final[0])

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

            # 最终计算整体R²
            R2_test = r2_score(norm_prediction_test[:,-1,:], norm_output_test[:,-1,:], multioutput="raw_values").clone().detach().to("cpu").numpy()
            R2_list.append(R2_test[0])
            # 计算整体的RMSE
            all_predictions_test_real = cable_prediction_process.back_process(args, norm_prediction_test[:,-1,:])
            all_targets_test_real = cable_prediction_process.back_process(args, norm_output_test[:,-1,:])
            mse_test_final = mean_squared_error(all_predictions_test_real, all_targets_test_real, multioutput="raw_values").clone().detach().to("cpu").numpy()
            rmse_test_final = np.sqrt(mse_test_final)
            RMSE_list.append(rmse_test_final[0])

    return R2_list, RMSE_list


def main():
    model_option = "RWKV_TS"  # LSTNet、DA_RNN、RWKV_TS

    with open("./result/R2/{}_R2.csv".format(model_option), mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cable", "Seed 41-train", "Seed 41-test", "Seed 42-train", "Seed 42-test", "Seed 43-train", "Seed 43-test"])

    with open("./result/R2/{}_RMSE.csv".format(model_option), mode='w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cable", "Seed 41-train", "Seed 41-test", "Seed 42-train", "Seed 42-test", "Seed 43-train", "Seed 43-test"])

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

        R2_list = [args.target_option]
        RMSE_list = [args.target_option]
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

            # 记录各个随机种子的R2
            # R2_list = cal_training_test_R2(args, train_val_dataset, test_dataset, R2_list)
            R2_list, RMSE_list = cal_training_test_R2_RMSE(args, train_val_dataset, test_dataset, R2_list, RMSE_list)

        with open("./result/R2/{}_R2.csv".format(model_option), mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(R2_list)

        with open("./result/R2/{}_RMSE.csv".format(model_option), mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(RMSE_list)


if __name__ == '__main__':
    main()