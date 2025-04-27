"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: train_tools.py
@time: 2024/10/24 9:42
"""
import csv

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

from torch import nn

from torcheval.metrics.functional import mean_squared_error, r2_score

from model_additional_paras import modified_RWKV_args
from utils.DA_RNN import AutoEncForecast
from utils.LSTNet import LSTNet
from utils.RWKV_TS import Model
from utils.cable_prediction_process import back_process
from utils.early_stopping import EarlyStopping


def train_model(args, train_val_dataset):
    torch.manual_seed(args.seed_num)

    # 使用 random_split 进行训练集和验证集的随机划分
    train_size = int(len(train_val_dataset) * args.ratio_train_val)
    val_size = len(train_val_dataset) - train_size

    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    # 定义加载器
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

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
    rnn_model.to(device)

    # 定义损失函数
    criterion = nn.MSELoss()
    criterion.to(device)

    # 指定优化器
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=args.lr)

    # model_save_path = args.trained_model_root + args.model_option + ".pth"
    model_save_path = args.trained_model_root + args.model_option + "_" + args.target_option + "_" + str(args.seed_num) + ".pth"

    early_stopping = EarlyStopping(args.early_stopping, True)
    epochs = []
    loss_epochs = []
    loss_epochs_val = []

    # 开始模型训练
    for epoch in range(args.epochs):
        # if epoch % args.logging == 0:
        #     print("正在训练第{} - {} 个epoch".format(epoch, epoch + args.logging))

        # train model
        rnn_model.train()
        mse_loss_total = 0
        count_train = 0
        for _, (cable_force, env_temp,cable_force_output) in enumerate(dataloader_train):
            count_train += 1
            x_1_tr = env_temp.to(device)
            x_2_tr = cable_force.to(device)

            output_tr = cable_force_output.to(device)

            optimizer.zero_grad()

            if args.model_option == "LSTNet":
                input_tr = torch.cat((x_1_tr, x_2_tr), -1)
                prediction = rnn_model(input_tr)
            elif args.model_option == "DA_RNN":
                prediction = rnn_model(x_1_tr, x_2_tr)
            else:
                input_tr = torch.cat((x_1_tr, x_2_tr), -1)
                prediction = rnn_model(input_tr, None, input_tr, None)

            loss_value = criterion(prediction, output_tr)

            loss_value.backward()
            optimizer.step()

            mse_loss_total += loss_value.item()

        epochs.append(epoch)  # 收集输出损失函数结果的epoch节点
        loss_epochs.append(mse_loss_total / count_train)

        # valuate model
        rnn_model.eval()
        mse_loss_total_val = 0

        count_val = 0
        with torch.no_grad():
            for _, (cable_force_val, env_temp_val, cable_force_output_val) in enumerate(dataloader_val):
                count_val += 1
                x_1_val = env_temp_val.to(device)
                x_2_val = cable_force_val.to(device)

                output_val = cable_force_output_val.to(device)

                if args.model_option == "LSTNet":
                    input_val = torch.cat((x_1_val, x_2_val), -1)
                    prediction_val = rnn_model(input_val)
                elif args.model_option == "DA_RNN":
                    prediction_val = rnn_model(x_1_val, x_2_val)
                else:
                    input_val = torch.cat((x_1_val, x_2_val), -1)
                    prediction_val = rnn_model(input_val, None, input_val, None)

                loss_value_val = criterion(prediction_val, output_val)

                mse_loss_total_val += loss_value_val.item()

            loss_epochs_val.append(mse_loss_total_val / count_val)

            early_stopping(mse_loss_total_val, rnn_model, model_save_path)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

    # 验证集上验证所训练模型的R2
    rnn_model.eval()
    rnn_model.load_state_dict(torch.load(model_save_path))
    with torch.no_grad():
        val_total_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        for _, (cable_force_val, env_temp_val, cable_force_output_val) in enumerate(val_total_loader):
            x_1_val = env_temp_val.to(device)
            x_2_val = cable_force_val.to(device)

            output_val = cable_force_output_val.to(device)

            if args.model_option == "LSTNet":
                input_val = torch.cat((x_1_val, x_2_val), -1)
                prediction_val = rnn_model(input_val)
            elif args.model_option == "DA_RNN":
                prediction_val = rnn_model(x_1_val, x_2_val)
            else:
                input_val = torch.cat((x_1_val, x_2_val), -1)
                prediction_val = rnn_model(input_val, None, input_val, None)

            R2 = r2_score(prediction_val[:,-1,:], output_val[:,-1,:], multioutput="raw_values").clone().detach().to("cpu").numpy()
            with open(args.R2_root + "R2_val.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(R2)

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.subplots_adjust(left=0.068, right=0.988, top=0.998, bottom=0.11)

    ax.plot(epochs, loss_epochs, label="train_loss", linestyle="-", color="blue")
    ax.plot(epochs, loss_epochs_val, label="val_loss", linestyle="--", color="red")

    ax.legend()

    plt.ylim(bottom=0)

    if args.model_option == "LSTNet":
        fig_filepath = args.trained_model_loss_root + "loss_" + str(args.lr) + ".png"
    elif args.model_option == "DA_RNN":
        fig_filepath = args.trained_model_loss_root + "loss_" + str(args.lr) + "_" + str(args.hidden_size_encoder) + "_" + str(args.hidden_size_decoder) + ".png"
    else:
        fig_filepath = args.trained_model_loss_root + "loss_" + str(args.lr) + ".png"
        # fig_filepath = args.trained_model_loss_root + "loss_" + str(args.d_model) + "_" + str(args.d_ff) + ".png"
        # fig_filepath = args.trained_model_loss_root + "missing_" + str(args.missing_option) + "-cable_" + args.target_option + "-seed_" + str(args.seed_num) + ".png"
    # plt.show()
    plt.savefig(fig_filepath)
    plt.close()


def test_model(args, test_dataset):
    torch.manual_seed(args.seed_num)

    # 定义加载器
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

    trained_model_path = args.trained_model_root + args.model_option + ".pth"
    rnn_model.load_state_dict(torch.load(trained_model_path))
    rnn_model.to(device)

    # test model
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
            # 由于本研究针对的是单步预测，因此不存在预测结果重合问题，可直接处理分析
            # 以下是索力增量的分析程序
            R2 = r2_score(norm_prediction_test[:,-1,:], norm_output_test[:,-1,:], multioutput="raw_values").clone().detach().to("cpu").numpy()
            with open(args.R2_root + "R2.csv", "a+", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(R2)

            y_monitor = np.squeeze(norm_output_test.clone().detach().to("cpu").numpy(), axis=1)
            y_predict = np.squeeze(norm_output_test.clone().detach().to("cpu").numpy(), axis=1)

            # # 以下是回归到原始尺度的分析程序
            # y_test_norm = norm_output_test.clone().detach().to("cpu").numpy()
            # y_predict_norm = norm_prediction_test.clone().detach().to("cpu").numpy()
            #
            # y_test_norm = np.squeeze(y_test_norm, axis=1)  # 我的数据的步长为一，直接把第二维舍弃变成二维执行后续分析
            # y_predict_norm = np.squeeze(y_predict_norm, axis=1)
            #
            # y_monitor_ini = back_process(args, y_test_norm)
            # y_predict_ini = back_process(args, y_predict_norm)
            # y_monitor = y_monitor_ini
            # y_predict = y_predict_ini
            #
            # y_monitor_ini_torch = torch.from_numpy(y_monitor_ini[:, :])
            # y_predict_ini_torch = torch.from_numpy(y_predict_ini[:, :])
            #
            # R2 = r2_score(y_predict_ini_torch, y_monitor_ini_torch, multioutput="raw_values").clone().detach().to("cpu").numpy()
            # with open(args.R2_root+"R2.csv", "a+", newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(R2)

            # 利用 y_monitor_ini 的长度生成一个等间隔的 x
            x = np.arange(0, len(y_predict) * 0.5, 0.5)

            # 绘图
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 调整图片边距

            # 绘制第一个数组
            plt.plot(x, y_monitor, label='cable force monitor', color='skyblue', linestyle='-')

            # 绘制第二个数组
            plt.plot(x, y_predict, label='cable force prediction', color='pink', linestyle='--')

            # 添加标题和标签
            plt.xlabel('Time(h)')
            plt.ylabel('Cable force(kN)')

            plt.legend()

            # 显示图像
            # plt.show()


# 以下用于模拟监测数据丢失后需要基于滚动预测方式重构数据的情形
def test_model_rolling(args, test_dataset):
    torch.manual_seed(args.seed_num)

    # 定义加载器
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

    trained_model_path = args.trained_model_root + args.model_option + ".pth"
    rnn_model.load_state_dict(torch.load(trained_model_path))
    rnn_model.to(device)

    # valuate model
    rnn_model.eval()
    with torch.no_grad():
        for _, (cable_force_test, env_temp_test, cable_force_output_test) in enumerate(dataloader_test):
            x_1_test = env_temp_test.to(device)
            x_2_test = cable_force_test.to(device)

            norm_output_test = cable_force_output_test.to(device)

            # 滚动预测，预测步数与测试集长度一致
            maximum_rolling_steps = cable_force_test.shape[0]
            predicted_sequence = []

            # 进行滚动预测
            if args.model_option == "LSTNet":
                input_test = torch.cat((x_1_test, x_2_test), -1)
                current_input = input_test[0:1].clone()  # 克隆初始输入数据
            elif args.model_option == "DA_RNN":
                current_input = x_2_test[0:1].clone()  # 克隆初始输入数据
            else:
                input_test = torch.cat((x_1_test, x_2_test), -1)
                current_input = input_test[0:1].clone()  # 克隆初始输入数据

            dif_count = 0
            for step in range(maximum_rolling_steps):
                if args.model_option == "LSTNet":
                    norm_prediction_test = rnn_model(current_input)
                elif args.model_option == "DA_RNN":
                    norm_prediction_test = rnn_model(x_1_test[step:step + 1], current_input)
                else:
                    norm_prediction_test = rnn_model(current_input, None, current_input, None)

                # 将预测结果保存下来
                predicted_sequence.append(norm_prediction_test.item())  # 存储预测值（浮点数）

                # 将预测结果与其它维度的监测数据组成新的输入数据
                if args.model_option == "LSTNet":
                    other_dims = input_test[step + 1:step + 2, 0:1, :-1]
                    con_vector = torch.cat((other_dims, norm_prediction_test), -1)
                elif args.model_option == "DA_RNN":
                    con_vector = norm_prediction_test
                else:
                    other_dims = input_test[step + 1:step + 2, 0:1, :-1]
                    con_vector = torch.cat((other_dims, norm_prediction_test), -1)

                # 更新输入，保持维度为 (1, time_steps, input_size)
                new_input = torch.cat([current_input[:, 1:, :], con_vector], dim=1)
                # 更新 current_input
                current_input = new_input

                if abs((norm_prediction_test-norm_output_test[step].item())) > 0.1:
                    dif_count += 1
                else:
                    if dif_count > 0:
                        dif_count -= 1
                    else:
                        dif_count = 0

                if dif_count >= 10:
                    break

            # 对偏差可接受范围内的数据进行对比分析
            prediction_array = np.array(predicted_sequence)
            target_array = np.squeeze(norm_output_test.clone().detach().to("cpu").numpy())[:len(prediction_array)]

            y_monitor_ini = back_process(args, target_array)
            y_predict_ini = back_process(args, prediction_array)

            # 利用 y_monitor_ini 的长度生成一个等间隔的 x
            # x = np.linspace(0, len(y_predict_ini) - 1, len(y_predict_ini))
            x = np.arange(0, len(y_predict_ini) * 0.5, 0.5) + 0.5

            # 绘图
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.figure(figsize=(10, 6))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 调整图片边距

            # 绘制第一个数组
            plt.plot(x, y_monitor_ini, label='cable force monitor', color='skyblue', linestyle='-', marker='o')

            # 绘制第二个数组
            plt.plot(x, y_predict_ini, label='cable force prediction', color='pink', linestyle='--', marker='s')

            # 添加标题和标签
            plt.xlabel('Time(h)')
            plt.ylabel('Cable force(kN)')

            # plt.ylim(bottom=0)

            plt.legend()

            # 显示图像
            plt.show()


def test_model_rolling_repeat(args, test_dataset, repeat_num):
    torch.manual_seed(args.seed_num)

    # 定义加载器
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

    trained_model_path = args.trained_model_root + args.model_option + ".pth"
    rnn_model.load_state_dict(torch.load(trained_model_path))
    rnn_model.to(device)

    # valuate model
    rnn_model.eval()
    with torch.no_grad():
        for _, (cable_force_test, env_temp_test, cable_force_output_test) in enumerate(dataloader_test):
            x_1_test = env_temp_test.to(device)
            x_2_test = cable_force_test.to(device)

            norm_output_test = cable_force_output_test.to(device)

            for repeat_index in range(repeat_num):
                strat_index = repeat_index * (len(x_1_test)//repeat_num)

                # 滚动预测，预测步数与测试集长度一致
                maximum_rolling_steps = cable_force_test.shape[0] // repeat_num
                predicted_sequence = []

                # 进行滚动预测
                if args.model_option == "LSTNet":
                    input_test = torch.cat((x_1_test, x_2_test), -1)
                    current_input = input_test[strat_index:strat_index+1].clone()  # 克隆初始输入数据
                elif args.model_option == "DA_RNN":
                    current_input = x_2_test[strat_index:strat_index+1].clone()  # 克隆初始输入数据
                else:
                    input_test = torch.cat((x_1_test, x_2_test), -1)
                    current_input = input_test[strat_index:strat_index+1].clone()  # 克隆初始输入数据

                dif_count = 0
                for step in range(maximum_rolling_steps):
                    if args.model_option == "LSTNet":
                        norm_prediction_test = rnn_model(current_input)
                    elif args.model_option == "DA_RNN":
                        norm_prediction_test = rnn_model(x_1_test[strat_index+step:strat_index+step + 1], current_input)
                    else:
                        norm_prediction_test = rnn_model(current_input, None, current_input, None)

                    # 将预测结果保存下来
                    predicted_sequence.append(norm_prediction_test.item())  # 存储预测值（浮点数）

                    # 将预测结果与其它维度的监测数据组成新的输入数据
                    if args.model_option == "LSTNet":
                        other_dims = input_test[strat_index+step + 1:strat_index+step + 2, 0:1, :-1]
                        con_vector = torch.cat((other_dims, norm_prediction_test), -1)
                    elif args.model_option == "DA_RNN":
                        con_vector = norm_prediction_test
                    else:
                        other_dims = input_test[strat_index+step + 1:strat_index+step + 2, 0:1, :-1]
                        con_vector = torch.cat((other_dims, norm_prediction_test), -1)

                    # 更新输入，保持维度为 (1, time_steps, input_size)
                    new_input = torch.cat([current_input[:, 1:, :], con_vector], dim=1)
                    # 更新 current_input
                    current_input = new_input

                    if abs(norm_prediction_test - norm_output_test[strat_index+step].item()) > 0.1:
                        dif_count += 1
                    else:
                        if dif_count > 0:
                            dif_count -= 1
                        else:
                            dif_count = 0

                    if dif_count >= 10:
                        break

                # 对偏差可接受范围内的数据进行对比分析
                prediction_array = np.array(predicted_sequence)
                target_array = np.squeeze(norm_output_test.clone().detach().to("cpu").numpy())[strat_index:strat_index+len(prediction_array)]

                y_monitor_ini = back_process(args, target_array)
                y_predict_ini = back_process(args, prediction_array)

                # 利用 y_monitor_ini 的长度生成一个等间隔的 x
                # x = np.linspace(0, len(y_predict_ini) - 1, len(y_predict_ini))
                x = np.arange(0, len(y_predict_ini) * 0.5, 0.5) + 0.5

                # 绘图
                plt.rcParams['font.family'] = 'Times New Roman'
                plt.figure(figsize=(10, 6))
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 调整图片边距

                # 绘制第一个数组
                plt.plot(x, y_monitor_ini, label='cable force monitor', color='skyblue', linestyle='-', marker='o')

                # 绘制第二个数组
                plt.plot(x, y_predict_ini, label='cable force prediction', color='pink', linestyle='--', marker='s')

                # 添加标题和标签
                plt.xlabel('Time(h)')
                plt.ylabel('Cable force(kN)')

                # plt.ylim(bottom=0)

                plt.legend()

                # 显示图像
                plt.show()


def test_model_rolling_repeat_record(args, test_dataset, repeat_num):
    torch.manual_seed(args.seed_num)

    # 定义加载器
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

    trained_model_path = args.trained_model_root + "{}/".format(args.target_option) + "seed_{}/".format(str(args.seed_num)) + args.model_option + ".pth"
    rnn_model.load_state_dict(torch.load(trained_model_path))
    rnn_model.to(device)

    # valuate model
    rnn_model.eval()
    with torch.no_grad():
        for _, (cable_force_test, env_temp_test, cable_force_output_test) in enumerate(dataloader_test):
            x_1_test = env_temp_test.to(device)
            x_2_test = cable_force_test.to(device)

            norm_output_test = cable_force_output_test.to(device)

            duration_list = [args.target_option + "-" + "seed_{}".format(str(args.seed_num))]
            for repeat_index in range(repeat_num):
                strat_index = repeat_index * (len(x_1_test)//repeat_num)

                # 滚动预测，预测步数与测试集长度一致
                maximum_rolling_steps = cable_force_test.shape[0] // repeat_num
                predicted_sequence = []

                # 进行滚动预测
                if args.model_option == "LSTNet":
                    input_test = torch.cat((x_1_test, x_2_test), -1)
                    current_input = input_test[strat_index:strat_index+1].clone()  # 克隆初始输入数据
                elif args.model_option == "DA_RNN":
                    current_input = x_2_test[strat_index:strat_index+1].clone()  # 克隆初始输入数据
                else:
                    input_test = torch.cat((x_1_test, x_2_test), -1)
                    current_input = input_test[strat_index:strat_index+1].clone()  # 克隆初始输入数据

                dif_count = 0
                for step in range(maximum_rolling_steps):
                    if args.model_option == "LSTNet":
                        norm_prediction_test = rnn_model(current_input)
                    elif args.model_option == "DA_RNN":
                        norm_prediction_test = rnn_model(x_1_test[strat_index+step:strat_index+step + 1], current_input)
                    else:
                        norm_prediction_test = rnn_model(current_input, None, current_input, None)

                    # 将预测结果保存下来
                    predicted_sequence.append(norm_prediction_test.item())  # 存储预测值（浮点数）

                    # 将预测结果与其它维度的监测数据组成新的输入数据
                    if args.model_option == "LSTNet":
                        other_dims = input_test[strat_index+step + 1:strat_index+step + 2, 0:1, :-1]
                        con_vector = torch.cat((other_dims, norm_prediction_test), -1)
                    elif args.model_option == "DA_RNN":
                        con_vector = norm_prediction_test
                    else:
                        other_dims = input_test[strat_index+step + 1:strat_index+step + 2, 0:1, :-1]
                        con_vector = torch.cat((other_dims, norm_prediction_test), -1)

                    # 更新输入，保持维度为 (1, time_steps, input_size)
                    new_input = torch.cat([current_input[:, 1:, :], con_vector], dim=1)
                    # 更新 current_input
                    current_input = new_input

                    if abs(norm_prediction_test - norm_output_test[strat_index+step].item()) > 0.1:
                        dif_count += 1
                    else:
                        if dif_count > 0:
                            dif_count -= 1
                        else:
                            dif_count = 0

                    if dif_count >= 10:
                        break

                # 记录这一个segment的滚动预测时长
                duration = 0.5 * len(predicted_sequence)
                duration_list.append(duration)

            with open("./result/{}.csv".format(args.model_option), mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(duration_list)


