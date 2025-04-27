import argparse
from argparse import ArgumentParser


def define_paras():
    parser: ArgumentParser = argparse.ArgumentParser(description='RNN for time series prediction')
    parser.add_argument('-model_option', type=str, default='LSTNet', help="当前创建和训练的模型类型选项")
    parser.add_argument('-model_name', type=str, default='LSTNet_model', help="当前创建和训练的模型名称")

    # data
    parser.add_argument('-ratio_train_test', type=float, default=0.9, help="训练集与测试集的比例")
    parser.add_argument('-ratio_train_val', type=float, default=0.9, help="训练集与测试集的比例")
    parser.add_argument('-shuffle', type=bool, default=True, help="是否打乱训练数据加载器中的数据顺序")
    parser.add_argument('-sample_root', type=str, default='./source_data/uni_delta_monitor_data/', help="训练样本文件根目录")

    # learning
    parser.add_argument('-seed_num', type=int, default=42, help="随机种子")
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.2, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=300, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=256, help="批次大小")
    parser.add_argument('-early_stopping', type=int, default=100, help="早停数")
    parser.add_argument('-logging', type=int, default=50, help="每多少个epoch输出一次损失的变化")

    # model
    parser.add_argument('-hid_RNN', type=int, default=80, help="RNN的各层神经元特征数")
    parser.add_argument('-input_feature_num', type=int, default=10, help="输入X的特征维度")
    parser.add_argument('-output_feature_num', type=int, default=1, help="输出Y的特征维度")
    parser.add_argument('-layer_num', type=int, default=2, help="堆叠LSTM的层数")
    parser.add_argument('-window_size', type=int, default=64, help="窗口尺寸，即每个LSTM样本涵盖的数据量")
    parser.add_argument('-step_size', type=int, default=1, help="滑动步长")

    # device
    parser.add_argument('-use_gpu', type=bool, default=True, help="是否开启gpu训练")
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    # output
    parser.add_argument('-trained_model_root', type=str, default='./result/networks/LSTNet/', help="已训练模型的存储位置")
    parser.add_argument('-trained_model_loss_root', type=str, default='./result/train_loss/LSTNet/', help="模型损失图像存储路径")
    parser.add_argument('-trained_model_loss_value_root', type=str, default='./result/Loss_value/LSTNet/',
                        help="模型损失值存储路径")
    parser.add_argument('-reconstruction_outcome_root', type=str, default='./result/reconstruction_result/LSTNet/',
                        help="模型重构值存储路径")
    parser.add_argument('-R2_root', type=str, default='./result/R2/LSTNet/', help="模型损失值存储路径")

    return parser


def add_prediction_option(parser, prediction_option):
    prediction_target_list = ["S2", "S8", "S10", "S14", "S16", "S17", "S18"] # 分别对应选项0~6
    prediction_target = prediction_target_list[prediction_option]

    prediction_initial_value_list = [1777.525, 1722.005, 1741.021, 1674.005, 1728.351, 1718.727, 1576.611]
    prediction_target_initial_value = prediction_initial_value_list[prediction_option]

    parser.add_argument('-target_option', type=str, default=prediction_target, help="预测目标的标签名称")
    parser.add_argument('-target_initial_value', type=float, default=prediction_target_initial_value, help="预测目标的初始值")
    parser.add_argument('-max_min_value_index', type=int, default=prediction_option, help="归一化处理极值array的index")

    return parser