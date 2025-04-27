"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: model_additional_paras.py
@time: 2024/10/24 9:31
"""


def add_LSTNet_paras(parser):
    # model
    parser.add_argument('-skip', type=int, default=8, help="skip-RNN的跳跃参数")
    parser.add_argument('-highway_window', type=int, default=3, help="使用历史倒数多少步数据进行回归")
    parser.add_argument('-hid_CNN', type=int, default=48, help="CNN的输出特征图的通道数")
    parser.add_argument('-hidSkip', type=int, default=64, help="skip-RNN的隐藏神经元数目")
    parser.add_argument('-CNN_kernel', type=int, default=6, help="CNN的核数")
    parser.add_argument('-output_fun', type=str, default="tanh", help="输出激活函数")

    return parser


def add_DARNN_paras(parser):
    parser.add_argument('-denoising', type=bool, default=True, help="是否对源数据施加噪声以实现降噪功能")
    parser.add_argument('-directions', type=int, default=1, help="RNN神经元的方向数，1表示单向，2表示双向")
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1, help="梯度累积步数，用于模拟较大的批次处理。设置为 1 表示不使用梯度累积")
    parser.add_argument('-hidden_size_encoder', type=int, default=32, help="编码器LSTM隐藏层的单元数量")
    parser.add_argument('-hidden_size_decoder', type=int, default=64, help="解码器LSTM隐藏层的单元数量")
    parser.add_argument('-input_att', type=bool, default=True, help="控制是否使用输入特征选择注意力机制")
    parser.add_argument('-lrs_step_size', type=int, default=50, help="学习率调度步数，表示经过多少步后调整学习率")
    parser.add_argument('-max_grad_norm', type=float, default=0.1, help="梯度裁剪阈值，用于防止梯度爆炸")
    parser.add_argument('-reg1', type=bool, default=True, help="控制L1正则化是否使用")
    parser.add_argument('-reg2', type=bool, default=False, help="控制L2正则化是否使用")
    parser.add_argument('-reg_factor1', type=float, default=1e-4, help="L1正则化系数")
    parser.add_argument('-reg_factor2', type=float, default=1e-4, help="L2正则化系数")
    parser.add_argument('-seq_len', type=int, default=64, help="时序数据长度")
    parser.add_argument('-temporal_att', type=bool, default=True, help="控制是否使用时间步注意力机制")

    return parser


def add_RWKV_paras(parser):
    parser.add_argument('-seq_len', type=int, default=64, help="时序数据长度")
    parser.add_argument('-pred_len', type=int, default=1, help="预测未来步长")
    parser.add_argument('-patch_size', type=int, default=1, help="分片大小")
    parser.add_argument('-stride', type=int, default=1, help="滑窗大小")
    parser.add_argument('-d_ff', type=int, default=512, help="全连接层fcn的维度")  # 从2048减小到512
    parser.add_argument('-enc_in', type=int, default=7, help="编码器的输入维度")
    parser.add_argument('-dec_in', type=int, default=7, help="解码器的输入维度")
    parser.add_argument('-c_out', type=int, default=7, help="输出特征维度")
    parser.add_argument('-d_model', type=int, default=512, help="模型的维度")
    parser.add_argument('-embed', type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]")  # 我们的案例没有把时间纳入，可以不考虑
    parser.add_argument('-freq', type=str, default="h", help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h]")    # 我们的案例没有把时间纳入，可以不考虑
    parser.add_argument('-dropout', type=float, default=0.1, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-gpt_layers', type=int, default=6, help="RWKV 模型结构层数")
    parser.add_argument('-n_heads', type=int, default=8, help="多头注意力机制的头数")

    return parser


def modified_paras_by_model_option(args):
    if args.model_option != "LSTNet":
        if args.model_option == "DA_RNN":
            model_name = "DA_RNN_model"
        elif args.model_option == "RWKV_TS":
            model_name = "RWKV_TS_model"
        else:
            raise SystemExit('选择训练的模型选项不存在，请核查！')

        args.model_name = model_name
        args.trained_model_root = "./result/networks/{}/".format(args.model_option)
        args.trained_model_loss_root = "./result/train_loss/{}/".format(args.model_option)
        args.trained_model_loss_value_root = "./result/Loss_value/{}/".format(args.model_option)
        args.reconstruction_outcome_root = "./result/reconstruction_result/{}/".format(args.model_option)
        args.R2_root = "./result/R2/{}/".format(args.model_option)


def modified_RWKV_args(args):
    args.seq_len = args.window_size
    args.pred_len = args.step_size
    args.enc_in = args.input_feature_num
    args.dec_in = args.output_feature_num
    args.c_out = args.output_feature_num
    args.dropout = args.drop_out

    return args
