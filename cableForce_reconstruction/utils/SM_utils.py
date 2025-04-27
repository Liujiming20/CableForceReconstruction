import torch

from utils import util_main, model_additional_paras
from utils.DA_RNN import AutoEncForecast
from utils.LSTNet import LSTNet
from utils.RWKV_TS import Model


def create_required_SM(rec_target, missing_num):
    cable_label = "S" + rec_target.split()[1]
    label_list = ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]
    cable_index = label_list.index(cable_label)

    # 加载配置参数
    # 初始化基础参数
    parser = util_main.define_paras()
    parser = util_main.add_prediction_option(parser, prediction_option=cable_index)  # ["S2", "S8", "S10", "S14", "S16", "S17", "S18"]各元素下标即为选项

    model_option = "RWKV_TS"  # LSTNet、DA_RNN、RWKV_TS

    parser = model_additional_paras.add_RWKV_paras(parser)

    args = parser.parse_args()
    args.model_option = model_option

    model_additional_paras.modified_paras_by_missing_fea_num(args, missing_num)

    args.step_size = 1  # 设置未来提前预测的时段长度
    args.window_size = 50
    args.input_feature_num = 10 - missing_num  # 输入特征减少

    args = model_additional_paras.modified_RWKV_args(args)

    torch.manual_seed(args.seed_num)  # 定义随机种子

    # 设置最优超参数组合
    if missing_num == 0:
        args.lr = 0.001
        args.n_heads = 4
        args.gpt_layers = 3
        args.d_model = 512
        args.d_ff = 128
        args.dropout = 0.2
    elif missing_num == 1:
        args.lr = 0.001
        args.n_heads = 4
        args.gpt_layers = 3
        args.d_model = 256
        args.d_ff = 128  # 必须小于等于d_model
        args.dropout = 0.2
    elif missing_num == 2:
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

    device = torch.device("cpu")
    if args.model_option == "LSTNet":
        rnn_model = LSTNet(args)
    elif args.model_option == "DA_RNN":
        rnn_model = AutoEncForecast(args, device)
    else:
        rnn_model = Model(args, device)

    trained_model_path = args.trained_model_root + "{}".format(args.target_option) + "/seed_{}/".format(args.seed_num) + args.model_option + ".pth"
    rnn_model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
    rnn_model.to(device)
    # print(rnn_model)

    return rnn_model, device
