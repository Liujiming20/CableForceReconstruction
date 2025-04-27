"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: LSTNet.py
@time: 2024/10/24 8:29
"""
import torch
import torch.nn as nn


class LSTNet(nn.Module):
    def __init__(self, args):
        super(LSTNet, self).__init__()
        self.batch_size = args.batch_size
        self.P = args.window_size
        self.horizon = args.step_size  # 滑窗之间的步长与需要预存未来多长时间的输出是一致的

        self.skip = args.skip  # skip-RNN的跳跃参数
        self.hw = args.highway_window  # 执行自回归的highway组件的参数，表示利用多少个历史数据进行自回归

        self.m_x = args.input_feature_num  # 10
        self.m_y = args.output_feature_num  # 1

        self.hidR = args.hid_RNN  # RNN层的隐藏神经元数目
        self.hidC = args.hid_CNN  # CNN的输出特征图的通道数
        self.hidS = args.hidSkip  # skip-RNN的隐藏神经元数目
        self.Ck = args.CNN_kernel  # CNN的核数
        self.pt = (self.P - self.Ck) // self.skip  # skip-RNN的跳跃步长

        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m_x))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.drop_out)

        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m_y * self.horizon)
        else:  # 单窗口ANN
            self.linear1 = nn.Linear(self.hidR, self.m_y * self.horizon)

        if self.hw > 0:
            self.highway = nn.Linear(self.hw * self.m_x, self.m_y * self.horizon)

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        if args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m_x)
        c = torch.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.view(-1, self.hw * self.m_x)
            z = self.highway(z)
            z = z.view(-1, self.m_y * self.horizon)  # 调整输出大小
            res = res + z

        if self.output:
            res = self.output(res)
        return res.view(-1, self.horizon, self.m_y)  # 调整输出形状