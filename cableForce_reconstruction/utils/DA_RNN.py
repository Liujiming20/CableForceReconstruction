"""
@author: liujiming
@email: jimingLiu@sjtu.edu.cn
@software: PyCharm
@file: DA_RNN.py
@time: 2024/10/26 9:49
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf


def init_hidden(device: torch.device, x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.

    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)


###########################################################################
################################ ENCODERS #################################
###########################################################################

class Encoder(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the model.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.device = device

        self.input_size = args.input_feature_num
        self.hidden_size = args.hidden_size_encoder
        self.seq_len = args.seq_len

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input daa
        """
        h_t, c_t = (init_hidden(self.device, input_data, self.hidden_size),
                    init_hidden(self.device, input_data, self.hidden_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size))

        for t in range(self.seq_len):
            _, (h_t, c_t) = self.lstm(input_data[:, t, :].unsqueeze(0), (h_t, c_t))
            input_encoded[:, t, :] = h_t
        return _, input_encoded


class AttnEncoder(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.device = device

        self.input_size = args.input_feature_num
        self.hidden_size = args.hidden_size_encoder
        self.seq_len = args.seq_len
        self.add_noise = args.denoising
        self.directions = args.directions

        self.lstm = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=1)
        self.attn = nn.Linear(in_features=2 * self.hidden_size + self.seq_len,out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def _get_noise(self, input_data: torch.Tensor, sigma=0.01, p=0.1):
        """
        Get noise.

        Args:
            input_data: (torch.Tensor): tensor of input data
            sigma: (float): variance of the generated noise
            p: (float): probability to add noise
        """
        normal = sigma * torch.randn(input_data.shape)
        mask = np.random.uniform(size=(input_data.shape))
        mask = (mask < p).astype(int)
        noise = normal * torch.tensor(mask)
        return noise

    def forward(self, input_data: torch.Tensor):
        """
        Forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        h_t, c_t = (init_hidden(self.device, input_data, self.hidden_size, num_dir=self.directions),
                    init_hidden(self.device, input_data, self.hidden_size, num_dir=self.directions))

        attentions, input_encoded = (Variable(torch.zeros(input_data.size(0), self.seq_len, self.input_size)),
                                     Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size)))

        if self.add_noise and self.training:
            input_data += self._get_noise(input_data).to(self.device)

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1).to(self.device)), dim=2).to(self.device)  # bs * input_size * (2 * hidden_dim + seq_len)

            e_t = self.attn(x.view(-1, self.hidden_size * 2 + self.seq_len))  # (bs * input_size) * 1
            a_t = self.softmax(e_t.view(-1, self.input_size)).to(self.device)  # (bs, input_size)

            weighted_input = torch.mul(a_t, input_data[:, t, :].to(self.device))  # (bs * input_size)
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            input_encoded[:, t, :] = h_t
            attentions[:, t, :] = a_t

        return attentions, input_encoded


###########################################################################
################################ DECODERS #################################
###########################################################################

class Decoder(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the network.

        Args:
            config:
        """
        super(Decoder, self).__init__()
        self.device = device

        self.seq_len = args.seq_len
        self.hidden_size = args.hidden_size_decoder
        self.output_size = args.output_feature_num

        self.lstm = nn.LSTM(1, self.hidden_size, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, _, y_hist: torch.Tensor):
        """
        Forward pass

        Args:
            _:
            y_hist: (torch.Tensor): shifted target
        """
        h_t, c_t = (init_hidden(self.device, y_hist, self.hidden_size),
                    init_hidden(self.device, y_hist, self.hidden_size))

        for t in range(self.seq_len):
            inp = y_hist[:, t].unsqueeze(0).unsqueeze(2)
            lstm_out, (h_t, c_t) = self.lstm(inp, (h_t, c_t))
        return self.fc(lstm_out.squeeze(0))


class AttnDecoder(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the network.

        Args:
            config:
        """
        super(AttnDecoder, self).__init__()
        self.device = device

        self.seq_len = args.seq_len
        self.encoder_hidden_size = args.hidden_size_encoder
        self.decoder_hidden_size = args.hidden_size_decoder
        self.output_size = args.output_feature_num

        self.attn = nn.Sequential(
            nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=self.output_size, hidden_size=self.decoder_hidden_size)
        self.fc = nn.Linear(self.encoder_hidden_size + self.output_size, self.output_size)
        self.fc_out = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.output_size)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor):
        """
        Perform forward computation.

        Args:
            input_encoded: (torch.Tensor): tensor of encoded input
            y_history: (torch.Tensor): shifted target
        """
        h_t, c_t = (init_hidden(self.device, input_encoded, self.decoder_hidden_size), init_hidden(self.device, input_encoded, self.decoder_hidden_size))
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2), input_encoded.to(self.device)), dim=2)

            x = tf.softmax(self.attn(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)).view(-1, self.seq_len), dim=1)

            context = torch.bmm(x.unsqueeze(1), input_encoded.to(self.device))[:, 0, :]  # (batch_size, encoder_hidden_size)

            y_tilde = self.fc(torch.cat((context.to(self.device), y_history[:, t].to(self.device)), dim=1))  # (batch_size, out_size)

            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(y_tilde.unsqueeze(0), (h_t, c_t))

        return self.fc_out(torch.cat((h_t[0], context.to(self.device)), dim=1))  # predicting value at t=self.seq_length+1


class AutoEncForecast(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AutoEncForecast, self).__init__()
        self.encoder = AttnEncoder(args, device).to(device) if args.input_att else Encoder(args, device).to(device)
        self.decoder = AttnDecoder(args, device).to(device) if args.temporal_att else Decoder(args, device).to(device)

    def forward(self, encoder_input: torch.Tensor, y_hist: torch.Tensor, return_attention: bool = False):
        """
        Forward computation. encoder_input_inputs.

        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
            return_attention: (bool): whether or not to return the attention
        """
        attentions, encoder_output = self.encoder(encoder_input)
        outputs = self.decoder(encoder_output, y_hist.float())

        if return_attention:
            return outputs, attentions
        return outputs.unsqueeze(2)  # 与dataset处理的数据维度一致