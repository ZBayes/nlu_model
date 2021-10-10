import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as torch_data_util
import math
torch.set_printoptions(precision=8)
class TextRCNN(nn.Module):
    def __init__(self, param):
        super(TextRCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = param['kernel_num'] # output chanel size
        kernel_size = param['kernel_size']
        vocab_size = param['vocab_size']
        embed_dim = param['embed_dim']
        dropout = param['dropout']
        class_num = param['class_num']
        lstm_hidden = param.get('lstm_hidden', 128)
        lstm_num_layers = param.get('lstm_num_layers', 2)
        pad_size = param.get("pad_size", 20)
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.embed.weight = nn.Parameter(torch.FloatTensor(param['pre_word_embeds']))


        self.lstm = nn.LSTM(embed_dim, lstm_hidden, lstm_num_layers,
                        bidirectional=True, batch_first=True, dropout=dropout)
        self.maxpool = nn.MaxPool1d(pad_size)
        self.fc1 = nn.Linear(lstm_hidden * 2 + embed_dim, class_num)
        self.softmax = nn.Softmax(dim = 1)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        x_lstm, _ = self.lstm(x)
        x_cat = torch.cat((x, x_lstm), 2)
        out = F.relu(x_cat)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc1(out)
        logit = self.softmax(out)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()