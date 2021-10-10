import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.utils.data as torch_data_util
import math
import copy
torch.set_printoptions(precision=8)
class TransformerCls(nn.Module):
    def __init__(self, param):
        super(TransformerCls, self).__init__()
        # ci = 1  # input chanel size
        # kernel_num = param['kernel_num'] # output chanel size
        # kernel_size = param['kernel_size']
        vocab_size = param['vocab_size']
        embed_dim = param['embed_dim']
        # dropout = param['dropout']
        class_num = param['class_num']
        # lstm_hidden = param.get('lstm_hidden', 128)
        # lstm_num_layers = param.get('lstm_num_layers', 2)
        # pad_size = param.get("pad_size", 20)
        # num_head = param.get("num_head", 5)
        # num_encoder = param.get("num_encoder", 2)
        # encoder_hidden = param.get("encoder_hidden", 1024)
        # num_encoder = param.get("num_encoder", 2)
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.embed.weight = nn.Parameter(torch.FloatTensor(param['pre_word_embeds']))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        # self.transformer_encoder2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)

        self.fc1 = nn.Linear(embed_dim, class_num)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

        # self.fc1 = nn.Linear(lstm_hidden * 2 + embed_dim, class_num)
        self.softmax = nn.Softmax(dim = 1)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    def forward(self, x):
        # x: (batch, sentence_length)
        emb_out = self.embed(x)

        trans = self.transformer_encoder(emb_out)
        cat_trans = torch.mean(trans, 1)  # (batch, 3 * kernel_num)

        out = self.fc1(cat_trans)
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
            elif isinstance(m, nn.TransformerEncoder):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
