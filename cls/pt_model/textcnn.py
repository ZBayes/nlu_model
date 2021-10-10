import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as torch_data_util
import math
torch.set_printoptions(precision=8)
class TextCNN(nn.Module):
    def __init__(self, param):
        super(TextCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = param["model_para"]['kernel_num'] # output chanel size
        kernel_size = param["model_para"]['kernel_size']
        vocab_size = param["embedding"]['vocab_size']
        embed_dim = param["embedding"]['embed_dim']
        dropout = param["model_para"]['dropout']
        class_num = param["train_para"]['class_num']
        self.param = param
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # pretrained_weight = np.array(param['pretrained_weight'])
        self.embed.weight = nn.Parameter(torch.FloatTensor(param["embedding"]["word2vector_model"].embedding_weights))
        self.convs = [nn.Conv2d(ci, kernel_num, (i, embed_dim)) for i in kernel_size]
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)
        self.softmax = nn.Softmax(dim = 1)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        x = x.unsqueeze(1)
        x_conv = [self.conv_and_pool(x, conv) for conv in self.convs]
        x_conv = torch.cat(x_conv, 1)  # (batch, 3 * kernel_num)
        logit = self.softmax(self.fc1(x_conv))
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