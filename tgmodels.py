import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from tgmatrix import *

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TGGCNResnet(nn.Module):
    def __init__(self,  num_classes, in_channel=300, t=0, adj_file=None):
        super(TGGCNResnet, self).__init__()
        self.model = models.resnet101(weights=None)
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        inp = inp.type(torch.FloatTensor)
        tensor = feature.type(torch.FloatTensor)
        tensor = tensor.cuda()
        inp=inp.cuda()
        feature = tensor.permute(0, 3, 1, 2)
        feature = self.features(feature)
        # print("backbone_feature:{}".format(feature.shape))
        feature = self.pooling(feature)
        # print("pooling:{}".format(feature.shape))
        feature = feature.view(feature.size(0), -1)
        # print("reshape:{}".format(feature.shape))

        inp = inp[0]

        adj = gen_adj(self.A).detach()
        # print("adj:{}".format(adj.shape))
        x = self.gc1(inp, adj)
        # print("gc1:{}".format(x.shape))
        x = self.relu(x)
        # print("relu:{}".format(x.shape))
        x = self.gc2(x, adj)
        # print("gc2:{}".format(x.shape))

        x = x.transpose(0, 1)
        # print("transpose:{}".format(x.shape))
        x = torch.matmul(feature, x)
        # print("matmul:{}".format(x.shape))
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


