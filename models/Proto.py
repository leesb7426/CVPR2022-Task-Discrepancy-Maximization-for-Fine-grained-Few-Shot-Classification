import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import Conv_4, ResNet
import math
from .TDM import TDM

class Proto(nn.Module):

    def __init__(self, args=None): # way = None, shots = None, resnet = False

        super().__init__()

        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet

        self.resolution = 25
        if self.resnet:
            self.num_channel = 640
            self.dim = 640
            self.feature_extractor = ResNet.resnet12(drop=True)
        else:
            self.num_channel = 64
            self.dim = 64 * 25
            self.feature_extractor = Conv_4.BackBone(self.num_channel)

        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.TDM = TDM(self.args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def get_feature_vector(self, inp):

        feature_map = self.feature_extractor(inp)

        return feature_map

    def get_neg_l2_dist(self, inp, way, shot, query_shot):

        feature_map = self.get_feature_vector(inp)
        _, d, h, w = feature_map.shape
        m = h * w
        support = feature_map[:way * shot].view(way, shot, d, m)
        query = feature_map[way * shot:].view(1, -1, d, m)
        query_num = query.shape[1]

        if self.args.TDM:
            weight = self.TDM(support, query.transpose(0, 1))
            weight = weight.view(way, query_num, d, 1)

        centroid = support.mean(dim=1).unsqueeze(dim=1)
        if self.args.TDM:
            centroid = centroid * weight
            query = query * weight

        if self.resnet:
            centroid = centroid.mean(dim=-1)
            query = query.mean(dim=-1)
        else:
            centroid = centroid.view(way, -1, self.dim)
            query = query.view(-1, query_num, self.dim)

        l2_dist = torch.sum(torch.pow(centroid - query, 2), dim=-1).transpose(0, 1)
        neg_l2_dist = l2_dist.neg()

        return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot
                                           )

        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index

    def forward(self, inp):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=self.way,
                                           shot=self.shots[0],
                                           query_shot=self.shots[1]
                                           )

        logits = neg_l2_dist / self.dim * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction
