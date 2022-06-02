import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4, ResNet
import math
from .TDM import TDM

class FRN(nn.Module):

    def __init__(self, args=None):

        super().__init__()

        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet

        if self.resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12(drop=True)
        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.d = num_channel
        self.resolution = 25

        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.r = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.TDM = TDM(self.args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def get_feature_map(self, inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)

        if self.resnet:
            feature_map = feature_map / np.sqrt(640)

        return feature_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()  # N,HW,C

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):

        way, shot, resolution, d = support.shape
        query_num = query.shape[1]

        support = support.view(way, shot * resolution, d)
        query = query.view(-1, query_num * resolution, d)

        reg = support.size(1) / support.size(2)
        lam = reg * alpha.exp() + 1e-6
        rho = beta.exp()

        st = support.permute(0, 2, 1)  # way, d, shot*resolution

        sts = st.matmul(support)  # way, d, d
        m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
        hat = m_inv.matmul(sts)  # way, d, d
        Q_bar = query.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d

        if self.args.TDM:
            support = support.view(way, shot, resolution, d)
            query = query.view(-1, query_num, resolution, d)
            weight = self.TDM(support, query)
            weight = weight.view(way, query_num, 1, d)

            Q_bar = Q_bar.reshape(way, query_num, resolution, d)
            query = query.reshape(-1, query_num, resolution, d)
            Q_bar = Q_bar * weight
            query = query * weight
            Q_bar = Q_bar.reshape(way, query_num * resolution, d)
            query = query.reshape(-1, query_num * resolution, d)

        dist = (Q_bar - query).pow(2).sum(2).permute(1, 0)
        return dist

    def get_neg_l2_dist(self, inp, way, shot, query_shot, return_support=False):

        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]

        feature_map = self.get_feature_map(inp)
        support = feature_map[:way * shot].view(way, shot, resolution, d)
        query = feature_map[way * shot:].view(1, -1, resolution, d)

        recon_dist = self.get_recon_dist(query=query, support=support, alpha=alpha, beta=beta)  # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().\
            view(-1, resolution, way).mean(1)  # way*query_shot, way

        if return_support:
            return neg_l2_dist, support.view(way, shot * resolution, d)
        else:
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

        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True
                                                    )

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction, support