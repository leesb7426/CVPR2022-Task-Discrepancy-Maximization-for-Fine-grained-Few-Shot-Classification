import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss, BCEWithLogitsLoss, BCELoss


def auxrank(support):
    way = support.size(0)
    shot = support.size(1)
    support = support/support.norm(2).unsqueeze(-1)
    L1 = torch.zeros((way**2-way)//2).long().cuda()
    L2 = torch.zeros((way**2-way)//2).long().cuda()
    counter = 0
    for i in range(way):
        for j in range(i):
            L1[counter] = i
            L2[counter] = j
            counter += 1
    s1 = support.index_select(0, L1) # s^2-s, s, d
    s2 = support.index_select(0, L2) # s^2-s, s, d
    dists = s1.matmul(s2.permute(0,2,1)) # s^2-s, s, s
    assert dists.size(-1)==shot
    frobs = dists.pow(2).sum(-1).sum(-1)
    return frobs.sum().mul(.03)


def default_train(train_loader, model, optimizer, writer, iter_counter, gpu_num):

    if gpu_num > 1:
        way = model.module.way
        query_shot = model.module.shots[-1]
        support_shot = model.module.shots[0]

    else:
        way = model.way
        query_shot = model.shots[-1]

    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = nn.NLLLoss().cuda()
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1

        if gpu_num > 1:
            inp_spt = inp[:way * support_shot]
            inp_qry = inp[way * support_shot:]
            qry_num = inp_qry.shape[0]
            inp_list = []
            for i in range(gpu_num):
                inp_qry_fraction = inp_qry[int(qry_num/i):int(qry_num/(i+1))]
                inp_list.append(torch.cat((inp_spt, inp_qry_fraction), dim=0))
            inp = torch.cat(inp_list, dim=0)

        inp = inp.cuda()
        log_prediction, s = model(inp)
        s = s[:way]
        frn_loss = criterion(log_prediction, target)
        aux_loss = auxrank(s)
        loss = frn_loss + aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc

    avg_acc = avg_acc / (i + 1)

    return iter_counter, avg_acc