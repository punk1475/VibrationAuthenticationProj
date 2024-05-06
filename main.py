import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader

from data import SiameseNetDataSet
from myUtil import MyUtil
from net import SiameseNet, ContrastiveLoss

MODEL_PATH = "model/store_siamese.pt"
DATA_PATH = "../data_repository/data_for_proj2"


def train(model, epoch_num, optimizer_par, data_loader_par, loss_net_par):
    train_history = []
    for epoch in range(0, epoch_num):
        for __i, __data in enumerate(data_loader_par, 0):
            __feature1, __feature2, __labels = __data
            __feature1 = __feature1.cuda()
            __feature2 = __feature2.cuda()
            __label1, __label2 = MyUtil.set_label_on_cuda(__labels)
            optimizer_par.zero_grad()
            output1, output2 = model(__feature1, __feature2)
            loss = loss_net_par(output1, output2, __label1, __label2)
            loss.backward()
            optimizer_par.step()
            train_history.append(loss.item())
        print(f"Epoch number: {epoch}\n")
    plt.plot(range(0, train_history.__len__()), train_history)
    plt.show()
    MyUtil.store_model(model, loss_net_par, optimizer_par, MODEL_PATH)


def get_value_of_net(model, _feature1, _feature2):
    output1, output2 = model(_feature1, _feature2)
    p_dist = nn.PairwiseDistance(p=2)
    distance = p_dist(output1, output2)
    return distance, output1, output2


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    siamese_data = SiameseNetDataSet(DATA_PATH, 130, 3)
    train_loader = DataLoader(siamese_data, shuffle=False, batch_size=5)
    net = SiameseNet(3, 8, 3, 5, 7, 130, 500, 10).cuda()
    loss_net = ContrastiveLoss(10).cuda()

    max_epoch_num = 1000
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    MyUtil.load_existed_model(net, loss_net, optimizer, MODEL_PATH)
    # train(net, max_epoch_num, optimizer, train_loader, loss_net)
    test_loader = DataLoader(siamese_data, shuffle=False, batch_size=1)
    data_iter = iter(test_loader)
    score = 0.0
    pos = 0.0
    f_pos = 0.0
    neg = 0.0
    f_neg = 0.0
    for i in range(400):
        feature1, feature2, labels = next(data_iter)
        feature1 = feature1.cuda()
        feature2 = feature2.cuda()
        label1, label2 = MyUtil.set_label_on_cuda(labels)
        id1 = label1["name"].tolist()
        id2 = label2["name"].tolist()
        dist, o1, o2 = get_value_of_net(net, feature1, feature2)
        if dist[0] < 8.0:
            pos += 1.0
            if id1[0] == id2[0]:
                score += 1.0
            else:
                f_pos += 1.0
        else:
            neg += 1.0
            if id1[0] != id2[0]:
                score += 1.0
            else:
                f_neg += 1.0
        print(f"id1:{id1},id2:{id2},distance:{dist},out1:{o1},out2{o2}\n")
    print(f"score is :{score/400.0*100.0},假阳性率为 ：{f_pos/pos},假阴性率为{f_neg/neg}")
