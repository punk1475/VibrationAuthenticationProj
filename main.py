import gc
import os

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader

import myutil
from data import SiameseNetDataSet
from myutil import MyUtil
from net import SiameseNet, ContrastiveLoss

MODEL_PATH = "model/siamese.pt"
TRAIN_DATA_PATH = "../data_repository/data_for_proj2/train"
TEST_DATA_PATH = "../data_repository/data_for_proj2/test"


def train(model, epoch_num, optimizer_par, train_data_loader_par, loss_net_par, vali_data_loader_par, train_last_loss,
          threshold=10):
    train_history = []
    bk_time = 0
    check_interval = 5
    reminder = check_interval-1
    for epoch in range(0, epoch_num):
        model.train()
        for __data in train_data_loader_par:
            __data_pair1, __data_pair2, __data_pair3, __same_flag = __data
            __data_pair = MyUtil.set_data_on_cuda(__data_pair1, __data_pair2, __data_pair3)

            # gc.collect()
            # torch.cuda.empty_cache()

            optimizer_par.zero_grad()
            output1, output2 = model(__data_pair)
            loss = loss_net_par(output1, output2, __same_flag)
            loss.backward()
            optimizer_par.step()

            # optimizer_par.zero_grad()
            # output1, output2 = model(__data_pair2)
            # loss = loss_net_par(output1, output2, __same_flag)
            # loss.backward()
            # optimizer_par.step()
            #
            # optimizer_par.zero_grad()
            # output1, output2 = model(__data_pair3)
            # loss = loss_net_par(output1, output2, __same_flag)
            # loss.backward()
            # optimizer_par.step()

        # print(f"Epoch number: {epoch}\n")
        if epoch % check_interval == reminder:
            _loss_val = val(model, vali_data_loader_par, loss_net_par)
            train_history.append(_loss_val)
            if _loss_val < train_last_loss:
                train_last_loss = _loss_val
                MyUtil.store_model(model, loss_net_par, optimizer_par, last_loss, MODEL_PATH)
                bk_time = epoch
            else:
                if bk_time + threshold*check_interval < epoch:
                    break
    plt.plot(range(0, train_history.__len__()), train_history)
    plt.savefig("train.png")


def val(model, val_data_loader, val_criterion):
    model.eval()
    total_loss = 0
    for __data in val_data_loader:
        _loss_value = 0
        __data_pair1, __data_pair2, __data_pair3, __same_flag = __data
        __data_pair = MyUtil.set_data_on_cuda(__data_pair1, __data_pair2, __data_pair3)
        # gc.collect()
        # torch.cuda.empty_cache()
        with torch.no_grad():
            _output1, _output2 = model(__data_pair)
            _loss_value += val_criterion(_output1, _output2, __same_flag).cpu().item()

            # _output1, _output2 = model(__data_pair2)
            # _loss_value += val_criterion(_output1, _output2, __same_flag).cpu().item()
            #
            # _output1, _output2 = model(__data_pair3)
            # _loss_value += val_criterion(_output1, _output2, __same_flag).cpu().item()

            total_loss += _loss_value * len(__same_flag)
    avg_loss = total_loss / len(val_data_loader.dataset)
    # print(f"score:{avg_loss}")
    return avg_loss


def get_value_of_net(model, data_pair):
    output1, output2 = model(data_pair)
    p_dist = nn.PairwiseDistance(p=2)
    distance = p_dist(output1, output2)
    return distance, output1, output2


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    train_data = SiameseNetDataSet(TRAIN_DATA_PATH, 3)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=5, num_workers=4, pin_memory=True)

    test_data = SiameseNetDataSet(TRAIN_DATA_PATH, 3)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=5, num_workers=4, pin_memory=True)

    # FIXME:这里在服务器训练时需要修改
    net = SiameseNet(3, 2, 3, 512, 8).cuda()
    criterion = ContrastiveLoss(2).cuda()
    max_epoch_num = 1000
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    last_loss = MyUtil.load_existed_model(net, criterion, optimizer, MODEL_PATH)

    train(net, max_epoch_num, optimizer, train_loader, criterion, test_loader, last_loss)

    # data_iter = iter(test_loader)
    # score = 0.0
    # pos = 0.0
    # f_pos = 0.0
    # neg = 0.0
    # f_neg = 0.0
    # test_num = 400
    # for i in range(test_num):
    #     data_pair1, data_pair2, data_pair3, same_flag = next(data_iter)
    #     data_pair1, data_pair2, data_pair3 = MyUtil.set_data_on_cuda(data_pair1, data_pair2, data_pair3)
    #
    #
    #
    #
    #     dist, o1, o2 = get_value_of_net(net, data_pair2)
    #     dist = dist.tolist()
    #     print(f"dist:{dist}, same:{same_flag}")
    #     if dist[0] < 8.0:
    #         pos += 1.0
    #         if id1[0] == id2[0]:
    #             score += 1.0
    #         else:
    #             f_pos += 1.0
    #     else:
    #         neg += 1.0
    #         if id1[0] != id2[0]:
    #             score += 1.0
    #         else:
    #             f_neg += 1.0
    #     print(f"id1:{id1},id2:{id2},distance:{dist},out1:{o1},out2{o2}\n")
    # print(f"score is :{score / test_num * 100.0},假阳性率为 ：{f_pos / pos},假阴性率为{f_neg / neg}")
