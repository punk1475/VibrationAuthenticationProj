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
from data import SiameseNetDataSet, DataController, DataSetWithName
from myutil import MyUtil
from net import SiameseNet, ContrastiveLoss

MODEL_PATH = "model/siamese.pt"
TRAIN_DATA_PATH = "dataSet/train"
TEST_DATA_PATH = "dataSet/validate"
TEMPLATE_PATH = "template.json"


def train(model, epoch_num, optimizer_par, train_data_loader_par, loss_net_par, vali_data_loader_par, train_last_loss,
          threshold=20):
    train_history = []
    bk_time = 0
    check_interval = 5
    reminder = check_interval - 1
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

        if epoch % check_interval == reminder:
            _loss_val = val(model, vali_data_loader_par, loss_net_par)
            train_history.append(_loss_val)
            if _loss_val < train_last_loss:
                train_last_loss = _loss_val
                MyUtil.store_model(model, loss_net_par, optimizer_par, train_last_loss, MODEL_PATH)
                bk_time = epoch
            else:
                if bk_time + threshold * check_interval < epoch:
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


def check(model, test_data_loader):
    model.eval()
    score = 0.0
    pos = 0.0
    f_pos = 0.0
    neg = 0.0
    f_neg = 0.0
    test_num = len(test_data_loader.dataset)
    for __data in test_data_loader:
        _loss_value = 0
        __data_pair1, __data_pair2, __data_pair3, __same_flag = __data
        __data_pair = MyUtil.set_data_on_cuda(__data_pair1, __data_pair2, __data_pair3)

        with torch.no_grad():
            flag = torch.Tensor(__same_flag).cuda()
            dist, o1, o2 = get_value_of_net(model, __data_pair)
            dist = torch.where(dist >= 1.0, 1, 0)  # dist为1表示神经网络认为两人不相似
            score_tensor = torch.where(dist != flag, 1, 0)
            fpos_tensor = torch.where((dist == flag) & (dist == 0), 1, 0)
            fneg_tensor = torch.where((dist == flag) & (dist == 1), 1, 0)
            pos_tensor = torch.where(dist == 0, 1, 0)
            neg_tensor = torch.where(dist == 1, 1, 0)
            score += torch.sum(score_tensor)
            pos += torch.sum(pos_tensor)
            f_pos += torch.sum(fpos_tensor)
            f_neg += torch.sum(fneg_tensor)
            neg += torch.sum(neg_tensor)
    return score / test_num, f_pos / pos, f_neg / neg


def get_value_of_net(model, data_pair):
    output1, output2 = model(data_pair)
    p_dist = nn.PairwiseDistance(p=2)
    distance = p_dist(output1, output2)
    return distance, output1, output2


def run_for_check_user():
    net = SiameseNet(3, 2, 3, 1024, 8).cuda()
    criterion = ContrastiveLoss(2).cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    MyUtil.load_existed_model(net, criterion, optimizer, MODEL_PATH)
    test_data_controller = DataController(TEST_DATA_PATH, 3)
    user_data_list = []
    score_list = []
    template_info = MyUtil.get_info_of_template(TEMPLATE_PATH)
    net.eval()
    for k, v in test_data_controller.data_dic.items():
        user_data_list.append(DataLoader(DataSetWithName(k, v), shuffle=False, batch_size=1, pin_memory=True))
    for user_data in user_data_list:
        for data in user_data:
            triple_data, name = data
            triple_data = MyUtil.set_single_data_on_cuda(triple_data)
            for i in range(0, 3):
                output = net.forward_one(triple_data[i]['cepstrum_tensor'], triple_data[i]['cwt_tensor'])
                bst_user, dist = MyUtil.compare_and_select_best_user(template_info, output)
                check_dic = {"bst_user": bst_user, "dist": dist, "name": name}
                score_list.append(check_dic)
    print("11111")


def run():
    train_data_controller = DataController(TRAIN_DATA_PATH, 3)
    train_data = SiameseNetDataSet(train_data_controller)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=32, pin_memory=True)
    test_data_controller = DataController(TRAIN_DATA_PATH, 3)
    test_data = SiameseNetDataSet(test_data_controller)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=32, pin_memory=True)

    # FIXME:这里在服务器训练时需要修改
    net = SiameseNet(3, 4, 3, 512, 8).cuda()
    criterion = ContrastiveLoss(2).cuda()
    max_epoch_num = 1000
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 64, eta_min=0, last_epoch=-1)
    last_loss = MyUtil.load_existed_model(net, criterion, optimizer, MODEL_PATH)

    train(net, max_epoch_num, optimizer, train_loader, criterion, test_loader, last_loss)
    score, f_pos, f_neg = check(net, test_loader)
    print(f"score:{score}, fpos:{f_pos}, f_neg:{f_neg}")


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    run_for_check_user()
    # net = SiameseNet(3, 2, 3, 1024, 8).cuda()
    # criterion = ContrastiveLoss(2).cuda()
    # optimizer = optim.Adam(net.parameters(), lr=0.0005)
    # MyUtil.load_existed_model(net, criterion, optimizer, MODEL_PATH)
    # # train_data_controller = DataController(TRAIN_DATA_PATH, 3)
    # # MyUtil.get_template_from_data_controller_and_net(net, train_data_controller, TEMPLATE_PATH)
    # test_data_controller = DataController(TEST_DATA_PATH, 3)
