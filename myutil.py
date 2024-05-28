import json
import os.path
import random

import pandas as pd
import torch
from scipy.stats import pearsonr
from torch import nn


class MyUtil:

    @staticmethod
    def set_data_on_cuda(data_pair1, data_pair2, data_pair3):
        choice_seed = random.randint(0, 2)
        match choice_seed:
            case 0:
                for i in range(0, 2):
                    for k, v in data_pair1[i].items():
                        data_pair1[i][k] = v.float().cuda()
                return data_pair1
            case 1:
                for i in range(0, 2):
                    for k, v in data_pair2[i].items():
                        data_pair2[i][k] = v.float().cuda()
                return data_pair2
            case 2:
                for i in range(0, 2):
                    for k, v in data_pair3[i].items():
                        data_pair3[i][k] = v.float().cuda()
                return data_pair3

    @staticmethod
    def load_existed_model(model, loss_net, optimizer, path):
        if os.path.exists(path):
            check_point = torch.load(path)
            model.load_state_dict(check_point["model_state_dict"])
            loss_net.load_state_dict(check_point["loss_state_dict"])
            optimizer.load_state_dict(check_point["optimizer_state_dict"])
            return check_point["last_loss"]
        return float("inf")

    @staticmethod
    def store_model(model, loss, optimizer, last_loss, path):
        torch.save({
            "model_state_dict": model.state_dict(),
            "loss_state_dict": loss.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "last_loss": last_loss
        }, path)

    @staticmethod
    def dataIsSimilar(x1, x2, sigma=0.6):
        # 0为相关系数，1为显著性水平
        r = abs(pearsonr(x1, x2)[0])
        if r < sigma:
            return False, r
        else:
            return True, r

    @staticmethod
    def get_label_pair_from_one_dataframe(df):
        label_vec1 = df.iloc[0, 0:5].tolist()
        label_vec2 = df.iloc[1, 0:5].tolist()
        return [label_vec1, label_vec2], df["filename"].tolist()

    @staticmethod
    def get_label_pair_from_dataframe_pair(df1, df2):
        label_vec1 = df1.iloc[0, 0:5].tolist()
        label_vec2 = df2.iloc[0, 0:5].tolist()
        return [label_vec1, label_vec2], df1["filename"].tolist() + df2["filename"].tolist()

    @staticmethod
    def set_single_data_on_cuda(data):
        # 前三项是三段时间的特征
        for i in range(0, 3):
            for k, v in data[i].items():
                data[i][k] = v.float().cuda()
        return data

    @staticmethod
    def get_template_from_data_controller_and_net(net, data_controller, path, output_size=8):
        data_dic = data_controller.data_dic
        template_dic = {}
        net.eval()
        for username, file_dic in data_dic.items():
            template_tensor = torch.zeros(output_size, dtype=torch.float).cuda()
            for file_name, triple_data in file_dic.items():
                triple_data = MyUtil.set_single_data_on_cuda(triple_data)
                with torch.no_grad():
                    for i in range(0, 3):
                        template_tensor += net.forward_one(
                            triple_data[i]['cepstrum_tensor'].unsqueeze(dim=0),
                            triple_data[i]['cwt_tensor'].unsqueeze(dim=0)).squeeze(dim=0)
            template_tensor = template_tensor / (len(file_dic) * 3)
            template_dic[username] = template_tensor.tolist()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(template_dic, f, ensure_ascii=False)

    @staticmethod
    def get_info_of_template(path):
        with open(path, 'r', encoding='utf-8') as f:
            template_dic = json.load(f)
            for k, v in template_dic.items():
                template_dic[k] = torch.Tensor(v).unsqueeze(dim=0).cuda()
            return template_dic

    @staticmethod
    def compare_and_select_best_user(template_dic, output_tensor):
        p_dist = nn.PairwiseDistance(p=1)
        # 为两个（1，x）大小的张量
        dist = float("inf")
        bst_key = None
        for k, v in template_dic.items():
            distance = p_dist(output_tensor, v)[0].tolist()
            if distance < dist:
                bst_key = k
                dist = distance
        return bst_key, dist
