import os.path
import random

import pandas as pd
import torch
from scipy.stats import pearsonr


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

