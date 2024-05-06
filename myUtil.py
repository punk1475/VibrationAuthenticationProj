import os.path

import pandas as pd
import torch


class MyUtil:
    @staticmethod
    def export_csv_to_list(csv_path, data_len, channels):
        csv_pd = pd.read_csv(csv_path)

        acc_x = csv_pd.loc[:, "acc_x"].values
        acc_x = acc_x.tolist()
        acc_x = MyUtil.cut_data_in_specific_length(acc_x, data_len)
        acc_x = torch.Tensor(acc_x)

        acc_y = csv_pd.loc[:, "acc_y"].values
        acc_y = acc_y.tolist()
        acc_y = MyUtil.cut_data_in_specific_length(acc_y, data_len)
        acc_y = torch.Tensor(acc_y)

        acc_z = csv_pd.loc[:, "acc_z"].values
        acc_z = acc_z.tolist()
        acc_z = MyUtil.cut_data_in_specific_length(acc_z, data_len)
        acc_z = torch.Tensor(acc_z)
        # print(f"this path is {csv_path}")
        x_distance = csv_pd.loc[0, "x_distance"]
        y_distance = csv_pd.loc[0, "y_distance"]
        name = csv_pd.loc[0, "id"]
        # 如果使用多个张量，应在此拼接
        data = torch.cat((acc_x, acc_y, acc_z), 0)
        data = data.reshape(channels, data_len)

        # print(f"{csv_path} close")
        return data, {"x_distance": x_distance, "y_distance": y_distance, "name": name}

    @staticmethod
    def cal_dissimilarity(label1, label2, threshold=11):
        x_distance1 = label1["x_distance"]
        y_distance1 = label1["y_distance"]
        x_distance2 = label2["x_distance"]
        y_distance2 = label2["y_distance"]
        diff = ((x_distance1 - x_distance2) ** 2 + (y_distance1 - y_distance2) ** 2) ** 0.5
        return torch.clamp(diff/threshold, max=1.0)

    @staticmethod
    def cut_data_in_specific_length(data, length):
        if data.__len__() <= length:
            data += [0] * (length - len(data))
        else:
            data = data[0:length]
        return data

    @staticmethod
    def cal_data_length_after_cov(ori_len, cov_kernel_size, layer_num, stride):
        return ori_len-layer_num*(cov_kernel_size-stride)

    @staticmethod
    def set_label_on_cuda(labels):
        label1, label2 = labels[0], labels[1]
        for k, v in label1.items():
            label1[k] = v.cuda()
        for k, v in label2.items():
            label2[k] = v.cuda()
        return label1, label2

    @staticmethod
    def load_existed_model(model, loss, optimizer, path):
        if os.path.exists(path):
            check_point = torch.load(path)
            model.load_state_dict(check_point["model_state_dict"])
            loss.load_state_dict(check_point["loss_state_dict"])
            optimizer.load_state_dict(check_point["optimizer_state_dict"])

    @staticmethod
    def store_model(model, loss, optimizer, path):
        torch.save({
            "model_state_dict": model.state_dict(),
            "loss_state_dict": loss.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, path)



