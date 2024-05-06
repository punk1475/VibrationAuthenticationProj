import os

from myUtil import MyUtil
import torch.utils.data as data
import random


class SiameseNetDataSet(data.Dataset):
    def __init__(self, dir_str, data_len, channels):
        self.data_len = data_len
        self.channels = channels
        dir_list = os.listdir(dir_str)
        dir_dict = {}
        for item in dir_list:
            dir_dict[item] = os.listdir(dir_str + '/' + item)
        ori_len = dir_list.__len__() * dir_dict[dir_list[0]].__len__()
        self.root_dir_str = dir_str
        self.length = ori_len
        self.path_dict = dir_dict
        self.dir_list = dir_list

    def __getitem__(self, index):
        dir1 = random.choice(self.dir_list)
        dir2 = random.choice(self.dir_list)
        same = random.randint(0, 1)
        if same:
            if dir1 != dir2:
                dir2 = dir1
        else:
            while dir1 == dir2:
                dir2 = random.choice(self.dir_list)
        path1 = self.root_dir_str + '/' + dir1 + '/' + random.choice(self.path_dict[dir1])
        path2 = self.root_dir_str + '/' + dir2 + '/' + random.choice(self.path_dict[dir2])
        while path1 == path2:
            path2 = self.root_dir_str + '/' + dir2 + '/' + random.choice(self.path_dict[dir2])
        data1 = MyUtil.export_csv_to_list(path1, self.data_len, self.channels)
        data2 = MyUtil.export_csv_to_list(path2, self.data_len, self.channels)
        # print(f"file1:{path1},file2:{path2}")
        return data1[0], data2[0], [data1[1], data2[1]]

    def __len__(self):
        return self.length
