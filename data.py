import os

import numpy.random
import pandas as pd

from dataPreProcess import DataPreprocess
from myutil import MyUtil
import torch.utils.data as data
import random


class SiameseNetDataSet(data.Dataset):
    def __init__(self, data_controller):
        self.channels = data_controller.channels
        self.root_dir_str = data_controller.root_dir_str
        self.data_dic = data_controller.data_dic
        self.meta_dic = data_controller.meta_dic
        self.length = data_controller.length
        self.dir_list = data_controller.dir_list

    def __getitem__(self, index):
        dir1 = random.choice(self.dir_list)
        same = random.randint(0, 1)
        if same:
            # 不满足时需要切换
            meta = self.meta_dic[dir1]
            pair_candidate = meta.sample(n=2, random_state=numpy.random.RandomState(), axis=0, replace=False)
            label_vec_list, file_name_list = MyUtil.get_label_pair_from_one_dataframe(pair_candidate)
            while MyUtil.dataIsSimilar(label_vec_list[0], label_vec_list[1])[0]:
                pair_candidate = meta.sample(n=2, random_state=numpy.random.RandomState(), axis=0, replace=False)
                label_vec_list, file_name_list = MyUtil.get_label_pair_from_one_dataframe(pair_candidate)
            # path1 = self.root_dir_str + '/' + dir1 + '/' + file_name_list[0]
            # path2 = self.root_dir_str + '/' + dir1 + '/' + file_name_list[1]
            data_pre1 = self.data_dic[dir1][file_name_list[0]]
            data_pre2 = self.data_dic[dir1][file_name_list[1]]
        else:
            dir2 = random.choice(self.dir_list)
            while dir2 == dir1:
                dir2 = random.choice(self.dir_list)
            meta1 = self.meta_dic[dir1]
            meta2 = self.meta_dic[dir2]
            candidate1 = meta1.sample(n=1, random_state=numpy.random.RandomState(), axis=0)
            candidate2 = meta2.sample(n=1, random_state=numpy.random.RandomState(), axis=0)
            label_vec_list, file_name_list = MyUtil.get_label_pair_from_dataframe_pair(candidate1, candidate2)
            while not MyUtil.dataIsSimilar(label_vec_list[0], label_vec_list[1])[0]:
                candidate1 = meta1.sample(n=1, random_state=numpy.random.RandomState(), axis=0)
                candidate2 = meta2.sample(n=1, random_state=numpy.random.RandomState(), axis=0)
                label_vec_list, file_name_list = MyUtil.get_label_pair_from_dataframe_pair(candidate1, candidate2)
            data_pre1 = self.data_dic[dir1][file_name_list[0]]
            data_pre2 = self.data_dic[dir2][file_name_list[1]]
        return [data_pre1[0], data_pre2[0]], [data_pre1[1], data_pre2[1]], [data_pre1[2], data_pre2[2]], same


    def __len__(self):
        return self.length


class DataController:
    def __init__(self, dir_str, channels):
        self.channels = channels
        self.root_dir_str = dir_str
        dir_list = os.listdir(dir_str)
        meta_dic = {}
        data_dic = {}
        ori_len = 0
        if dir_list.__len__() != 0:
            ori_len = int(dir_list.__len__() * os.listdir(dir_str + '/' + dir_list[0]).__len__()/2)
            for user_dir in dir_list:
                meta_pd = pd.read_csv(self.root_dir_str + '/' + user_dir + '/' + user_dir + '.meta')
                meta_dic[user_dir] = meta_pd
                file_names = os.listdir(dir_str+"/"+user_dir)
                file_dic = {}
                for data_file in file_names:
                    if data_file.endswith("csv"):
                        data_pd = pd.read_csv(dir_str+"/"+user_dir+"/"+data_file, usecols=[4, 5, 6, 7])
                        data_pre = DataPreprocess.data_pre_entry(data_pd)
                        file_dic[data_file] = data_pre
                data_dic[user_dir] = file_dic
            self.data_dic = data_dic
            self.meta_dic = meta_dic
        self.length = ori_len
        self.dir_list = dir_list

