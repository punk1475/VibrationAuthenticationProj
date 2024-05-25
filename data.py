import os

import numpy.random
import pandas as pd

from dataPreProcess import DataPreprocess
from myutil import MyUtil
import torch.utils.data as data
import random


class SiameseNetDataSet(data.Dataset):
    def __init__(self, dir_str, channels):
        self.channels = channels
        dir_list = os.listdir(dir_str)
        ori_len = 0
        if dir_list.__len__() != 0:
            ori_len = int(dir_list.__len__() * os.listdir(dir_str + '/' + dir_list[0]).__len__()/2)
        self.root_dir_str = dir_str
        self.length = ori_len
        self.dir_list = dir_list

    def __getitem__(self, index):
        dir1 = random.choice(self.dir_list)
        same = random.randint(0, 1)
        if same:
            # 不满足时需要切换
            meta_path = self.root_dir_str + '/' + dir1 + '/' + dir1 + '.meta'
            meta = pd.read_csv(meta_path)
            pair_candidate = meta.sample(n=2, random_state=numpy.random.RandomState(), axis=0, replace=False)
            label_vec_list, path_list = MyUtil.get_label_pair_from_one_dataframe(pair_candidate)
            while MyUtil.dataIsSimilar(label_vec_list[0], label_vec_list[1])[0]:
                pair_candidate = meta.sample(n=2, random_state=numpy.random.RandomState(), axis=0, replace=False)
                label_vec_list, path_list = MyUtil.get_label_pair_from_one_dataframe(pair_candidate)
            path1 = self.root_dir_str + '/' + dir1 + '/' + path_list[0]
            path2 = self.root_dir_str + '/' + dir1 + '/' + path_list[1]
        else:
            dir2 = random.choice(self.dir_list)
            while dir2 == dir1:
                dir2 = random.choice(self.dir_list)
            meta_path1 = self.root_dir_str + '/' + dir1 + '/' + dir1 + '.meta'
            meta_path2 = self.root_dir_str + '/' + dir2 + '/' + dir2 + '.meta'
            meta1 = pd.read_csv(meta_path1)
            meta2 = pd.read_csv(meta_path2)
            candidate1 = meta1.sample(n=1, random_state=numpy.random.RandomState(), axis=0)
            candidate2 = meta2.sample(n=1, random_state=numpy.random.RandomState(), axis=0)
            label_vec_list, path_list = MyUtil.get_label_pair_from_dataframe_pair(candidate1, candidate2)
            while not MyUtil.dataIsSimilar(label_vec_list[0], label_vec_list[1])[0]:
                candidate1 = meta1.sample(n=1, random_state=numpy.random.RandomState(), axis=0)
                candidate2 = meta2.sample(n=1, random_state=numpy.random.RandomState(), axis=0)
                label_vec_list, path_list = MyUtil.get_label_pair_from_dataframe_pair(candidate1, candidate2)
            path1 = self.root_dir_str + '/' + dir1 + '/' + path_list[0]
            path2 = self.root_dir_str + '/' + dir2 + '/' + path_list[1]
        csv1 = pd.read_csv(path1, usecols=[4, 5, 6, 7])
        csv2 = pd.read_csv(path2, usecols=[4, 5, 6, 7])
        feature_1_1, feature_1_2, feature_1_3, main_axis_1 = DataPreprocess.data_pre_entry(csv1)
        feature_2_1, feature_2_2, feature_2_3, main_axis_2 = DataPreprocess.data_pre_entry(csv2)

        return [feature_1_1, feature_2_1], [feature_1_2, feature_2_2], [feature_1_3, feature_2_3], same


    def __len__(self):
        return self.length
        # return 400
