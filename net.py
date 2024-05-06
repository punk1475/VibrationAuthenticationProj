# 使用的损失函数类似Contrastive Loss
import torch
from torch import nn

from myUtil import MyUtil


class SiameseNet(nn.Module):
    def __init__(self, inputChannels, outputChannels, s_kernelSize, m_kernelSize, l_kernelSize, inputLength,
                 hideNeuronalNum, outNeuronalNum):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(inputChannels, outputChannels, kernel_size=s_kernelSize),
            nn.BatchNorm1d(outputChannels),
            nn.ReLU(),

            nn.Conv1d(outputChannels, outputChannels, kernel_size=s_kernelSize),
            nn.BatchNorm1d(outputChannels),
            nn.ReLU(),

            nn.Conv1d(outputChannels, 2 * outputChannels, kernel_size=m_kernelSize),
            nn.BatchNorm1d(2 * outputChannels),
            nn.ReLU(),

            nn.Conv1d(2 * outputChannels, 3 * outputChannels, kernel_size=m_kernelSize),
            nn.BatchNorm1d(3 * outputChannels),
            nn.ReLU(),

            nn.Conv1d(3 * outputChannels, 3 * outputChannels, kernel_size=l_kernelSize),
            nn.BatchNorm1d(3 * outputChannels),
            nn.ReLU(),

            nn.Conv1d(3 * outputChannels, 3 * outputChannels, kernel_size=l_kernelSize),
            nn.BatchNorm1d(3 * outputChannels),
            nn.ReLU(),

            nn.Flatten()
        )
        self.fc_height = MyUtil.cal_data_length_after_cov(
            MyUtil.cal_data_length_after_cov(MyUtil.cal_data_length_after_cov(inputLength, s_kernelSize, 2,
                                                                              1),
                                             m_kernelSize, 2, 1), l_kernelSize, 2, 1) * 3 * outputChannels

        self.fc = nn.Sequential(
            nn.Linear(self.fc_height,
                      hideNeuronalNum),
            nn.ReLU(inplace=True),

            nn.Linear(hideNeuronalNum, hideNeuronalNum),
            nn.ReLU(inplace=True),

            nn.Linear(hideNeuronalNum, outNeuronalNum),
            nn.ReLU(inplace=True)
        )

    def forward_one(self, dt):
        __output = self.cnn(dt)
        __output = self.fc(__output)
        return __output

    def forward(self, input1, input2):
        output1_var = self.forward_one(input1)
        output2_var = self.forward_one(input2)
        return output1_var, output2_var


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1_parm, output2_param, label1_param, label2_param):
        p_dist = nn.PairwiseDistance(p=2)
        euclidean_distance = p_dist(output1_parm, output2_param)
        dissimilarity = MyUtil.cal_dissimilarity(label1_param, label2_param)
        flag = label1_param["name"].ne(label2_param["name"])
        flag = torch.where(flag, 1, 0)
        loss_contrastive = torch.mean((1 - flag) * dissimilarity * torch.pow(euclidean_distance, 2) +
                                      flag * (1 - dissimilarity) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # loss_contrastive = torch.mean((1 - flag) * torch.pow(euclidean_distance, 2) +
        #                               flag * torch.pow(
        #     torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
