# 使用的损失函数类似Contrastive Loss
import torch
from torch import nn

from myutil import MyUtil


class SiameseNet(nn.Module):
    def __init__(self, inputChannels, outputChannels, kernelSize,
                 hideNeuronalNum, outNeuronalNum):
        super().__init__()
        self.cnn_for_steady = nn.Sequential(
            nn.Conv1d(inputChannels, outputChannels, kernel_size=kernelSize),
            nn.BatchNorm1d(outputChannels),
            nn.ReLU(),

            nn.Conv1d(outputChannels, 2*outputChannels, kernel_size=kernelSize),
            nn.BatchNorm1d(2*outputChannels),
            nn.ReLU(),

            nn.Conv1d(2*outputChannels, 4 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm1d(4 * outputChannels),
            nn.ReLU(),

            nn.Conv1d(4 * outputChannels, 6 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm1d(6 * outputChannels),
            nn.ReLU(),

            nn.Conv1d(6 * outputChannels, 10 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm1d(10 * outputChannels),
            nn.ReLU(),

            nn.Conv1d(10 * outputChannels, 16 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm1d(16 * outputChannels),
            nn.ReLU(),


            nn.Flatten()
        )

        self.cnn_for_transient = nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, kernel_size=kernelSize),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(),

            nn.Conv2d(outputChannels, 2 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm2d(2*outputChannels),
            nn.ReLU(),

            nn.Conv2d(2 * outputChannels, 4 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm2d(4 * outputChannels),
            nn.ReLU(),

            nn.Conv2d(4 * outputChannels, 6 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm2d(6 * outputChannels),
            nn.ReLU(),

            nn.Conv2d(6 * outputChannels, 10 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm2d(10 * outputChannels),
            nn.ReLU(),

            nn.Conv2d(10 * outputChannels, 16 * outputChannels, kernel_size=kernelSize),
            nn.BatchNorm2d(16 * outputChannels),
            nn.ReLU(),
            # 为了支持不同采样率的设备，增设了自适应平均池化层，如果影响效果可以考虑去掉
            # 考虑最低采样为400hz，取稳态为100，瞬态为205*36，在经过6层cnn后，稳态长100-6*2=88，瞬态长193*24
            nn.AdaptiveAvgPool2d((None, 24)),
            nn.Flatten()
        )

        self.fc_height = 16*outputChannels*(193*24+88)

        self.fc = nn.Sequential(
            nn.Linear(self.fc_height, hideNeuronalNum),
            nn.ReLU(inplace=True),

            nn.Linear(hideNeuronalNum, outNeuronalNum),
            nn.ReLU(inplace=True)
        )

    def forward_one(self, feature_steady, feature_transient):
        __output_steady = self.cnn_for_steady(feature_steady)
        __output_transient = self.cnn_for_transient(feature_transient)
        __output = self.fc(torch.cat((__output_transient, __output_steady), dim=-1))
        return __output

    def forward(self, input_pair):
        output1_var = self.forward_one(input_pair[0]['cepstrum_tensor'], input_pair[0]['cwt_tensor'])
        output2_var = self.forward_one(input_pair[1]['cepstrum_tensor'], input_pair[1]['cwt_tensor'])
        return output1_var, output2_var


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1_parm, output2_param, flag_same):
        p_dist = nn.PairwiseDistance(p=1)
        distance = p_dist(output1_parm, output2_param)
        # flag = label1_param["name"].ne(label2_param["name"])
        # flag = torch.where(flag, 1, 0)
        flag = torch.Tensor(flag_same).cuda()

        loss_contrastive = torch.mean(flag * torch.pow(distance, 2) +
                                      (1-flag) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive
