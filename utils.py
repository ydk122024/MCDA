"""
基于ava_Dice_loss.py
区别在于计算loss的时候，背景也被包含进去
"""

import torch
import torch.nn as nn

# 分割的类别数
num_organ = 1

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 首先将金标准拆开
        organ_target = torch.zeros((pred.size()))

        for organ_index in range(num_organ + 1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index,  :, :] = temp_target

        organ_target = organ_target.cuda()

        dice = 0.0
        for organ_index in range(num_organ + 1):
            dice += 2 * (pred[:, organ_index, :, :] * organ_target[:, organ_index, :, :]).sum(dim=1).sum(
                dim=1) / (pred[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) +
                          organ_target[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)

        dice = 1 - dice / (num_organ + 1)
        
        return dice.mean()


class SWL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lamba = nn.Sigmoid(Parameter(torch.Tensor(1)))
        self.lamba1 = nn.Sigmoid(Parameter(torch.Tensor(1)))
        self.lamba2 = nn.Sigmoid(Parameter(torch.Tensor(1)))
        self.lamba3 = nn.Sigmoid(Parameter(torch.Tensor(1)))
        self.lamba4 = 1-self.lamba-self.lamba1-self.lamba2-self.lamba3

        self.reset_para()

    def reset_para(self):
        nn.init.constant_(tensor, 0.2)

    def forward(self, outputs, labels, seg5, seg4, seg3, seg2):
        return self.lamba*DiceLoss(outputs, labels) + self.lamba1*DiceLoss(seg5, labels) + self.lamba2*DiceLoss(seg4, labels)\
                + self.lamba3*DiceLoss(seg3, labels) + self.lamba4*DiceLoss(seg2, labels)
