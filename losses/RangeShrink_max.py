import torch
import torch.nn as nn

class RangeShrink_max(nn.Module):
    def __init__(self):
        super(RangeShrink_max, self).__init__()

    def forward(self, P, hard_label, sent_idx):
        """
            [tensor] P: 是预测某一个topic的概率,是一个一维tensor
            [tensor] hard_label: 是当前这句话所处段落的所有topic的序列,由-1填充,取值范围0-80
            [tensor] sent_idx: 是说当前这句话是这一段的第几句话,从0开始
        """
        mask1 = hard_label>=0
        mask1[0:sent_idx] = False
        topic_list = set(hard_label[mask1==True].tolist())
        loss = 0
        max_prob = 0
        for i in range(81):
            if i in topic_list:
                if P[i] > max_prob:
                    max_prob = P[i]
        loss = -max_prob
        return loss