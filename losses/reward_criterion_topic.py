import torch
import torch.nn as nn

class RewardCriterionTopic(nn.Module):
    def __init__(self):
        super(RewardCriterionTopic, self).__init__()

    def forward(self, seq, logP, rewards):
        mask = seq >= 0
        wt_mask = (seq>=0).float()
        #wt_mask[seq == 0] = 0.1
        #mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        wt_mask = torch.masked_select(wt_mask, mask)
        loss = torch.sum(-logP * rewards * wt_mask) / torch.sum(wt_mask) # loga+logb=loga*b，该loss是希望reward的期望越大越好
        return loss