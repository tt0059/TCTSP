import torch
import torch.nn as nn

class SuperFeat(nn.Module):
    def __init__(self):
        super(SuperFeat, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, acc_diff):
        acc_diff = torch.mean(acc_diff * acc_diff, dim=-1)/1024
        acc_mask = (acc_diff != 0).float()
        if(torch.sum(acc_mask) != 0):
            loss = torch.sum(acc_diff)/torch.sum(acc_mask)
        else:
            loss = torch.sum(acc_diff)
        return loss, {'L2 diff Loss': loss.item()}