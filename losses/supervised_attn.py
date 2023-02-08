import torch
import torch.nn as nn

class SuperAttn(nn.Module):
    def __init__(self):
        super(SuperAttn, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, stacked_alpha, attn_labels):
        attn_labels.transpose_(1,2)
        debug = 1
        
        logit = logit.view(-1, logit.shape[-1])
        target_seq = target_seq.view(-1)
        loss = self.criterion(logit, target_seq)
        return loss, {'CrossEntropy Loss': loss.item()}