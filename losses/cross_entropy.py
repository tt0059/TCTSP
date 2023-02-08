import torch
import torch.nn as nn

class CrossEntropy(nn.Module):
    def __init__(self, if_topic=False):
        super(CrossEntropy, self).__init__()
        #if(if_topic):
        self.criterion = nn.CrossEntropyLoss(ignore_index = -1)
        #self.if_topic = if_topic

    def forward(self, logit, target_seq):
        logit = logit.view(-1, logit.shape[-1])
        target_seq = target_seq.view(-1)
        #if(self.if_topic == False):
        loss = self.criterion(logit, target_seq)
        return loss, {'CrossEntropy Loss': loss.item()}
        #else:
            #mask = torch.ones_like(target_seq).float()
            #mask[target_seq == 0] = 0.05
            #mask[target_seq == -2] = 0.0
            #unc_num = torch.sum((target_seq == 0).float())
            #all_num = torch.sum((mask > 0).float())
            #unc_ratio = unc_num/all_num
            #loss = self.criterion(logit, target_seq)
            ##print(target_seq[:20])
            #loss = torch.sum(loss * mask) / max([1,torch.sum(mask)])
            #return loss, {'CrossEntropy Loss': loss.item(), 'Unclustered Ratio': unc_ratio.item()}
        