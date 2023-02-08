import torch
import torch.nn as nn
import torch.nn.functional as F

class RangeShrink_sum(nn.Module): # 这里只对KL散度求了平均，实际计算kl散度是在model里面
    def __init__(self):
        super(RangeShrink_sum, self).__init__()
            
    def forward(self, rangeShrink, targets):
        mask_clu = (targets>=0).float()  # 这里会求mask，不用担心对齐未聚类的句子
        #mask_unclu = 0.001 * (targets==0).float() #unclu ratio 0.005
        mask = mask_clu# + mask_unclu
	
        #unc_num = torch.sum((targets == 0).float())
        #all_num = torch.sum((mask > 0).float())
        #unc_ratio = unc_num/all_num
        
        rangeShrink_loss = rangeShrink * mask
        rangeShrink_loss = torch.sum(rangeShrink_loss) / torch.sum(mask)        
        return rangeShrink_loss, {'RangeShrink Loss': rangeShrink_loss.item()}#, 'Unclustered Ratio': unc_ratio.item()}
