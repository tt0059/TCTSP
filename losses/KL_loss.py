import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
            
    def forward(self, kl_divs, targets):
        mask_clu = (targets>=0).float()  # 这里会求mask，不用担心对齐未聚类的句子
        #mask_unclu = 0.001 * (targets==0).float() #unclu ratio 0.005
        mask = mask_clu# + mask_unclu
	
       # unc_num = torch.sum((targets == 0).float())
        #all_num = torch.sum((mask > 0).float())
        #unc_ratio = unc_num/all_num
        
        kl_loss = kl_divs * mask
        kl_loss = torch.sum(kl_loss) / torch.sum(mask)        
        return kl_loss, {'KL Loss': kl_loss.item()}#, 'Unclustered Ratio': unc_ratio.item()}
