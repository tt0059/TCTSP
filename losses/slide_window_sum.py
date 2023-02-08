import torch
import torch.nn as nn
import torch.nn.functional as F

class slide_window_sum(nn.Module): # 这里只对KL散度求了平均，实际计算kl散度是在model里面
    def __init__(self):
        super(slide_window_sum, self).__init__()
            
    def forward(self, slide_window, targets):
        mask = (targets>=0).float()  # 这里会求mask，不用担心对齐未聚类的句子
        slide_window_loss = slide_window * mask
        slide_window_loss = torch.sum(slide_window_loss) / torch.sum(mask)        
        return slide_window_loss, {'slide_window Loss': slide_window_loss.item()}#, 'Unclustered Ratio': unc_ratio.item()}
