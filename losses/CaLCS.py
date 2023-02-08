import torch
import torch.nn as nn

class CaLCS(nn.Module):
    def __init__(self):
        super(CaLCS, self).__init__()
        self.batch_size = 20

    def forward(self, topic_prob, hard_label):
        """
            需要gts所在类别的概率
            gts长度
            dp_matrix[i][j]表示生成的序列的长度为i的前缀和gts中长度为j的前端的LCS长度

        """
        mask = (hard_label>=0).float()
        gts_len = torch.sum(mask, axis=1)
        dp_matrix = torch.zeros(self.batch_size,21,21,requires_grad=True).cuda()
        CaLCSs = torch.zeros(self.batch_size,requires_grad=True).cuda()
        for i in range(self.batch_size):
            for j in range(int(gts_len[i].item())): # dp_matrix每一行
                for k in range(int(gts_len[i].item())): # dp_matrix每一列
                    dp_matrix[i][j+1][k+1] = topic_prob[i][j][int(hard_label[i][k])]*(dp_matrix[i][j][k]+1).clone()+\
                                             (1-topic_prob[i][j][int(hard_label[i][k])])*max(dp_matrix[i][j+1][k].clone(),dp_matrix[i][j][k+1].clone())
            CaLCSs[i] = - torch.log(dp_matrix[i][int(gts_len[i].item())][int(gts_len[i].item())]/int(gts_len[i].item()))
        loss = torch.sum(CaLCSs)/self.batch_size

        return loss, {'CaLCS Loss': loss.item()}