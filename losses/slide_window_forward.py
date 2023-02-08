import torch
import torch.nn as nn

class slide_window(nn.Module):
    def __init__(self):
        super(slide_window, self).__init__()

    def forward(self, P, hard_label, sent_idx, window_size=3):
        """
            设计理念:滑动窗口保证了每次topic预测loss都是相同数量个概率的和,希望提升的概率为当前句子对应的gts topic概率以及该段topic序列中相邻的topic的概率,
            相邻的topic很有可能含义相近。即我们对topic进行了约束,但是约束并不严格，留有余地，且余地的范围也十分合理，余地大小取决于窗口大小。
            [tensor] P: 是预测某一个topic的概率,是一个一维tensor
            [tensor] hard_label: 是当前这句话所处段落的所有topic的序列,由-1填充,取值范围0-80
            [tensor] sent_idx: 是说当前这句话是这一段的第几句话,从0开始
            [int] window_size: 指定窗口大小,是奇数,因为中间位置留给当前句子对应的gts
        """
        mask1 = hard_label>=0
        # mask1[0:sent_idx] = False
        topic_list = hard_label[mask1==True]
        # one_side = (window_size-1)/2 # window一侧的长度
        idx_list = torch.tensor(list(range(int(sent_idx),int(sent_idx)+int(window_size))))
        idx_list[idx_list<0]=0
        idx_list[idx_list>len(topic_list)-1] = len(topic_list)-1
        idx_list = idx_list.tolist()
        topic_list_select = topic_list[idx_list]
        topic_list_select = topic_list_select.tolist()
        loss = 0
        for i in topic_list_select:
            loss += P[i]
        loss = -loss

        return loss

# a = slide_window()
# P = torch.tensor([0.1,0.05,0.05,0.5,0.3]) # 共五个topic
# hard_label = torch.tensor([0,2,3,1,4,-1,-1,-1])
# sent_idx = torch.tensor([4])
# loss = a(P,hard_label,sent_idx)