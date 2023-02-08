import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.Wah = nn.Linear(cfg.MODEL.RNN_SIZE, cfg.MODEL.ATT_HIDDEN_SIZE, bias=False)
        self.alpha = nn.Linear(cfg.MODEL.ATT_HIDDEN_SIZE, 1, bias=False)
        self.dropout = nn.Dropout(cfg.MODEL.ATT_HIDDEN_DROP) if cfg.MODEL.ATT_HIDDEN_DROP > 0 else None
        self.rnn_size = cfg.MODEL.RNN_SIZE
        self.att_hid_size = cfg.MODEL.ATT_HIDDEN_SIZE 
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.dropout = nn.Dropout(cfg.MODEL.ATT_HIDDEN_DROP) if cfg.MODEL.ATT_HIDDEN_DROP > 0 else None
        
        if cfg.MODEL.ATT_ACT == 'RELU':
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()

    # h -- batch_size * cfg.MODEL.RNN_SIZE
    # att_feats -- batch_size * att_num * att_feats_dim
    # p_att_feats -- batch_size * att_num * cfg.ATT_HIDDEN_SIZE
    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.size(1) #att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = self.act(dot)                                # batch * att_size * att_hid_size
        if(self.dropout is not None):
            dot = self.dropout(dot)
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).to(weight)
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res