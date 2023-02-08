import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from models.att_basic_model_hier_updown_SAP import AttBasicModel_UPDOWN_SAP
from layers.attention import Attention
from lib.config import cfg
import lib.utils as utils
import blocks

class subsequent_attribute_predictor(nn.Module):
    def __init__(self,opt):
        super(subsequent_attribute_predictor,self).__init__()

        self.rnn_size = cfg.MODEL.RNN_SIZE
        self.drop_prob_lm = opt.drop_prob_lm        

  
        self.f2logit = nn.Linear(2*self.rnn_size,self.rnn_size) # 由 att-lstm以及图卷积两个输出作为输入，之后输出维度等同于att-lstm输出维度
                                 
        self.attr_fc = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                 nn.LeakyReLU(0.1,inplace=True),
                                 nn.Dropout(self.drop_prob_lm))
        self.attr_fc2 = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                 nn.LeakyReLU(0.1,inplace=True),
                                 nn.Dropout(self.drop_prob_lm))
        self.attr_fc3 = nn.Linear(self.rnn_size,self.rnn_size)        
        self.init_weight()

    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p,0)
        return 

    def forward(self,word_embedding,h_att,previous_attr,subsequent_mat):
        x1 = torch.mm(subsequent_mat, word_embedding) #\      # 需要明确的输入来源：1.word embedding
        x2 = self.attr_fc(x1)                         # \                         2.previous_topic
        x3 = torch.mm(subsequent_mat,x2)              #  }  2层GCN Layers         3.subsequent_mat: 转移概率矩阵
        x4 = self.attr_fc2(x3)                        # /
        x5 = self.attr_fc3(x4)                        #/
        susequent_attr_embedding = x5[previous_attr] # E_t^{prev}
        
        input_feat = torch.cat([h_att,susequent_attr_embedding],dim=1) 
        logits = self.f2logit(input_feat) # FC Layer 
        
        return logits

class UpDown_SAP(AttBasicModel_UPDOWN_SAP):
    def __init__(self,arg):
        super(UpDown_SAP, self).__init__()
        self.arg = arg
        self.topic_num = arg.topic_num
        self.SAP = subsequent_attribute_predictor(arg)
        self.subsequent_mat = torch.from_numpy(np.load(self.arg.markov_mat_path)).cuda().float() # 转化为tensor
        ##### Updown ####
        self.num_layers = 2
        
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        # First LSTM layer
        ############################## topic updown ######################################
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.WORD_EMBED_DIM + self.att_dim
        self.lstm1 = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(cfg.MODEL.RNN_SIZE, cfg.MODEL.RNN_SIZE)
        self.att = Attention()
        ############################## word updown ######################################
        self.wlstm1 = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        # Second LSTM Layer
        self.wlstm2 = nn.LSTMCell(cfg.MODEL.RNN_SIZE + self.att_dim, cfg.MODEL.RNN_SIZE)
        self.watt = Attention()
        
        if cfg.MODEL.BOTTOM_UP.DROPOUT_FIRST_INPUT > 0:
            self.dropout1 = nn.Dropout(cfg.MODEL.BOTTOM_UP.DROPOUT_FIRST_INPUT)
        else:
            self.dropout1 = None
            
        if cfg.MODEL.BOTTOM_UP.DROPOUT_SEC_INPUT > 0:
            self.dropout2 = nn.Dropout(cfg.MODEL.BOTTOM_UP.DROPOUT_SEC_INPUT)
        else:
            self.dropout2 = None
    
    def Forward_Topic(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        wt_topic = kwargs[cfg.PARAM.WT_TOPIC]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        state_word = kwargs[cfg.PARAM.STATE]
        state_topic = kwargs[cfg.PARAM.STATE_TOPIC]   


        xt_topic = self.topic_embed(wt_topic)
        # lstm1
        ctx_word = state_word[1][0]
        h2_tm1 = state_topic[0][-1]
        input1 = torch.cat([ctx_word, self.ctx_drop(h2_tm1) + gv_feat, xt_topic], 1)
        if self.dropout1 is not None:
            input1 = self.dropout1(input1)
        h1_t, c1_t = self.lstm1(input1, (state_topic[0][0], state_topic[1][0]))
        sap_logits = self.SAP(self.topic_embed(torch.Tensor([range(self.topic_num)]).long().cuda()).detach().squeeze(0),h1_t,wt_topic,self.subsequent_mat) #!! 输入接好后需要看一下是否真的起到作用
        att = self.att(sap_logits, att_feats, p_att_feats, att_mask)

        # lstm2
        input2 = torch.cat([att, sap_logits], 1)
        if self.dropout2 is not None:
            input2 = self.dropout2(input2)
        h2_t, c2_t = self.wlstm2(input2, (state_topic[0][1], state_topic[1][1]))

        state_topic = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))
        return h2_t, state_topic

    # state[0] -- h, state[1] -- c
    def Forward_Word(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        topic_feats = kwargs[cfg.PARAM.TOPIC_FEATS]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        state_word = kwargs[cfg.PARAM.STATE]
        state_topic = kwargs[cfg.PARAM.STATE_TOPIC]   
        xt = self.word_embed(wt)
        
        # lstm1
        h2_tm1 = state_word[0][-1]
        input1 = torch.cat([topic_feats, self.ctx_drop(h2_tm1) + gv_feat, xt], 1)
        if self.dropout1 is not None:
            input1 = self.dropout1(input1)
        h1_t, c1_t = self.wlstm1(input1, (state_word[0][0], state_word[1][0]))
        att = self.watt(h1_t, att_feats, p_att_feats, att_mask)

        # lstm2
        input2 = torch.cat([att, h1_t], 1)
        if self.dropout2 is not None:
            input2 = self.dropout2(input2)
        h2_t, c2_t = self.wlstm2(input2, (state_word[0][1], state_word[1][1]))

        state_word = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))
        return h2_t, state_word

    
        

        
        