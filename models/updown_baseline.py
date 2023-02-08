import torch
import torch.nn as nn
import torch.nn.functional as F

from models.att_basic_model_hier_updown_baseline import AttBasicModel_UPDOWN_BASELINE
from layers.attention import Attention
from lib.config import cfg
import lib.utils as utils

class UpDown_BASELINE(AttBasicModel_UPDOWN_BASELINE):
    def __init__(self):
        super(UpDown_BASELINE, self).__init__()
        self.num_layers = 2
        
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.WORD_EMBED_DIM + self.att_dim
        self.lstm1 = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(cfg.MODEL.RNN_SIZE, cfg.MODEL.RNN_SIZE)
        self.att = Attention()
        
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
        att = self.att(h1_t, att_feats, p_att_feats, att_mask)

        # lstm2
        input2 = torch.cat([att, h1_t], 1)
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

    
        

        
