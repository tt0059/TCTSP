import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model_hier_topic_v3 import AttBasicModel_HIER_TOPIC_V3
import blocks

class XLAN_HIER_SUPTOPIC_V3(AttBasicModel_HIER_TOPIC_V3):
    def __init__(self):
        super(XLAN_HIER_SUPTOPIC_V3, self).__init__()
        self.num_layers = 2 #2 for single

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE*2 + cfg.MODEL.BILINEAR.DIM
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        
        
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        self.attention = blocks.create(            
            cfg.MODEL.BILINEAR.DECODE_BLOCK, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )

               
        self.watt_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)         
        self.wctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        self.wattention = blocks.create(            
            cfg.MODEL.BILINEAR.DECODE_BLOCK, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
        self.watt2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )        

    # state[0] -- h, state[1] -- c
    def Forward_Topic(self, **kwargs):
        wt_topic = kwargs[cfg.PARAM.WT_TOPIC]
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        state_word = kwargs[cfg.PARAM.STATE]
        state_topic = kwargs[cfg.PARAM.STATE_TOPIC]        


        #####topic level##########
        xt_word = self.word_embed(wt)
        xt_topic = self.topic_embed(wt_topic.detach())
        h_att, c_att = self.att_lstm(torch.cat([xt_word, xt_topic, gv_feat + self.ctx_drop(state_topic[0][1])], 1), (state_topic[0][0], state_topic[1][0]))
        att, _, alpha_spatial = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        ctx_input = torch.cat([att, h_att], 1)

        topic_feats = self.att2ctx(ctx_input)  #avg keywords embedding + alpha_spatial feature attn weights
        state_topic = [torch.stack((h_att, topic_feats)), torch.stack((c_att, state_topic[1][1]))]
        return topic_feats, state_topic
    
    def Forward_Word(self, **kwargs):
        wt_topic = kwargs[cfg.PARAM.WT_TOPIC]
        wt = kwargs[cfg.PARAM.WT]
        topic_feats = kwargs[cfg.PARAM.TOPIC_FEATS]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        state_word = kwargs[cfg.PARAM.STATE]
        state_topic = kwargs[cfg.PARAM.STATE_TOPIC]        


        #####word level##########
        xt_word = self.word_embed(wt)
        xt_topic = self.topic_embed(wt_topic)
        h_att, c_att = self.watt_lstm(torch.cat([xt_word, xt_topic, gv_feat + self.wctx_drop(state_word[0][1])], 1), (state_word[0][0], state_word[1][0]))
        att, _, alpha_spatial = self.wattention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        ctx_input = torch.cat([att, h_att], 1)

        output_word = self.watt2ctx(ctx_input)  #avg keywords embedding + alpha_spatial feature attn weights
        state_word = [torch.stack((h_att, output_word)), torch.stack((c_att, state_word[1][1]))]
        return output_word, state_word