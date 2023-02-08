import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model_hier_topic import AttBasicModel_HIER_TOPIC
import blocks

class XLAN_HIER_SUPTOPIC(AttBasicModel_HIER_TOPIC):
    def __init__(self):
        super(XLAN_HIER_SUPTOPIC, self).__init__()
        self.num_layers = 4 #2 for single

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM
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
        self.topic_fc = nn.Linear(cfg.MODEL.RNN_SIZE, self.topic_size)
        
        
        self.watt_lstm = nn.LSTMCell(rnn_input_size + cfg.MODEL.RNN_SIZE, cfg.MODEL.RNN_SIZE)         
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
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        old_state = kwargs[cfg.PARAM.OLD_STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        stacked_spatial = kwargs[cfg.PARAM.STACKED_SPATIAL]
        sent_index = kwargs[cfg.PARAM.SENT_INDEX]
        


        cs_index = wt==2 #2 is the encoded '.'
        
        if gv_feat is None or gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed(wt)
        
        
        #####sentence level##########
        h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        att, _, alpha_spatial = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        ctx_input = torch.cat([att, h_att], 1)

        topic_feats = self.att2ctx(ctx_input)  #avg keywords embedding + alpha_spatial feature attn weights
        topic_logits = self.topic_fc(topic_feats)
       
        #topic_embedding = self.topic_embed(topic_index)
                     
        ######word level########
        batch_size = gv_feat.size(0)
        for batch_id in range(batch_size):
            if cs_index[batch_id]==1:
                old_state[batch_id] = topic_feats[batch_id] #keywords_embedding[batch_id,sent_index[batch_id]]
                #####
                if(self.training):
                #    acc_diff[batch_id, sent_index] = keywords_embedding[batch_id,sent_index[batch_id]] - old_state[batch_id]
                    stacked_spatial[batch_id,sent_index[batch_id]] = torch.mean(alpha_spatial[batch_id], dim=0)
                sent_index[batch_id] += 1
         
        topic_vector = old_state
        prev_h_word = state[0][3]
        watt_lstm_input = torch.cat([xt, gv_feat + self.wctx_drop(prev_h_word), topic_vector], 1)
        h_watt, c_watt = self.watt_lstm(watt_lstm_input, (state[0][2], state[1][2]))        
        watt, _, walpha_spatial = self.wattention(h_watt, att_feats, att_mask, p_att_feats, precompute=True)
        wctx_input = torch.cat([watt, h_watt], 1)

        output_word = self.watt2ctx(wctx_input)
        
        state = [torch.stack((h_att, topic_feats, h_watt, output_word)), torch.stack((c_att, state[1][1], c_watt, state[1][3]))]
        return output_word, state, old_state, stacked_spatial, sent_index, topic_logits #perhaps return walpha_spatial