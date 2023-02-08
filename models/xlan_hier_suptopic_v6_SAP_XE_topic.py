import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model_hier_topic_v6_XE_topic import AttBasicModel_HIER_TOPIC_V6_XE_topic
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


# !! 整体上感觉是两个XLAN解码器，都在LSTM位置加了额外输入
class XLAN_HIER_SUPTOPIC_V6_SAP_XE_topic(AttBasicModel_HIER_TOPIC_V6_XE_topic):
    def __init__(self,arg):
        super(XLAN_HIER_SUPTOPIC_V6_SAP_XE_topic, self).__init__()
        self.arg = arg
        self.topic_num = arg.topic_num
        self.SAP = subsequent_attribute_predictor(arg)

        ##### XLAN_HIER_SUPTOPIC_V6####
        self.num_layers = 2 #2 for single

        # Topic X-LAN
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

        # word X-LAN       
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
        #wt_topic = kwargs[cfg.PARAM.WT_TOPIC]
        wt = kwargs[cfg.PARAM.WT]
        wt_topic = kwargs[cfg.PARAM.WT_TOPIC]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        state_word = kwargs[cfg.PARAM.STATE]
        state_topic = kwargs[cfg.PARAM.STATE_TOPIC]        

        # !!ctx是context的意思吗，比xlan在lstm多输入了ctx_word
        #####topic level##########
        subsequent_mat = np.load(self.arg.markov_mat_path) # 加载转移概率矩阵   # init
        subsequent_mat = torch.from_numpy(subsequent_mat).cuda().float() # 转化为tensor
        xt_topic = self.topic_embed(wt_topic) # hard label转化为1024（embed_size）维
        ctx_word = state_word[1][0]  #now use 1,0 which is the ctx of word lstm;1 1 was not used in word-lstm, consider use it in topic 这就是图中从word decoder输入给topic decoder的状态
        h_att, c_att = self.att_lstm(torch.cat([xt_topic, ctx_word, gv_feat + self.ctx_drop(state_topic[0][1])], 1), (state_topic[0][0], state_topic[1][0]))
        sap_logits = self.SAP(self.topic_embed(torch.Tensor([range(self.topic_num)]).long().cuda()).detach().squeeze(0),h_att,wt_topic,subsequent_mat) #!! 输入接好后需要看一下是否真的起到作用
        att, _, alpha_spatial = self.attention(sap_logits, att_feats, att_mask, p_att_feats, precompute=True)
        
        ctx_input = torch.cat([att, sap_logits], 1)

        topic_feats = self.att2ctx(ctx_input)  #avg keywords embedding + alpha_spatial feature attn weights
        state_topic = [torch.stack((h_att, topic_feats)), torch.stack((c_att, state_topic[1][1]))]
        return topic_feats, state_topic
    
    def Forward_Word(self, **kwargs):
        #wt_topic = kwargs[cfg.PARAM.WT_TOPIC]
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
        #xt_topic = self.topic_embed(wt_topic) 下面主要是在解码器LSTM中增加了topic_feats的输入
        h_att, c_att = self.watt_lstm(torch.cat([xt_word, topic_feats, gv_feat + self.wctx_drop(state_word[0][1])], 1), (state_word[0][0], state_word[1][0]))
        att, _, alpha_spatial = self.wattention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        # !!上面比xlan官方多了alpha_spatial,是一个调试用变量？
        ctx_input = torch.cat([att, h_att], 1)

        output_word = self.watt2ctx(ctx_input)  #avg keywords embedding + alpha_spatial feature attn weights 特征
        state_word = [torch.stack((h_att, output_word)), torch.stack((c_att, state_word[1][1]))]
        return output_word, state_word