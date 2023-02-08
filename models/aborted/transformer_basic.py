import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.nn import init

    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, size, N, use_AoA=True):
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(size, use_AoA), N)
        self.norm = LayerNorm(size)
        self.N = N
        #self.init_weight()
    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                #pass
                init.xavier_normal_(p)
            #else:
            #    init.constant_(p,0)
        return         
        
    def forward(self, vis, src_mask=None,pe_obj=None):
        "Pass the input (and mask) through each layer in turn."
        t = 0
        for layer in self.layers:
            if(t==0):
                vis = layer(vis,src_mask,pe_obj=pe_obj)
                #vis,attr = checkpoint(layer,vis,attr,pe_obj)
                #x = self.sublayer1(x, lambda x: checkpoint(self.self_attn,x, m, m))
            else:
                vis = layer(vis,src_mask)
                #vis,attr = checkpoint(layer,vis,attr)
            t = t + 1
        return self.norm(vis)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, use_AoA):
        super(EncoderLayer, self).__init__()
        if(use_AoA):
            self.attentions = MultiHeadedDotAttention(8, size,0.1,project_k_v=1,scale=1,do_aoa=1,norm_q=0,dropout_aoa=0.3)
        else:
            self.attentions = MultiHeadedAttention(8, size)
            self.feed_forward = PositionwiseFeedForward(size, 2048, 0.3)
            self.sublayer2 = SublayerConnection(size, 0.3)
            self.dropout = nn.Dropout(0.5)
        self.sublayer = SublayerConnection(size, 0.1)
        self.size = size
        self.use_AoA = use_AoA
        #self.init_weight()
    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                #pass
                init.xavier_normal_(p)
            #else:
            #    init.constant_(p,0)
        return         

    def forward(self, vis, src_mask=None,pe_obj=None):
        "Follow Figure 1 (left) for connections."
        vis = self.sublayer(vis, lambda vis: self.attentions(vis, vis, vis, src_mask ,pe_obj)) 
        if(self.use_AoA == True):
            return vis
        else:
            return self.dropout(self.sublayer2(vis, self.feed_forward))
    


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, size, N):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(size, 0.3), N)
        #self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(0.5)
    def apply_to_states(self, fn):
        self.x = fn(self.x)
        for layer in self.layers:
            layer.apply_to_states(fn)
            
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.dropout(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiHeadedAttention(8, size, 0.3)
        self.src_attn = MultiHeadedAttention(8, size, 0.3)
        self.feed_forward = PositionwiseFeedForward(size, 2048)
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)



class AttentionLayer_AoA(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size):
        super(AttentionLayer_AoA, self).__init__()
        self.size = size
        self.attentions = MultiHeadedDotAttention(8, size, project_k_v=1, scale=1, use_output_layer=0, do_aoa=1, norm_q=1,dropout_aoa=0.0)
        #self.norm = clones(LayerNorm(size),2)
        #self.init_weight()
    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                #pass
                init.xavier_normal_(p)
            #else:
            #    init.constant_(p,0)
        return 
    def forward(self, ht, memory, src_mask=None):
 
        "Follow Figure 1 (right) for connections."
        
        x = ht.unsqueeze(1)
        src_mask = None if src_mask is None else src_mask
        att =  self.attentions(x, memory, memory, src_mask).squeeze(1)
        return att



class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))     
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)      
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
    
    
def attention(query, key, value, mask=None, dropout=None,pe_obj=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if pe_obj is not None:
        pe_obj_mh = pe_obj.unsqueeze(1).expand(pe_obj.size(0),8,pe_obj.size(1),pe_obj.size(2))
        scores = scores*torch.log(pe_obj_mh+1e-3)
    if mask is not None:    
        scores = scores.masked_fill(mask==0, -1e9)
        #scores = scores.masked_fill(mask, -np.inf)     
    p_attn = F.softmax(scores,dim=-1)   
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn




class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=0, use_output_layer=0, do_aoa=0, norm_q=0, dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x:x

        self.linears = clones(nn.Linear(d_model, d_model * scale), 1+2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer =  nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x:x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x:x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, value, key, mask=None,pe_obj=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k 
        if self.project_k_v == 0:
            query_ =  self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query_, key_, value_, mask=mask, 
                            dropout=self.dropout,pe_obj=pe_obj)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.3):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None,pe_obj=None):
        "Implements Figure 2"
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] 
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask, 
                                 self.dropout,pe_obj)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

            


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class PositionalEncoding_obj(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model,h, dropout, max_len=1000):
        super(PositionalEncoding_obj, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.h = h
        # Compute the positional encodings once in log space.
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model/(h/2), 2).float() *
                             -(math.log(10000.0) / (d_model/(h/2))))  #d_model/2
        self.register_buffer('div_term', div_term)
        #pe[:, 0::2] = torch.sin(position * div_term)
        #pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0)
        #self.register_buffer('pe', pe)
        self.projpe = nn.Linear(d_model,1)
        nn.init.xavier_uniform_(self.projpe.weight)  
        self.rl = nn.ReLU(inplace=True)
        
    def forward(self, x):
        #x is the position matrix bs*36*36*4
        mul = x.unsqueeze(-1)*self.div_term #bs*36*36*4*64
        #pe = torch.zeros(x.size(0), x.size(1),x.size(2),x.size(3),self.div_term.size(-1)*2).cuda() #
        pe = torch.cat([torch.sin(mul).unsqueeze(-2),torch.sin(mul).unsqueeze(-2)],-2) #bs*36*36*4*2*64       
        pe = pe.reshape(x.size(0), x.size(1),x.size(2),self.div_term.size(-1)*self.h) #bs*36*36*512
        pe_proj = self.projpe(pe).squeeze(-1)

        return self.dropout(self.rl(pe_proj))

class soft_Attention(nn.Module):
    def __init__(self, size):
        super(soft_Attention, self).__init__()
        self.rnn_size = size#opt.rnn_size
        self.att_hid_size = size#opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        
        self.att2att = nn.Linear(self.rnn_size, self.att_hid_size)
        
    def forward(self, h, att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        p_att_feats = self.att2att(att_feats)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
       # if(att_size == 10):
       #     print(weight)
       #     debug = 1
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res, weight

class soft_Attention3d(nn.Module):
    def __init__(self, size):
        super(soft_Attention3d, self).__init__()
        self.rnn_size = size#opt.rnn_size
        self.att_hid_size = size#opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        
        self.att2att = nn.Linear(self.rnn_size, self.att_hid_size)
        
    def forward(self, h, att_feats, att_masks=None):
        # The p_att_feats here is already projected
        h_length = h.size(1)
        att_size = att_feats.size(1)#att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        p_att_feats = self.att2att(att_feats).unsqueeze(1).expand(-1,h_length,-1,-1) #bs * att_size * att_hidsize
        att = p_att_feats.view(-1,h_length, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * h_len * att_hid_size
        att_h = att_h.unsqueeze(2).expand_as(att)            # batch* h_len* att_size * att_hid_size
        dot = att + att_h                                   # batch *h_len* att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch *h_len* att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size*h_len) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size*h_len) * 1
        dot = dot.view(-1,h_length, att_size)                        # batch * h_length, att_size
        
        weight = F.softmax(dot, dim=-1)                             # batch *h_length* att_size
       # if(att_size == 10):
       #     print(weight)
       #     debug = 1
        if att_masks is not None:
            weight = weight * att_masks.unsqueeze(1).expand(-1,h_length,-1).float()
            weight = weight / weight.sum(-1, keepdim=True) # normalize to 1
        #att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        #att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return weight
