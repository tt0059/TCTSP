import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import blocks
import lib.utils as utils
from lib.config import cfg
from models.basic_model import BasicModel
from models.transformer_basic import *
class Transformer(BasicModel):
    def __init__(self):
        super(Transformer, self).__init__()
        self.ss_prob = 0.0                               # Schedule sampling probability
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1       # include <BOS>/<EOS>
        
        self.att_dim = cfg.MODEL.ATT_FEATS_EMBED_DIM \
            if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0 else cfg.MODEL.ATT_FEATS_DIM

        # word embed
        sequential = [nn.Embedding(self.vocab_size, cfg.MODEL.WORD_EMBED_DIM)]
        sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
        if cfg.MODEL.WORD_EMBED_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
        if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
        self.word_embed = nn.Sequential(*sequential)

        # attention feats embed
        sequential = []
        if cfg.MODEL.ATT_FEATS_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
        if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))
        if cfg.MODEL.ATT_FEATS_NORM == True:
            sequential.append(torch.nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
        self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        self.dropout_lm  = nn.Dropout(cfg.MODEL.DROPOUT_LM) if cfg.MODEL.DROPOUT_LM > 0 else None
        self.logit = nn.Linear(cfg.MODEL.RNN_SIZE, self.vocab_size)
        self.p_att_feats = None#nn.Linear(self.att_dim, cfg.MODEL.ATT_HIDDEN_SIZE) \
            #if cfg.MODEL.ATT_HIDDEN_SIZE > 0 else None

 
        self.encoder = Encoder(cfg.MODEL.RNN_SIZE, 6, use_AoA=False)
        self.decoder = Decoder(cfg.MODEL.RNN_SIZE, 6)
    
    def make_kwargs(self, wt, gv_feat, att_feats, att_mask, p_att_feats, state, **kgs):
        kwargs = kgs
        kwargs[cfg.PARAM.WT] = wt
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.STATE] = state
        return kwargs
    
    def init_hidden(self, batch_size):
        return None
        
    def preprocess(self, **kwargs):
        soft_labels = kwargs[cfg.PARAM.GLOBAL_FEAT]

        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        # embed gv_feat
        gv_feat = torch.zeros([att_feats.shape[0],1]).float().cuda()
        att_feats = self.att_embed(att_feats)
        att_feats = self.encoder(att_feats, att_mask)
        
        return gv_feat, att_feats, att_mask, None, soft_labels    
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
        seq_mask =  subsequent_mask(ys.size(1)).cuda().type(torch.cuda.FloatTensor) #subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, :, :].unsqueeze(1)
        output = self.decoder(self.word_embed(ys), encoder_out, att_mask, seq_mask).squeeze(1)   
        #return output, state
        if(len(output.shape) == 2):
            return output, [ys.unsqueeze(0)]
        else:
            return output[:,-1,:], [ys.unsqueeze(0)]    
    
    
    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
       
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
        seq_mask =  subsequent_mask(ys.size(1)).cuda().type(torch.cuda.FloatTensor) #subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, :, :].unsqueeze(1)
        decoder_out = self.decoder(self.word_embed(ys), encoder_out, att_mask, seq_mask).squeeze(1)
        
        logprobs = F.log_softmax(self.logit(decoder_out), dim=-1)
        if(len(logprobs.shape) == 2):
            return logprobs, [ys.unsqueeze(0)]
        else:
            return logprobs[:,-1,:], [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn
    
    
    
    
    
    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        
       
        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        #seq_mask[:,0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)
        
        
        seq_embedding = self.word_embed(seq)
        decoder_out = self.decoder(seq_embedding, encoder_out, att_mask, seq_mask)
        decoder_out = self.logit(decoder_out)
        
          
        
        return decoder_out
    
    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)
    

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        
       

        outputs = []
    
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx / candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                p_att_feats_tmp = []
                for p_feat in p_att_feats:
                    p_key, p_value2 = p_feat
                    p_key = utils.expand_tensor(p_key, beam_size)
                    p_value2 = utils.expand_tensor(p_value2, beam_size)
                    p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

   
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        att_feats = self.att_embed(att_feats)
        encoder_out = self.encoder(att_feats, att_mask)
  
        
        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
    
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break

        return sents, logprobs