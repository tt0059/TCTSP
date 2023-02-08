import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import blocks
import lib.utils as utils
from lib.config import cfg
from models.basic_model import BasicModel
from losses.RangeShrink import RangeShrink

class AttBasicModel_UPDOWN_SAP(BasicModel):
    def __init__(self):
        super(AttBasicModel_UPDOWN_SAP, self).__init__()
        self.ss_prob = 0.0                               # Schedule sampling probability
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1       # include <BOS>/<EOS>
        self.topic_size = cfg.MODEL.TOPIC_SIZE + 1       # the last one is the start token for topic
        self.max_seq_num = 20
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
        
        #use discrete 
        sequential = [nn.Embedding(self.topic_size, cfg.MODEL.WORD_EMBED_DIM)]
        sequential.append(utils.activation(cfg.MODEL.WORD_EMBED_ACT))
        if cfg.MODEL.WORD_EMBED_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.WORD_EMBED_DIM))
        if cfg.MODEL.DROPOUT_WORD_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED))
        self.topic_embed = nn.Sequential(*sequential)        
    
        # global visual feat embed
        sequential = []
        if cfg.MODEL.GVFEAT_EMBED_DIM > 0:
            sequential.append(nn.Linear(cfg.MODEL.GVFEAT_DIM, cfg.MODEL.GVFEAT_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.GVFEAT_EMBED_ACT))
        if cfg.MODEL.DROPOUT_GV_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_GV_EMBED))
        self.gv_feat_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

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
        self.topic_logit = nn.Linear(cfg.MODEL.RNN_SIZE, self.topic_size-1)
        self.p_att_feats = nn.Linear(self.att_dim, cfg.MODEL.ATT_HIDDEN_SIZE) \
            if cfg.MODEL.ATT_HIDDEN_SIZE > 0 else None

        self.topic_criterion = RangeShrink()

    def init_hidden(self, batch_size):
        state_topic = [Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda())]
        state_word =  [Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda()),
                Variable(torch.zeros(self.num_layers, batch_size, cfg.MODEL.RNN_SIZE).cuda())] 
        return state_topic, state_word
                 

    
    def make_kwargs(self, wt_topic, wt, topic_feats, gv_feat, att_feats, att_mask, p_att_feats, state_topic, state_word, **kgs):
        kwargs = kgs
        kwargs[cfg.PARAM.WT] = wt
        kwargs[cfg.PARAM.WT_TOPIC] = wt_topic
        kwargs[cfg.PARAM.TOPIC_FEATS] = topic_feats
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        kwargs[cfg.PARAM.STATE] = state_word
        kwargs[cfg.PARAM.STATE_TOPIC] = state_topic
        return kwargs    
    
    
    
    
    def preprocess(self, **kwargs):
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        
        if gv_feat is None or gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
                
        # embed gv_feat
        if self.gv_feat_embed is not None:
            gv_feat = self.gv_feat_embed(gv_feat)
        
        # embed att_feats
        if self.att_embed is not None:    
            att_feats = self.att_embed(att_feats)

        p_att_feats = self.p_att_feats(att_feats) if self.p_att_feats is not None else None

        
        return gv_feat, att_feats, att_mask, p_att_feats

    # gv_feat -- batch_size * cfg.MODEL.GVFEAT_DIM
    # att_feats -- batch_size * att_num * att_feats_dim
    def forward(self, **kwargs): 
        #torch.autograd.set_detect_anomaly(True)
        seq = kwargs[cfg.PARAM.INPUT_SENT] 
        topic_seq = kwargs[cfg.PARAM.TOPIC_LABELS]#.long() 
        topic_hard_label = kwargs['TOPIC_LABELS_HARD']
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        
        gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        p_att_feats = utils.expand_tensor(p_att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)  
        
         
        batch_size = gv_feat.size(0)
        state_topic, state_word = self.init_hidden(batch_size)
              
        
        outputs = Variable(torch.zeros(batch_size, seq.size(1), self.vocab_size).cuda())
        stacked_spatial = Variable(torch.zeros(batch_size, 20, att_feats.shape[1]).cuda()) #max seq num is 20
        topic_kls = Variable(torch.zeros(batch_size, 20).cuda())
        sent_index = - torch.ones(batch_size).long().detach().cuda() # -1 because the first one is bos
        topic_feats = Variable(torch.zeros(batch_size, cfg.MODEL.RNN_SIZE).cuda())
        for t in range(seq.size(1)): # every word
            #for selecting words!
            if self.training and t >=1 and self.ss_prob > 0:
                prob = torch.empty(batch_size).cuda().uniform_(0, 1)
                mask = prob < self.ss_prob
                if mask.sum() == 0:
                    wt = seq[:,t].clone()
                else:
                    ind = mask.nonzero().view(-1)
                    wt = seq[:, t].data.clone()
                    prob_prev = torch.exp(outputs[:, t-1].detach())
                    wt.index_copy_(0, ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, ind))
            else:
                wt = seq[:,t].clone()
            if t >= 1 and seq[:, t].max() == 0:
                break

            if(t == 0):
                wt_topic = (self.topic_size - 1) * torch.ones(batch_size).long().cuda().detach()
            cs_index = (wt==2) + (t == 0)            
            topic_renew = False   
            for batch_id in range(batch_size):
                if(cs_index[batch_id]==1):
                    topic_renew = True
                    break             
            #forward topic LSTM only when an sentence ends
            if(topic_renew == True):
                kwargs= self.make_kwargs(wt_topic, wt, topic_feats, gv_feat, att_feats, att_mask, p_att_feats, state_topic, state_word)
                topic_feats_new, state_topic_new = self.Forward_Topic(**kwargs)
                #renew the topic only when a sentence ends or the first time step
                new_wt_topic = wt_topic.clone()
                new_wt = wt.clone()
                #new_topic_feats = topic_feats.clone()
                if self.dropout_lm is not None:
                    topic_logits = self.topic_logit(self.dropout_lm(topic_feats_new))
                else:
                    topic_logits = self.topic_logit(topic_feats_new)
                #renew the topics index after forward_topic
                state_topic_tp0 = state_topic[0].clone()
                state_topic_tp1 = state_topic[1].clone()                
                for batch_id in range(batch_size):
                    if cs_index[batch_id]==1:
                        sent_index[batch_id] += 1
                        topic_feats[batch_id] = topic_feats_new[batch_id]

                        state_topic_tp0[:, batch_id] = state_topic_new[0][:, batch_id] # = state_topic_new[0][:, batch_id]
                        state_topic_tp1[:, batch_id] = state_topic_new[1][:, batch_id]
                        if(sent_index[batch_id]<20):
                            kl_div = self.topic_criterion(torch.softmax(topic_logits[batch_id], dim=0), topic_hard_label[batch_id], sent_index[batch_id])
                            topic_kls[batch_id, sent_index[batch_id]] = kl_div               
                            new_wt_topic[batch_id] = torch.argmax(topic_seq[batch_id, sent_index[batch_id]])
                        #also init the state of the words_lstm
                        if(t != 0):
                            new_wt[batch_id] = seq[batch_id, 0].clone()
                            state_word[0][0][batch_id,:] = 0
                            state_word[0][1][batch_id,:] = 0
                            #state_word[1][0][batch_id,:] = 0
                            state_word[1][1][batch_id,:] = 0                            
                state_topic[0] = state_topic_tp0
                state_topic[1] = state_topic_tp1
                wt_topic = new_wt_topic.clone()
                wt = new_wt.clone()
                        
            kwargs = self.make_kwargs(wt_topic, wt, topic_feats, gv_feat, att_feats, att_mask, p_att_feats, state_topic, state_word)
            output_word, state_word = self.Forward_Word(**kwargs)
            if self.dropout_lm is not None:
                output = self.dropout_lm(output_word)
            logit = self.logit(output)
            outputs[:, t] = logit
            #outputs_attn[:, t] = alpha_spatial

        return outputs, stacked_spatial, topic_kls
    def decode(self, **kwargs):
        greedy_decode = kwargs['GREEDY_DECODE']
        block_trigrams = not self.training
        trigrams = [] # will be a list of batch_size dictionaries
        gv_feat, att_feats, att_mask, p_att_feats = self.preprocess(**kwargs)
        batch_size = gv_feat.size(0)
        state_topic, state_word = self.init_hidden(batch_size)
        #old_state = gv_feat.new_zeros(batch_size, cfg.MODEL.RNN_SIZE)
       
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.ones(batch_size, dtype=torch.long).cuda())*(self.vocab_size-1)
        unfinished = wt.eq(wt)
        
        stacked_spatial = Variable(torch.zeros(batch_size, 20, att_feats.shape[1] ).cuda()) #max seq num is 20
        
        sent_index = - torch.ones(batch_size).long().detach().cuda() # -1 because the first one is bos
        topic_sents = - 1 * Variable(torch.ones(batch_size, 20, dtype=torch.long).cuda())
        topic_logprobs = Variable(torch.zeros(batch_size, 20).cuda())
        topic_feats = Variable(torch.zeros(batch_size, cfg.MODEL.RNN_SIZE).cuda())
        wt_topic = (self.topic_size - 1) * torch.ones(batch_size).long().cuda().detach()   # init
        for t in range(cfg.MODEL.SEQ_LEN):         
            cs_index = (wt==2) + (t == 0)    
            topic_renew = False   
            for batch_id in range(batch_size):
                if(cs_index[batch_id]==1):
                    topic_renew = True
                    break                  
            #we need to renew the topic
            if(topic_renew == True):
                kwargs= self.make_kwargs(wt_topic, wt, topic_feats, gv_feat, att_feats, att_mask, p_att_feats, state_topic, state_word)
                logprobs_topic_t, state_topic_new, topic_feats_new = self.get_logprobs_state_topic(**kwargs)
    
                
                new_wt_topic = wt_topic.clone()
                new_wt = wt.clone()
                #renew the topics using argmax                
                state_topic_tp0 = state_topic[0].clone()
                state_topic_tp1 = state_topic[1].clone() 
                for batch_id in range(batch_size):
                    if cs_index[batch_id]==1:
                        sent_index[batch_id] += 1
                        topic_feats[batch_id] = topic_feats_new[batch_id]
                        state_topic_tp0[:, batch_id] = state_topic_new[0][:, batch_id] # = state_topic_new[0][:, batch_id]
                        state_topic_tp1[:, batch_id] = state_topic_new[1][:, batch_id]
                        
                        new_wt_topic[batch_id] = torch.argmax(logprobs_topic_t[batch_id]).clone()
                        if(sent_index[batch_id]<20):
                            topic_logprobs[batch_id, sent_index[batch_id]] = logprobs_topic_t[batch_id, new_wt_topic[batch_id]]
                            topic_sents[batch_id, sent_index[batch_id]] = new_wt_topic[batch_id]      
                        #also init the state of the words_lstm
                        if(t != 0):
                            new_wt[batch_id] = self.vocab_size-1#seq[batch_id, 0].clone()
                            state_word[0][0][batch_id,:] = 0
                            state_word[0][1][batch_id,:] = 0
                            #state_word[1][0][batch_id,:] = 0
                            state_word[1][1][batch_id,:] = 0                           
                state_topic[0] = state_topic_tp0
                state_topic[1] = state_topic_tp1
                wt_topic = new_wt_topic.clone()
                wt = new_wt.clone()
                                
                        
            kwargs = self.make_kwargs(wt_topic, wt, topic_feats, gv_feat, att_feats, att_mask, p_att_feats, state_topic, state_word)
            logprobs_t, state_word = self.get_logprobs_state_word(**kwargs) 
            # Mess with trigrams
            if block_trigrams and t >= 3 and greedy_decode:
                # Store trigram generated at last step
                prev_two_batch = sents[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = sents[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = sents[:,t-2:t]
                mask = torch.zeros(logprobs_t.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs_t = logprobs_t + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)
        
        
                # Length penalty
                #penalty = 1.00
                #formula = penalty**t  # (5 + t)**penalty / (5 + 1)**penalty
                #helper = (torch.ones(logprobs.shape) - (1.0 - formula) * (torch.arange(logprobs.shape[1]).expand(logprobs.shape) <= 1).float()).cuda() 
                #logprobs = logprobs * helper 
        
            # END MODIFIED ----------------------------------------------------   
            
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
            
        return sents, logprobs, topic_sents, topic_logprobs

    def get_logprobs_state_topic(self, **kwargs):
        topic_feats, state_topic = self.Forward_Topic(**kwargs)
        logprobs_topic = F.log_softmax(self.topic_logit(topic_feats), dim=1)
        return logprobs_topic, state_topic, topic_feats
    
    def get_logprobs_state_word(self, **kwargs):
        output_word, state_word = self.Forward_Word(**kwargs)
        logprobs_word = F.log_softmax(self.logit(output_word), dim=1)
        return logprobs_word, state_word
    
    def _expand_state(self, batch_size, beam_size, cur_beam_size, state, selected_beam):
        shape = [int(sh) for sh in state.shape]
        beam = selected_beam
        for _ in shape[2:]:
            beam = beam.unsqueeze(-1)
        beam = beam.unsqueeze(0)
        
        state = torch.gather(
            state.view(*([shape[0], batch_size, cur_beam_size] + shape[2:])), 2,
            beam.expand(*([shape[0], batch_size, beam_size] + shape[2:]))
        )
        state = state.view(*([shape[0], -1, ] + shape[2:]))
        return state

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        gv_feat, att_feats, att_mask, p_att_feats, topic_labels = self.preprocess(**kwargs)
        
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        state = self.init_hidden(batch_size)
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())

        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

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

            for s in range(len(state)):
                state[s] = self._expand_state(batch_size, beam_size, cur_beam_size, state[s], selected_beam)

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
                att_feats = utils.expand_tensor(att_feats, beam_size)
                gv_feat = utils.expand_tensor(gv_feat, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                p_att_feats = utils.expand_tensor(p_att_feats, beam_size)

                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        return outputs, log_probs



