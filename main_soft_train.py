
#-*- coding: UTF-8 -*-
import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler_VinVL import Evaler_VinVL
from scorer.scorer import Scorer
from scorer.topic_scorer import Topic_Scorer
from lib.config import cfg, cfg_from_file
from tensorboardX import SummaryWriter

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count() # 返回可用GPU数
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        self.rl_stage = False
        self.setup_logging()
        self.setup_dataset()
        self.setup_network()

        self.train_evaler = Evaler_VinVL( 
            eval_ids = cfg.DATA_LOADER.TRAIN_ID,
            gv_feat = None,#cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TRAIN_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TRAIN_ANNFILE,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG 
        )

        self.val_evaler = Evaler_VinVL(
            eval_ids = cfg.DATA_LOADER.VAL_ID,
            gv_feat = None,#cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE,
            seq_per_img = 1
        )
        self.test_evaler = Evaler_VinVL(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            gv_feat = None,#cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE,
            seq_per_img = 1
        )
        self.scorer = Scorer(['CIDEr','Meteor','Bleu'],[1,0.5,0.5])
        #self.scorer = Scorer(['CIDEr','Bleu'],[1,1])
        self.topic_scorer = Topic_Scorer()      


    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)

        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        model_list = [
            'XLAN_SUPTOPIC_V6_SAP',
            'XLAN_SUPTOPIC_V6_SAP_PARA',
            'XLAN_SUPTOPIC_V6_SAP_XE_topic',
            'XLAN_HIER_SUPTOPIC_V6_SAP_topic_feature',
            'Dual_decoder_XTransformer_v1',
        ]
        if cfg.MODEL.TYPE in model_list:
            model = models.create(cfg.MODEL.TYPE,self.args)
        else:
            model = models.create(cfg.MODEL.TYPE)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device), 
                device_ids = [self.args.local_rank], 
                output_device = self.args.local_rank,
                broadcast_buffers = False,
                find_unused_parameters=True
            )
        else:
            self.model = torch.nn.DataParallel(model).cuda()

        if self.args.resume >= 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                           map_location=lambda storage, loc: storage)
            )
            self.start_epoch = self.args.resume
            self.scheduled_sampling(self.start_epoch)
        else:
            self.start_epoch = 0

        self.optim = Optimizer(self.model)
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.xe_criterion_topic = losses.create(self.args.topic_criterion).cuda()
        self.rl_criterion = losses.create('RewardCriterion').cuda()
        self.rl_criterion_topic = losses.create('RewardCriterionTopic').cuda()

    def setup_dataset(self):
        self.coco_set = datasets.coco_dataset_VinVL.CocoDataset_VinVL(            
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID, 
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH, 
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS, 
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num=  cfg.DATA_LOADER.MAX_FEAT,
            topic_label_folder=cfg.DATA_LOADER.TOPIC_LABEL_PATH,
            seq_len = cfg.MODEL.SEQ_LEN
        )

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader_VinVL.load_train(
            self.distributed, epoch, self.coco_set)

    def eval(self, epoch, writer):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None

        #train_res = self.train_evaler(self.model, 'train_' + str(epoch + 1), cfg.MODEL.TYPE)
        #self.logger.info('######## Epoch (TRAIN)' + str(epoch + 1) + ' ########')
        #self.logger.info(str(train_res))        


        val_res, val_topic_sents = self.val_evaler(self.model, 'val_' + str(epoch + 1), cfg.MODEL.TYPE)
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))

        test_res, val_topic_sents = self.test_evaler(self.model,'test_' + str(epoch + 1), cfg.MODEL.TYPE)
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(test_res))
        writer.add_scalar('val/CIDEr',val_res['CIDEr'],epoch+1)
        writer.add_scalar('test/CIDEr',test_res['CIDEr'],epoch+1)
        with open(os.path.join(args.folder,'eval_results.txt'),'a') as f:
            f.write('Epoch {} \n'.format(epoch+1))
            f.write('validation results:\n')
            f.write(str(val_res)+'\n')
            f.write('test results:\n')
            f.write(str(test_res)+'\n')

        val = 0
        for score_type, weight in zip(['CIDEr','METEOR','Bleu_4'],[1,0.5,0.5]):
            val -= test_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))

    def make_kwargs(self, indices, input_seq, target_seq, att_feats, att_mask, topic_labels):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor) # 非填充单词统计，填充单词为0
        #seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1) # 每张图片对应的段落中的单词数
        max_len = int(seq_mask_sum.max()) # 该batch内图片的段落的最大词数
        input_seq = input_seq[:, 0:max_len].contiguous() # 减少填充单词数，对齐该batch最长段落,最长的句子仅在开头有<BOS>
        target_seq = target_seq[:, 0:max_len].contiguous() # 减少填充单词数，对齐该batch最长段落,最长的句子仅在结尾有0

        if cfg.DATA_LOADER.SEQ_PER_IMG==4:
            topic_labels_hard = topic_labels
        else:
            topic_labels_hard = torch.argmax(topic_labels,dim=-1) # 默认填充句topic硬标签为0
            topic_mask = (torch.sum(topic_labels, dim=-1) > 0.5) # 填充句的topic概率分布和为 0/-189(carrot)
            if cfg.MODEL.TYPE == 'Dual_decoder_XTransformer_v1':
                topic_labels_hard[topic_mask==False] = 81
            else:
                topic_labels_hard[topic_mask==False] = -1 #填充句topic硬标签全设为-1
        
        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: None,
            cfg.PARAM.ATT_FEATS: att_feats,
            cfg.PARAM.ATT_FEATS_MASK: att_mask,
            cfg.PARAM.TOPIC_LABELS: topic_labels,
            cfg.PARAM.TOPIC_LABELS_HARD: topic_labels_hard #这里有软硬标签
        }
        return kwargs

    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START: 
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob



    def display(self, epoch, iteration, data_time, batch_time, losses, losses_word, loss_info, losses_topic, loss_info_topic):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}, losses_word = {:.5}, losses_topic = {:.5}'\
            .format(data_time.avg, batch_time.avg, losses.avg, losses_word.avg, losses_topic.avg)
        self.logger.info('Epoch '+str(epoch)+', Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        for name in sorted(loss_info_topic):
            self.logger.info('  ' + name + ' = ' + str(loss_info_topic[name]))        
        data_time.reset()
        batch_time.reset()
        losses.reset()
        losses_word.reset()
        losses_topic.reset()


    def forward(self, kwargs): 
        if self.rl_stage == False:
            logit, stacked_alpha, topic_kls = self.model(**kwargs)
            loss_word, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])
            loss_topic, loss_info_topic = self.xe_criterion_topic(topic_kls, kwargs[cfg.PARAM.TOPIC_LABELS_HARD]) # 为什么用的是硬标签？
            return loss_word, loss_info, loss_topic, loss_info_topic
        else:
            if(args.use_new_scst == False):
                ids = kwargs[cfg.PARAM.INDICES]
                gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
                att_feats = kwargs[cfg.PARAM.ATT_FEATS]
                att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
    
    
                # max
                kwargs['BEAM_SIZE'] = 1
                kwargs['GREEDY_DECODE'] = True
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
    
    
                self.model.eval()
                with torch.no_grad():
                    seq_max, _, topic_seq_max, _ = self.model.module.decode(**kwargs)          
                self.model.train()
    
    
                rewards_max, rewards_info_max = self.scorer(ids, seq_max.data.cpu().numpy().tolist())
                rewards_max = utils.expand_numpy(rewards_max)
    
    
                topic_rewards_max, topic_rewards_info_max, max_places = self.topic_scorer(ids, topic_seq_max.data.cpu().numpy().tolist())
                topic_rewards_max = utils.expand_numpy(topic_rewards_max)            
    
                # sample
                kwargs['BEAM_SIZE'] = 1
                kwargs['GREEDY_DECODE'] = False
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
    
                seq_sample, logP_sample, topic_seq_sample, topic_logP_sample = self.model.module.decode(**kwargs)
                seq_sample = seq_sample.cpu().numpy()
                rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.tolist())
    
                topic_rewards_sample, topic_rewards_info_sample, sample_places = self.topic_scorer(ids, topic_seq_sample.data.cpu().numpy().tolist())
                topic_rewards_sample = utils.expand_numpy(topic_rewards_sample)              
    
    
                rewards = rewards_sample - rewards_max 
                rewards = torch.from_numpy(rewards).float().cuda()
                loss_word = self.rl_criterion(torch.from_numpy(seq_sample).cuda(), logP_sample, rewards)
    
                topic_rewards = topic_rewards_sample - topic_rewards_max 
                topic_rewards = torch.from_numpy(topic_rewards).float().cuda()
                loss_topic = self.rl_criterion_topic(topic_seq_sample, topic_logP_sample, topic_rewards)
    
                loss_info = {}
                for key in rewards_info_sample:
                    loss_info[key + '_sample'] = rewards_info_sample[key]
                for key in rewards_info_max:
                    loss_info[key + '_max'] = rewards_info_max[key]
    
                loss_info_topic = {}
                for key in topic_rewards_info_sample:
                    loss_info_topic['topic_'+key + '_sample'] = topic_rewards_info_sample[key]
                for key in topic_rewards_info_max:
                    loss_info_topic['topic_'+key + '_max'] = topic_rewards_info_max[key]
            else:
                #print(kwargs[cfg.PARAM.TOPIC_LABELS_HARD][0])
                ids = kwargs[cfg.PARAM.INDICES]
                att_feats = kwargs[cfg.PARAM.ATT_FEATS]
                att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
                
                ids = utils.expand_numpy(ids, args.expand_num)   # 将特征或信息重复5次来做RL生成采样
                att_feats = utils.expand_tensor(att_feats, args.expand_num)
                att_mask = utils.expand_tensor(att_mask, args.expand_num)                
    
                # max
                kwargs['BEAM_SIZE'] = 1
                kwargs['GREEDY_DECODE'] = False
                kwargs[cfg.PARAM.ATT_FEATS] = att_feats
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask  
                
                seq_sample, logP_sample, topic_seq_sample, topic_logP_sample = self.model.module.decode(**kwargs) # 返回预测出的句子，预测出的单词的logprobs，预测出的topic序列、预测出的topic单独的logprobs
                rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.cpu().numpy().tolist())   # 根据预测出的句子来计算score #!!别忘了每个图预测了5次             
                
                rewards_avg = rewards_sample.copy()
                for i in range(int(rewards_sample.shape[0]/args.expand_num)):
                    for j in range(args.expand_num):
                        rewards_avg[i * args.expand_num + j] = (sum(rewards_sample[i*args.expand_num : (i+1)*args.expand_num]) - rewards_sample[i*args.expand_num+j])/(args.expand_num-1)   #就是5个为一组，每个都有自己的avg，但是自己的avg是剩余四个的平均值
                       
                rewards = rewards_sample - rewards_avg # 最终reward是自己减去自己的avg
                rewards = torch.from_numpy(rewards).float().cuda() # 转换为torch
                loss_word = self.rl_criterion(seq_sample, logP_sample, rewards) # 计算损失
                
                topic_rewards_sample, topic_rewards_info_sample, sample_places = self.topic_scorer(ids, topic_seq_sample.data.cpu().numpy().tolist())
                # topic_rewards_info_sample 是 topic_rewards_sample的均值
                rewards_avg_topic = topic_rewards_sample.copy()
                for i in range(int(topic_rewards_sample.shape[0]/args.expand_num)):
                    for j in range(args.expand_num):
                        rewards_avg_topic[i * args.expand_num + j] = (sum(topic_rewards_sample[i*args.expand_num : (i+1)*args.expand_num]) - topic_rewards_sample[i*args.expand_num+j])/(args.expand_num-1)  # 同上              
                topic_rewards = topic_rewards_sample - rewards_avg_topic
                topic_rewards = torch.from_numpy(topic_rewards).float().cuda() # 转换为torch               
                
                loss_topic = self.rl_criterion_topic(topic_seq_sample, topic_logP_sample, topic_rewards)
                loss_info = {}
                for key in rewards_info_sample:
                    loss_info[key + '_sample'] = rewards_info_sample[key]
                loss_info_topic = {}
                for key in topic_rewards_info_sample:
                    loss_info_topic[key + '_sample'] = topic_rewards_info_sample[key]                   
            return loss_word, loss_info, loss_topic, loss_info_topic # info里面都是均值

    def train(self):
        self.model.train()
        self.optim.zero_grad()
        writer = SummaryWriter(self.args.tensorboard_path)
        iteration = 0
        if(args.accumulate_iter != 0):
            accumulate_num = int(cfg.TRAIN.BATCH_SIZE / args.accumulate_iter)
        for epoch in range(self.start_epoch, cfg.SOLVER.MAX_EPOCH): # every epoch
            if epoch >= cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)
            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()

            losses_word = AverageMeter()
            losses_topic = AverageMeter()
            losses = AverageMeter()
            for _, (b_indices, b_image_id, b_input_seq, b_target_seq, b_att_feats, b_att_mask, b_topic_labels) in enumerate(self.training_loader): #every batch
                data_time.update(time.time() - start)
                loss = 0.0
                loss_word_all = 0.0
                loss_topic_all = 0.0
                if(args.accumulate_iter != 0):
                    for i in range(args.accumulate_iter): # 一般设 0 
                        input_seq = b_input_seq[i * accumulate_num : (i+1) * accumulate_num].cuda()  # 对的，一个是accumulate_iter,一个是num
                        target_seq = b_target_seq[i * accumulate_num : (i+1) * accumulate_num].cuda()
                        att_feats = b_att_feats[i * accumulate_num : (i+1) * accumulate_num].cuda()
                        att_mask = b_att_mask[i * accumulate_num : (i+1) * accumulate_num].cuda()
                        topic_labels = b_topic_labels[i * accumulate_num : (i+1) * accumulate_num].cuda()
                        indices = b_indices[i * accumulate_num : (i+1) * accumulate_num]
                     
                        kwargs = self.make_kwargs(indices, input_seq, target_seq, att_feats, att_mask, topic_labels)
                        loss_word, loss_info, loss_topic, loss_info_topic = self.forward(kwargs)
                        loss_cur = loss_word + args.topic_weight * loss_topic
                        #print(loss_word)
                        #print(loss_cur)
                        loss = loss + loss_cur
                        loss_word_all = loss_word_all + loss_word
                        loss_topic_all = loss_topic_all + loss_topic

                    #with torch.autograd.detect_anomaly():
                    loss = loss / args.accumulate_iter
                    loss_word = loss_word_all / args.accumulate_iter
                    loss_topic = loss_topic_all / args.accumulate_iter
                else:
                    if cfg.DATA_LOADER['SEQ_PER_IMG']==4: # todo:expand to 4 
                        input_seq = b_input_seq.cuda()
                        target_seq = b_target_seq.cuda()
                        att_feats = b_att_feats.cuda()
                        att_mask = b_att_mask.cuda()
                        topic_labels = b_topic_labels.cuda() 
                        indices = b_indices
                    else:
                        input_seq = b_input_seq.cuda()
                        target_seq = b_target_seq.cuda()
                        att_feats = b_att_feats.cuda()
                        att_mask = b_att_mask.cuda()
                        topic_labels = b_topic_labels.cuda()
                        indices = b_indices

                    
                    kwargs = self.make_kwargs(indices, input_seq, target_seq, att_feats, att_mask, topic_labels)
                    loss_word, loss_info, loss_topic, loss_info_topic = self.forward(kwargs) # 送入batch数据进行前传，计算word层loss和topic层loss，之后进行加权求和得到总loss 
                    loss = loss_word + args.topic_weight * loss_topic 
                    writer.add_scalar('Train/loss',loss,iteration)
                    writer.add_scalar('Train/loss_word',loss_word,iteration)
                    writer.add_scalar('Train/loss_topic',loss_topic,iteration)                       
                
                loss.backward()
                utils.clip_gradient(self.optim.optimizer, self.model,
                                    cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                self.optim.step()
                self.optim.zero_grad()
                self.optim.scheduler_step('Iter')

                batch_time.update(time.time() - start)
                start = time.time()

                losses.update(loss.item())
                losses_word.update(loss_word.item())
                losses_topic.update(loss_topic.item())


                self.display(epoch, iteration, data_time, batch_time, losses, losses_word, loss_info, losses_topic, loss_info_topic)
                iteration += 1

                if self.distributed:
                    dist.barrier()
                # break #! 此处用于调试eval，非此用途请注释掉

            if (epoch>=3): 
                self.save_model(epoch)
                val = self.eval(epoch,writer)
                self.optim.scheduler_step('Epoch', val)
                self.scheduled_sampling(epoch)
            
            if self.distributed:
                dist.barrier()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default='experiments/Xlan_suptopic_V6_RL_kmeans')
    parser.add_argument("--topic_weight", type=float, default=0.0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--accumulate_iter", type=int, default=0)
    parser.add_argument("--expand_num", type=int, default=5)
    parser.add_argument("--use_new_scst", type=bool, default=True)
    parser.add_argument("--tensorboard_path", type=str, default='')
    # SAP
    parser.add_argument('--rnn_size', type=int, default=1024, help='size of the rnn in number of hidden nodes in each layer') # 别和cfg混淆了
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='strength of dropout in the Language Model RNN')
    parser.add_argument('--markov_mat_path',type=str,default='/home/huangyq/my-image-to-paragraph/data/markov_mat.npy')
    parser.add_argument('--topic_num',type=int,default=190,help='number of topic')
    parser.add_argument('--topic_criterion',type=str,default='KLLoss',help='criterion of topic')
    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    num_gpus = torch.cuda.device_count()
    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config_server.yml'))
    cfg.ROOT_DIR = args.folder
    with open(os.path.join(args.folder,'eval_results.txt'),'a') as f:
        pass
    assert args.accumulate_iter == 0 or cfg.TRAIN.BATCH_SIZE % args.accumulate_iter == 0 # accumulate_iter是什么？
    trainer = Trainer(args)
    trainer.train()
