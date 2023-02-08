#!/usr/bin/env python
# 
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np
import pdb
from collections import Counter
def lcs_extend(x,y):
    d = [0] * (len(x) + 1)
    p = [0] * (len(x) + 1)
    for i in range(0,len(d)):
        d[i] = [0] * (len(y) + 1)
        p[i] = [0] * (len(y) + 1)

    for i in range(1,len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i-1] == y[j-1]:
                d[i][j] = d[i-1][j-1] + 1
                p[i][j] = 1
            elif d[i-1][j] > d[i][j-1]:
                d[i][j] = d[i-1][j]
                p[i][j] = 2
            else:
                d[i][j] = d[i][j-1]
                p[i][j] = 3
    place = []
    place = lcs_print(x,y,len(x),len(y),p,place)
    place.sort()
    return place

def lcs_print(x,y,lenX,lenY, p, place):
    if lenX == 0 or lenY == 0:
        return place
    if p[lenX][lenY] == 1:
        #print(x[lenX-1])
        place.append(lenX-1)
        lcs_print(x,y,lenX-1,lenY-1,p, place)
    elif p[lenX][lenY] == 2:
        lcs_print(x,y,lenX-1,lenY,p,place)
    else:
        lcs_print(x,y,lenX,lenY-1,p,place)
    return place

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate)==1)	
        assert(len(refs)>0)         
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0]#candidate[0].split(" ")
    	
        for reference in refs:
            # split into tokens
            token_r = reference#[0]#reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/len(token_c))
            rec.append(lcs/len(token_r))
            place = 0
            #now we do not have unclu points
            #place = lcs_extend(token_c, token_r)
            #lcs = len(place)
            #wt_all = 0.0
            #for i in range(len(place)):
                #if(token_c[i] == 0):
                    #wt_all += 0.1 #can be modified. theoratically shoud be 0.01
                #else:
                    #wt_all += 1
            
            #zeros_token_c = Counter(token_c)[0]
            #wt_token_c = float(max([len(token_c) - zeros_token_c*(1-0.1),1]))
            
            #zeros_token_r = Counter(token_r)[0]
            #wt_token_r = float(max([len(token_r) - zeros_token_r*(1-0.1),1]))
            
            #prec.append(lcs/wt_token_c)
            #rec.append(lcs/wt_token_r)

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            score = 0.0
        return score, place
    
    def remove_ignore(self, seq):
        sent = []
        for word in seq:
            if word == -1:# or word == 0:
                continue
            sent.append(word)
        return sent
    
    def remove_replace_ignore_gts(self, seq):
        sent = []
        for word in seq:
            #if word == 0:# or word == 0:
            #    continue
            if word == -1: # 这里唐挺由-1换为0，因为gts填充换为了0
                continue#sent.append(0)
            sent.append(word)
        return sent
    
    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py 
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values 
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        #assert(gts.keys() == res.keys())
        #imgIds = gts.keys()

        score = []
        places = []
        for i, hypo in enumerate(res):
            #print(list(gts[i][0]))
            #print(hypo)
            ref = [self.remove_replace_ignore_gts(list(gts[i][0]))] # 去除了0填充,#!!有点问题,但是由于计算的是rouge，所以没问题了
            hypo = [self.remove_ignore(hypo)] # 去除了-1填充
            
            cur_score, place = self.calc_score(hypo, ref)
            score.append(cur_score)
            places.append(place)
            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score), places

    def method(self):
        return "Rouge"
