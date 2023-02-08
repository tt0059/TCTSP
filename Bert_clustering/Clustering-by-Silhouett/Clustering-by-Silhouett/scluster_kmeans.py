from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
import numpy as np
import time
# dictionary of clustering functions:
CLUSTERING = {'kmeans' : lambda df, k: KMeans(n_clusters = k).fit(df),#.labels_,
             'hdbscan' : lambda df, s: HDBSCAN(min_cluster_size = s, min_samples=None).fit(df),#.labels_,
            'meanshift': lambda df, q: MeanShift(bandwidth=0.025*min(q,50)).fit(df)}

# the clustering by silhouette object
class SCluster:
    """
    The object that calculate the clustering by silhouette.
    for each step, the object calculate the clustering labels,
    and then, calculate for each result the silhouette score.
    next, this code choose the labels with the best score.
    """
    def __init__(self,typ='kmeans', org=2, lim=20, stp=1, dup=0.95):
        """
        initialize the object
        :param typ: clustering type
        :param org: first value in the loop
        :param lim: last value in the loop
        :param stp: # values between each step
        :param dup: value for fix dataframe row length for silhouette
        """
        # initial parameters
        self.type= typ
        self.org = org
        self.lim = lim+1
        self.stp = stp
        self.dup = dup

        # clustering function
        self.function = CLUSTERING[self.type.lower()]

        # values for calculation
        self.max = -1
        self.min_unclu_ratio = 1.0
        self.scores = {}
        self.labels_= []
        self.centers_all = {}
        self.centers_ = []

    def adapt_silhouette(self,labels):
        """
        calculate the silhouette value for the given dataframe
        :param labels: cluster labels
        :return: the dataframe silhouette score for the given labels
        """
        data, labels= self.df[labels > -1], labels[labels > -1]
        if data.shape[0] == 0: return -1
        while True:
            try:
                return silhouette_score(data, labels, sample_size=self.size)*(labels.shape[0]/self.n)
            except:
                self.size = int(self.size*self.duf)

    def fit(self,data):
        """
        fit the optimal cluster labels to the data
        :param data: input dataframe
        """
        self.n = data.shape[0]
        self.size = self.n
        self.df = data
        min_topic_size = 0
        num = float(data.shape[0])
        for i in range(self.org, self.lim , self.stp):
            st = time.time()
            renew = False
            model = self.function(self.df, i)
            label = model.labels_
            centers = model.cluster_centers_
            silho = self.adapt_silhouette(label)
            self.scores[silho] = label
            self.centers_all[silho] = centers
            unclu_num = sum(label == -1)
            unclu_ratio = unclu_num/num
            
            num_cluster = len(list(set(list(label))))
            if ((self.max - self.min_unclu_ratio) < (silho - unclu_ratio)): #self.max < silho or ((self.max - silho) <1e-3 and (self.min_unclu_ratio - unclu_ratio)>1e-3):
                self.max = silho
                self.min_unclu_ratio = unclu_ratio
                min_topic_size = i
                np.save('hard_labels_norm_tp.npy', label)
                np.save('clu_centers_norm_tp.npy', centers)               
                print('labels and centers are saved')
                renew = True
                
            if(renew == False):
                print(f'cluster kind: {self.type}, input value = {i}, takes {time.time()-st}s, cluster_num = {num_cluster}, silhouette = {round(silho,4)}, unlu_ratio {round(unclu_ratio,4)}')
            else:
                print(f'\nthe current best choice is setting {min_topic_size} clusters, takes {time.time()-st}s, cluster_num = {num_cluster}, with {round(self.max,4)} silhouette score and {round(self.min_unclu_ratio,4)} uncluster ratio\n')
          
        self.labels_ = self.scores[self.max]
        self.centers_ = self.centers_all[self.max]
        print(f'the best choice is setting min topic size to {min_topic_size}, with {self.max} silhouette score and {self.min_unclu_ratio} uncluster ratio')
        return self


