#from scluster import SCluster
#from scluster_kmeans import SCluster
import pandas as pd
import numpy as np
# define variables
data= pd.read_csv(r'umap_data_norm_2.csv')   #49 is the best
typ = 'hdbscan'
if(typ == 'hdbscan'):
    from scluster import SCluster
else:
    from scluster_kmeans import SCluster
org = 50
lim = 80
stp = 1

# application
print('start runing clustering')
tp = SCluster(typ=typ, org=org ,lim=lim, stp=stp).fit(data)#.labels_
#np.save('umap_hard_labels_norm_kmeans_10.npy', tp.labels_)
#np.save('umap_centers_norm_kmeans_10.npy', tp.centers_)
#np.save('umap_probs_norm.npy', tp.probabitilies_)