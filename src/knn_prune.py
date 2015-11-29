from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from itertools import groupby
from util import load_data
import numpy as np

def prune_individuals(full_data):
    N, M = full_data.shape
    all_ids = full_data[:, M-1]
    counts = np.array([])
    for i in all_ids:
        counts = np.append(counts, sum(i == all_ids))
    return full_data[counts != 1]
    
if __name__ == "__main__":
    k_vals = [3, 5, 7, 11, 14, 21]
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    
    N, M = data_in.shape
    full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)
    pruned_full_data = prune_individuals(full_data[full_data[:, M+1] != -1])
       
    Np, Mp = pruned_full_data.shape

    pruned_in = pruned_full_data[:, :M]
    pruned_targ = pruned_full_data[:, M].reshape(Np, 1)
    pruned_ids = pruned_full_data[:, M+1].reshape(Np, 1)

    for k in k_vals:
        kNN = KNeighborsClassifier(n_neighbors=k)
        train_in, valid_in, train_targ, valid_targ = train_test_split(pruned_in, pruned_targ, test_size=0.7, stratify=pruned_ids)
        kNN.fit(train_in, train_targ.flatten())
        print "k = {} ; prediction = {}".format(k, kNN.score(valid_in, valid_targ))
