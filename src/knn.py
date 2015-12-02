from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from util import *
import numpy as np

if __name__ == "__main__":
    k_vals = [3, 5, 7, 9, 11, 13, 14, 16, 21]
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    
    N, M = data_in.shape
    full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)

    pruned_in, pruned_targ, pruned_ids = prune_individuals(full_data)

    for k in k_vals:
        kNN = KNeighborsClassifier(n_neighbors=k)
        train_in, valid_in, train_targ, valid_targ = train_test_split(pruned_in, 
                pruned_targ, test_size=0.33, stratify=pruned_ids)
        kNN.fit(train_in, train_targ.flatten())
        print "k = {} ; classification rate = {}".format(k, kNN.score(valid_in, valid_targ))
