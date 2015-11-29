from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from itertools import groupby
from util import load_data
import numpy as np

if __name__ == "__main__":
    k_vals = [3, 5, 7, 11, 14, 21]
    rates = []
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    
    N, M = data_in.shape

    for k in k_vals:
        kNN = KNeighborsClassifier(n_neighbors=k)
        full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)
        sorted_by_id = full_data[full_data[:, M+1].argsort()]
        '''
        groups = np.array([])
        unknows = None
        for k, v in groupby(sorted_by_id, lambda x: x[M+1]):
            if k == -1:
                unknown = v
            else:
                groups = np.append(groups, L)
            print k, v
            print groups
        '''
