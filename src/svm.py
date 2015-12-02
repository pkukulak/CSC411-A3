from sklearn import svm
from sklearn.cross_validation import train_test_split
from util import *
import numpy as np

def run_SVC(train_in, valid_in, train_targ, valid_targ):
    H_c = 3e2
    H_kernel = 'poly'
    H_degree = 2
    H_coef0 = 0.0
    H_g = 0.001

    classifier = svm.SVC(kernel=H_kernel, C=H_c, coef0=H_coef0, degree=H_degree, gamma=H_g)
    classifier.fit(train_in, train_targ)
    return classifier.score(valid_in, valid_targ)

if __name__ == "__main__":
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    N, M = data_in.shape

    full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)
    pruned_in, pruned_targ, pruned_ids = prune_individuals(full_data)
    num_iters = 10
    rates = np.array([])
    for i in xrange(num_iters):
        train_in, valid_in, train_targ, valid_targ = train_test_split(pruned_in,
            pruned_targ, test_size=0.27, stratify=pruned_ids)

        rates = np.append(rates, run_SVC(train_in, valid_in, train_targ, valid_targ))
        print "Success rate = {}".format(rates[i])

    print "Average Success Rate = {}".format(np.mean(rates))
