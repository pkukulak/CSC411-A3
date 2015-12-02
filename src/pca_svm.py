from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from util import *
import numpy as np

if __name__ == "__main__":
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    N, M = data_in.shape

    full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)
    pruned_in, pruned_targ, pruned_ids = prune_individuals(full_data)

    pca = PCA(n_components=200)
    print "Initializing PCA with {} components.".format(pca.n_components)
    svc = svm.SVC(C=4.0, kernel='linear', degree=2)
    print "Initializing SVM with a {} kernel of degree {}.".format(svc.kernel, svc.degree)

    print "Transforming input data."
    decomposed_in = pca.fit_transform(pruned_in)
    num_iters = 10
    rates = np.array([])
    for i in xrange(num_iters):
        print "ITERATION {}".format(i)
        train_in, valid_in, train_targ, valid_targ = train_test_split(decomposed_in,
                pruned_targ, test_size=0.33, stratify=pruned_ids)
        print "Fitting SVM."
        svc.fit(train_in, train_targ)
        rates = np.append(rates, svc.score(valid_in, valid_targ))
        print "Success rate = {}".format(rates[i])

    print "Average Success Rate = {}".format(np.mean(rates))
