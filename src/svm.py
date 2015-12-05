from sklearn import svm
from sklearn.cross_validation import train_test_split, cross_val_score
from util import *
import numpy as np

NUM_TEST = 1253

def run_SVC(train_in, valid_in, train_targ, valid_targ):
    H_c = 3e2
    H_kernel = 'poly'
    H_degree = 2
    H_coef0 = 0.0
    H_g = 0.001

    classifier = svm.SVC(kernel=H_kernel, C=H_c, coef0=H_coef0, degree=H_degree, gamma=H_g)
    classifier.fit(train_in, train_targ)
    return classifier.score(valid_in, valid_targ), classifier

def run_nu_SVC(train_in, valid_in, train_targ, valid_targ):
    H_kernel = 'poly'
    H_degree = 2
    H_coef0 = 0.0
    H_nu = 0.27

    classifier = svm.NuSVC(nu=H_nu, kernel=H_kernel, coef0=H_coef0, degree=H_degree)
    classifier.fit(train_in, train_targ)
    return classifier.score(valid_in, valid_targ), classifier

if __name__ == "__main__":
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    N, M = data_in.shape

    full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)
    pruned_in, pruned_targ, pruned_ids = prune_individuals(full_data)
    num_iters = 35
    rates_svc = np.array([])
    svc_models = np.array([])
    rates_nusvc = np.array([])
    nusvc_models = np.array([])

    for i in xrange(num_iters):
        train_in, valid_in, train_targ, valid_targ = train_test_split(pruned_in,
            pruned_targ, test_size=0.28, stratify=pruned_ids)

        curr_svc_rate, svc = run_SVC(train_in, valid_in, train_targ, valid_targ)
        curr_nu_svc_rate, nu_svc = run_nu_SVC(train_in, valid_in, train_targ, valid_targ)

        rates_svc = np.append(rates_svc, curr_svc_rate)
        rates_nusvc = np.append(rates_nusvc, curr_nu_svc_rate)

        svc_models = np.append(svc_models, svc)
        nusvc_models = np.append(nusvc_models, nu_svc)

        print "SVC Success Rate = {} ; NuSVC Success Rate = {}".format(rates_svc[i], rates_nusvc[i])

    best_svc = svc_models[np.argmax(rates_svc)]
    best_nu_svc = nusvc_models[np.argmax(rates_nusvc)]

    print "Best SVC's accuracy = {}".format(np.max(rates_svc))
    print "Best NuSVC's accuracy = {}".format(np.max(rates_nusvc))
    print "Average SVC Success Rate = {}".format(np.mean(rates_svc))
    print "Average NuSVC Success Rate = {}".format(np.mean(rates_nusvc))

    max_accuracies = [np.max(rates_svc), np.max(rates_nusvc)]
    test_classifier = [best_svc, best_nu_svc][np.argmax(max_accuracies)]
    test_targ = best_svc.predict(test_in).astype(int)

    with open("submission.csv", "w+") as f:
        f.write("Id,Prediction\n")
        for i in xrange(NUM_TEST):
            if i > test_targ.size - 1:
                f.write("{},0\n".format(i+1))
            else:
                f.write("{},{}\n".format(i+1, test_targ[i]))
        f.close()
