from sklearn import svm
from sklearn.cross_validation import train_test_split, cross_val_score
from util import *
from scipy.stats.mstats import mode
import numpy as np

NUM_TEST = 1253

def run_SVC(train_in, valid_in, train_targ, valid_targ):
    H_c = 3e2
    H_kernel = 'linear'
    H_degree = 2
    H_coef0 = 3e2
    H_g = 0.001

    classifier = svm.SVC(kernel=H_kernel, C=H_c, coef0=H_coef0, degree=H_degree, gamma=H_g)
    classifier.fit(train_in, train_targ)
    return classifier.score(valid_in, valid_targ), classifier

def run_nu_SVC(train_in, valid_in, train_targ, valid_targ):
    H_kernel = 'linear'
    H_degree = 2
    H_coef0 = 3e2
    H_nu = 0.27

    classifier = svm.NuSVC(nu=H_nu, kernel=H_kernel, coef0=H_coef0, degree=H_degree)
    classifier.fit(train_in, train_targ)
    return classifier.score(valid_in, valid_targ), classifier

if __name__ == "__main__":
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    N, M = data_in.shape
    
    print "Loading data."
    full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)
    pruned_in, pruned_targ, pruned_ids = prune_individuals(full_data)
    num_experts = 10
    i = 0
    rates_svc = np.array([])
    svc_models = np.array([])
    rates_nusvc = np.array([])
    nusvc_models = np.array([])
    predictions = np.empty([418, 0])

    while i < num_experts:
        print "Training."
        train_in, valid_in, train_targ, valid_targ = train_test_split(pruned_in,
            pruned_targ, test_size=0.28, stratify=pruned_ids)

        curr_nu_svc_rate, nu_svc = run_nu_SVC(train_in, valid_in, train_targ, valid_targ)

        print "Score = {}".format(curr_nu_svc_rate)
        if curr_nu_svc_rate > 0.77:
            rates_nusvc = np.append(rates_nusvc, curr_nu_svc_rate)
            nusvc_models = np.append(nusvc_models, nu_svc)
            #nu_svc_predictions = np.append(nu_svc_predictions, nu_svc_prediction)
            i += 1
            print"{}.".format(i)
            print "Found a NuSVC Expert."
    
    for expert in nusvc_models:
        test_targ = expert.predict(test_in)
        predictions = np.append(predictions, test_targ.reshape(418, 1), axis=1)

    final_test_targ, _ = mode(predictions, axis=1)

    with open("submission.csv", "w+") as f:
        f.write("Id,Prediction\n")
        for i in xrange(NUM_TEST):
            if i > test_targ.size - 1:
                f.write("{},0\n".format(i+1))
            else:
                f.write("{},{}\n".format(i+1, int(final_test_targ.flatten()[i])))
        f.close()
