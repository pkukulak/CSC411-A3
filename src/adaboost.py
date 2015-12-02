from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from util import *
import numpy as np

def run_AdaBoost(train_in, valid_in, train_targ, valid_targ):
    classifier = AdaBoostClassifier(learning_rate=0.8)
    classifier.fit(train_in, train_targ)
    print classifier.score(valid_in, valid_targ)

if __name__ == "__main__":
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()
    N, M = data_in.shape

    full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)
    pruned_in, pruned_targ, pruned_ids = prune_individuals(full_data)

    train_in, valid_in, train_targ, valid_targ = train_test_split(pruned_in,
            pruned_targ, test_size=0.33, stratify=pruned_ids)

    run_AdaBoost(train_in, valid_in, train_targ, valid_targ)
