from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from util import load_data, prune_individuals
import numpy as np

if __name__ == "__main__":
    data_in, data_targ, data_ids, unlabeled_in, test_in = load_data()

    N, M = data_in.shape
    full_data = np.append(np.append(data_in, data_targ, axis=1), data_ids, axis=1)

    pruned_in, pruned_targ, pruned_ids = prune_individuals(full_data)

    logit = LogisticRegression()
    
    train_in, valid_in, train_targ, valid_targ = train_test_split(pruned_in,
            pruned_targ, test_size=0.33, stratify=pruned_ids)
    logit.fit(train_in, train_targ.flatten())
    print "Validation MCE = {}".format(logit.score(valid_in, valid_targ))

