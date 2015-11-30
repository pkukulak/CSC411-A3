import scipy.io as io
import sklearn as sk
import numpy as np
from sklearn.cross_validation import train_test_split

LABELED = "/Users/FILIP/U of T/CSC411/A3/data/labeled_images.mat"
UNLABELED = "/Users/FILIP/U of T/CSC411/A3/data/unlabeled_images.mat"
TEST = "/Users/FILIP/U of T/CSC411/A3/data/public_test_images.mat"

def load_data():
    '''
    Load all data and return it as a tuple of arrays.
    input:
        labeled_path - the path (relative or absolute) to labeled data.
                       this data MUST be stored in a .mat file.
        unlabeled_path - the path (relative or absolute) to unlabeled data.
                         this data MUST be stored in a .mat file.
        test_path - the path (relative or absolute) to test data.
                    this data MUST be stored in a .mat file.
    output:
        train_in - an NxM matrix of all training data.
        train_targ - an Nx1 matrix of labels corresponding to training data.
        train_ids - an Nx1 matrix of ids corresponding to training data.
        unlabeled_in - a KxM matrix of additional unlabeled training data.
        test_in - a JxM matrix of test data.
    '''
    # First, load the raw data into dictionaries.
    # These dictionaries all contain keys that allow us to index
    # into the data and metadata.
    loaded_labeled = io.loadmat(LABELED)
    loaded_unlabeled = io.loadmat(UNLABELED)
    loaded_test = io.loadmat(TEST)

    train_in = loaded_labeled['tr_images']
    train_targ = loaded_labeled['tr_labels']
    train_ids = loaded_labeled['tr_identity']

    x, y, z = train_in.shape
    N, M = z, x * y
    train_in = train_in.flatten().reshape(M, N).T
    
    unlabeled_in = loaded_unlabeled['unlabeled_images']
    
    test_in = loaded_test['public_test_images']
    
    return train_in, train_targ, train_ids, unlabeled_in, test_in

def prune_individuals(full_data):
    '''
    Replace all unique identifiers with a single identifier value.
    This is done to enable stratifying based on identifier.
    input:
        full_data - an NxM+2 matrix, where each row is a datapoint
                    of M features. the M+1th column is the label
                    for that datapoint and the M+2th column is the id.
    output:
        pruned_in - an NxM matrix of data points.
        pruned_targ - an Nx1 matrix of labels corresponding to pruned_in.
        pruned_ids - an Nx1 matrix of ids corresponding to pruned_in
    '''
    N, M = full_data.shape
    all_ids = full_data[:, M-1]
    new_ids = np.array([])
    for i in xrange(all_ids.size):
        if np.sum(all_ids == all_ids[i]) == 1:
            new_ids = np.append(new_ids, 42069)
        else:
            new_ids = np.append(new_ids, all_ids[i])
    pruned_full_data = np.append(full_data[:, :M-1], new_ids.reshape(-1, 1), axis=1)

    pruned_in = pruned_full_data[:, :M-2]
    pruned_targ = pruned_full_data[:, M-2]
    pruned_ids = pruned_full_data[:, M-1]

    return pruned_in, pruned_targ, pruned_ids
