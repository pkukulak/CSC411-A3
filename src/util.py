import scipy.io as io
import sklearn as sk
import numpy as np
from sklearn.cross_validation import train_test_split

LABELED = "../data/labeled_images.mat"
UNLABELED = "../data/unlabeled_images.mat"
PUBLIC_TEST = "../data/public_test_images.mat"
HIDDEN_TEST = "../data/hidden_test_images.mat"
CSV_FILE_NAME = "submission.csv"

# This global variable represents the number of classes
# in this classification task.
K = 7

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
    loaded_pub_test = io.loadmat(PUBLIC_TEST)
    loaded_hid_test = io.loadmat(HIDDEN_TEST)

    train_in = reshape_data(loaded_labeled['tr_images'])
    train_targ = loaded_labeled['tr_labels']
    train_ids = loaded_labeled['tr_identity']

    pub_test_in = reshape_data(loaded_pub_test['public_test_images'])
    hid_test_in = reshape_data(loaded_hid_test['hidden_test_images'])

    test_in = np.vstack((pub_test_in, hid_test_in))

    unlabeled_in = reshape_data(loaded_unlabeled['unlabeled_images'])
   
    return train_in, train_targ, train_ids, unlabeled_in, test_in

def reshape_data(data_in):
    '''
    Return a z by x*y matrix of data points, where x & y are
    the dimensions of each ith datapoint, and z is the number of examples.
    input:
        data_in - an x by y by z matrix of datapoints.
    output:
        reshaped_data_in - a z by x*y matrix of datapoints.
    '''
    x, y, z = data_in.shape
    N, M = z, x *y
    return data_in.flatten().reshape(M, N).T

def concatenate_data_info(data_in, data_targ, data_ids):
    '''
    Concatenate two rows to data_in: the targets for each row,
    stored in data_targ; the ids for each row, stored in data_ids.
    Return the resulting matrix.
    input:
        data_in - an NxM matrix of data points per row.
        data_targ - an Nx1 matrix of data point targets.
        data_ids - an Nx1 matrix of data point identities.
    output:
        full_data - an NxM+2 matrix of data and its corresponding
                    target and identity.
    '''
    data_in_targ = np.append(data_in, data_targ, axis=1)
    return np.append(data_in_targ, data_ids, axis=1)

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

def write_prediction_to_csv(test_targ):
    '''
    Write a file to disk that formats the prediction stored in
    test_targ.
    input:
        test_targ - a Nx1 array of predictions for test data.
    '''
    N, _ = test_targ.shape
    with open(CSV_FILE_NAME, "w+") as f:
        f.write("Id,Prediction\n")
        for i in xrange(N):
            if i > N - 1:
                f.write("{},0\n".format(i+1))
            else:
                f.write("{},{}\n".format(i+1, int(test_targ.flatten()[i])))
        f.close()

def encode_prediction(model, data_in):
    '''
    Return a NxK matrix, where N is the number of examples and K
    is the number of classes. This matrix represents the predictions
    yielded by the input model in a 1-of-K format: that is, each row
    has a 1 in the kth position, representing the prediction for the
    corresponding row in data_in.
    input:
        model - a model that has been fit to training data.
                must be a scikit-learn model, i.e. have a "prediciton"
                method.
        data_in - an NxM matrix, where each row is a data point of M features.
    output:
        encoded - an NxK matrix, where each row has a 1 in the kth position.
                  this indications that example i is of class k.
    '''
    N, M = data_in.shape
    prediction = model.predict(data_in)
    encoded = np.zeros((N, K+1))
    encoded[np.arange(N), prediction.astype(int)] = 1
    return encoded
