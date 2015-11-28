import scipy.io as io
import sklearn as sk
import numpy as np
from sklearn.cross_validation import train_test_split

LABELED = "/Users/FILIP/U of T/CSC411/A3/data/labeled_images.mat"
UNLABELED = "/Users/FILIP/U of T/CSC411/A3/data/unlabeled_images.mat"
TEST = "/Users/FILIP/U of T/CSC411/A3/data/public_test_images.mat"

#def load_data(labeled_path, unlabeled_path, test_path):
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
        train_input - an array of data representing the training set
        train_targ - an array of labels corresponding to the training set
        valid_input - an array of data representing the validation set
        valid_targ - an array of labels corresponding to the validation set
        test_input - an array of data representing the test set
    '''
    # First, load the raw data into dictionaries.
    # These dictionaries all contain keys that allow us to index
    # into the data and metadata.
    #loaded_labeled = io.loadmat(labeled_path)
    #loaded_unlabeled = io.loadmat(unlabeled_path)
    #loaded_test = io.loadmat(test_path)
    loaded_labeled = io.loadmat(LABELED)
    loaded_unlabeled = io.loadmat(UNLABELED)
    loaded_test = io.loadmat(TEST)

    # We then reshape the data into a two-dimensional array
    # of shape (N, M), where N is the number of examples and
    # M is the number of features. That is, data is arranged
    # by row and features are arranged by column.
    x, y, z = loaded_labeled['tr_images'].shape
    N, M = z, x * y
    labels = loaded_labeled['tr_labels']
    identities = loaded_labeled['tr_identity']
    labeled_data = loaded_labeled['tr_images'].reshape(N, M)

    data_to_targ = np.append(labeled_data, labels, axis=1)
    data_to_targ_to_id = np.append(data_to_targ, identities, axis=1)

    data_to_targ_to_id = np.array(sorted(data_to_targ_to_id,
                                         key=lambda entry: entry[M+1]))

    known_data = data_to_targ_to_id[data_to_targ_to_id[:, M+1] != -1]
    unknown_data = data_to_targ_to_id[data_to_targ_to_id[:, M+1] == -1]
    known_data_to_targ = known_data[:, :M+1]
    known_labeled_data = known_data[:, :M]
    unknown_data_to_targ = unknown_data[:, :M+1]
    unknown_labeled_data = unknown_data[:, :M]


    train_in = labeled_data[:num_train]
    train_targ = data_to_targ[:num_train, M]

    valid_in = labeled_data[num_train:]
    valid_targ = data_to_targ[num_train:, M]

    test_input = loaded_test['public_test_images']

    return train_in, train_targ, valid_in, valid_targ, test_input
