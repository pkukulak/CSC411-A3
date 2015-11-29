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

    train_in = loaded_labeled['tr_images']
    train_targ = loaded_labeled['tr_labels']
    train_ids = loaded_labeled['tr_identity']

    x, y, z = train_in.shape
    N, M = z, x * y
    train_in = train_in.reshape(N, M)
    
    unlabeled_in = loaded_unlabeled['unlabeled_images']
    
    test_in = loaded_test['public_test_images']
    
    return train_in, train_targ, train_ids, unlabeled_in, test_in
