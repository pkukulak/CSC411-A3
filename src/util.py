import scipy.io as io

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
    num_data = z
    num_features = x * y
    labeled_data = loaded_labeled['tr_images'].reshape(num_data, num_features)
    
    # We use 80% of our data set as training data.
    # The remaining 20% is used as validation data.
    num_train = num_data * 8/10
    train_input = labeled_data[:num_train]
    train_targ = loaded_labeled['tr_labels'][:num_train]

    valid_input = labeled_data[num_train:]
    valid_targ = loaded_labeled['tr_labels'][num_train:]

    test_input = loaded_test['public_test_images']

    return train_input, train_targ, valid_input, valid_targ, test_input
