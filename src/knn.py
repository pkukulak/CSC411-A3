from run_knn import run_knn
from util import load_data
import matplotlib.pyplot as plt
import numpy as np

N = 50

def classification_rate(prediction, validation):
    '''
    Calculate the classification rate of the array prediction,
    given the targets in validation.

    inputs:
        prediction:     the vector of outputs from knn
                        classification
        validation:     the target vector of our validation set

    outputs:
        a number 0 <= k <= 1 representing the percentage of 
        successfully labeled inputs
    '''
    correct = 0
    for i in range (N):
        if prediction[i, 0] == validation[i, 0]:
            correct += 1.0
    return correct / N

if __name__ == "__main__":
    k_vals = np.array([7, 14, 21])
    rates = []

    train_in, train_targ, valid_in, valid_targ, test_in = load_data()

    for k in k_vals:
        knn_prediction = run_knn(k, train_in, train_targ, valid_in)
        rates += [classification_rate(knn_prediction, valid_targ)]
        print "k = {} , success rate = {}".format(k, rates[-1])

    plt.plot(k_vals, rates, 'ro')
    plt.axis([min(k_vals) - 1, max(k_vals) + 1, 0, 1])
    plt.xlabel('k')
    plt.ylabel('classification rate')
    plt.show()
