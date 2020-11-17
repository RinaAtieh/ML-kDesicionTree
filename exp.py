import dataset_preparation
import decision_tree
import k_decision_tree
import numpy as np
import matplotlib.pyplot as plt
from param import test_cls


def div_train():
    # best params
    n=6
    k=3
    p=0.6

    data_4 = dataset_preparation.loaded_X_train_4
    data_10 = dataset_preparation.loaded_X_train_10

    test_data_4 = dataset_preparation.loaded_X_test_4
    test_data_10 = dataset_preparation.loaded_X_test_10

    acc = []
    for i in [40, 80, 120, 120, 200]:
        data = data_4[:i, :] # first i rows
        trees = k_decision_tree.n_decistion_tree(n, k, p, data)
        accuracy1_4 = test_cls(test_data_4, trees, k)
        acc.append(accuracy1_4)
    a = [40, 80, 120, 120, 200]
    plt.plot(a, acc)
    plt.show()

    acc = []
    for i in [40, 80, 120, 120, 200]:
        data = data_10[:i, :]  # first i rows
        trees = k_decision_tree.n_decistion_tree(n, k, p, data)
        accuracy1_10 = test_cls(test_data_10, trees, k)
        acc.append(accuracy1_10)
    a = [40, 80, 120, 120, 200]
    plt.plot(a, acc)
    plt.show()

div_train()
