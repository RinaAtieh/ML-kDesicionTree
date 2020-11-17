import dataset_preparation
import decision_tree
import k_decision_tree
import numpy as np
import matplotlib.pyplot as plt

def test_cls(test_data, train_tree, k):
    predict_array = []
    for example in test_data:
        predict = k_decision_tree.k_decistion_tree(example, train_tree, k)
        predict_array.append(predict)
    actual_labels = test_data[: ,-1]
    matches = np.count_nonzero(actual_labels == predict_array)
    examples_num = len(test_data)
    accuracy = matches / examples_num
    return accuracy

def experiments():
    # default values
    n=7
    k=3
    p=0.5

    data_4 = dataset_preparation.loaded_X_train_4
    data_10 = dataset_preparation.loaded_X_train_10

    test_data_4 = dataset_preparation.loaded_X_test_4
    test_data_10 = dataset_preparation.loaded_X_test_10

    # exp1 K is the variable.
    # dataset 4
    trees = k_decision_tree.n_decistion_tree(n, 1, p, data_4)
    accuracy1_4 = test_cls(test_data_4, trees, 1)
    trees = k_decision_tree.n_decistion_tree(n, 3, p, data_4)
    accuracy2_4 = test_cls(test_data_4, trees, 3)
    trees = k_decision_tree.n_decistion_tree(n, 5, p, data_4)
    accuracy3_4 = test_cls(test_data_4, trees, 5)
    a = [1, 3, 5]
    b = [accuracy1_4, accuracy2_4, accuracy3_4]
    plt.plot(a, b)
    plt.show()

    # exp2 N is the variable.
    # dataset 4
    trees = k_decision_tree.n_decistion_tree(5, k, p, data_4)
    accuracy1_4 = test_cls(test_data_4, trees, k)
    trees = k_decision_tree.n_decistion_tree(6, k, p, data_4)
    accuracy2_4 = test_cls(test_data_4, trees, k)
    trees = k_decision_tree.n_decistion_tree(7, k, p, data_4)
    accuracy3_4 = test_cls(test_data_4, trees, k)
    a = [5, 6, 7]
    b = [accuracy1_4, accuracy2_4, accuracy3_4]
    plt.plot(a, b)
    plt.show()

    # exp3 P is the variable.
    # dataset 4
    trees = k_decision_tree.n_decistion_tree(n, k, 0.5, data_4)
    accuracy1_4 = test_cls(test_data_4, trees, k)
    trees = k_decision_tree.n_decistion_tree(n, k, 0.6, data_4)
    accuracy2_4 = test_cls(test_data_4, trees, k)
    trees = k_decision_tree.n_decistion_tree(n, k, 0.7, data_4)
    accuracy3_4 = test_cls(test_data_4, trees, k)
    a = [0.5, 0.6, 0.7]
    b = [accuracy1_4, accuracy2_4, accuracy3_4]
    plt.plot(a, b)
    plt.show()

    # exp1 K is the variable.
    # dataset 10
    trees = k_decision_tree.n_decistion_tree(n, 1, p, data_10)
    accuracy1_10 = test_cls(test_data_10, trees, 1)
    trees = k_decision_tree.n_decistion_tree(n, 3, p, data_10)
    accuracy2_10 = test_cls(test_data_10, trees, 3)
    trees = k_decision_tree.n_decistion_tree(n, 5, p, data_10)
    accuracy3_10 = test_cls(test_data_10, trees, 5)
    a = [1, 3, 5]
    b = [accuracy1_10, accuracy2_10, accuracy3_10]
    plt.plot(a, b)
    plt.show()

    # exp2 N is the variable.
    # dataset 10
    trees = k_decision_tree.n_decistion_tree(5, k, p, data_10)
    accuracy1_10 = test_cls(test_data_10, trees, k)
    trees = k_decision_tree.n_decistion_tree(6, k, p, data_10)
    accuracy2_10 = test_cls(test_data_10, trees, k)
    trees = k_decision_tree.n_decistion_tree(7, k, p, data_10)
    accuracy3_10 = test_cls(test_data_10, trees, k)
    a = [5, 6, 7]
    b = [accuracy1_10, accuracy2_10, accuracy3_10]
    plt.plot(a, b)
    plt.show()

    # exp3 P is the variable.
    # dataset 10
    trees = k_decision_tree.n_decistion_tree(n, k, 0.5, data_10)
    accuracy1_10 = test_cls(test_data_10, trees, k)
    trees = k_decision_tree.n_decistion_tree(n, k, 0.6, data_10)
    accuracy2_10 = test_cls(test_data_10, trees, k)
    trees = k_decision_tree.n_decistion_tree(n, k, 0.7, data_10)
    accuracy3_10 = test_cls(test_data_10, trees, k)
    a = [0.5, 0.6, 0.7]
    b = [accuracy1_10, accuracy2_10, accuracy3_10]
    plt.plot(a, b)
    plt.show()


    # found optimal params
    trees = k_decision_tree.n_decistion_tree(6, 3, 0.6, data_4)
    accuracy1_4 = test_cls(test_data_4, trees, 3)
    trees = k_decision_tree.n_decistion_tree(6, 3, 0.6, data_10)
    accuracy1_10 = test_cls(test_data_10, trees, 3)
    a = [4, 10]
    b = [accuracy1_4, accuracy1_10]
    plt.plot(a, b)
    plt.show()

    """
    # my experiment to find the optimal params 
    
    trees = k_decision_tree.n_decistion_tree(6, 3, 0.7, data_4)
    accuracy1_4 = test_cls(test_data_4, trees, 3)
    trees = k_decision_tree.n_decistion_tree(6, 3, 0.7, data_10)
    accuracy1_10 = test_cls(test_data_10, trees, 3)
    a = [4, 10]
    b = [accuracy1_4, accuracy1_10]
    plt.plot(a, b)
    plt.show()

    trees = k_decision_tree.n_decistion_tree(7, 3, 0.7, data_4)
    accuracy1_4 = test_cls(test_data_4, trees, 3)
    trees = k_decision_tree.n_decistion_tree(7, 3, 0.7, data_10)
    accuracy1_10 = test_cls(test_data_10, trees, 3)
    a = [4, 10]
    b = [accuracy1_4, accuracy1_10]
    plt.plot(a, b)
    plt.show()
    """

experiments()