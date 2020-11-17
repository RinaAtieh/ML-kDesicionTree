import dataset_preparation
import decision_tree
import numpy as np

def n_decistion_tree(n, k, p, data):
    our_n_tree = []
    set_data = data
    for i in range(n):
        if int(p*200) < len(set_data):
            cur_size = int(p * 200)
            chosen_idx = np.random.choice(200, replace=True, size=cur_size)
            set_data = data[chosen_idx]
        columns = len(set_data[0]) #19
        centroid_arr = []
        for j in range(columns -1): # without label
            col = set_data[: ,j] #all rows in column j
            max_col = max(col)
            min_col = min(col)
            len_col = len(col)
            avg = (sum(col) - (len_col*min_col))/((max_col-min_col)*len_col) #normalization of avg
            #avg = sum(col)/len(col)
            centroid_arr.append(avg)
        tree = decision_tree.decision_tree_build(set_data)
        our_n_tree.append([tree,centroid_arr])
    return our_n_tree

def find_distance(x, vec):
    dis = 0
    for i in range(len(vec)):
        dis += (vec[i]-x[i]) * (vec[i]-x[i])
    dis = (dis)**(1/2)
    return dis

def find_best_k_distance(all_dist, k):
    topK = []
    for i in range(k):
        min = [[-1, np.inf], -1]
        if len(all_dist) == 0:
            return topK
        for index in range(len(all_dist)):
            if min[0][1] > all_dist[index][1]:
                min[0][0] = all_dist[index][0]
                min[0][1] = all_dist[index][1]
                min[1] = index
        topK.append(min[0])
        all_dist.pop(min[1])
    return topK

def k_decistion_tree(example, all_trees, k):
    all_dist = []
    for elm in range(len(all_trees)):
        dist = find_distance(example,all_trees[elm][1])
        all_dist.append([elm,dist])
    k_trees = find_best_k_distance(all_dist, k)
    predictions = []
    for ptr in k_trees:
        tree = all_trees[ptr[0]][0]
        val = decision_tree.predict(example, tree)
        predictions.append(val)
    count_N = predictions.count("N")
    count_P = predictions.count("P")
    if count_N > count_P:
        return "N"
    if count_N < count_P:
        return "P"
    val = np.random.choice([True, False])
    if val:
        return "P"
    return "N"


