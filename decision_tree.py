import numpy as np

def possible_values_to_split(data):
    array_split_values = {}
    n_columns = len(data[0]) # 18 feature
    for column_index in range(n_columns - 1):  # last column is label
        values = data[:, column_index] # values in this column index
        unique_values = np.unique(values) # unique values
        array_split_values[column_index] = unique_values
    return array_split_values


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column] # all rows in this column
    data_below = data[split_column_values <= split_value] #
    data_above = data[split_column_values > split_value]
    return data_below, data_above

def calc_entropy(data):
    label_column = data[:, -1]
    unique, counts = np.unique(label_column, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_entropy = (p_data_below * calc_entropy(data_below) + p_data_above * calc_entropy(data_above))
    return overall_entropy

def calculate_best_split(data, splits):
    best_metric = np.inf
    for column_index in splits: # try split according to this column (feature)
        for value in splits[column_index]: # try all possible values
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_metric = calculate_overall_entropy(data_below, data_above)
            if current_metric <= best_metric:
                best_metric = current_metric
                best_split_column = column_index
                best_split_value = value
    return best_split_column, best_split_value

def create_leaf(data):
    # best classification
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    index = counts_unique_classes.argmax()
    leaf = unique_classes[index]
    return leaf


def decision_tree_build(df):
    data = df
    splits = possible_values_to_split(data)
    split_column, split_value = calculate_best_split(data, splits)
    data_below, data_above = split_data(data, split_column, split_value) # split the data

    # if the best split with no benefit calcuate the most classification, and return
    if len(data_below) == 0 or len(data_above) == 0:
        leaf = create_leaf(data)
        return leaf

    # sub tree is dictionary of the question that we should ask (question values) to the two possibilty of answer
    question = (split_column,split_value)
    sub_tree = {question: []}

    #recursion on data below and data above
    yes_answer = decision_tree_build(data_below) # build the left decision tree of yes answer
    no_answer = decision_tree_build(data_above) # build the right decision tree of no answer

    if yes_answer == no_answer: # so no point for asking
        sub_tree = yes_answer # so sub_tree is leaf
    else:
        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)

    return sub_tree

def predict(example, tree):

    # if tree is not instance of dict so the tree have the classification directly
    if not isinstance(tree, dict):
        return tree

    question = list(tree.keys())[0] # the current question of split
    col = question[0]
    value = question[1]

    # ask question
    if example[int(col)] <= float(value): # yes answer
         answer = tree[question][0]
    else: # no answer
         answer = tree[question][1]

    # if answer is value (not tree)
    if not isinstance(answer, dict):
        return answer

    # recursive, we should continue asking on the remaining tree
    else:
        remaining_tree = answer
        return predict(example, remaining_tree)