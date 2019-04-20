import numpy as np
import os
from collections import defaultdict
import math
import graphviz
import random
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


def partition(x):
    x_subsets = defaultdict(list)
    for x1 in range(len(x)):
        x_subsets[x[x1]].append(x1)
    return x_subsets


def entropy(y, w=None):
    labels = np.unique(y)
    count = defaultdict(int)
    total = 0
    entropy = 0
    for j in range(len(labels)):
        for i in range(len(y)):
            if y[i] == labels[j]:
                count[j] = count[j] + w[i]
    for key, value in count.items():
        total += value
    for key, value in count.items():
        entropy += (value / total) * math.log(value / total, 2)
    return entropy * -1


def mutual_information(x, y, w=None):
    y_entropy = entropy(y, w)
    MI = defaultdict(int)
    indices = np.unique(x)
    for i in indices:
        cond_entropy = 0
        t = defaultdict(list)
        weights_temp = defaultdict(list)
        for j in range(len(x)):
            if x[j] == i:
                t[0].append(y[j])
                weights_temp[0].append(w[j])
            else:
                t[1].append(y[j])
                weights_temp[1].append(w[j])
        for value, value1 in zip(t.values(), weights_temp.values()):
            cond_entropy += (len(value) / len(x)) * entropy(value, value1)
        MI[i] = y_entropy - cond_entropy
    return MI


def boosting(x, y, max_depth, num_stumps):
    trees = []
    alphas = []
    z_t = 0
    weights = [1 / len(y) for x in range(len(y))]
    for i in range(num_stumps):
        tree = id3(x, y, None, 0, max_depth, weights)
        trees.append(tree)
        y_pred = [predict_tree(x1, tree) for x1 in x]
        e_t = compute_error(y, y_pred, weights)
        alpha = 0.5 * math.log(((1 - e_t) / e_t), 2)
        alphas.append(alpha)
        for k1 in range(len(y)):
            z_t += weights[k1] * math.exp(-1 * alpha * y[k1] * y_pred[k1])
        for i in range(len(y)):
            if y[i] != y_pred[i]:
                weights[i] = weights[i] * math.exp(alpha) / z_t
            else:
                weights[i] = weights[i] * math.exp(-1 * alpha) / z_t
    return (alphas, trees)


def bagging(x, y, max_depth, num_trees):
    index = 0
    trees = []
    alphas = []
    weights = [1 for x in range(len(x))]
    for j in range(num_trees):
        x_new = []
        y_new = []
        for i in range(len(x)):
            index = random.randrange(len(x))
            x_new.append(x[index])
            y_new.append(y[index])
        trees.append(id3(np.asarray(x_new), np.asarray(y_new), None, 0, max_depth, weights))
        alphas.append(1)
    return (alphas, trees)


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
    decision_tree = dict()
    vector = 0
    vector_value = 0
    if (attribute_value_pairs == None):
        attribute_value_pairs = []
        for i in range(len(x[0])):
            unique_list = []
            for l in x[:, i]:
                if l not in unique_list:
                    unique_list.append(l)
                    attribute_value_pairs.append(tuple((i, l)))
    list_vectors = []
    if (depth == max_depth or len(attribute_value_pairs) == 0):
        return (np.argmax(np.bincount(y)))
    if (all(p == y[0] for p in y)):
        return y[0]
    for i in attribute_value_pairs:
        list_vectors.append(i[0])
    uniq_list = np.unique(list_vectors)
    mi = defaultdict(int)
    max = 0
    for i in uniq_list:
        mi = mutual_information(x[:, i], y, weights)
        for key, value in mi.items():
            if ((i, key) in attribute_value_pairs):
                if max < value:
                    max = value
                    vector = i
                    vector_value = key
    attribute_value_pairs.remove((vector, vector_value))
    x_left = []
    x_right = []
    y_left = []
    y_right = []
    for k, p in zip(x, y):
        if (k[vector] == vector_value):
            x_left.append(k)
            y_left.append(p)
        else:
            x_right.append(k)
            y_right.append(p)
    array_copy = attribute_value_pairs.copy()
    decision_tree[vector, vector_value, True] = id3(np.asarray(x_left), y_left, array_copy, depth + 1, max_depth,
                                                    weights)
    array_copy = attribute_value_pairs.copy()
    decision_tree[vector, vector_value, False] = id3(np.asarray(x_right), y_right, array_copy, depth + 1, max_depth,
                                                     weights)
    return decision_tree


def predict_tree(x, tree):
    if type(tree) is not dict:
        return tree
    for key in tree.keys():
        if ((x[key[0]] == key[1])):
            return (predict_tree(x, tree[key[0], key[1], True]))
        else:
            return (predict_tree(x, tree[key[0], key[1], False]))


def predict_example(x, h_ens):
    alphas = h_ens[0]
    trees = h_ens[1]
    pred = 0
    for i in range(len(trees)):
        pred += alphas[i] * predict_tree(x, trees[i])
    if (pred / sum(alphas)) >= 0.5:
        return 1
    else:
        return 0


def compute_error(y_true, y_pred, w=None):
    e_t = 0
    no_errors = 0
    if w is None:
        return compute_error_tree(y_true, y_pred)
    else:
        for k in range(len(y_true)):
            if y_true[k] != y_pred[k]:
                no_errors += w[k]
        e_t = no_errors / np.sum(w)
    return e_t


def compute_error_tree(y_true, y_pred):
    count = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            count = count + 1
    return (count / len(y_pred))


def pretty_print(tree, depth=0):
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    uid += 1  # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    M = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    M = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]


    print("bagging (3,5)----------------------------------------------------")

    t = bagging(Xtrn, ytrn, 3, 5)
    y_pred = [predict_example(x, t) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print(confusion_matrix(ytst, y_pred))

    print("----------------------scikit learn------------------------")
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=5)
    clf.fit(Xtrn, ytrn)
    y_predicted = clf.predict(Xtst)
    print(confusion_matrix(ytst, y_predicted))

    print("bagging (3,10)----------------------------------------------------")

    t = bagging(Xtrn, ytrn, 3, 10)
    y_pred = [predict_example(x, t) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print(confusion_matrix(ytst, y_pred))

    print("----------------------scikit learn------------------------")
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=10)
    clf.fit(Xtrn, ytrn)
    y_predicted = clf.predict(Xtst)
    print(confusion_matrix(ytst, y_predicted))

    print("Bagging (5,5)----------------------------------------------------")

    t = bagging(Xtrn, ytrn, 5, 5)
    y_pred = [predict_example(x, t) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print(confusion_matrix(ytst, y_pred))

    print("----------------------scikit learn------------------------")
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=5)
    clf.fit(Xtrn, ytrn)
    y_predicted = clf.predict(Xtst)
    print(confusion_matrix(ytst, y_predicted))


    print("Bagging (5,10)----------------------------------------------------")

    t = bagging(Xtrn, ytrn, 5, 10)
    y_pred = [predict_example(x, t) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print(confusion_matrix(ytst, y_pred))

    print("----------------------scikit learn------------------------")
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=10)
    clf.fit(Xtrn, ytrn)
    y_predicted = clf.predict(Xtst)
    print(confusion_matrix(ytst, y_predicted))


    print("Boosting (1,5)----------------------------------------------------")

    t = boosting(Xtrn, ytrn, 1, 5)
    y_pred = [predict_example(x, t) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print(confusion_matrix(ytst, y_pred))

    print("----------------------scikit learn------------------------")
    clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=5)
    clf.fit(Xtrn,ytrn)
    y_predicted = clf.predict(Xtst)
    print(confusion_matrix(ytst, y_predicted))
    print("Boosting (1,10)----------------------------------------------------")

    t = boosting(Xtrn, ytrn, 1, 10)
    y_pred = [predict_example(x, t) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print(confusion_matrix(ytst, y_pred))

    print("----------------------scikit learn------------------------")
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=10)
    clf.fit(Xtrn, ytrn)
    y_predicted = clf.predict(Xtst)
    print(confusion_matrix(ytst, y_predicted))

    print("Boosting (2,5)----------------------------------------------------")

    t = boosting(Xtrn, ytrn, 2, 5)
    y_pred = [predict_example(x, t) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print(confusion_matrix(ytst, y_pred))

    print("----------------------scikit learn------------------------")
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=5)
    clf.fit(Xtrn, ytrn)
    y_predicted = clf.predict(Xtst)
    print(confusion_matrix(ytst, y_predicted))

    print("Boosting (2,10)----------------------------------------------------")

    t = boosting(Xtrn, ytrn, 2, 10)
    y_pred = [predict_example(x, t) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print(confusion_matrix(ytst, y_pred))

    print("----------------------scikit learn------------------------")
    clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=10)
    clf.fit(Xtrn,ytrn)
    y_predicted = clf.predict(Xtst)
    print(confusion_matrix(ytst, y_predicted))

