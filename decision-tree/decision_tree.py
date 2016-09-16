# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-06 14:49:29
# Last modified: 2016-09-13 09:30:19

"""
Decision Tree implementation using ID3 alogrithm.

TODO:   1. Add a logger to record running info.
        2. Draw the decision tree in the web.
"""

import collections
import math
import queue
import os

import numpy as np


class IOManager:
    """IO Class for reading and writing data.
    """
    def __init__(self, feat_size):
        self._feat_size = feat_size

    ########################################################
    # Public methods
    ########################################################
    def read(self, path):
        """Read the data from given path.
        Returns:
            - A tuple of data instances.
              Each instance is also represented as a tuple.
        """
        reval = []
        with open(path, encoding='utf8') as f:
            for line in f:
                s = line.strip().lower().split(',')
                if len(s) != self._feat_size + 1:
                    print('data size=' + str(len(s)))
                    print('feat size=' + str(self._feat_size))
                    raise Exception('Unmatched feature size !')
                reval.append(s)
        return reval

    def write(self, path, data):
        """Write the data into the given file.
        """
        with open(path, 'w', encoding='utf8') as f:
            for line in data:
                f.write(str(line)+'\n')


class ID3Node:
    """Tree node for Decision Tree.

    Each tree node should contains:
        - a subset of intances.
        - entropy of this node.
        - an indicator shows which attributes has been used.
        - a label for this node.
        - a dict from attribute values to its children
        - a indicator shows which attribute is selected
        - a indicator shows which level this node is
        - a split method
        - a predict method
    """
    def __init__(self, dataset, attr_visited, depth, attr_value=None):
        self._subset = dataset
        self._attr_visited = attr_visited
        self._depth = depth
        self._attr_value = attr_value

        most_common = self._get_most_common()
        self._entropy = self._calculate_entropy(most_common)

        self._label = most_common[0][0]
        self._leaf = (most_common[0][-1] == len(self._subset))
        self._children = {}

        # If current node is a leaf node, _select_feat should be -1
        # otherwise, it will be the index of selected feature.
        self._select_feat = -1

    ########################################################
    # Property methods
    ########################################################
    @property
    def depth(self):
        return self._depth

    @property
    def entropy(self):
        return self._entropy

    @property
    def label(self):
        return self._label

    @property
    def leaf(self):
        return self._leaf

    @leaf.setter
    def leaf(self, value):
        """Set this node to be a leaf node.
        This is used for the constraint for depth.
        """
        self._leaf = bool(value)

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, value):
        self._children = value

    ########################################################
    # Public methods
    ########################################################
    def split(self):
        """Split this node according to information gain.
        """
        if self.leaf is True:
            return None

        indexs = [i for i in range(len(self._attr_visited))
                  if self._attr_visited[i] is False]

        best_children = {}
        best_feat = -1
        best_info_gain = 0.0
        for i in indexs:
            children = {}
            new_entropy = 0.0
            attr_visited = self._attr_visited[:]
            attr_visited[i] = True

            # Instead of scaning all the possible attribute values,
            # we can noly scan the possible attribute values of subset.
            values = set([instance[i] for instance in self._subset])
            for val in values:
                subset = tuple(instance for instance in self._subset
                               if instance[i] == val)
                children[val] = ID3Node(
                        subset, attr_visited, self.depth+1, val)
            for key in children:
                child = children[key]
                prob = float(len(child)/len(self._subset))
                new_entropy += (prob * child.entropy)
            info_gain = self.entropy - new_entropy
            if info_gain > best_info_gain:
                best_children = children
                best_info_gain = info_gain
                best_feat = i
        self._select_feat = best_feat
        return best_children

    def predict(self, instance):
        """Predict the label for the given instance.

        Args:
            instance: tuple - an instance of test data.
        """
        if self.leaf is True:
            return self.label
        else:
            value = instance[self._select_feat]
            if value not in self.children:
                return self.label
            else:
                return self.children[value].predict(instance)

    ########################################################
    # Private helping methods
    ########################################################
    def _calculate_entropy(self, most_common):
        """Calculate the entropy for this node.
        """
        # get the class prob distribution
        probs = [item[-1]/len(self._subset) for item in most_common]
        entropy = sum([-1*p*math.log2(p) for p in probs])

        return entropy

    def _get_most_common(self):
        labels = [item[-1] for item in self._subset]
        counter = collections.Counter(labels)
        return counter.most_common()

    ###############################################
    # Magic methods
    ########################################################
    def __len__(self):
        """
        Return the number of data instance.
        """
        return len(self._subset)

    def __str__(self):
        s = ('Node: leaf:{a}, select_feat:{b}, subset_size:{f}, '
             'label:{c}, depth:{g}, entropy: {d}, attr_value={h}, '
             'children number: {e}').format(a=str(self.leaf),
                                            b=str(self._select_feat),
                                            c=str(self.label),
                                            d=str(self.entropy),
                                            e=str(len(self.children)),
                                            f=str(len(self)),
                                            g=str(self.depth),
                                            h=str(self._attr_value))
        return s


class DecisionTree:
    """The Decision Tree Class
    """
    def __init__(self, depth_limit, feat_size):
        """Initialize a new decision tree.

        Args:
            depth_limit: int - The limitation of depth.
        """
        self._depth_limit = depth_limit
        self._depth = -1
        self._feat_size = feat_size

    ########################################################
    # Property methods
    ########################################################
    @property
    def depath_limit(self):
        """TODO: Docstring for depath_limit.
        """
        return self._depth_limit

    @property
    def depth(self):
        return self._depth

    ########################################################
    # Public methods
    ########################################################
    def train(self, train_instances):
        """Using hierarchical traverse instead of recursion.
        """
        que = queue.Queue()
        attr_visited = [False] * self._feat_size
        self._root = ID3Node(train_instances, attr_visited, 1)
        que.put(self._root)
        while not que.empty():
            node = que.get()
            if node.depth > self._depth:
                self._depth = node.depth
            if node.leaf is False and node.depth < self._depth_limit:
                children = node.split()
                node.children = children
                for key in children:
                    que.put(children[key])

    def predict(self, test_instance):
        return self._root.predict(test_instance)

    ########################################################
    # Magic methods
    ########################################################
    def __str__(self):
        node_queue = collections.OrderedDict()
        que = queue.Queue()
        que.put(self._root)
        while not que.empty():
            node = que.get()
            if node.depth not in node_queue:
                node_queue[node.depth] = []
            node_queue[node.depth].append(node)
            for key in node.children:
                que.put(node.children[key])

        s = ''
        for key in node_queue:
            s += ('Level:{a}\n'.format(a=str(key)))
            for node in node_queue[key]:
                s += (str(node)+'\n')
            s += ('-'*50+'\n')
        return s


class Manager:
    """
    Manager class which is used to
    manage the training and test process.
    """
    def __init__(self, feat_size, max_depth=-1, miss_option=3):
        self._feat_size = feat_size
        self._max_depth = max_depth
        self._miss_option = miss_option

        self.io = IOManager(feat_size)

    def train(self, data):
        # Training
        if self._max_depth <= 0:
            tree = DecisionTree(float('INF'), self._feat_size)
        else:
            tree = DecisionTree(self._max_depth, self._feat_size)
        # tree = DecisionTree(5)
        train_data = self._preprocessing(data, self._miss_option)
        tree.train(train_data)
        return tree

    def test(self, data, tree):
        accuracy = 0
        if self._miss_option == 2:
            option = 1
        else:
            option = self._miss_option
        test_data = self._preprocessing(data, option)
        for instance in test_data:
            pred_label = tree.predict(instance)
            accuracy += int(instance[-1] == pred_label)
        accuracy = accuracy/len(test_data)
        return accuracy

    def train_test(self, train_path, test_path):
        """Train the decision tree and make the test.

        Returns:
            - accuracy of the predictation.
            - depth of this tree.
        """
        # Train
        data = self.io.read(train_path)
        tree = self.train(data)

        # Test
        data = self.io.read(test_path)
        accuracy = self.test(data, tree)
        return accuracy, tree.depth

    def cross_validation(self, path):
        """
        Proceed the cross validation.

        args:
            path: str - The path of the cross validation files.
            n: int - The number of files.
        """
        # Construt a decision tree
        file_names = os.listdir(path)
        accs = []
        depths = []
        for test_name in file_names:
            train_data = []
            test_data = self.io.read(os.path.join(path, test_name))
            for train_name in file_names:
                if test_name == train_name:
                    continue
                else:
                    train_data += self.io.read(os.path.join(path, train_name))
            tree = self.train(train_data)
            depths.append(tree.depth)
            accs.append(self.test(test_data, tree))
        average, std = self._statistics(accs)
        # print(accs)
        # print(depths)
        # s = ('average accuracy:{a}\t'
        #      'standard deviation:{b}').format(a=str(average),
        #                                       b=str(std))
        # print(s)
        return accs, depths, average, std

    ########################################################
    # Private methods
    ########################################################
    def _preprocessing(self, data, option):
        """Preprocessing the data for miss feature.

        There are three different methods:
        1. Setting the missing feature as the majority feature value.
        2. Setting the missing feature as the majority value of that label.
        3. Treating the missing feature as a special feature.

        Args:
            data: list - A list of data need to be preprocessing.
            feat_size: int - The size of features.
            option: int - Indicate which method need to be used.
                          Range:[1, 2, 3]
        """
        option_dict = {
                1: self._method_1,
                2: self._method_2,
                3: self._method_3
                }
        option_dict[option](data)
        return data

    def _method_1(self, data):
        majority_value = self._majority(data)
        for ins in data:
            # in case there are more than 2 '?' in one instance.
            while '?' in ins:
                index = ins.index('?')
                ins[index] = majority_value[index]

    def _method_2(self, data):
        for ins in data:
            # in case there are more than 2 '?' in one instance.
            while '?' in ins:
                index = ins.index('?')
                lable_set = [item[index] for item in data
                             if (item[-1] == ins[-1] and
                                 item[index] != '?')]
                counter = collections.Counter(lable_set)
                ins[index] = counter.most_common()[0][0]

    def _method_3(self, data):
        return

    def _majority(self, data):
        """Compute the majority value for each feature.
        """
        majority_value = [-1] * self._feat_size
        for index in range(self._feat_size):
            values = [item[index] for item in data if item[index] != '?']
            counter = collections.Counter(values)
            majority = counter.most_common()[0][0]
            majority_value[index] = majority
        # print(majority_value)
        return majority_value

    def _statistics(self, accuracy):
        mydata = np.array(accuracy)
        mean = np.mean(mydata)
        std = np.std(mydata)
        return mean, std

############################################################
# Problems related functions
############################################################


def limiting_depth(path, feat_size, report_path):
    limit_depths = range(1, 21)
    with open(report_path, 'w', encoding='utf8') as f:
        for depth in limit_depths:
            mg = Manager(feat_size, depth)
            accs, depths, average, std = mg.cross_validation(path)
            report = ('Max Depth:{a}\naccuracy:{b}\nreal depth:{c}\n'
                      'average accuracy:{d}'
                      '\tstd:{e}\n\n').format(a=str(depth),
                                              b=str(accs),
                                              c=str(depths),
                                              d=str(average),
                                              e=str(std))
            f.write(report)


def diff_methods(path, feat_size, report_path):
    with open(report_path, 'w', encoding='utf8') as f:
        for i in range(1, 4):
            mg = Manager(feat_size, miss_option=i)
            accs, depths, average, std = mg.cross_validation(path)
            report = ('Method:{a}\naccuracy:{b}\ndepth:{c}\n'
                      'average accuracy:{d}'
                      '\tstd:{e}\n\n').format(a=str(i),
                                              b=str(accs),
                                              c=str(depths),
                                              d=str(average),
                                              e=str(std))
            f.write(report)
