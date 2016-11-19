# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2016-11-13 10:33:58
# Last modified: 2016-11-13 13:50:02

"""
Decision Tree
"""
import collections
import queue
import random
import math


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
    def __init__(self, dataset, depth, feat_size, k):
        self._subset = dataset
        self._depth = depth
        self._feat_size = feat_size
        self._k = k

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

        indexs = random.sample(range(self._feat_size), self._k)

        best_children = {}
        best_feat = -1
        best_info_gain = 0.0
        for i in indexs:
            children = {}
            new_entropy = 0.0

            # Instead of scaning all the possible attribute values,
            # we can noly scan the possible attribute values of subset.
            values = set([instance[i] for instance in self._subset])
            for val in values:
                subset = tuple(instance for instance in self._subset
                               if instance[i] == val)
                children[val] = ID3Node(
                        subset, self.depth+1, self._feat_size, self._k)
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
             'label:{c}, depth:{g}, entropy: {d}, '
             'children number: {e}').format(a=str(self.leaf),
                                            b=str(self._select_feat),
                                            c=str(self.label),
                                            d=str(self.entropy),
                                            e=str(len(self.children)),
                                            f=str(len(self)),
                                            g=str(self.depth),
                                            )
        return s


class DecisionTree:
    """The Decision Tree Class
    """
    def __init__(self, depth_limit, feat_size, k):
        """Initialize a new decision tree.

        Args:
            depth_limit: int - The limitation of depth.
        """
        if depth_limit <= 0:
            self._depth_limit = float('INF')
        else:
            self._depth_limit = depth_limit
        self._depth = -1
        self._feat_size = feat_size
        self._k = k

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
        self._root = ID3Node(train_instances, 1, self._feat_size, self._k)
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
