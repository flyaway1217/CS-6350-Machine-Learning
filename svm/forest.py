# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2016-11-13 08:56:08
# Last modified: 2016-11-13 14:24:43

"""
Random Forest Implementation.
"""

import math
import random

from tree import DecisionTree


class Forest:
    """Impementation for random forest.
    """
    def __init__(self, data, N):
        """Construct the forest.

        Args:
            data: DataManager - An instance of DataManager.
            m: int - The number of training instance for each tree.
            N: int - The number of trees.
        """
        self._data = data
        self._feat_size = data.dim
        self._k = int(math.log2(self._feat_size))
        self._N = N
        self._forest = None

    def train(self):
        """Train N different decision trees.

        Returns:
            a list of DecisionTree.
        """
        reval = []
        for i in range(self._N):
            dataset = self._random(self._data, self._data._train_size)
            dataset = self._format_data(dataset)
            tree = DecisionTree(-1, self._feat_size, self._k)
            tree.train(dataset)
            reval.append(tree)

        self._forest = reval

    def predict(self, inss):
        """Using this forest to predict.
        """
        reval = []
        dataset = self._format_data(inss)
        for ins in dataset:
            pred = []
            for tree in self._forest:
                pred.append(tree.predict(ins))
            reval.append(pred)
        return reval

    ########################################################
    # Private helping methods
    ########################################################
    def _random(self, data, m):
        """Draw m samples with replacement.

        Returns:
            reval: list(Instance) - A list of Instance.
        """
        reval = []
        train_ins = data.train_ins
        for _ in range(m):
            index = random.randint(0, len(train_ins)-1)
            reval.append(train_ins[index])
        return reval

    def _format_data(self, inss):
        """Form the data format for decision tree.

        Args:
            inss: list(Instance) - A list of Instance.

        Returns:
            reavl: list
        """
        reval = []
        for ins in inss:
            feats = ins.feature[:]
            feats.append(ins.label)
            reval.append(feats)
        return reval
