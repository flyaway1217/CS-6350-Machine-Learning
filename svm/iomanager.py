# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2016-11-13 10:05:35
# Last modified: 2016-11-13 22:19:52

"""
IO Manager.
"""

import collections

Instance = collections.namedtuple('Instance', ['label', 'feature'])


class DataManager:
    """IO Class for reading and writing data.
    """
    def __init__(self,
                 train_data_paths=None,
                 train_label_paths=None,
                 test_data_paths=None,
                 test_label_paths=None):
        self._train_data_paths = train_data_paths
        self._train_label_paths = train_label_paths
        self._test_data_paths = test_data_paths
        self._test_label_paths = test_label_paths

        self._train_ins = []
        self._test_ins = []
        self._dim = -1
        self._train_size = -1
        self._test_size = -1

    def load(self):
        """Load the data.
        """
        if self._train_data_paths is not None:
            for data_path, label_path in zip(
                    self._train_data_paths, self._train_label_paths):
                # print(data_path)
                # print(label_path)
                self._train_ins += self._read(data_path, label_path)

        if self._test_data_paths is not None:
            for data_path, label_path in zip(
                    self._test_data_paths, self._test_label_paths):
                # print(data_path)
                # print(label_path)
                self._test_ins += self._read(data_path, label_path)

        tmp = self._train_ins + self._test_ins
        if self._check(tmp) is False:
            raise Exception('Features of data set is not equal !')
        self._dim = len(self._train_ins[0].feature)

        self._train_size = len(self._train_ins)
        self._test_size = len(self._test_ins)

    @classmethod
    def write(cls, data, path):
        with open(path, 'w', encoding='utf8') as f:
            for item in data:
                f.write(str(item)+'\n')

    ########################################################
    # Private helping methods
    ########################################################
    def _read(self, data_path, label_path):
        """
        Read the data from given path.

        Args:
            data_path: str - The path of input feature.
            label_path : srt - The path of labels.

        Returns:
            a list of Instance.
        """
        reval = []
        with open(data_path, encoding='utf8') as data:
            with open(label_path, encoding='utf8') as label:
                for feat, label in zip(data, label):
                    feat = [float(i) for i in feat.split()]
                    # feat = array.array('f', feat)
                    label = int(label)
                    reval.append(Instance(label, feat))
        return reval

    def _check(self, inss):
        """Check if all these instances have same dimension.

        Args:
            inss: list(Instance) - A list of instance.

        Returns:
            True if all the instances have the same dimension.
        """
        if inss is None:
            return None
        dim = len(inss[0].feature)
        values = [1 for ins in inss if len(ins) != dim]
        return bool(sum(values))

    ########################################################
    # Property methods
    ########################################################
    @property
    def train_size(self):
        return self._train_size

    @property
    def test_size(self):
        return self._test_size

    @property
    def train_ins(self):
        if len(self._train_ins) == 0:
            raise Exception('Training data has not been loaded !')
        return self._train_ins[:]

    @property
    def test_ins(self):
        if len(self._test_ins) == 0:
            raise Exception('Test data has not been loaded !')
        return self._test_ins[:]

    @property
    def dim(self):
        """Return the dimension of feature.
        """
        if self._dim == -1:
            raise Exception('Data has not been loaded !')
        return self._dim
