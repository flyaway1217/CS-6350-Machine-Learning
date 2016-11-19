# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-24 14:45:47
# Last modified: 2016-11-13 15:59:27

"""
Configer for perceptron.
"""

import configparser
import numpy as np


class Config:
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.read(path)

        # SVM
        self.rate = config.getfloat('SVM', 'rate')
        self.C = config.getfloat('SVM', 'C')
        self.epoch = config.getint('SVM', 'epoch')
        self.random = config.getboolean('SVM', 'isRandom')
        self.randomseed = config.getfloat('SVM', 'randomseed')
        self.isshuffle = config.getboolean('SVM', 'isshuffle')

        self.mode = config.get('SVM', 'mode')
        if self._check_mode(self.mode) is False:
            raise Exception('Unrecognzied mode !')

        # Paths
        self.train_data_path = config.get('Paths', 'train_data_path')
        self.train_label_path = config.get('Paths', 'train_label_path')
        self.test_data_path = config.get('Paths', 'test_data_path')
        self.test_label_path = config.get('Paths', 'test_label_path')
        self.cvpath = config.get('Paths', 'cvpath')
        self.model_path = config.get('Paths', 'model_path')
        self.details_report_path = config.get(
                'Paths', 'details_report_path')
        self.simple_report_path = config.get(
                'Paths', 'simple_report_path')

        # Hyperparamaeters
        values = config.get('HyperParamaeters', 'rate_range')
        values = self._parse_range(values)
        rate_s = values[0]
        rate_e = values[1]
        rate_step = values[2]

        self.rate_ranges = np.arange(
                rate_s, rate_e, rate_step
                )

        values = config.get('HyperParamaeters', 'C_range')
        values = self._parse_range(values)
        s = values[0]
        num = int(values[1])
        rate = values[2]
        C_ranges = [s*(rate**i) for i in range(num)]
        self.C_ranges = np.array(C_ranges)

        # Tree
        self.N = config.getint('Tree', 'N')
        self.tree_train_outpath = config.get('Tree', 'tree_train_outpath')
        self.tree_test_outpath = config.get('Tree', 'tree_test_outpath')

    def _check_mode(self, mode):
        modes = set(['Train', 'TrainTest', 'Test',
                     'CrossValidation', 'HyperParameters'])

        return mode in modes

    def _parse_range(self, string):
        values = string.split(':')
        values = [float(v) for v in values]
        return values[0], values[1], values[2]
