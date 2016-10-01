# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-24 14:45:47
# Last modified: 2016-09-25 11:05:47

"""
Configer for perceptron.
"""

import configparser


class Config:
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.read(path)

        # Perceptron
        self.feat_size = config.getint('Perceptron', 'feat_size')
        self.rate = config.getfloat('Perceptron', 'rate')
        self.margin = config.getfloat('Perceptron', 'margin')
        self.epoch = config.getint('Perceptron', 'epoch')
        self.random = config.getboolean('Perceptron', 'isRandom')
        self.randomseed = config.getfloat('Perceptron', 'randomseed')
        self.shuffle = config.getboolean('Perceptron', 'shuffle')
        self.aggressive = config.getboolean('Perceptron', 'aggressive')

        self.mode = config.get('Perceptron', 'mode')
        if self._check_mode(self.mode) is False:
            raise Exception('Unrecognzied mode !')

        # Paths
        self.train_path = config.get('Paths', 'train_path')
        self.test_path = config.get('Paths', 'test_path')
        self.cvpath = config.get('Paths', 'cvpath')
        self.model_path = config.get('Paths', 'model_path')
        self.details_report_path = config.get(
                'Paths', 'details_report_path')
        self.simple_report_path = config.get(
                'Paths', 'simple_report_path')

        # Hyperparamaeters
        values = config.get('HyperParamaeters', 'rate_range')
        values = self._parse_range(values)
        self.rate_s = values[0]
        self.rate_e = values[1]
        self.rate_step = values[2]

        values = config.get('HyperParamaeters', 'margin_range')
        values = self._parse_range(values)
        self.margin_s = values[0]
        self.margin_e = values[1]
        self.margin_step = values[2]

    def _check_mode(self, mode):
        modes = set(['Train', 'TrainTest', 'Test',
                     'CrossValidation', 'HyperParameters'])

        return mode in modes

    def _parse_range(self, string):
        values = string.split(':')
        values = [float(v) for v in values]
        return values[0], values[1], values[2]
