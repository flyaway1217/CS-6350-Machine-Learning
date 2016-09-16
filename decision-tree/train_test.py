# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-11 17:23:44
# Last modified: 2016-09-11 21:29:37

"""
Simple script for training and test.
"""

import decision_tree as dt

N = 22
train_path = './datasets/SettingB/training.data'
test_path = './datasets/SettingB/test.data'
mg = dt.Manager(N)
accuracy, depth = mg.train_test(train_path, test_path)

s = ('Training data:{a}\nTest data:{b}\n'
     'accuracy:{c}\ndepth:{d}').format(a=train_path,
                                       b=test_path,
                                       c=str(accuracy),
                                       d=str(depth))
print(s)
