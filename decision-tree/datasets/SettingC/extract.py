# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-11 15:51:35
# Last modified: 2016-09-11 16:03:29

"""

"""

import collections


def read(path):
    data = []
    with open(path) as f:
        for line in f:
            s = line.strip().split(',')
            data.append(s)
    return data


def most_common(data):
    feats = [ins[10] for ins in data if ins[10] != '?' and ins[-1] == 'e']
    counter = collections.Counter(feats)
    most_common = counter.most_common()
    print(most_common)


if __name__ == '__main__':
    data = read('./training.data')
    most_common(data)
