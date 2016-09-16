# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-11 17:35:27
# Last modified: 2016-09-11 21:30:39

"""
Simple script for cross validation.
"""

import decision_tree as dt

data_path = './datasets/SettingB/CVSplits/'
feat_size = 22
report_path = './SettingB_report.txt'
dt.limiting_depth(data_path, feat_size, report_path)

s = ('Data path:{a}\nReport path:{b}\n'
     'Report has been written into '
     'files successfully.').format(a=data_path,
                                   b=report_path)
print(s)
