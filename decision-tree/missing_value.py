# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-11 17:47:51
# Last modified: 2016-09-11 21:32:33

"""
Simple script for missing value.
"""

import decision_tree as dt

data_path = './datasets/SettingC/CVSplits/'
feat_size = 22
report_path = './SettingC_report.txt'
dt.diff_methods(data_path, feat_size, report_path)

s = ('Data path:{a}\nReport path:{b}\n'
     'Report has been written into '
     'files successfully.').format(a=data_path,
                                   b=report_path)
print(s)
