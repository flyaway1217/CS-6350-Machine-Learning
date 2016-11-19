# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2016-11-13 10:37:10
# Last modified: 2016-11-13 14:25:33

"""
The main entrance for this assiginment.
"""

from forest import Forest
from iomanager import DataManager
from config import Config


def run_forest():
    config = Config('./config.ini')
    data = DataManager(
            [config.train_data_path],
            [config.train_label_path],
            [config.test_data_path],
            [config.test_label_path]
            )
    data.load()
    forest = Forest(data, config.N)
    forest.train()

    preds = forest.predict(data.test_ins)
    preds = [[str(v) for v in pred] for pred in preds]
    preds = [' '.join(v) for v in preds]
    DataManager.write(preds, config.tree_test_outpath)

    preds = forest.predict(data.train_ins)
    preds = [[str(v) for v in pred] for pred in preds]
    preds = [' '.join(v) for v in preds]
    DataManager.write(preds, config.tree_train_outpath)


if __name__ == '__main__':
    run_forest()
