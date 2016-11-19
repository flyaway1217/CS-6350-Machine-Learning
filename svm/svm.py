# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-11-10 14:04:39
# Last modified: 2016-11-14 20:33:29

"""
Simple SVM classifier implementation.
"""

import random
import collections
import os
# import array

# import numpy as np

from config import Config
from iomanager import DataManager

Result = collections.namedtuple('Result', ['rate', 'C',
                                           'aver_prec', 'prec',
                                           'aver_recall', 'recall',
                                           'averF1', 'F1',
                                           'aver_acc', 'acc'])


class SVM:
    """Simple SVM  Class.
    """
    def __init__(self,
                 rate=1,
                 C=1,
                 epoch=1,
                 random=1,
                 randomseed=-1,
                 isshuffle=True
                 ):
        """Initialize a new SVM classifier.

        Args:
            rate: float - Learning rate.
            C: float - Tradeoff between loss and regularizer.
            epoch: int - Number of passes over the training data.
            random: int - 1 for randon initialization, 0 for all
                          zero initialization.
            randomseed: int - Random seed. If randomseed < 0,
                              we use system time.
            isshuffle: True - If need to shuffle the data.
        """
        self._rate = rate
        self._epoch = epoch
        self._random = random
        self._C = C
        self._isshuffle = isshuffle
        if randomseed < 0:
            self._randomseed = None
        else:
            self._randomseed = randomseed

    ########################################################
    # Propery methods
    ########################################################
    @property
    def weight(self):
        return self._weight

    @property
    def rate(self):
        return self._rate

    @property
    def C(self):
        return self._C

    @property
    def random(self):
        return self._random

    @property
    def randomseed(self):
        return self._randomseed

    @property
    def epoch(self):
        return self._epoch

    @property
    def isshuffle(self):
        return self._isshuffle

    ########################################################
    # Public methods
    ########################################################
    def train(self, data):
        """Train the svm.

        Args:
            data: DataManager
        """
        # Initialize the weights
        self._weight = self._weight_init(
                data.dim, self._randomseed)

        mydata = data.train_ins
        count = 1
        for _ in range(self.epoch):
            if self._isshuffle is True:
                random.shuffle(mydata)
            for ins in mydata:
                step_rate = self._get_rate(count)
                count += 1
                if self._is_mistake(ins):
                    self._weight = self._update(ins, step_rate)
                else:
                    self._weight = self._shrink(
                            self.weight, 1-step_rate)

    def predict(self, data):
        reval = [self._sgn(self._cal(ins.feature)) for ins in data]
        return reval

    def evaluate_acc(self, data):
        """Evaluate the classifier.

        Return:
            accuracy
        """
        preds = self.predict(data.test_ins)
        val = [int(pred == ins.label) for (pred, ins)
               in zip(preds, data.test_ins)]
        return sum(val)/len(val)

    def evaluate_F1(self, data):
        """Evaluate the classifier.

        Return:
            prec: float
            recall: float
            f1: float
        """
        preds = self.predict(data.test_ins)
        TP = sum([1 for (pred, ins) in zip(preds, data.test_ins)
                  if ins.label == 1 and pred == 1])
        FP = sum([1 for (pred, ins) in zip(preds, data.test_ins)
                  if ins.label == -1 and pred == 1])
        FN = sum([1 for (pred, ins) in zip(preds, data.test_ins)
                  if ins.label == 1 and pred == -1])
        if TP+FP == 0:
            prec = 0
            recall = 0
            f1 = 0
        else:
            prec = TP / (TP + FP)
            recall = TP / (TP + FN)
            if prec == 0 or recall == 0:
                f1 = 0
            else:
                f1 = 2 * (prec*recall)/(prec+recall)

        return prec, recall, f1

    def save_model(self, path):
        with open(path, 'w', encoding='utf8') as f:
            s = [str(v) for v in self._weight]
            s = '\n'.join(s)
            f.write(s+'\n')

    def load_model(self, path):
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            self._weight = [float(v.strip()) for v in lines]

    ########################################################
    # Private helping Methods
    ########################################################
    def _get_rate(self, step):
        """This method is used for changing the rate

        Args:
            step: int - The current step.

        Returns:
            A float.
        """
        rate = 1 + (self.rate * step / self._C)
        rate = self.rate / rate
        return rate

    def _is_mistake(self, ins):
        """Check if a current instance is inside the margin.
        """
        value = self._cal(ins.feature)
        return (value * ins.label) <= 1

    def _cal(self, features):
        """Calculte the W*x value.
        Args:
            features:

        Returns:
            W*X
        """
        myfeat = features[:]
        myfeat.insert(0, 1.0)
        if len(myfeat) != len(self.weight):
            raise Exception(
                    'Feature size and weight size is not equal !')

        value = [w*v for w, v in zip(self.weight, myfeat)]

        return sum(value)

    def _update(self, ins, step_rate):
        """Use the given instance to update the weights.
        """
        w1 = self._shrink(self.weight, 1-step_rate)

        myfeat = ins.feature[:]
        myfeat.insert(0, 1)
        if len(myfeat) != len(self.weight):
            raise Exception(
                    'Feature size and weight size is not equal !')

        w2 = [self.C * step_rate * ins.label * v for v in myfeat]

        return [v1+v2 for v1, v2 in zip(w1, w2)]

    def _shrink(self, weight, shrink_rate):
        """Shrink the given weight with shrink_rate.
        """
        new_weight = [v*shrink_rate for v in weight]
        return new_weight

    def _weight_init(self, feat_size, seed):
        """Initializer for weights.
        """
        if self._random is True:
            random.seed(seed)
            weight = [random.random() for i in range(feat_size+1)]
        else:
            weight = [0] * (feat_size+1)
        return weight

    def _sgn(self, value):
        if value >= 0:
            return 1
        else:
            return -1


class Manager:
    """Manager class which is used to
    manage the training and test process.
    """
    def __init__(self, config_path):
        self._config = Config(config_path)

    def run(self):
        """This is ugly implementation...
        """
        if self._config.mode == 'TrainTest':
            svm = SVM(
                    rate=self._config.rate,
                    C=self._config.C,
                    epoch=self._config.epoch,
                    random=self._config.random,
                    randomseed=self._config.randomseed,
                    isshuffle=self._config.isshuffle
                    )
            data = DataManager(
                    [self._config.train_data_path],
                    [self._config.train_label_path],
                    [self._config.test_data_path],
                    [self._config.test_label_path]
                    )
            data.load()
            svm.train(data)
            accuracy = svm.evaluate_acc(data)
            prec, recall, f1 = svm.evaluate_F1(data)
            s = ('Train data:{a}\nTest data:{b}\nAccuracy:{c}'
                 '\nPrec:{d}\nRecall:{e}\nF1:{f}').format(
                    a=self._config.train_data_path,
                    b=self._config.test_data_path,
                    c=accuracy,
                    d=prec,
                    e=recall,
                    f=f1)
            print(s)
        elif self._config.mode == 'HyperParameters':
            rate_ranges = self._config.rate_ranges
            C_ranges = self._config.C_ranges
            results = self._playing_with_hypers(rate_ranges,
                                                C_ranges,
                                                self._config.cvpath)
            bestF1 = max(results, key=lambda x: x.averF1)
            bestAcc = max(results, key=lambda x: x.aver_acc)
            print(bestF1)
            print(bestAcc)
            details, simples = self._report(results, bestF1, bestAcc)
            DataManager.write(details, self._config.details_report_path)
            DataManager.write(simples, self._config.simple_report_path)

    def _playing_with_hypers(self, rate_ranges, C_ranges,
                             path):
        """Playing the hyperparameters.

        Args:
            rate_ranges: Iteration - The ranges of rate.
            C_ranges: Iteration - The ranges of C.
            path: str - The path of cross-validation files.

        Returns:
                A list of Result instance.
        """
        reval = []
        for rate in rate_ranges:
            for C in C_ranges:
                print('Rate={a}, C={b}'.format(a=rate, b=C))
                svm = SVM(
                        rate=rate,
                        C=C,
                        epoch=self._config.epoch,
                        random=self._config.random,
                        randomseed=self._config.randomseed,
                        isshuffle=self._config.isshuffle
                        )
                result = self._cross_validation(svm, path)
                reval.append(result)
        return reval

    def _report(self, results, bestF1, bestAcc):
        """Format the result.
        Args:
            results: list - a list of Result class.
            best: Result - The best result.

        Returns:
            details: a list of str, the report for each result.
            simples: the same with details, only for draw the figure.
        """
        details = []
        simples = []
        for item in results:
            s = ('Rate={a}\nC={b}\nAcc={c}\nAverAcc={d}\n'
                 'F1={e}\nAverF1={f}\n'
                 'prec={g}\nAverPrec={h}\n'
                 'recall={i}\nAverRecall={j}\n\n')
            s = s.format(a=item.rate, b=item.C,
                         c=item.acc, d=item.aver_acc,
                         e=item.F1, f=item.averF1,
                         g=item.prec, h=item.aver_prec,
                         i=item.recall, j=item.aver_recall)
            details.append(s)
            s = '{a} {b} {c} {d} {e} {f}\n'.format(
                    a=item.rate, b=item.C,
                    c=item.aver_acc, d=item.averF1,
                    e=item.aver_prec, f=item.aver_recall)
            simples.append(s)

        s = ('Best hyperparameters By F1\n:BestRate={a}\nC={b}\n'
             'Acc={c}\nAverAcc={d}\n'
             'F1={e}\nAverF1={f}\n'
             'prec={g}\nAverPrec={h}\n'
             'recall={i}\nAverRecall={j}\n\n')
        s = s.format(a=bestF1.rate, b=bestF1.C,
                     c=bestF1.acc, d=bestF1.aver_acc,
                     e=bestF1.F1, f=bestF1.averF1,
                     g=bestF1.prec, h=bestF1.aver_prec,
                     i=bestF1.recall, j=bestF1.aver_recall)

        details.append(s)

        s = ('Best hyperparameters By Accuracy\n:BestRate={a}\nC={b}\n'
             'Acc={c}\nAverAcc={d}\n'
             'F1={e}\nAverF1={f}\n'
             'prec={g}\nAverPrec={h}\n'
             'recall={i}\nAverRecall={j}\n\n')
        s = s.format(a=bestAcc.rate, b=bestAcc.C,
                     c=bestAcc.acc, d=bestAcc.aver_acc,
                     e=bestAcc.F1, f=bestAcc.averF1,
                     g=bestAcc.prec, h=bestAcc.aver_prec,
                     i=bestAcc.recall, j=bestAcc.aver_recall)

        details.append(s)

        return details, simples

    def _cross_validation(self, svm, path):
        """ Run cross validation for the given svm
        on the given data.

        Returns:
            result: Result
        """
        file_names = os.listdir(path)
        data_names = [v for v in file_names if v.endswith('data')]
        label_names = [v for v in file_names if v.endswith('label')]

        accs = []
        precs = []
        recalls = []
        f1s = []
        for test_data_name, test_label_name in zip(
                data_names, label_names):
            test_data_paths = [os.path.join(path, test_data_name)]
            test_label_paths = [os.path.join(path, test_label_name)]

            train_data_paths = [os.path.join(path, v) for v in data_names
                                if v != test_data_name]
            train_label_paths = [os.path.join(path, v) for v in label_names
                                 if v != test_label_name]

            data = DataManager(
                    train_data_paths,
                    train_label_paths,
                    test_data_paths,
                    test_label_paths
                    )
            data.load()
            svm.train(data)

            accs.append(svm.evaluate_acc(data))
            prec, recall, f1 = svm.evaluate_F1(data)
            precs.append(prec)
            recalls.append(recall)
            f1s.append(f1)

        aver_prec = sum(precs) / len(precs)
        aver_recall = sum(recalls) / len(recalls)
        aver_F1 = sum(f1s) / len(f1s)
        aver_acc = sum(accs) / len(accs)

        result = Result(
                svm.rate,
                svm.C,
                aver_prec, precs,
                aver_recall, recalls,
                aver_F1, f1s,
                aver_acc, accs)
        return result


if __name__ == '__main__':
    manager = Manager('config.ini')
    manager.run()
