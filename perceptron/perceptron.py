# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.1
#
# Date: 2016-09-17 15:15:06
# Last modified: 2016-10-01 11:02:45

"""
Simple Perceptron classifier implementation.
"""

import random
import collections
import os

import numpy as np

from config import Config

Instance = collections.namedtuple('Instance', ['label', 'feature'])
Result = collections.namedtuple('Result', ['rate', 'margin',
                                           'aver', 'accs'])


class IOManager:
    """IO Class for reading and writing data.
    """
    def __init__(self):
        pass

    def read(self, path):
        """
        Read the data from given path.

        Because the data is represented in sprase form, this reading
        method will treat each instance as a dict. So, this method
        finally returns a list of Instance objects.

        Args:
            path: str - The path of input data.

        Returns:
            a list of Instance.
        """
        reval = []
        with open(path, encoding='utf8') as f:
            for line in f:
                values = line.strip().split()
                label = int(values[0])
                dic_ins = dict()
                values = [item.split(':') for item in values[1:]]
                for index, value in values:
                    dic_ins[int(index)] = int(value)
                reval.append(Instance(label, dic_ins))
        return reval

    def write(self, data, path):
        with open(path, 'w', encoding='utf8') as f:
            for item in data:
                f.write(str(item))


class Perceptron:
    """Simple Perceptron Class.

    Although the features are in sparse, I still decide to use
    a fixed length vector to store the wegihts.
    The reason is that I want to initialize the weights at the same time
    for the convenient of next experiments.
    """
    def __init__(self,
                 feat_size=200,
                 rate=1,
                 margin=0,
                 epoch=1,
                 random=1,
                 randomseed=-1,
                 aggressive=False):
        """Initialize a new Perceptron classifier.

        Args:
            feat_size: int - The size of features.
            rate: float - Learning rate.
            batch: int - Number of passes over the training data.
            margin: postive float - Margin.
            random: int - 1 for randon initialization, 0 for all
                          zero initialization.
            randomseed: int - Random seed. If randomseed < 0,
                              we use system time.
        """
        self._feat_size = feat_size
        self._rate = rate
        if margin < 0:
            raise Exception('Margin must be postive float number !')
        self._margin = margin
        self._epoch = epoch
        self._random = random
        self._aggressive = aggressive
        if randomseed < 0:
            self._randomseed = None
        else:
            self._randomseed = randomseed

        # Initialize the weights
        self._weight, self._b = self._weight_init(
                self._feat_size, self._randomseed)
        self._count = 0

    ########################################################
    # Propery methods
    ########################################################
    @property
    def weight(self):
        return self._weight

    @property
    def b(self):
        return self._b

    @property
    def count(self):
        """Return the number of mistakes.
        """
        return self._count

    ########################################################
    # Public methods
    ########################################################
    def train(self, data, isshuffle=False):
        """Train the perceptron.

        Args:
            data: list(Instance) - The training data.
            isshuffle: boolean - Indicate if need shuffling
                                 during different epochs.
        """
        mydata = data[:]
        for i in range(self._epoch):
            if isshuffle is True:
                random.shuffle(mydata)
            for ins in mydata:
                if self._is_mistake(ins):
                    self._update(ins)

    def predict(self, data):
        reval = [self._sgn(self._cal(ins.feature)) for ins in data]
        return reval

    def evaluate(self, data):
        """Evaluate the classifier.

        Return:
            precision.
        """
        preds = self.predict(data)
        val = [int(pred == ins.label) for (pred, ins)
               in zip(preds, data)]
        return sum(val)/len(val)

    def save_model(self, path):
        with open(path, 'w', encoding='utf8') as f:
            s = [str(v) for v in self._weight]
            s = '\n'.join(s)
            f.write(s+'\n')
            f.write(str(self._b))

    def load_model(self, path):
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            self._weight = [float(v.strip()) for v in lines[:-1]]
            self._b = float(lines[-1])

    ########################################################
    # Private Helping Methods
    ########################################################

    def _get_rate(self, ins, margin):
        """This method is used for aggressive algorithm.

        Args:
            ins: Instance - The current training instance.
            margin: float - Margin.

        Returns:
            A float.
        """
        if self._aggressive is False:
            return self._rate
        else:
            value = margin - ins.label * self._cal(ins.feature)
            s = 0
            for i in ins.feature:
                s += (ins.feature[i] * ins.feature[i])
            return value / (s + 1)

    def _is_mistake(self, ins):
        """Check if a mistake is made on the given instance.
        """
        value = self._cal(ins.feature)
        # Deal with the special case.
        if value == 0:
            sgn = self._sgn(value)
        else:
            sgn = value
        return sgn * ins.label < self._margin

    def _cal(self, features):
        """Calculte the W*x + b value.
        Args:
            features: dit - The features values of current instance.

        Returns:
            W*x+b
        """
        value = 0
        for index in features:
            value += self._weight[index] * features[index]
        return value + self._b

    def _sgn(self, value):
        """
        Linear Threshold Unit.
        """
        if value >= 0:
            return 1
        else:
            return -1

    def _update(self, ins):
        """Use the given instance to update the weights.
        """
        self._count += 1
        rate = self._get_rate(ins, self._margin)
        self._b += (ins.label * rate)
        for index in ins.feature:
            update = ins.label * rate * ins.feature[index]
            self._weight[index] += update

    def _weight_init(self, feat_size, seed):
        """Initializer for weights.
        """
        random.seed(seed)
        weight = [random.random() for i in range(feat_size)]
        b = random.random()
        return weight, b


class Manager:
    """Manager class which is used to
    manage the training and test process.
    """
    def __init__(self, config_path):
        self.io = IOManager()
        self._config = Config(config_path)

    def run(self):
        """This is ugly implementation...
        """
        if self._config.mode == 'Train':
            perceptron = Perceptron(
                    feat_size=self._config.feat_size,
                    rate=self._config.rate,
                    margin=self._config.margin,
                    epoch=self._config.epoch,
                    random=self._config.random,
                    randomseed=self._config.randomseed,
                    aggressive=self._config.aggressive
                    )
            train_data = self.io.read(self._config.train_path)
            perceptron.train(train_data, self._config.shuffle)
            perceptron.save_model(self._config.model_path)
            s = ('Train completed. Made {b} mistakes.'
                 'Model is saved in {a}').format(
                    a=self._config.model_path,
                    b=perceptron.count)
            print(s)
        elif self._config.mode == 'TrainTest':
            perceptron = Perceptron(
                    feat_size=self._config.feat_size,
                    rate=self._config.rate,
                    margin=self._config.margin,
                    epoch=self._config.epoch,
                    random=self._config.random,
                    randomseed=self._config.randomseed,
                    aggressive=self._config.aggressive
                    )
            train_data = self.io.read(self._config.train_path)
            test_data = self.io.read(self._config.test_path)
            perceptron.train(train_data, self._config.shuffle)
            accuracy = perceptron.evaluate(test_data)
            s = ('Train data:{a}\nTest data:{b}\nMade {d} mistakes'
                 '\nAccuracy:{c}').format(
                    a=self._config.train_path,
                    b=self._config.test_path,
                    c=accuracy,
                    d=perceptron.count)
            print(s)
        elif self._config.mode == 'CrossValidation':
            perceptron = Perceptron(
                    feat_size=self._config.feat_size,
                    rate=self._config.rate,
                    margin=self._config.margin,
                    epoch=self._config.epoch,
                    random=self._config.random,
                    randomseed=self._config.randomseed,
                    aggressive=self._config.aggressive
                    )
            aver, accs = self._cross_validation(
                    perceptron,
                    self._config.cvpath)
            s = 'Cross-Validation on {a}\naccs={b}\naverage={c}'.format(
                    a=self._config.cvpath,
                    b=accs, c=aver)
            print(s)
        elif self._config.mode == 'HyperParameters':
            rate_ranges = np.arange(
                    self._config.rate_s,
                    self._config.rate_e,
                    self._config.rate_step)
            margin_ranges = np.arange(
                    self._config.margin_s,
                    self._config.margin_e,
                    self._config.margin_step)
            results = self._playing_with_hypers(rate_ranges,
                                                margin_ranges,
                                                self._config.cvpath)
            best = max(results, key=lambda x: x.aver)
            print(best)
            details, simples = self._report(results, best)
            self.io.write(details, self._config.details_report_path)
            self.io.write(simples, self._config.simple_report_path)

        elif self._config.mode == 'Test':
            perceptron = Perceptron(
                    feat_size=self._config.feat_size,
                    rate=self._config.rate,
                    margin=self._config.margin,
                    epoch=self._config.epoch,
                    random=self._config.random,
                    randomseed=self._config.randomseed,
                    aggressive=self._config.aggressive
                    )
            perceptron.load_model(self._config.model_path)
            test_data = self.io.read(self._config.test_path)
            accuracy = perceptron.evaluate(test_data)
            s = 'Train data:{a}\nTest data:{b}\nAccuracy:{c}'.format(
                    a=self._config.train_path,
                    b=self._config.test_path,
                    c=accuracy)
            print(s)

    def _playing_with_hypers(self, rate_ranges, margin_ranges,
                             path):
        """Playing the hyperparameters.

        Args:
            rate_ranges: Iteration - The ranges of rate.
            margin_ranges: Iteration - The ranges of margin.
            path: str - The path of cross-validation files.

        Returns:
                A list of Result instance.
        """
        reval = []
        for rate in rate_ranges:
            for margin in margin_ranges:
                print('Rate={a}, Margin={b}'.format(a=rate, b=margin))
                perceptron = Perceptron(
                        feat_size=self._config.feat_size,
                        rate=rate,
                        margin=margin,
                        epoch=self._config.epoch,
                        random=self._config.random,
                        randomseed=self._config.randomseed,
                        aggressive=self._config.aggressive
                        )
                aver, accs = self._cross_validation(perceptron, path)
                reval.append(Result(rate, margin, aver, accs))
        return reval

    def _report(self, results, best):
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
            s = 'Rate={a}\nMargin={b}\nAccs={c}\nAverage={d}\n\n'
            s = s.format(a=item.rate, b=item.margin,
                         c=item.accs, d=item.aver)
            details.append(s)
            s = '{a} {b} {c}\n'.format(
                    a=item.rate, b=item.margin, c=item.aver)
            simples.append(s)

        s = ('Best hyperparameters:\nRate={a}\nMargin={b}'
             '\nAccs={c}\nAverage={d}\n\n').format(
                     a=best.rate, b=best.margin,
                     c=best.accs, d=best.aver)
        details.append(s)

        return details, simples

    def _cross_validation(self, perceptron, path):
        """ Run cross validation for the given perceptron
        on the given data.

        Returns:
            average: The average accuracy of cross validation.
            accs: a list of accuracy.
        """
        file_names = os.listdir(path)
        accs = []
        for test_name in file_names:
            train_data = []
            test_data = self.io.read(os.path.join(path, test_name))
            for train_name in file_names:
                if test_name == train_name:
                    continue
                else:
                    train_data += self.io.read(os.path.join(path, train_name))
            perceptron.train(train_data, self._config.shuffle)
            accs.append(perceptron.evaluate(test_data))
        average = sum(accs) / len(accs)
        return average, accs


if __name__ == '__main__':
    manager = Manager('config.ini')
    manager.run()
