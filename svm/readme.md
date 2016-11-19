In this problem, there are a lot files conerning the cource code.

# Files

## Source code

Actually there three source code file.

- ``svm.py``: The core source file of SVM.
- ``config.py``: The configuration cource code file which deals with read and store the configuration.
- ``draw.py``: To draw figures for playing around with hyperparameters.
- ``forest.py``: The source file for random forest.
- ``iomanager.py``: The standard input and output wrapper.
- ``main.py``: The entrance point for forest experiments.
- ``tree.py``: The source file for decision tree.


## Data files

- ``details.txt`` and ``simple.txt``: These two files contains the result of experiment when I play with rate and margin. ``detials.txt`` has detial result while ``simple.txt`` is used to draw the figures.


## Config file

``config.ini``: This a config file contains all the configuration of this perceptron. If you want to use different configurations, just change the this file and no need to change the source code file.


# Run

Every time you set a configuration in the ``config.ini``, just run ``python3 svm.py``.
My ``perceptron.py`` code can be run in the CADE machines with Python3.
But for ``draw.py``, because CADE machines do not install ``matplotlib`` for Python3.
So, if you want to draw the figures, PLEASE USE python TO RUN ``draw.py`` INSTEAD OF pythone3.
