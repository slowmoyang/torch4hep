import os
import numpy as np
from collections import defaultdict
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ROOT
import pandas as pd

# TODO xticklabels --> epoch
# TODO max or min valid highlighting
#   book(x="step", y="auc", good="max")

class LearningCurve(object):
    def __init__(self, steps_per_epoch, directory, verbose=True):
        self._steps_per_epoch = steps_per_epoch
        self._directory = directory

        self._memory = defaultdict(lambda: list())
        self._booking_list = []

        self._verbose = verbose

    def update(self, **kwargs):
        for key, value in kwargs.iteritems():
            if isinstance(value, list):
                self._memory[key] += value
            else:
                self._memory[key].append(value)

    def draw(self, x, y):
        x_train = self._training[x].values
        y_train = self._training[y].values

        x_valid = self._validation[x].values
        y_valid = self._validation[y].values

        # NOTE LOWESS (locally weighted scatterplot smoothing)
        # https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
        smooth_train_curve = lowess(
            endog=y_train,
            exog=x_train,        
            frac=0.075,
            it=0,
            is_sorted=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        # NOTE plt.plot interface doesn't work well..
        train_curve = Line2D(xdata=x_train, ydata=y_train,
                label="Training",
                color="navy", alpha=0.33, lw=2)
        ax.add_line(train_curve)

        smooth_train_curve = Line2D(
            xdata=smooth_train_curve[:, 0],
            ydata=smooth_train_curve[:, 1],
            label="Training (LOWESS)",
            color="navy", lw=3)
        ax.add_line(smooth_train_curve)

        valid_curve = Line2D(
            xdata=x_valid,
            ydata=y_valid,
            label="Validation",
            color="orange",
            ls="--", lw=3,
            marker="^", markersize=10)
        ax.add_line(valid_curve)


        # TODO is there auto option..?
        # y_min = min(each.min() for each in [y_train, y_valid])
        # y_max = max(each.max() for each in [y_train, y_valid])

        # y_margin = (y_max - y_min) * 0.05
        # y_min -= y_margin
        # y_max += y_margin

        # ax.set_xlim(x_train.min(), x_train.max())
        # ax.set_ylim(y_min, y_max)
        ax.set_xlabel(x, fontdict={"size": 20})
        ax.set_ylabel(y, fontdict={"size": 20})
        ax.legend(fontsize=20)

        path = os.path.join(self._directory, "{}.{{}}".format(y))

        ax.grid()

        fig.savefig(path.format('png'))
        fig.savefig(path.format('pdf'), format="pdf")

        fig.show()

    def save_record(self):
        raise NotImplementedError

    def book(self, y, x="step"):
        self._booking_list.append([x, y])

    def finish(self, booking_list=None):
        if booking_list is not None:
            self._booking_list += booking_list

        self._training = pd.DataFrame(self._memory["training"])
        self._validation = pd.DataFrame(self._memory["validation"])

        training_path = os.path.join(self._directory, "training.csv")
        validation_path = os.path.join(self._directory, "validation.csv")

        self._training.to_csv(training_path)
        self._validation.to_csv(validation_path)

        for x, y in self._booking_list:
            self.draw(x, y)
