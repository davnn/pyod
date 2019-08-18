from collections import defaultdict
from copy import deepcopy

import numpy as np

from .base import BaseDetector
from .lof import LOF


class PartitioningDetector(BaseDetector):
    def __init__(self, base_estimator=None, contamination=0.1, partition_column=0):
        super(PartitioningDetector, self).__init__(contamination=contamination)

        self.partition_column = partition_column

        if base_estimator == None:
            self.base_estimator_ = LOF(contamination=contamination)
        else:
            assert isinstance(base_estimator,
                              BaseDetector), "The estimator is no pyod.models.base.BaseEstimator instance"
            self.base_estimator_ = base_estimator

    def _partition(self, X):
        di = defaultdict(list)
        dv = defaultdict(list)
        ks = X[:, self.partition_column]
        vs = np.delete(X, self.partition_column, 1)

        for i, (k, v) in enumerate(zip(ks, vs)):
            di[k].append(i)
            dv[k].append(v)

        di = {k: np.array(v) for k, v in di.items()}
        dv = {k: np.array(v) for k, v in dv.items()}

        return di, dv

    # returns results of functions 'fs' of shape len(X) x len(fs)
    def _partition_apply(self, di, dv, f):
        try:
            # apply a list of functions fs to partitioned values
            result = [(index, f(self.estimators_[key], dv[key])) for key, index in di.items()]
            # stack the values of partitions
            result = np.hstack(result).transpose()
            # sort by first column
            result = result[result[:, 0].argsort()]
            # return the result vector without the index
            return result[:, 1]
        except KeyError as ke:
            raise KeyError(f"Could not find the key {str(ke)} in the trained detectors with keys: {self.estimators_.keys()}.")

    def fit(self, X, y=None):
        di, self.features_ = self._partition(X)
        self.estimators_ = {k: deepcopy(self.base_estimator_).fit(v) for k, v in self.features_.items()}
        self.decision_scores_ = self._partition_apply(di, self.features_, lambda d, _: d.decision_scores_)

        # set required attributes
        self._process_decision_scores()
        self._set_n_classes(y)

        return self

    def decision_function(self, X):
        di, dv = self._partition(X)
        return self._partition_apply(di, dv, lambda d, v: d.decision_function(v))

    def predict(self, X):
        di, dv = self._partition(X)
        return self._partition_apply(di, dv, lambda d, v: d.predict(v))
