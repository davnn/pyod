from collections import defaultdict
from copy import deepcopy

import numpy as np

from .base import BaseDetector
from .lof import LOF


class ContextualDetector(BaseDetector):
    def __init__(self, contamination=0.1, base_detector=None, context_column=0):
        super(ContextualDetector, self).__init__(contamination=contamination)

        self.context_column = context_column

        if base_detector == None:
            self.base_detector_ = LOF(contamination=contamination)
        else:
            assert isinstance(base_detector,
                              BaseDetector), "The estimator is no pyod.models.base.BaseEstimator instance"
            self.base_detector_ = base_detector

    def _contextualize(self, X):
        di = defaultdict(list)
        dv = defaultdict(list)
        ks = X[:, self.context_column]
        vs = np.delete(X, self.context_column, 1)

        for i, (k, v) in enumerate(zip(ks, vs)):
            di[k].append(i)
            dv[k].append(v)

        di = {k: np.array(v) for k, v in di.items()}
        dv = {k: np.array(v) for k, v in dv.items()}

        return di, dv

    # returns results of functions 'fs' of shape len(X) x len(fs)
    def _contextual_apply(self, di, dv, f):
        # apply a list of functions fs to contextualized values
        result = [(index, f(self.detectors_[key], dv[key])) for key, index in di.items()]
        # stack the values of contexts
        result = np.hstack(result).transpose()
        # sort by first column
        result = result[result[:, 0].argsort()]
        # return all but the index column
        return result[:, 1]

    def fit(self, X, y=None):
        di, self.detectors_features_ = self._contextualize(X)
        self.detectors_ = {k: deepcopy(self.base_detector_).fit(v) for k, v in self.detectors_features_.items()}
        self.decision_scores_ = self._contextual_apply(di, self.detectors_features_, lambda d, _: d.decision_scores_)

        # set required attributes
        self._process_decision_scores()
        self._set_n_classes(y)

        return self

    def decision_function(self, X):
        di, dv = self._contextualize(X)
        return self._contextual_apply(di, dv, lambda d, v: d.decision_function(v))

    def predict(self, X):
        di, dv = self._contextualize(X)
        return self._contextual_apply(di, dv, lambda d, v: d.predict(v))
