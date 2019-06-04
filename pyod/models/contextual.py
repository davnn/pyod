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

    def fit(self, X, y=None):
        dv = defaultdict(list)
        di = defaultdict(list)
        ks = X[:, self.context_column]
        vs = np.delete(X, self.context_column, 1)

        for i, (k, v) in enumerate(zip(ks, vs)):
            di[k].append(i)
            dv[k].append(v)

        self.detectors_features_ = {k: np.array(v) for k, v in dv.items()}
        self.detectors_ = {k: deepcopy(self.base_detector_).fit(v) for k, v in self.detectors_features_.items()}

        tmp = np.array([(di[key],
                         self.detectors_[key].decision_scores_,
                         self.detectors_[key].labels_) for key in self.detectors_.keys()])

        tmp = np.hstack([*tmp]).transpose()
        tmp = tmp[tmp[:, 0].argsort()]  # sort by first column

        self.decision_scores_ = tmp[:, 1]
        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)
        self.labels_ = tmp[:, 2]
        self.threshold_ = {k: d.threshold_ for k, d in self.detectors_.items()}

        return self

    def decision_function(self, X):
        f = self._create(lambda d, v: d.decision_function(v))
        return f(X)

    def predict(self, X):
        f = self._create(lambda d, v: d.predict(v))
        return f(X)

    def predict_proba(self, X, method="linear"):
        f = self._create(lambda d, v: d.predict_proba(v, method))
        return f(X)

    def _create(self, fun):
        # Todo: Row wise prediction is slow, predict in batch for each detector

        def row_wise(row):
            key = row[self.context_column]
            val = np.array([x for i, x in enumerate(row) if i != self.context_column]).reshape(1, -1)
            result = fun(self.detectors_[key], val)[0]
            return result

        def f(X):
            return np.array(np.apply_along_axis(row_wise, 1, X))

        return f
