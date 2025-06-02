import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from joblib import Parallel, delayed

class LOCI:
    def __init__(self, estimator, random_state=None, loss=None, n_jobs=1):
        self.estimator = estimator
        self.random_state = random_state
        self.loss = loss
        self.n_jobs = n_jobs
        self.feature_names_ = None
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        self.feature_names_ = X.columns
        return self

    def _score_single_feature(self, j, X_test, y_test, v0):
        fname = self.feature_names_[j]
        model_j = clone(self.estimator)
        model_j.fit(self.X_train_[[fname]], self.y_train_)
        preds_j = model_j.predict(X_test[[fname]])
        vj = self.loss(y_test, preds_j)
        return fname, v0 - vj

    def score(self, X_test, y_test):
        if self.X_train_ is None:
            raise ValueError("You must call `fit` before `score`.")

        # Baseline: constant (mean) model
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(self.X_train_, self.y_train_)
        pred_dummy = dummy.predict(np.zeros((len(X_test), 1)))  # doesn't use features
        v0 = self.loss(y_test, pred_dummy)

        # LOCI: leave-one-covariate-in
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_single_feature)(j, X_test, y_test, v0)
            for j in range(X_test.shape[1])
        )

        return dict(results)
